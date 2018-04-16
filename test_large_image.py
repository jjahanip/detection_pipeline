import os
import sys
import argparse
import tensorflow as tf

import numpy as np
import skimage, skimage.io

from collections import defaultdict
from io import StringIO
from PIL import Image

# Add necessary paths
sys.path.append("lib")
sys.path.append("lib/object_detection")
sys.path.append("lib/slim")

from lib.object_detection.utils import ops as utils_ops
from lib.ops import check_path
from lib.image_uitls import *
from lib.object_detection.utils import label_map_util
from lib.object_detection.utils import visualization_utils as vis_util

import matplotlib.pyplot as plt
import matplotlib.patches as patches


def get_crop(image_filenames, crop_size=[300, 300], adjust_hist=False, vis_idx=0):

    # crop width and height
    crop_width, crop_height = crop_size


    # grayscale image (1 channel)
    if len(image_filenames) == 1:
        image = skimage.io.imread(image_filenames[0])   # read single channel image
        image = skimage.img_as_ubyte(image)             # cast to 8-bit
        if adjust_hist:
            image = imadjust(image)                     # adjust the histogram of the image
            image = np.expand_dims(image, axis=2)       # add depth dimension

    # RGB image (3 channels)
    if len(image_filenames) > 1:
        img = []
        for i, image_filename in enumerate(image_filenames):
            im_ch = skimage.io.imread(image_filename)        # read each channel
            im_ch = skimage.img_as_ubyte(im_ch)              # cast to 8-bit
            img.append(im_ch)
            if adjust_hist:
                img[i] = imadjust(img[i])                    # adjust the histogram of the image
        if len(image_filenames) == 2:                        # if two channels were provided
            img.append(np.zeros_like(img[0]))                # set third channel to zero
        image = np.stack((im for im in img), axis=2)         # change to np array rgb image

    # get image information
    img_rows, img_cols, img_ch = image.shape  # img_rows = height , img_cols = width

    # get each crop
    ovrlp = 50
    for i in range(0, img_rows, crop_height - ovrlp):
        for j in range(0, img_cols, crop_width - ovrlp):

            # crop the image
            crop_img = image[i:i + crop_height, j:j + crop_width, :]   # create crop image
            # if we were at the edges of the image
            # crop new image with the size of crop from the end of image
            if crop_img.shape[:2][::-1] != tuple(crop_size):
                # if both dims are at the end
                if np.all(crop_img.shape[:2][::-1] != np.array(crop_size)):
                    crop_img = image[-crop_height:, -crop_width:, :]
                    i = img_rows - crop_height
                    j = img_cols - crop_width
                # if xdim is at the end
                if crop_img.shape[:2][::-1][0] != tuple(crop_size)[0]:
                    crop_img = image[i:i + crop_height, -crop_width:, :]
                    j = img_cols - crop_width
                # if ydim is at the end
                if crop_img.shape[:2][::-1][1] != tuple(crop_size)[1]:
                    crop_img = image[-crop_height:, j:j + crop_width, :]
                    i = img_rows - crop_height

            yield [j, i], crop_img


def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict,
                                   feed_dict={image_tensor: np.expand_dims(image, 0)})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
                'detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict

def non_max_suppression_fast(boxes, overlapThresh):
    # Malisiewicz et al.
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='data/test', help='path to the large input images direcotry')
    parser.add_argument('--crop_size', type=str, default='300,300', help='size of the cropped image e.g. 300,300')
    parser.add_argument('--model_dir', type=str, default='freezed_model1', help='path to exported model directory')
    parser.add_argument('--output_file', type=str, default='bbxs.txt', help='path to txt file of coordinates')
    parser.add_argument('--labels_file', type=str, default='data/nucleus_map.pbtxt', help='path to label map pbtxt file')
    parser.add_argument('--num_classes', type=int, default=1, help='number of the classes')
    parser.add_argument('--adjust_image', action='store_true',  help='adjust histogram of image')
    parser.add_argument('--visualize_crop', type=int, default=0, help='visualize n sample images with bbxs')

    args = parser.parse_args()

    # read input
    input_fnames = []
    for file in os.listdir(args.input_dir):
        file_name, file_extension = os.path.splitext(file)
        if file_extension in ['.jpeg', '.jpg', '.bmp', '.tif', '.tiff', '.png']:
            input_fnames.append(check_path(os.path.join(args.input_dir, file)))
    assert len(input_fnames) <= 3, ('Provide no more than 3 images')

    # read crop size
    crop_size = list(map(int, args.crop_size.split(',')))

    # read Boolean to adjust the image histogram
    adjust_image = args.adjust_image
    adjust_image = True

    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    PATH_TO_CKPT = check_path(args.model_dir + '/frozen_inference_graph.pb')

    # # List of the strings that is used to add correct label for each box.
    # PATH_TO_LABELS = check_path(args.labels_file)
    #
    # NUM_CLASSES = args.num_classes

    # read number of images to be visualized
    vis_idx = args.visualize_crop

    # load frozen model to memory
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    # # load label map
    # label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    # categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
    #                                                         use_display_name=True)
    # category_index = label_map_util.create_category_index(categories)


    bbxs = []

    # for each crop:
    crop_idx = 0
    for corner, crop in get_crop(input_fnames, crop_size=crop_size, adjust_hist=adjust_image):
        output_dict = run_inference_for_single_image(crop, detection_graph)
        keep_boxes = output_dict['detection_scores'] > .5
        detection_boxes = output_dict['detection_boxes'][keep_boxes]
        # detection_scores = output_dict['detection_scores'][keep_boxes]

        crop_bbxs = []
        for box in detection_boxes:
            box = box.tolist()
            ymin, xmin, ymax, xmax = box

            crop_bbxs.append([xmin * crop_size[0],
                              ymin * crop_size[1],
                              (xmax - xmin) * crop_size[0],
                              (ymax - ymin) * crop_size[1]])

            xmin = np.rint(xmin * crop_size[0] + corner[0]).astype(int)
            xmax = np.rint(xmax * crop_size[0] + corner[0]).astype(int)
            ymin = np.rint(ymin * crop_size[1] + corner[1]).astype(int)
            ymax = np.rint(ymax * crop_size[1] + corner[1]).astype(int)

            # append each bbx to the list
            bbxs.append([xmin, ymin, xmax, ymax])


        # visualize bbxs
        if crop_idx < vis_idx:
            visualize_bbxs(crop, bbxs=np.array(crop_bbxs))
        crop_idx = crop_idx + 1

    # remove overlapping bounding boxes due to cropping with overlap
    new_bbxs = non_max_suppression_fast(np.array(bbxs), .5)

    # save to the file
    np.savetxt(args.output_file, new_bbxs, fmt='%d,%d,%d,%d', header='xmin,ymin,xmax,ymax')



if __name__ == '__main__':

    main()
    print()
