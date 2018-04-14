import os
import sys
import argparse

import numpy as np
import skimage.io
import warnings
from lib.ops import write_xml, check_path
from lib.image_uitls import *
from lib.segmentation import GenerateBBoxfromSeeds

import matplotlib.pyplot as plt
import matplotlib.patches as patches

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, default='data/input_data', help='path to the directory of input images and centers file')
parser.add_argument('--crop_size', type=str, default='1000,1000', help='size of the cropped image e.g. 300,300')
parser.add_argument('--save_dir', type=str, default='data/', help='path to the folders of new images and xml files')
parser.add_argument('--adjust_image', action='store_true', help='adjust histogram of image')
parser.add_argument('--visualize', type=int, default=0, help='visualize n sample images with bbxs')
args = parser.parse_args()


def write_crops(save_folder, image_filenames, centers_filename, crop_size=[300, 300], adjust_hist=False, vis_idx=0):

    # crop width and height
    crop_width, crop_height = crop_size

    # check for subdirectories
    dir_list = os.listdir(save_folder)
    if 'imgs' not in dir_list:
        os.mkdir(os.path.join(save_folder, 'imgs'))
    if 'xmls' not in dir_list:
        os.mkdir(os.path.join(save_folder, 'xmls'))

    # grayscale image (1 channel)
    if len(image_filenames) == 1:
        image = skimage.io.imread(image_filenames[0])  # read single channel image
        image = skimage.img_as_ubyte(image)            # cast to 8-bit
        if adjust_hist:
            image = imadjust(image)                    # adjust the histogram of the image
            image = np.expand_dims(image, axis=2)      # add depth dimension

    # RGB image (3 channels)
    if len(image_filenames) > 1:
        img = []
        for i, image_filename in enumerate(image_filenames):
            im_ch = skimage.io.imread(image_filename)     # read each channel
            im_ch = skimage.img_as_ubyte(im_ch)           # cast to 8-bit
            img.append(im_ch)
            if adjust_hist:
                img[i] = imadjust(img[i])                               # adjust the histogram of the image
        if len(image_filenames) == 2:                                   # if two channels were provided
            img.append(np.zeros_like(img[0]))                           # set third channel to zero
        image = np.stack((im for im in img), axis=2)                    # change to np array rgb image

    # get image information
    img_rows, img_cols, img_ch = image.shape                            # img_rows = height , img_cols = width

    # load centers
    centers = np.loadtxt(centers_filename, skiprows=1)                  # load feature table
    centers = centers[:, 1:3].astype(int) - 1                           # extract centers

    # for each crop:
    crop_idx = 0
    for i in range(0, img_rows, crop_height):
        for j in range(0, img_cols, crop_width):
            # extract centers of the cells in the crop
            crop_centers = centers[(centers[:, 0] >= j) & (centers[:, 0] < j + crop_width) &
                                   (centers[:, 1] >= i) & (centers[:, 1] < i + crop_height)]
            if crop_centers.size == 0:                      # if no cell in the crop, SKIP
                continue

            # shift the x & y values based on crop size
            crop_centers[:, 0] = crop_centers[:, 0] - j
            crop_centers[:, 1] = crop_centers[:, 1] - i

            # crop the image
            crop_img = image[i:crop_height + i, j:crop_width + j]   # create crop image
            if crop_img.shape[:2][::-1] != tuple(crop_size):
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
                crop_centers = centers[(centers[:, 0] >= j) & (centers[:, 0] < j + crop_width) &
                                       (centers[:, 1] >= i) & (centers[:, 1] < i + crop_height)]
                # shift the x & y values based on crop size
                crop_centers[:, 0] = crop_centers[:, 0] - j
                crop_centers[:, 1] = crop_centers[:, 1] - i

            crop_name = str(i) + '_' + str(j) + '.jpeg'             # filename contains x & y coords of top left corner
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                skimage.io.imsave(os.path.join(save_folder, 'imgs', crop_name), np.squeeze(crop_img))   # save the image

            # generate bounding boxes using segmentation
            crop_bbxs = GenerateBBoxfromSeeds(crop_img[:, :, 0], crop_centers)

            # remove bbxs with width <10 or height<10
            crop_bbxs = crop_bbxs[(crop_bbxs[:, 2] > 10) & (crop_bbxs[:, 3] > 10)]

            # remove bbxs fall out of image
            crop_bbxs = crop_bbxs[(crop_bbxs[:, 0] >= 0) & (crop_bbxs[:, 1] >= 0) &
                                  (crop_bbxs[:, 0] + crop_bbxs[:, 2] < crop_width) &
                                  (crop_bbxs[:, 1] + crop_bbxs[:, 3] < crop_height)]

            # write bounding boxes in xml file
            xml_name = str(i) + '_' + str(j) + '.xml'  # filename contains x & y coords of top left corner
            labels = ['Nucleus'] * crop_bbxs.shape[0]
            write_xml(os.path.join(save_folder, 'xmls', xml_name), crop_bbxs, labels, image_size=crop_img.shape)

            # visualize bbxs
            if crop_idx < vis_idx:
                visualize_bbxs(crop_img, centers=crop_centers, bbxs=crop_bbxs)

            crop_idx = crop_idx + 1


def main():

    # read input
    input_fnames = []
    for file in os.listdir(args.input_dir):
        file_name, file_extension = os.path.splitext(file)
        if file_extension in ['.jpeg', '.jpg', '.bmp', '.tif', '.tiff', '.png']:
            input_fnames.append(check_path(os.path.join(args.input_dir, file)))
        if file_extension == '.txt':
            centers_fname = check_path(os.path.join(args.input_dir, file))
    assert len(input_fnames) <= 3, ('Provide no more than 3 images')

    # read centers file
    # centers_fname = args.centers_file

    # read crop size
    crop_size = list(map(int, args.crop_size.split(',')))

    # read path to save imgs and xmls folders
    save_folder = check_path(args.save_dir)

    # read Boolean to adjust the image histogram
    adjust_image = args.adjust_image

    # read number of images to be visualized
    vis_idx = args.visualize



    write_crops(save_folder, input_fnames, centers_fname, crop_size=crop_size, adjust_hist=adjust_image, vis_idx=vis_idx)
    print('Successfully created the cropped images and corresponding xml files in:\n{}\n{}'
          .format(args.save_dir+'/imgs', args.save_dir+'/xmls'))


if __name__ == '__main__':

    main()
    print()
