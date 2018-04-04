from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import logging
import os

import tensorflow as tf
from lib.object_detection.utils import dataset_util

import cv2
import numpy as np
import scipy.io as sio

from lib.image_uitls import imadjust
import matplotlib.pyplot as plt
import matplotlib.patches as patches

'''BIOMARKER , CHANNEL  , CLASS 
 order :   DAPI  , 4110_C10 , 0
       S100   , 4110_C6  , 1
       RECA1  , 4110_C5  , 2
       NeuN   , 4110_C7  , 3
       IBA1   , 4110_C8  , 4
'''

# crop_size = [500, 700]

# def imReadAdjust(image_name):
#     image   = cv2.imread(image_name, -1)                  # read image
#     image = imadjust(image)                               # adjust histogram of image
#     image = (image/256).astype('uint8')                   # convert to 8bit image
#     return image
#
#
# def importBbxClass(mat_filename):
#     mat_file = sio.loadmat(mat_filename)                    # load file from disk
#     bbxs = mat_file["bbxs"]                                 # extract the bounding boxes
#     bbxs[:,0:2] -= 1                                        # matlab starst from 1 and python from 0
#     classes = mat_file["classes"]                           # extract the classes
#     classes = [ np.squeeze(elem) for elem in classes[:, 0]] # squeeze the classes from array to element
#     classes = np.array(classes, dtype=object)               # cast to numpy array
#     return bbxs, classes
#
#
# def crop (images, bbxs, classes, crop_size):
#     imWidth, imHeight = images[0].shape                                 # get width and height of original image
#     width, height = crop_size                                           # get width and height of crop image
#
#     for i in range(imHeight // height):
#         for j in range(imWidth // width):
#             crop_images = []                                            # empty array to save the crop images
#             for im in range(len(images)):
#                 crop_images.append( images[im][j*width:(j+1)*width,     # crop the original image
#                                                i*height:(i+1)*height])
#
#         # get the bounding boxes in the crop
#             crop_bbxs = bbxs[ (bbxs[:, 0] >= (i * height)) &
#                               (bbxs[:, 0] + bbxs[:, 2] - 1 < ((i + 1) * height)) &
#                               (bbxs[:, 1] >= (j * width)) &
#                               (bbxs[:, 1] + bbxs[:, 3] - 1 < ((j + 1) * width))]
#
#         # reset x,y-coord of bounding box to new crop
#             crop_bbxs[:, 0]  -= (i * height)
#             crop_bbxs[:, 1]  -= (j * width)
#
#         # get the class of samples in the crop
#             crop_classes = classes[ (bbxs[:, 0] >= (i * height)) &
#                             (bbxs[:, 0] < ((i + 1) * height)) &
#                             (bbxs[:, 1] >= (j * width)) &
#                             (bbxs[:, 1] < ((j + 1) * width))]
#
#             yield crop_images, crop_bbxs, crop_classes
            
    
def generate_tf_example(crop_images, crop_bboxes, crop_classes):
    
    # stack the image
    image_CH0 = crop_images[0].reshape((crop_size[0],crop_size[1],1))
    image_CH1 = crop_images[1].reshape((crop_size[0],crop_size[1],1))
    image_CH2 = crop_images[2].reshape((crop_size[0],crop_size[1],1))
    image_CH3 = crop_images[3].reshape((crop_size[0],crop_size[1],1))
    image_CH4 = crop_images[4].reshape((crop_size[0],crop_size[1],1))
    inputs_stacked = np.concatenate([image_CH0, image_CH1, image_CH2, image_CH3, image_CH4], axis = -1)
    encoded_inputs = inputs_stacked.tostring()
    
    width = int(crop_size[1])
    height = int(crop_size[0])
    
    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    truncated = []
    poses = []
    difficult_obj = []

    
    number_object = len(crop_bboxes)
    for i in range(0, number_object):
        object0 = crop_bboxes[i]
            
        xmin.append(float(object0[0])/width)
        xmax.append(float(object0[0]+object0[2]-1)/width)
        ymin.append(float(object0[1])/height)
        ymax.append(float(object0[1]+object0[3]-1)/height)
    
        temp = crop_classes[i]
        if temp.size == 1:
            difficult_obj.append(int(0))
            classes_text.append(str(temp+1).encode('utf8'))
            classes.append(int(temp+1))
        else:
            difficult_obj.append(int(1))
            random_pick = int(np.random.choice(temp.size,1))
            classes_text.append(str(temp[random_pick]+1).encode('utf8'))
            classes.append(int(temp[random_pick]+1))
        
        truncated.append(int(0))
        poses.append('Unspecified'.encode('utf8'))

    filename_example = 'Unspecified'
        
    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename_example.encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(filename_example.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_inputs),
        'image/channels': dataset_util.int64_feature(5),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
        'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
        'image/object/truncated': dataset_util.int64_list_feature(truncated),
        'image/object/view': dataset_util.bytes_list_feature(poses),}))
    
    return example


def main(_):
    
    # initialize the process
    logging.info('Start creating tfRecord ...')
    tfrecords_filename = './data/input_data.record'
    
    writer = tf.python_io.TFRecordWriter(tfrecords_filename)    

    for crop_num in range(crop_nums):
        crop_images, crop_bbxs, crop_classes = next(crop_gen)
        # visualize(crop_images[0], crop_bbxs)
        print('writing Vehicle crop image ' + str(crop_num).zfill(5) + '...')
        tf_example = generate_tf_example(crop_images, crop_bbxs, crop_classes)
        writer.write(tf_example.SerializeToString())

    writer.close()

    # input_folder = 'D:\Jahandar\Lab\research\codes\git\detection_pipeline\data\LiVPa\original_images'
    # ##### Part 1: Dataset Vehicle######################################################################################
    # image_names = [ './data/vehicle/ARBc_FPI#6_Vehicle_20C_4110_C10_IlluminationCorrected_stitched.tif',
    #             './data/vehicle/ARBc_FPI#6_Vehicle_20C_4110_C6_IlluminationCorrected_stitched.tif',
    #             './data/vehicle/ARBc_FPI#6_Vehicle_20C_4110_C5_IlluminationCorrected_stitched.tif',
    #             './data/vehicle/ARBc_FPI#6_Vehicle_20C_4110_C7_IlluminationCorrected_stitched.tif',
    #             './data/vehicle/ARBc_FPI#6_Vehicle_20C_4110_C8_IlluminationCorrected_stitched.tif'
    #           ]
    # mat_filename = './data/vehicle/bbxsNclasses.mat'
    #
    # # read all images
    # images = []
    # for image_name in image_names:
    #     images.append(imReadAdjust(image_name))      # read the images
    # # read information from bounding boxes and classes
    # bbxs, classes = importBbxClass(mat_filename)        # read the bounding boxes and classes
    #
    # crop_gen = crop(images, bbxs, classes, crop_size)   # generator for the croping
    # crop_nums = (images[0].shape[0]//crop_size[0]) * (images[0].shape[1]//crop_size[1]) # number of crops in original image

    
    # for crop_num in range(crop_nums):
    #     crop_images, crop_bbxs, crop_classes = next(crop_gen)
    #     # visualize(crop_images[0], crop_bbxs)
    #     print('writing Vehicle crop image ' + str(crop_num).zfill(5) + '...')
    #     tf_example = generate_tf_example(crop_images, crop_bbxs, crop_classes)
    #     writer.write(tf_example.SerializeToString())
        


if __name__ == '__main__':
    tf.app.run()

