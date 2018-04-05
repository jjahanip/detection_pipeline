import os
import sys
import numpy as np
import skimage.io
import warnings
from lib.ops import write_xml, check_path
from lib.image_uitls import *
from lib.segmentation import GenerateBBoxfromSeeds

import matplotlib.pyplot as plt
import matplotlib.patches as patches


def write_crops(save_folder, image_filenames, centers_filename, crop_size, adjust_hist=False):

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
        if adjust_hist:
            image = imadjust(image)                    # adjust the histogram of the image
            image = np.expand_dims(image, axis=2)      # add depth dimension

    # RGB image (3 channels)
    if len(image_filenames) > 1:
        img = []
        for i, image_filename in enumerate(image_filenames):
            img.append(skimage.io.imread(image_filename))               # read each channel
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
            if crop_img.shape[:2][::-1] != crop_size:
                continue

            crop_name = str(i) + '_' + str(j) + '.jpeg'             # filename contains x & y coords of top left corner
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                skimage.io.imsave(os.path.join(save_folder, 'imgs', crop_name), np.squeeze(crop_img))   # save the image

            # generate bounding boxes using segmentation
            crop_bbxs = GenerateBBoxfromSeeds(crop_img[:, :, 0], crop_centers)

            # remove bbxs with width <10 or height<10
            crop_bbxs = crop_bbxs[(crop_bbxs[:, 0] + crop_bbxs[:, 2] > crop_width) &
                                  (crop_bbxs[:, 1] + crop_bbxs[:, 3] > crop_height)]

            # remove bbxs fall out of image
            crop_bbxs = crop_bbxs[(crop_bbxs[:, 2] > 10) & (crop_bbxs[:, 3] > 10)]

            # write bounding boxes in xml file
            xml_name = str(i) + '_' + str(j) + '.xml'  # filename contains x & y coords of top left corner
            labels = ['Nucleus'] * crop_bbxs.shape[0]
            write_xml(os.path.join(save_folder, 'xmls', xml_name), crop_bbxs, labels, image_size=crop_img.shape)

            # visualize bbxs
            # visualize_bbxs(crop_img, centers=crop_centers, bbxs=crop_bbxs)


if __name__ == '__main__':
    crop_size = (300, 300)
    save_folder = os.path.join(os.getcwd(), 'data', 'LiVPa')

    input_fnames = []
    input_fnames.append(check_path('D:\\Jahandar\\Lab\\images\\crops_for_badri_proposal\\LiVPa\\ARBc_#4_Li+VPA_37C_4110_C10_IlluminationCorrected_stitched.tif'))
    input_fnames.append(check_path('D:\\Jahandar\\Lab\\images\\crops_for_badri_proposal\\LiVPa\\ARBc_#4_Li+VPA_37C_4110_C7_IlluminationCorrected_stitched.tif'))

    centers_fname = check_path('D:\\Jahandar\\Lab\\images\\crops_for_badri_proposal\\LiVPa\\centers.txt')

    write_crops(save_folder, input_fnames, centers_fname, crop_size=crop_size, adjust_hist=True)
