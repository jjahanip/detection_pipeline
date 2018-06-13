import os
import sys
import argparse

import numpy as np
import skimage.io
import warnings
from lib.ops import write_xml, check_path
from lib.image_uitls import *
from lib.segmentation import GenerateBBoxfromSeeds
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as patches


def write_crops(save_folder, image, centers=None, bbxs=None, crop_size=[300, 300], adjust_hist=False, vis_idx=0):

    # check for subdirectories
    dir_list = os.listdir(save_folder)
    if 'imgs' not in dir_list:
        os.mkdir(os.path.join(save_folder, 'imgs'))
    if 'xmls' not in dir_list:
        os.mkdir(os.path.join(save_folder, 'xmls'))
    if 'adjusted_imgs' not in dir_list and adjust_hist:
        os.mkdir(os.path.join(save_folder, 'adjusted_imgs'))

    # crop width and height
    crop_width, crop_height = crop_size

    # get image information
    img_rows, img_cols, img_ch = image.shape                            # img_rows = height , img_cols = width

    # 1. User provided the bounding boxes -> just generate the crops
    if bbxs is not None:
        centers = np.zeros((bbxs.shape[0], 2), dtype=int)
        centers[:, 0] = np.round((bbxs[:, 0] + bbxs[:, 2]) / 2)
        centers[:, 1] = np.round((bbxs[:, 1] + bbxs[:, 3]) / 2)
    # 2. User didn't provide centers or bounding boxes -> whole image segmentation
    elif centers is None and bbxs is None:
        # to be added by generating full image segmentation
        pass

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

            # if crop was on the edges, take the whole size of crop
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
                if adjust_hist:
                    adjusted_crop = np.copy(crop_img)           # create a new array on memory for adjusted crop
                    adjusted_crop = imadjust(adjusted_crop)
                    skimage.io.imsave(os.path.join(save_folder, 'adjusted_imgs', crop_name),
                                      adjusted_crop)  # save the adjusted_image image

            # if user provides the bounding boxes
            if bbxs is not None:
                # extract bbxs in the crop
                crop_bbxs = bbxs[(bbxs[:, 0] >= j) & (bbxs[:, 0] < j + crop_width) &
                                 (bbxs[:, 2] >= j) & (bbxs[:, 2] < j + crop_width) &
                                 (bbxs[:, 1] >= i) & (bbxs[:, 1] < i + crop_height) &
                                 (bbxs[:, 3] >= i) & (bbxs[:, 3] < i + crop_height)]

                # shift the x & y values based on crop size
                crop_bbxs[:, [0, 2]] = crop_bbxs[:, [0, 2]] - j
                crop_bbxs[:, [1, 3]] = crop_bbxs[:, [1, 3]] - i
            else:
                # generate bounding boxes using segmentation
                crop_bbxs = GenerateBBoxfromSeeds(crop_img[:, :, 0], crop_centers)

            # remove bbxs with width <10 or height<10
            crop_bbxs = crop_bbxs[(crop_bbxs[:, 2] - crop_bbxs[:, 0] > 10) &
                                  (crop_bbxs[:, 3] - crop_bbxs[:, 1] > 10)]

            # remove bbxs fall out of image
            crop_bbxs = crop_bbxs[(crop_bbxs[:, 0] >= 0) &
                                  (crop_bbxs[:, 1] >= 0) &
                                  (crop_bbxs[:, 2] < crop_width) &
                                  (crop_bbxs[:, 3] < crop_height)]

            # write bounding boxes in xml file
            xml_name = str(i) + '_' + str(j) + '.xml'  # filename contains x & y coords of top left corner
            labels = ['Nucleus'] * crop_bbxs.shape[0]
            write_xml(os.path.join(save_folder, 'xmls', xml_name), crop_bbxs, labels, image_size=crop_img.shape)

            # visualize bbxs
            if crop_idx < vis_idx:
                visualize_bbxs(crop_img, centers=crop_centers, bbxs=crop_bbxs, adjust_hist=adjust_hist, save=True)

            crop_idx = crop_idx + 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', type=str, default='data/train/input_data', help='path to the directory of input images')
    parser.add_argument('--centers_file', type=str, default='data/train/input_data/centers.txt', help='path to the centers file to generate bounding boxes')
    parser.add_argument('--bbxs_file', type=str, default='data/train/input_data/bbxs.txt', help='path to the bbxs file')
    parser.add_argument('--crop_size', type=str, default='300,300', help='size of the cropped image e.g. 300,300')
    parser.add_argument('--save_dir', type=str, default='data/train', help='path to the folders of new images and xml files')
    parser.add_argument('--adjust_image', action='store_true', help='adjust histogram of image')
    parser.add_argument('--visualize', type=int, default=2, help='visualize n sample images with bbxs')
    args = parser.parse_args()

    # read images
    imgs_fname = []
    for file in os.listdir(args.images_dir):
        file_name, file_extension = os.path.splitext(file)
        if file_extension in ['.jpeg', '.jpg', '.bmp', '.tif', '.tiff', '.png']:
            imgs_fname.append(check_path(os.path.join(args.images_dir, file)))
    assert len(imgs_fname) <= 3, ('Provide no more than 3 images')
    image = read_image_from_filenames(imgs_fname)

    # read centers
    if os.path.isfile(args.centers_file):
        # read table
        centers_table = pd.read_csv(args.centers_file, sep='\t')
        # read center coordinates
        centers = centers_table[['centroid_x', 'centroid_y']].values
    else:
        centers = None
    # read bbxs
    if os.path.isfile(args.bbxs_file):
        # read table
        bbxs_table = pd.read_csv(args.bbxs_file, sep='\t')
        # read bbxs coordinates
        bbxs = bbxs_table[['xmin', 'ymin', 'xmax', 'ymax']].values
    else:
        bbxs = None

    # read crop size
    crop_size = list(map(int, args.crop_size.split(',')))

    # read path to save imgs and xmls folders
    save_folder = check_path(args.save_dir)

    # read Boolean to adjust the image histogram
    adjust_image = args.adjust_image

    # read number of images to be visualized
    vis_idx = args.visualize



    write_crops(save_folder, image, centers=centers, bbxs=bbxs, crop_size=crop_size, adjust_hist=adjust_image, vis_idx=vis_idx)
    print('Successfully created the cropped images and corresponding xml files in:\n{}\n{}'
          .format(args.save_dir+'/imgs', args.save_dir+'/xmls'))


if __name__ == '__main__':

    main()
    print()
