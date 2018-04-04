import os
import sys
import numpy as np
import skimage.io
from lib.ops import write_xml, check_path
from lib.image_uitls import imadjust
from lib.segmentation import GenerateBBoxfromSeeds

import matplotlib.pyplot as plt
import matplotlib.patches as patches

# from skimage.feature import blob_log
# from skimage import segmentation, filters, morphology
# from scipy import ndimage, spatial
# import cv2


def write_crops(image_filenames, centers_filename, crop_size, adjust_hist=False):

    save_folder = os.path.join(os.getcwd(), 'data', 'LiVPa', 'patches')

    # grayscale image (1 channel)
    if len(image_filenames) == 1:
        image = skimage.io.imread(image_filenames[0])  # read single channel image
        if adjust_hist:
            image = imadjust(image)                    # adjust the histogram of the image
        img_rows, img_cols = image.shape               # img_rows = height , img_cols = width

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
        img_rows, img_cols, img_ch = image.shape                        # img_rows = height , img_cols = width

    centers = np.loadtxt(centers_filename, skiprows=1)                  # load feature table
    centers = centers[:, 1:3].astype(int) - 1                           # extract centers

    crop_width, crop_height = crop_size                  # crop width and height

    for i in range(0, img_rows, crop_height):
        for j in range(0, img_cols, crop_width):
            crop_img = image[i:crop_height + i, j:crop_width + j]   # create crop image
            file_name = str(i) + '_' + str(j) + '.jpeg'             # filename contains x & y coords of top left corner
            skimage.io.imsave(os.path.join(save_folder, file_name), crop_img)   # save the image

            # extract centers of the cells in the crop
            crop_centers = centers[(centers[:, 0] >= j) & (centers[:, 0] < j+crop_width) &
                                   (centers[:, 1] >= i) & (centers[:, 1] < i+crop_height)]

            # shift the x & y values based on crop size
            crop_centers[:, 0] = crop_centers[:, 0] - j
            crop_centers[:, 1] = crop_centers[:, 1] - i

            # generate bounding boxes using segmentation
            if len(image_filenames) == 1:
                crop_bbxs = GenerateBBoxfromSeeds(crop_img, crop_centers)
            else:
                crop_bbxs = GenerateBBoxfromSeeds(crop_img[:, :, 0], crop_centers)

            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.imshow(np.divide(crop_img, 256).astype(np.uint8))
            ax.plot(crop_centers[:, 0], crop_centers[:, 1], 'b.')
            for idx in range(crop_bbxs.shape[0]):
                ax.add_patch(patches.Rectangle((crop_bbxs[idx, 0], crop_bbxs[idx, 1]),
                                               crop_bbxs[idx, 2], crop_bbxs[idx, 3],
                                               edgecolor="blue", fill=False))

                print(crop_bbxs[idx, :])


    # img = cv2.imread(os.path.join(input_image, cropImgName), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    # bin_mask = img >filters.threshold_otsu(img)   #generate binary mask from original image
    # D = ndimage.distance_transform_edt(bin_mask)  # generate distant map, centrois locates in peaks
    # D = morphology.erosion(D,morphology.disk(3))
    #
    # #1) use blob_LoG to find the centroids in the window
    # blobRadius_min = 2
    # blobRadius_max = 6
    # blob_radius_range_pixel = np.array([blobRadius_min, blobRadius_max])
    # blob_radius_range = blob_radius_range_pixel /1.414     #  radius approximate root 2 * sigma
    # blobs = skimage.feature.blob_log(img,
    #                                  min_sigma = 3,
    #                                  max_sigma = blob_radius_range[1] ,
    #                                  num_sigma = 20,
    #                                  threshold = 0.01 , overlap= 0.5
    #                                 )
    #
    #
    # seed_centroidImg = np.zeros_like(img)
    # for i,(x,y) in enumerate( zip( np.uint(blobs[:,0]), np.uint(blobs[:,1]) ) ):
    #     seed_centroidImg[x,y] = (i+1)
    # kernel = morphology.disk(4)
    # marker = morphology.dilation (seed_centroidImg,kernel)    # make sure there is no connected components
    #
    # # Implement Watershed
    # labels = morphology.watershed(-D, marker, mask=bin_mask)  # labeled components, background = 0, other= ID
    #
    # #find the label ID of the center components
    # centerImg = np.array( [np.uint(img.shape[0]/2),np.uint(img.shape[1]/2)])
    # centerLabelID = labels[centerImg[0],centerImg[1]]
    # if centerLabelID == 0 :  # just in case the center component does't cover the center point of img
    #     Y = spatial.distance.cdist(blobs[:,0:2], [centerImg], 'euclidean')
    #     centerLabelID = np.argmin(Y) + 1
    #
    # singleCellMask = (labels==centerLabelID)
    #
    # singleCellMask = morphology.opening(singleCellMask,morphology.disk(3))
    # singleCellMask = morphology.binary_dilation(singleCellMask,morphology.disk(3))
    # border = segmentation.find_boundaries(singleCellMask)
    #
    #
    # saveImgName = os.path.join(saveFile_Loc, os.path.basename(cropImgName))
    # saveImg  = singleCellMask*img
    # cv2.imwrite(saveImgName, saveImg)
    a=1


if __name__ == '__main__':

    input_fnames = []
    input_fnames.append(check_path('D:\\Jahandar\\Lab\\images\\crops_for_badri_proposal\\LiVPa\\ARBc_#4_Li+VPA_37C_4110_C10_IlluminationCorrected_stitched.tif'))
    input_fnames.append(check_path('D:\\Jahandar\\Lab\\images\\crops_for_badri_proposal\\LiVPa\\ARBc_#4_Li+VPA_37C_4110_C7_IlluminationCorrected_stitched.tif'))

    centers_fname = check_path('D:\\Jahandar\\Lab\\images\\crops_for_badri_proposal\\LiVPa\\centers.txt')

    write_crops(input_fnames, centers_fname, (300, 200), adjust_hist=True)
