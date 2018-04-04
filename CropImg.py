import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
import numpy as np
import skimage
from skimage import util,segmentation,exposure,filters, morphology,measure,feature,io
from scipy import ndimage,stats,cluster,misc,spatial
from sklearn.cluster import KMeans
from sklearn.neighbors  import NearestNeighbors
import xml.etree.ElementTree as ET

import numpy as np
import cv2
import heapq
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from math import sqrt

import sys

import time


    
def cropImg (Read_img_file_Loc,Write_Cropimg_file_Loc ,  cropRange):
    for originalImgName  in os.listdir(Read_img_file_Loc) :
#        if 'tif' in originalImgName and ("registered" in originalImgName)==False :
        if '_registered.tif' in originalImgName:
            imgName = Read_img_file_Loc + '\\' + originalImgName
            original_image = io.imread(imgName)
            
            if len(original_image.shape) ==3 :
                image = original_image[cropRange[0][0]:cropRange[0][1],cropRange[1][0]:cropRange[1][1],:]     # left corder
            else:
                image = original_image[cropRange[0][0]:cropRange[0][1],cropRange[1][0]:cropRange[1][1]]     # left corder
            
            writeImgdone = cv2.imwrite( Write_Cropimg_file_Loc + '\\' + originalImgName, image)  #Convert to 16-bit uint.

            if writeImgdone == True:
                print('Generate Cropped Images for '+ originalImgName +' done!')
            else:
                print('[Caucious! ]  Generate Cropped Images  failed!!!!!!!!!!!!!!')
                

Read_img_file_Loc = ('F:\FACS-SCAN_rebeccaNIH2017Summer\Dragan50CHN\SingleSectionsFor2DImageAnalysis\R2Soma')
Write_Cropimg_file_Loc_root = Read_img_file_Loc

cropRanges = [[[9800,11000],[16000,21000]]]
for cropRange in cropRanges :
    Write_Cropimg_file_Loc =   Write_Cropimg_file_Loc_root + '\\Crop_HPC'                       # write in new folder 'Outputs'               

    if os.path.isdir(Write_Cropimg_file_Loc) == False:
        os.makedirs(Write_Cropimg_file_Loc)   
        
    cropImg (Read_img_file_Loc,Write_Cropimg_file_Loc ,  cropRange)



