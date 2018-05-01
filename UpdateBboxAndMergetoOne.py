# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 18:12:43 2018

Assemble back to whole brain
4) Update seeds and run watershed segmentation again


@author: xli63
"""


#%% Assemble the cropped images to big ones  and the updated bbox t

import os

#os.chdir(r'D:\research in lab\NIHIntern(new)\RebeccaCode')  # set current working directory
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import numpy as np
import skimage
from skimage import util,segmentation,exposure,filters, morphology,measure,feature,io,data
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


import CellSegmentationfcts as myfcts                                          # my functions


#%%
#def readBBoxXML (XML_dir):        
    
#%%    

def MergeUpdatedBbox(XML_dir,updatedTable_dir):
    for f_ID , npyFileName in enumerate (  os.listdir(XML_dir) ) :   
        # load X corner, Y corner            
    #    Y_cor  = int (npyFileName.split(']')[0].split('[')[1].split(' ')[-2] ) 
    #    X_cor  = int (npyFileName.split('[')[2].split('.')[0].split(' ')[-2] ) 
        
        Y_cor  = int (npyFileName.split('_')[0]) 
        X_cor  = int (npyFileName.split('_')[1].split('.')[0]) 
        
        tree = ET.parse( XML_dir +'./'+ npyFileName)    
        size = tree.find('size')
        imgWidth  = str ( size.find('width' ).text ) 
        imgHeight = str ( size.find('height').text ) 
    
        center_array = np.zeros( (len(tree.findall('object')),2 ) ,dtype = int)
        bbox_array   = np.zeros( (len(tree.findall('object')),4 ) ,dtype = int)
    
        for i, Obj in  enumerate  (tree .findall('object')):                            # take the current animal 
            bndbox = Obj.find('bndbox')
            xmin = int ( bndbox.find('xmin').text )
            ymin = int ( bndbox.find('ymin').text )
            xmax = int ( bndbox.find('xmax').text ) 
            ymax = int ( bndbox.find('ymax').text ) 
            bbox_array  [i,:] = [xmin,ymin,xmax,ymax]
            center = np.array(  [ (xmin +xmax )/2 + X_cor, int (ymin +ymax )/2  + Y_cor],dtype = int) 
            center_array[i,:] = center
            
        if f_ID == 0 :
            center_array_cat = center_array
            bbox_array_cat   = bbox_array
        else:
            center_array_cat = np.concatenate ( ( center_array_cat,center_array ) ,axis = 0)
            bbox_array_cat   = np.concatenate ( ( bbox_array_cat  ,bbox_array ) ,axis = 0)
    
    #plt.scatter(center_array_cat[:,0],center_array_cat[:,1])
    fullTable = np.concatenate ( ( center_array_cat,bbox_array_cat ) ,axis = 1)
    # Write into txt file
    
    featureTableFilename = updatedTable_dir + './' +  'updatedFeatureTable.txt'
    header = (       'centroid_x' + '\t' + 'centroid_y' + 
                       '\t' + 'xmin	'  + '\t' + 'ymin	' + 
                       '\t' + 'xmax	'  + '\t' + 'ymax' + '\t')
    np.savetxt(featureTableFilename,fullTable,fmt='%.18g', delimiter=' ', header = header)


if __name__ == "__main__":
    
    XML_dir =  r'F:\FACS-SCAN_rebeccaNIH2017Summer\Dragan50CHN\SingleSectionsFor2DImageAnalysis\HPC_whole\crops\bbox_dir_Corrected'
    updatedTable_dir = os.path.dirname(XML_dir) 
    MergeUpdatedBbox(XML_dir,updatedTable_dir)
 

        

    
        
        