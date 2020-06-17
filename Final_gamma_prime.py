# -*- coding: utf-8 -*-
"""
Created on Wed Jul 04 13:13:10 2018

@author: Nishan
"""

import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import cv2
from skimage.filters import threshold_otsu
from skimage import morphology,color
import skimage.morphology, skimage.data
from scipy import ndimage
from skimage import measure
from skimage.segmentation import clear_border





image=io.imread('img/Z18/Q937_Z18_150kx_scancorr.png',as_gray=True)
thresh = threshold_otsu(image)
binary = image < thresh
img_width = image.shape[0]
img_height = image.shape[1]


circle=np.load('circle.npy')

mask=np.zeros([img_width,img_height,3],dtype=np.uint8)

list_circle=[]
for n in  range (0,len(circle)):
     
    
    circle_3d= circle[n]
    circle_3d=circle_3d.astype(int)
    L=circle_3d[:,0]
    R=circle_3d[:,1]
    corected_LR= np.column_stack([R,L])
    corected_LR = corected_LR.reshape(corected_LR.shape[0],1,2)
   
     
    list_circle.append(corected_LR)
    



img2=cv2.drawContours(mask, list_circle, -1, (0,255,0), -1)

img_gray=color.rgb2gray(img2)
binary= img_gray > 0.01
binary=binary.astype(int)
filled= ndimage.binary_fill_holes(binary)
filled=filled.astype(float)
cleared_filled = clear_border(filled)

def PolyArea(x, y):
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))



contours = measure.find_contours(cleared_filled, 0) 
# 
#
#list_circle_area = []
#for n, contour in enumerate(contours):
#    area = PolyArea(contour[:, 0], contour[:, 1])
#    list_circle_area.append([n+1,area])
#    
#
#list_circle_area=np.array(list_circle_area)


plt.imshow(filled)

np.save('GP_array.npy',filled)

plt.savefig('img/Results/Z18/Q937_Z18_150kx_scancorr.png', dpi = 300)
