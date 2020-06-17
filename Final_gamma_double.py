# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 14:51:12 2018

@author: Nishan
"""


import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import cv2
from skimage.filters import threshold_otsu
from skimage import color
from scipy import ndimage
from skimage import measure
from shapely.geometry import Polygon
import math
from skimage.segmentation import clear_border

image=io.imread('img/Z18/Q937_Z18_150kx_scancorr.png', as_gray=True)
thresh = threshold_otsu(image)
binary = image < thresh

#Image height and width
img_width = image.shape[0]
img_height = image.shape[1]


ellipse=np.load('ellipse.npy')

#Creating a mask(background image)
mask=np.zeros([img_width,img_height,3],dtype=np.uint8)

#Define a list
list_ellipse=[]
for n in  range (0,len(ellipse)):
     
    
    ellipse_3d= ellipse[n]
    ellipse_3d=ellipse_3d.astype(int)
    L=ellipse_3d[:,0]
    R=ellipse_3d[:,1]
    corected_LR= np.column_stack([R,L])
    corected_LR = corected_LR.reshape(corected_LR.shape[0],1,2)
   
     
    list_ellipse.append(corected_LR)
    

img2=cv2.drawContours(mask, list_ellipse, -1, (0,255,0), -1)

img_gray=color.rgb2gray(img2)
binary= img_gray > 0.01
binary=binary.astype(int)

filled= ndimage.binary_fill_holes(binary)
filled=filled.astype(float)

def PolyArea(x, y):
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))



cleared_filled = clear_border(filled)  
contours = measure.find_contours(cleared_filled, 0)  

list_ellipse= []
for n, contour in enumerate(contours):
    contour_size=contours[n]
    area = PolyArea(contour[:, 0], contour[:, 1])
    
    eb = None

    try:
        eb = Polygon(contour_size).minimum_rotated_rectangle.boundary.coords.xy
    except:
        msg = contour_size
            # print msg

    if eb is not None:
        diag_1 = math.sqrt(math.pow(eb[1][0] - eb[1][1], 2) + math.pow(eb[0][0] - eb[0][1], 2))
        diag_2 = math.sqrt(math.pow(eb[1][1] - eb[1][2], 2) + math.pow(eb[0][1] - eb[0][2], 2))
        
        if diag_1 >= diag_2:
                aspect_ratio = diag_1 / diag_2
                max_dia=diag_1
                min_dia=diag_2
        elif diag_1 < diag_2:
                aspect_ratio = diag_2 / diag_1
                max_dia=diag_2
                min_dia=diag_1
        else:
                print("Invalid object !")
        
        
        
        list_ellipse.append(max_dia)   


list_ellipse=np.array(list_ellipse)
np.savetxt("img/Results/ellipse_dia.csv", list_ellipse)
#plt.plot(list_ellipse_area[:,1])

ellipse_area_pixel= np.sum(filled == 1.0)

ellipse_area_fraction = (float(ellipse_area_pixel) / float((img_width * img_height))) * 100


plt.imshow(cleared_filled)

np.save('GDP_array.npy',filled)


plt.savefig('img/Results/Z18/Q937_Z18_150kx_scancorr.png', dpi = 300)



