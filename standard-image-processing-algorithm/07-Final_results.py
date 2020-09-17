# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 12:19:27 2018

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
from shapely.geometry import Polygon
import math
from config import *



image=io.imread(IMAGE_PATH,as_gray=True)
thresh = threshold_otsu(image)
binary = image < thresh
img_width = image.shape[0]
img_height = image.shape[1]

def PolyArea(x, y):
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

gamma_double=np.load(TEMP_PATH + 'GDP_array.npy')
gamma_prime=np.load(TEMP_PATH + 'GP_array.npy')

gamma_prime= gamma_prime-gamma_double

gamma_prime= gamma_prime > 0

gamma_prime = morphology.remove_small_objects(gamma_prime , min_size=36, connectivity=2)

binary= binary.astype(int)

contours = measure.find_contours(gamma_prime, 0)  


list_circle= []
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
        
        list_circle.append([area,max_dia, min_dia])

list_circle=np.array(list_circle)

circle_area_pixel= np.sum(gamma_prime == 1.0)

circle_area_fraction = (float(circle_area_pixel) / float((img_width * img_height))) * 100


gamma_prime= gamma_prime.astype(float)
gamma_prime = skimage.color.gray2rgb(gamma_prime, alpha=None)
gamma_double = skimage.color.gray2rgb(gamma_double, alpha=None)




gamma_prime[np.where((gamma_prime == [1,1,1]).all(axis = 2))] = [1,0,0]


gamma_double[np.where((gamma_double == [1,1,1]).all(axis = 2))] = [0,1,0]

final = gamma_prime+gamma_double

plt.imshow(final)
plt.savefig(SAVE_FIG_PATH + '07-Final_results.png', dpi = 300)



