# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 11:50:55 2018

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
from skimage.segmentation import clear_border
import os

def area(vs):
    
    a = 0
    x0,y0 = vs[0]
    for [x1,y1] in vs[1:]:
        dx = x1-x0
        dy = y1-y0
        a += 0.5*(y0*dx - x0*dy)
        x0 = x1
        y0 = y1
    return a

img_path='img/718 images/111-Results-2/'
res_path='img/718 images/111-Results-2/Gamma_prime_size/'

for n in range (1,15):
    int_gamma_prime= io.imread(img_path + `n` + '/DGP.png',as_grey=True)
    crop_gamma_prime= int_gamma_prime[170:1277, 433:1539]
    gamma_prime = cv2.resize(crop_gamma_prime,(1384,1384))
    

    int_double_prime= io.imread(img_path + `n` + '/GDP.png',as_grey=True)
    crop_double_prime= int_double_prime[170:1280, 433:1550]
    double_prime = cv2.resize(crop_double_prime,(1384,1384))

    gamma_prime_rest= gamma_prime-double_prime
    gamma_prime= gamma_prime_rest > 0
    gamma_prime = morphology.remove_small_objects(gamma_prime , min_size=36, connectivity=2)

    gamma_prime= clear_border(gamma_prime) 
    contours = measure.find_contours(gamma_prime, 0)


    list_circle_dia= []
    
    for i in range (0,len(contours)):     
    
        area_cnt =area(contours[i])
        area_cnt=math.sqrt(math.pow(area_cnt,2))
        r_sqr=area_cnt/math.pi
        r=math.sqrt(r_sqr)
        list_circle_dia.append(r)
        
    list_circle_dia=np.trim_zeros(list_circle_dia)  
    list_circle_dia= filter(lambda a: a != 0.0, list_circle_dia)
    list_circle_dia = np.array(list_circle_dia)
    np.savetxt(res_path+'list_circle'+`n`+'.csv', list_circle_dia)    
    plt.imshow(gamma_prime)
    plt.savefig(res_path+ 'gamma_prime'+`n`+'.png', dpi = 300)    
    
    
    
    
    
    
    
    