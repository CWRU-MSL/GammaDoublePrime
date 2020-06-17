# -*- coding: utf-8 -*-
"""
Created on Thu May 31 15:44:09 2018

@author: Nishan
"""
import numpy as np

import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from skimage import io
from shapely.geometry import Polygon
import math
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)

def frame_image(img, frame_width):
    b = frame_width # border size in pixel
    ny, nx = img.shape[0], img.shape[1] # resolution / number of pixels in x and y
    if img.ndim == 3: # rgb or rgba array
        framed_img = np.zeros((b+ny+b, b+nx+b, img.shape[2]))
    elif img.ndim == 2: # grayscale image
        framed_img = np.zeros((b+ny+b, b+nx+b))
    framed_img[b:-b, b:-b] = img
    return framed_img


contours_m=np.load('contours.npy', allow_pickle=True).item()
image=io.imread('img/Z18/Q937_Z18_150kx_scancorr.png',as_gray=True)
thresh = threshold_otsu(image)

binary = image < thresh
img_width = binary.shape[0]
img_height = binary.shape[1]
binary= frame_image(binary, 15)





fig, ax = plt.subplots()
list_circles = []
list_ellipse = []
list_ellipse_dia=[]
list_circle_dia=[]
xsizes_min=[]
xsizes_max=[]
ysizes_min=[]
ysizes_max=[]
def PolyArea(x, y):
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
implot = plt.imshow(binary)
plt.xlim(0,img_height,) 
plt.ylim(img_width,0) 
for n in contours_m:
        
        
        area = PolyArea(contours_m[n][:, 0], contours_m[n][:, 1])
        contour_test=contours_m[n]
        eb = None

        try:
            eb = Polygon(contour_test).minimum_rotated_rectangle.boundary.coords.xy
        except:
            msg = contour_test
            # print msg

        if eb is not None:
            
            diag_1 = math.sqrt(math.pow(eb[1][0] - eb[1][1], 2) + math.pow(eb[0][0] - eb[0][1], 2))
            diag_2 = math.sqrt(math.pow(eb[1][1] - eb[1][2], 2) + math.pow(eb[0][1] - eb[0][2], 2))

            aspect_ratio = 0
            if diag_1 >= diag_2:
                aspect_ratio = diag_1 / diag_2
                max_dia=diag_1
            elif diag_1 < diag_2:
                aspect_ratio = diag_2 / diag_1
                max_dia=diag_2
            else:
                print("Invalid object !")

            if aspect_ratio > 1.9:
                if aspect_ratio <10:
                    list_ellipse.append([contour_test, aspect_ratio, area])
                    list_ellipse_dia.append([max_dia])
                # print "{n}  aspect ratio: {ap}, area: {area}, ellipse".format(n=n, ap=aspect_ratio, area=area)
                    ax.plot(contour_test[:, 1], contour_test[:, 0], linewidth=2, color='red')


            else:
                list_circles.append([contour_test, aspect_ratio, area])
#                xsizes_min.append(min(contour_test[:, 0]))
 #               xsizes_max.append(max(contour_test[:, 0]))
  #              ysizes_min.append(min(contour_test[:, 1]))
   #             ysizes_max.append(max(contour_test[:, 1]))
                if not(max(contour_test[:, 0]) >= img_width or max(contour_test[:, 1]) >= img_width or min(contour_test[:, 0]) <= 0 or min(contour_test[:, 1]) <= 0):
                    list_circle_dia.append([max_dia])
                #if not(any(contour_test[:, 0])<= 0.0 or any(contour_test[:, 1])<= 0.0 or any(contour_test[:, 0])>= float(img_width) or any(contour_test[:, 1])>= float (img_width)  ) :
                 #   list_circle_dia.append([max_dia])
                    
                # print "{n}  aspect ratio: {ap}, area: {area}, circle".format(n=n, ap=aspect_ratio, area=area)
                ax.plot(contour_test[:, 1], contour_test[:, 0], linewidth=2, color='blue')
                
                
                
#for n in list_ellipse:
# if      

plt.savefig('img/Results/Z18/Q937_Z18_150kx_scancorr.png', dpi = 300)
list_ellipse_cnt=[]
list_circle_cnt=[]
for n in  range (0,len(list_ellipse)):
    list_ellipse_cnt.append(list_ellipse[n][0])



for n in  range (0,len(list_circles)):
    list_circle_cnt.append(list_circles[n][0])    
   
np.save('ellipse.npy',list_ellipse_cnt)
np.save('circle.npy',list_circle_cnt)

#np.savetxt("img/718 images/results/3 - test/ellipse_dia.csv", list_ellipse_dia)
np.savetxt("img/Results/circle_dia.csv", list_circle_dia)


       


         
