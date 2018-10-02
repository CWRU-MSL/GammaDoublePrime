# -*- coding: utf-8 -*-
"""
Created on Mon May 28 19:12:36 2018

@author: Nishan
"""

from matplotlib import pyplot as plt
from skimage import io
import numpy as np
import math
from skimage.filters import threshold_otsu
from scipy.linalg import eigh,eig
from skimage.restoration import (denoise_tv_chambolle, denoise_bilateral,
                                 denoise_wavelet, estimate_sigma)



contourevidence=np.load('contourest.npy').item()

image=io.imread('img/718 images/111-png/14.png',as_grey=True)
thresh = threshold_otsu(image)
binary = image < thresh
binary = binary.astype(int)



def frame_image(img, frame_width):
    b = frame_width # border size in pixel
    ny, nx = img.shape[0], img.shape[1] # resolution / number of pixels in x and y
    if img.ndim == 3: # rgb or rgba array
        framed_img = np.zeros((b+ny+b, b+nx+b, img.shape[2]))
    elif img.ndim == 2: # grayscale image
        framed_img = np.zeros((b+ny+b, b+nx+b))
    framed_img[b:-b, b:-b] = img
    return framed_img

binary= frame_image(binary, 15)

def PolyArea(x, y):
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

#X=xe_i
#Y=ye_i
def mia_fitellip_lsf(xe_i,ye_i):
    
    X=xe_i
    Y=ye_i
    mx = np.mean(X)
    my = np.mean(Y)
    sx = (max(X)-min(X))/2.0
    sy = (max(Y)-min(Y))/2.0
    x = (X-mx)/sx
    y = (Y-my)/sy
    D = [ x *x,  x*y,  y*y,  x , y,np.ones(np.size(x)) ]
    D=np.array(D).T
   
    S= np.matmul(D.T,D)
   
    C_shape = (6,6)
   
    C=np.zeros(C_shape)
    C[0][2]=-2
    C[1][1]=1
    C[2][0]=-2
    if np.any(np.isnan(S)):
        a = []
        return
    geval, gevec = eig(S,C)
    
    
    NegC = np.where(np.isinf(geval)==False) and  np.where(geval < 0) 
    NegC=np.array(NegC)
    
    NegC=NegC[0][0]
    A=gevec[:,NegC]
    if (len(A)==0):
        a = []
        return
    a = [A[0]*sy*sy,
        A[1]*sx*sy,
        A[2]*sx*sx,
        -2*A[0]*sy*sy*mx - A[1]*sx*sy*my + A[3]*sx*sy*sy,
        -A[1]*sx*sy*mx - 2*A[2]*sx*sx*my + A[4]*sx*sx*sy,
        A[0]*sy*sy*mx*mx + A[1]*sx*sy*mx*my + A[2]*sx*sx*my*my- A[3]*sx*sy*sy*mx - A[4]*sx*sx*sy*my+ A[5]*sx*sx*sy*sy]
    a=np.array(a)
    return a


def mia_solveellipse_lsf(a):
    if (a.size!=0):
                    
        theta = math.atan2(a[1],a[0]-a[2])/2
        ct = math.cos(theta)
        st = math.sin(theta)
        ap = a[0]*ct*ct + a[1]*ct*st + a[2]*st*st
        cp = a[0]*st*st - a[1]*ct*st + a[2]*ct*ct
        T1=np.array([a[0], a[1]/2])
        T2= np.array([a[1]/2, a[2]])
        T = np.array([T1,T2])
        if (np.linalg.det(T)==0):
            v = []
            return
        t = np.matmul([- np.linalg.inv(2*T)] , [a[3], a[4]])
        cx = t[0][0]
        cy = t[0][1]
        t=t.T
        val = np.matmul(t.T, T)
        val= np.matmul(val, t)
        scale = float(1 / (val- a[5]))
        
        r1 = 1/np.sqrt(scale*ap)
        r2 = 1/np.sqrt(scale*cp)
        v = [r1, r2, cx, cy, theta]
        v=np.array(v)
    else:
        v=[]
    return v  
        
def mia_drawellip_lsf(v):
    X={}
    Y={}
    if (len(v)==0):
        X =0
        Y=0 
        v=0
    else:
        N = 110
        dx = 2*math.pi/N
        theta = v[4];
        R = [ [ math.cos(theta), math.sin(theta)], [-math.sin(theta),math.cos(theta)]]
        R=np.array(R).T
        for i in range (0,N+1):
            ang = (i+1)*dx
            x = v[0]*math.cos(ang)
            y = v[1]*math.sin(ang)
            xy=np.array([x, y])
            xy=np.transpose(np.asmatrix(xy))
            d1 =np.matmul (R,xy)
            X[i] = float(d1[0] + v[2])
            Y[i] = float(d1[1] + v[3])
    return X,Y






    
stats = {}
j = 0
xe={}
ye={}
X_array=[]
Y_array=[]
for i in range (0,len(contourevidence)):
     xe_i= contourevidence[i]['xe']
     ye_i=contourevidence[i]['ye']
     
     if  len(xe_i) > 5:
         print i
         a = mia_fitellip_lsf(xe_i,ye_i)
         v = mia_solveellipse_lsf(a)
         if (len(a)!=0) and  (not any(np.isnan(a))):
             X,Y=mia_drawellip_lsf(v)
         if (X!=0) :
             for i in X.keys():
                  X_array.append(X[i])
                  Y_array.append(Y[i])
                 
             
                
             XY=list(zip(X_array,Y_array))
             XY=np.array(XY)
               
             stats[j]= XY
             j = j + 1
             
             X_array=[]
             Y_array=[]

implot = plt.imshow(binary)




max_area=0
list_area=[]
for n in stats:
      area = PolyArea(stats[n][:, 0], stats[n][:, 1])
      list_area.append(area)
      if area>max_area:
          max_area=area
          max_i=n

del stats[max_i]
for i in stats:
      
    x=plt.plot(stats[i][:,1],stats[i][:,0],'.',color='red',linewidth=2,markersize=3)

img_width = image.shape[0]
img_height = image.shape[1]
plt.xlim(0,img_height,) 
plt.ylim(img_width,0)    
plt.show(x)
plt.savefig('img/718 images/111-Results-2/14/E.png', dpi = 300)
np.save('contours.npy',stats)

