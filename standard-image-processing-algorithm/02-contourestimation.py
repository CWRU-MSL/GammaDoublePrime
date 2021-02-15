# -*- coding: utf-8 -*-
"""
Created on Fri May 25 18:26:15 2018

@author: Nishan
"""


from skimage.filters import scharr
from skimage.morphology import skeletonize
import pylab as pl
from matplotlib import pyplot as plt
from skimage import io
import numpy as np
import math
from skimage.filters import threshold_otsu
import scipy 
from skimage.morphology import square,dilation
from bresenham import bresenham
import random
# import normalize_easy as ne
from collections import defaultdict
from skimage import morphology

from sklearn.preprocessing import normalize
from config import *

def normr(Mat):
    """Normalize the rows of the matrix so the L2 norm of each row is 1.
    >>> A = rand(4, 4)
    >>> B = normr(A)
    >>> np.allclose(norm(B[0, :]), 1)
    True
    """
    Mat = clean_and_check(Mat, nshape=2)

    B = normalize(Mat, norm='l2', axis=1)
    return B

def clean_and_check(x, nshape=2):
    """Make sure x is ready for computation; make it a dtype float and ensure it
    has the right shape
    """
    assert len(x.shape) == nshape, 'This input array must be a {X}-D \
    array'.format(X=nshape)

    if x.dtype != np.float:
        x = np.asarray(x, dtype=float)
    return x

seedpoints_final= np.load( TEMP_PATH + 'seedpoints.npy', allow_pickle=True).item()

image=io.imread(IMAGE_PATH,as_gray=True)
thresh = threshold_otsu(image)
binary = image < thresh
binary = morphology.remove_small_objects(binary, min_size=64, connectivity=2)

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


vis=1
r=110 #radius to defirne circular search region

Lambda=0.1



def smoothGradient(binary, sigma):
    
    binary=binary.astype(float)
    sigma=2
    filterLength = 8*math.ceil(sigma)
    n = (filterLength - 1)/2
    x = np.arange(-n,n,1)
    c = 1/(math.sqrt(2*math.pi)*sigma)
    gaussKernel = c * np.exp(-(x**2)/(2*sigma**2))
    
    gaussKernel = gaussKernel/sum(gaussKernel)
    derivGaussKernel =np.gradient(gaussKernel)
    
    negVals = derivGaussKernel < 0
    posVals = derivGaussKernel > 0
    
    
    
    derivGaussKernel[posVals]= np.divide(derivGaussKernel[posVals],sum(derivGaussKernel[posVals]))
    derivGaussKernel[negVals]= np.divide(derivGaussKernel[negVals],abs(sum(derivGaussKernel[negVals])))
    
    gaussKernel=np.reshape(gaussKernel,(1,len(gaussKernel)))
    derivGaussKernel=np.reshape(derivGaussKernel,(1,len(derivGaussKernel)))
    
    
    
    
    
    
    GX_smooth = scipy.ndimage.convolve(binary, gaussKernel.transpose(), mode='nearest')
    GX_smooth = scipy.ndimage.convolve(GX_smooth,derivGaussKernel, mode='nearest')
    
    
    GY_smooth = scipy.ndimage.correlate(binary, gaussKernel,mode='nearest')
    GY_smooth = scipy.ndimage.convolve(GY_smooth,derivGaussKernel.transpose(),mode='nearest')
    return GX_smooth, GY_smooth

def mia_cmpedge(binary,verbose):
    gradmag = scharr(binary)
    thresh_gradmag = threshold_otsu(gradmag)
    gradmag = gradmag > thresh_gradmag 
    gradmag=gradmag.astype(int)
    gradmag_thin= skeletonize(gradmag)
    plt.imshow(gradmag_thin)
    gradmag_thin= gradmag_thin.astype(int)
    [x,y] = np.nonzero(gradmag_thin)
    [dx, dy] = smoothGradient(binary,2)
    return gradmag_thin, x, y,dx,dy
    



def get_random_color(pastel_factor = 0.5):
    return [(x+pastel_factor)/(1.0+pastel_factor) for x in [random.uniform(0,1.0) for i in [1,2,3]]]

def color_distance(c1,c2):
    return sum([abs(x[0]-x[1]) for x in zip(c1,c2)])


#Generating 
def generate_new_color(existing_colors,pastel_factor = 0.5):
    max_distance = None
    best_color = None
    for i in range(0,100):
        color = get_random_color(pastel_factor = pastel_factor)
        if not existing_colors:
            return color
        best_distance = min([color_distance(color,c) for c in existing_colors])
        if not max_distance or best_distance > max_distance:
            max_distance = best_distance
            best_color = color
    return best_color



if __name__ == '__main__':
    
    colors = []

  

#Compare distance
def  mia_cmpdistance(xysr, xyei, binary):
    
    
    mask= dilation(binary, square(3))
    distance = (((xysr-xyei)**2).sum(axis=1))**0.5
    bresenham_list =[]
    for i in range(0,len(xysr)):
        bresenham_list = list(bresenham(int(xysr[i][0]),int(xysr[i][1]),int(xyei[i][0]),int(xyei[i][1])))
        psubx=[]
        psuby=[]
        for m in range(0,len(bresenham_list)):
            psubx.append(bresenham_list[m][0])
            psuby.append(bresenham_list[m][1])
        pidx=[]
        for n in range (0,len(psuby)):
            
            val= mask [psubx[n]][psuby[n]]
            pidx.append(val)

        
        
        for l in range (0,len(pidx)):
            if pidx[l]==0:
                distance[i] = float('Inf')
                
        distance_inf=[]    
        if not (all(np.isinf(distance))):
            for d in range(0,len(distance)):
                if  not np.isinf(distance[d]):
                    distance_inf.append(distance[d])
                    
            distance = distance/ np.amax(distance_inf)  

    return distance

#mia_cmpdivergence
def mia_cmpdivergence(xysr, xyei, dy,dx):
    grad = [dx[xyei[0][0]][xyei[0][1]], dy[xyei[0][0]][xyei[0][1]]]
    dvg  = np.absolute(np.sum(np.tile(grad, (len(xysr), 1)) * normr(xyei-xysr), axis=1)/(np.linalg.norm(grad)))

    return dvg
#mia_cmprelevence


  
for i in range(0,len(seedpoints_final)):
    col=generate_new_color(colors)
    print(col)
    colors.append(col)
col_edge=colors
imgsz = binary.shape
[gradmag, xe, ye, dy, dx] = mia_cmpedge(binary,0)
xye = np.vstack((xe ,ye)).T
nmepnts = len(xye)
nmspnts = len(seedpoints_final)
n_xys = (nmspnts,2)
xys = np.zeros(n_xys)


for i in range(0,nmspnts):
    xys[i][0] = np.floor(seedpoints_final[i]['xmc'])
    xys[i][1] = np.floor(seedpoints_final[i]['ymc'])
    
ids = range(0,nmspnts)
ids=np.array(ids)
ed2sp = np.zeros(nmepnts)

e2s= defaultdict(dict)
for i in range(0,nmepnts):
    idx=((xys-np.tile(xye[i], [nmspnts,1]))**2).sum(axis=1) < (r**2)
    xysr = xys[idx]
    idsr = ids[idx]
    nmsrpnts = len(idsr)
    xyei = np.tile(xye[i], [nmsrpnts, 1])
    eg = mia_cmpdistance(xysr, xyei, binary)
    dvg = mia_cmpdivergence(xysr, xyei,dy,dx)
    rel1 = 1/(1+eg) 
    rel2 = (dvg+1)/2
    rel2[rel1==0] = 0
    rel = (1-Lambda)*rel1 + Lambda*rel2
    if all(rel == 0):
        continue
    relmax= max(rel)
    idxrm=np.argmax(rel)  
    ed2sp[i] = idsr[idxrm]   
#[gradmag, ye, xe, dy, dx] = mia_cmpedge(binary,0)
for k in range (0,len(seedpoints_final)):
    val=k
    val_xe=xe[ed2sp==val]
    val_ye=ye[ed2sp==val]
    e2s[k]['xe']=val_xe 
    e2s[k]['ye']=val_ye
#    e2s[k]['eidx'] = sub2ind(imgsz,e2s(k).ye,e2s(k).xe);
implot = plt.imshow(gradmag)



for i in seedpoints_final:
    plt.plot(seedpoints_final[i]['ymc'],seedpoints_final[i]['xmc'],'.',color=col_edge[i])
    plt.plot(e2s[i]['ye'],e2s[i]['xe'],'.',color=col_edge[i])
# plt.show()
#plt.savefig(RESULT_PATH + '02-contourestimation.png', dpi = 300)
#contourest = TemporaryFile()
np.save(TEMP_PATH + 'contourest.npy',e2s)
  
    





