# -*- coding: utf-8 -*-
"""
Created on Mon May 28 18:51:44 2018

@author: Nishan
"""


from matplotlib import pyplot as plt
from skimage import io
import numpy as np
import math
from skimage.filters import threshold_otsu
import scipy 
from skimage.morphology import disk
import skimage.measure as sm
from scipy.spatial.distance import cdist
from skimage import morphology




#Defining the gaussian function
def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
        """
        2D gaussian mask - should give the same result as MATLAB's
        fspecial('gaussian',[shape],[sigma])
        """
        m,n = [(ss-1.)/2. for ss in shape]
        y,x = np.ogrid[-m:m+1,-n:n+1]
        h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
        h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
        sumh = h.sum()
        if sumh != 0:
            h /= sumh
        return h



#Reading the image
image=io.imread('img/718 images/111-png/14.png',as_grey=True)


#convert to a binary image 
thresh = threshold_otsu(image)
binary=image <thresh

#Removing small objects
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
#
#
binary= frame_image(binary, 15)
#assiging varibales
radii =list(range(8,9))
alpha=0.7 # Radial stricness
stdFactor=0.5  # Gradient threshold
mode='bright' # Distance between contour center points to define neighbouring segments
t=2; # Number of erosion operations and dilations
selem = disk(1) # structuring element for erosion and dilation
thr=28 # Thershold for removing duplicated seedpoints
bright = False

dark = False
if (mode =='bright'):
    bright = True
elif (mode == 'dark'):
    dark = True
elif (mode == 'both'):
    bright = True
    dark = True
else:
    print('invalid mode');
    

#Errosion and dilation

for i in range (0,2):
  binary=  morphology.dilation(binary, selem)

for i in range (0,2):
  binary=  morphology.erosion(binary, selem)
  
   
original = binary.astype(float)


# performs Fast radial symmetry transform to image

[gx, gy] = np.gradient(original)
        

maxRadius = math.ceil(np.amax(radii))
offset = maxRadius
    
filtered = np.array(np.shape(original)) + 2*offset
filtered = (filtered).astype(int)
filtered = np.zeros((filtered[0],filtered[1]))
    
    
S=np.zeros((len(radii),filtered.shape[0],filtered.shape[1]))
radiusIndex = 0
    
#        
for n in radii:
#    print n
    O_n  = np.zeros((filtered.shape[0],filtered.shape[1]))
    M_n  = np.zeros((filtered.shape[0],filtered.shape[1]))
    for i in range (0,(original.shape[0]-1)):
        for j in range (0,(original.shape[1]-1)):
            p = [i,j]
            g = [gx[i][j], gy[i][j]]
            gnorm = np.sqrt( np.matmul(g , (np.transpose(g))))
            if (gnorm > 0):
                
                gp=np.around((np.divide(g,gnorm))*n)
                if(bright):
                    ppve = p + gp
                    ppve = ppve + offset
                    O_n[ppve[0].astype(int)][ppve[1].astype(int)] = O_n[ppve[0].astype(int)][ppve[1].astype(int)] + 1
                    M_n[ppve[0].astype(int)][ppve[1].astype(int)] = M_n[ppve[0].astype(int)][ppve[1].astype(int)] + gnorm
                if(dark):
                    pnve = p - gp
                    pnve = pnve + offset
                    O_n[ppve[0].astype(int)][ppve[1].astype(int)] = O_n[ppve[0].astype(int)][ppve[1].astype(int)] - 1
                    M_n[ppve[0].astype(int)][ppve[1].astype(int)] = M_n[ppve[0].astype(int)][ppve[1].astype(int)] - gnorm
                          
    O_n= abs(O_n)
    O_n = np.divide( O_n,np.amax( O_n))
        
    M_n= abs(M_n)
    M_n = np.divide(M_n,np.amax(M_n))
        
    S_n= np.multiply(np.power(O_n, alpha) , M_n) 
    gaussian =matlab_style_gauss2D((math.ceil(n/2.0),math.ceil(n/2.0)),n*stdFactor)
    S_gaus=scipy.ndimage.convolve(S_n, gaussian)
        
        
        
    for i in range (0,(S_gaus.shape[0]-1)):
        for j in range (0,(S_gaus.shape[1]-1)):
            S[radiusIndex][i][j]=S_gaus[i][j]
     
    radiusIndex = radiusIndex + 1
                
          
    
S_sum= np.sum(S, axis=0)
filtered = S_sum= np.sum(S, axis=0)
offset_int= int(offset)
filtered= filtered[offset_int:-offset_int, offset_int:-offset_int]
thresh_ssedimg = threshold_otsu(filtered)
seedimg = filtered > thresh_ssedimg 
markerlabelmat, num_features = sm.label(seedimg,connectivity=2,return_num=True) 
seedpoints = {}
    
    
for i in range (1, num_features+1) :
    markimg = markerlabelmat==i
    [x,y] = np.nonzero(markimg)
    stats=sm.label(markimg)
    stats= sm.regionprops(stats)
    for props in stats:
        centroid=props.centroid
        
    s=i-1
    xmc = round(centroid[0])
    ymc = round(centroid[1])
    seedpoints[s] = {'xm':x,'ym':y,'xmc':xmc,'ymc':ymc}

xymc=(len(seedpoints),2)
xymc=np.zeros(xymc)
for i in range (0,len(seedpoints)):
        print i
        xymc[i][0] =seedpoints[i]['xmc'] 
        xymc[i][1] = seedpoints[i]['ymc']

markeridx=[]
markeridx = range(0, len(xymc))
markeridx=np.asarray(markeridx)
j = 0
markermerg = {}
while (len(markeridx) > 0):
            
    i = markeridx[0]
    d=cdist(xymc[i].reshape(1,2),xymc[markeridx])
    d1=d[0]
    indices = [n for n,v in enumerate(d1<thr) if v]
    xmc_mean= np.mean(xymc[markeridx[indices]],axis=0)[0]
    ymc_mean=np.mean(xymc[markeridx[indices]],axis=0)[1]
    markermerg[j]={'xmc':xmc_mean , 'ymc':ymc_mean}

    markeridx=np.delete(markeridx, indices)
    j=j+1
seedpoints_final = markermerg

implot = plt.imshow(binary)

#Plotting seed points

for i in seedpoints_final:
    plt.plot(seedpoints_final[i]['ymc'],seedpoints_final[i]['xmc'],'.')

plt.show()
plt.savefig('img/718 images/111-Results-2/14/S.png', dpi = 300)
np.save('seedpoints.npy',seedpoints_final)
