# -*- coding: utf-8 -*-
"""
Created on Thu May  7 01:25:38 2020

@author: 14198
"""

import cv2
import numpy as np
import os
import pandas as pd
from skimage import io
from sklearn import metrics
import pprint
pp = pprint.PrettyPrinter(indent=4)
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import pickle
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
import pprint
from sklearn.cluster import KMeans


#Feature Extaction function

def feature_extraction(img):
    
    df =pd.DataFrame()
    
    reshape_img= img.reshape(-1)
    df['Original Image'] = reshape_img
    
    num = 1
    kernels = []
    for theta in range(2):
        theta = theta / 4. * np.pi
        for sigma in (1, 3):
            for lamda in np.arange(0, np.pi, np.pi / 4):
                for gamma in (0.05, 0.5):
#               print(theta, sigma, , lamda, frequency)
                
                    gabor_label = 'Gabor' + str(num)
#                    print(gabor_label)
                    ksize=9
                    kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)    
                    kernels.append(kernel)
                    #Now filter image and add values to new column
                    fimg = cv2.filter2D(reshape_img, cv2.CV_8UC3, kernel)
                    filtered_img = fimg.reshape(-1)
                    df[gabor_label] = filtered_img  #Modify this to add new column for each gabor
                    num += 1
    edges =cv2.Canny(img,100,200)
    edges1=edges.reshape(-1)
    df['Canny Edge'] = edges1
    
    from skimage.filters import roberts, sobel, scharr, prewitt
    
    edge_roberts=  roberts(img)
    edge_roberts1= edge_roberts.reshape(-1)
    df['Roberts']= edge_roberts1
    
    
    edge_sobel = sobel(img)
    edge_sobel1 = edge_sobel.reshape(-1)
    df['Sobel'] = edge_sobel1
    
    
    edge_scharr = scharr(img)
    edge_scharr1 = edge_scharr.reshape(-1)
    df['Scharr'] = edge_scharr1

    #Feature 7 is Prewitt
    edge_prewitt = prewitt(img)
    edge_prewitt1 = edge_prewitt.reshape(-1)
    df['Prewitt'] = edge_prewitt1

    #Feature 8 is Gaussian with sigma=3
    from scipy import ndimage as nd
    gaussian_img = nd.gaussian_filter(img, sigma=3)
    gaussian_img1 = gaussian_img.reshape(-1)
    df['Gaussian s3'] = gaussian_img1

    #Feature 9 is Gaussian with sigma=7
    gaussian_img2 = nd.gaussian_filter(img, sigma=7)
    gaussian_img3 = gaussian_img2.reshape(-1)
    df['Gaussian s7'] = gaussian_img3
    
    
    median_img = nd.median_filter(img, size=3)
    median_img1 = median_img.reshape(-1)
    df['Median s3'] = median_img1

    #Feature 11 is Variance with size=3
    variance_img = nd.generic_filter(img, np.var, size=3)
    variance_img1 = variance_img.reshape(-1)
    df['Variance s3'] = variance_img1
    
    hariis_image_gray = np.float32(img)
    harris = cv2.cornerHarris(hariis_image_gray,25,1,0.06) 
    harris = harris.reshape(-1)
    df['Hariss'] = harris 
    
    orb = cv2.ORB_create(20000)
    kp, des = orb.detectAndCompute(img, None)
    orb_img = cv2.drawKeypoints(img, kp, None, flags=None)
    orb_img= cv2.cvtColor(orb_img,cv2.COLOR_BGR2GRAY)
    orb_img = orb_img.reshape(-1)
    df['ORB'] = orb_img 
    
    kmeans = KMeans(n_clusters=3, random_state=0).fit(img)
    kmeans_lbl = kmeans.cluster_centers_[kmeans.labels_]
    kmeans_lbl = kmeans_lbl.reshape(-1)
    df['kmeans'] = kmeans_lbl 
    
    return df




#features columns

df1=pd.DataFrame()
df2=pd.DataFrame()
mask_df= pd.DataFrame()
img_df= pd.DataFrame()
final_df= pd.DataFrame()

for directory_path in glob.glob("path of images"):
    for img_path in glob.glob(os.path.join(directory_path, "*.png")):
        img = cv2.imread(img_path,0)
        df1= feature_extraction(img)
        img_df= img_df.append(df1)

for directory_path in glob.glob("path of masks"):
    for mask_path  in glob.glob(os.path.join(directory_path, "*.png")):
        mask = cv2.imread(mask_path, 0)  
        label_img = mask.reshape(-1)
        df2['Label_Value'] = label_img
        mask_df= mask_df.append(df2)

final_df= pd.concat([img_df, mask_df], axis=1) 

print(final_df.head())


Y = final_df["Label_Value"].values  # Lables
X=  final_df.drop(labels = ["Label_Value"], axis=1)  # Features

#spliting data

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=20)


#hyper-parameters optimization 


n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap}
pp.pprint(random_grid)
model= RandomForestClassifier()
rf_random = RandomizedSearchCV(estimator = model, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

rf_random.fit(X_train, Y_train)

pp = pprint.PrettyPrinter(indent=4)
pp.pprint(model.get_params())


#Model Training with optimized parameters


model = RandomForestClassifier()#Fill parameters,random_state = 42)

model.fit(X_train, Y_train)

prediction_train = model.predict(X_train)

prediction_test = model.predict(X_test)

print ("Accuracy on training data = ", metrics.accuracy_score(Y_train, prediction_train))

print ("Accuracy = ", metrics.accuracy_score(Y_test, prediction_test))

feature_list = list(X.columns)
feature_imp = pd.Series(model.feature_importances_,index=feature_list).sort_values(ascending=False)
print(feature_imp)


#Testing with validation

filename = "RF"
pickle.dump(model, open(filename, 'wb'))

loaded_model = pickle.load(open(filename, 'rb'))


result = loaded_model.predict(X)
segmented = result.reshape((img.shape))
plt.imshow(segmented, cmap ='jet')


#Converting the segemted image to Red blus and green 

mask_yellow= np.zeros(segmented.shape)
mask_green=  np.where(segmented==1, 255, segmented)
mask_green=  np.where(segmented==2, 0, mask_green)
mask_red= np.where(segmented==2, 255, segmented)
mask_red=  np.where(segmented==1, 0, mask_red)



arr = np.zeros((448, 448, 3))

arr[:,:,0] = mask_red
arr[:,:,1] = mask_green
arr[:,:,2] = mask_yellow
arr=np.float32(arr)


from matplotlib import pyplot as plt
plt.imshow(arr, cmap ='jet')

io.imsave('segmented_RF.jpg', arr)
