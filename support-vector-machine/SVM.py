# -*- coding: utf-8 -*-
"""
Created on Thu May  7 01:25:38 2020

@author: 14198
"""

import cv2
import numpy as np
import pandas as pd
from skimage import io
from sklearn import metrics
import pprint
pp = pprint.PrettyPrinter(indent=4)
from PIL import Image
import pickle
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
import pprint
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix 
from sklearn.svm import SVC 
from sklearn.svm import LinearSVC
import pickle
from sklearn.model_selection import train_test_split

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


df_list = []

for i in range(1, 21):
    img = cv2.imread('images/input_unit8_448/' + str(i) + '.png' ,0)
    df= feature_extraction(img)
    labeled_img = cv2.imread('images/label_mask_448/' + str(i) + '.png' ,0)
    labeled_img1 = labeled_img.reshape(-1)
    df['Labels'] = labeled_img1
    df_list.append(df)

df = pd.concat(df_list, ignore_index=True)
#features columns

#img = cv2.imread('images/input_unit8_448/1.png' ,0)

#df= feature_extraction(img)
#label column
#labeled_img = cv2.imread('images/label_mask_448/1.png' ,0)
#labeled_img1 = labeled_img.reshape(-1)
#df['Labels'] = labeled_img1


#print(df.head())

Y = df["Labels"].values
X=  df.drop(labels = ["Labels"], axis=1) 



#spliting data

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=20)


#model = SVC() 
#print(model.score)
#model.fit(X_train, Y_train)
#Parameter optimization

#SVC(decision_function_shape)
# defining parameter range 
#param_grid = {'C': [0.1, 1, 10, 100, 1000],  
#              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
#              'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
#              'max_iter':[100,500,1000,1500]}  

param_grid = {'C': [0.1, 1, 10, 100, 1000],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
              'max_iter':[100,300,500,700,1000,1200,1500],
              'decision_function_shape':['ovo', 'ovr']}  
  
grid = RandomizedSearchCV(SVC() , param_grid, refit = True, verbose = 3) 
  
# fitting the model for grid search 
grid.fit(X_train, Y_train) 

# print best parameter after tuning 
print(grid.best_params_) 
  
# print how our model looks after hyper-parameter tuning 
print(grid.best_estimator_) 


grid_predictions = grid.predict(X_test) 
  
# print classification report 
print(classification_report(Y_test, grid_predictions))




prediction_train = grid.predict(X_train)

prediction_test = grid.predict(X_test)

print ("Accuracy on training data = ", metrics.accuracy_score(Y_train, prediction_train))

print ("Accuracy = ", metrics.accuracy_score(Y_test, prediction_test))


# for i in range(100, 1600,100):
#     model = LinearSVC(max_iter=i)  
#     model.fit(X_train, Y_train)
#     prediction_train = model.predict(X_train)

#     prediction_test = model.predict(X_test)

#     from sklearn import metrics

#     print ("Accuracy on training data = ", metrics.accuracy_score(Y_train, prediction_train))

#     print ("Accuracy" + str(i)+" = ", metrics.accuracy_score(Y_test, prediction_test))

# # feature_list = list(X.columns)
# # feature_imp = pd.Series(model.feature_imp,index=feature_list).sort_values(ascending=False)
# # print(feature_imp)




filename = "SVM"
pickle.dump(grid, open(filename, 'wb'))

loaded_model = pickle.load(open(filename, 'rb'))


result = loaded_model.predict(X)
segmented = result.reshape((img.shape))
plt.imshow(segmented, cmap ='jet')



print() 

segmented = result.reshape((img.shape))
segmented= segmented*100

    
plt.imshow(segmented, cmap ='rgb')
plt.imsave('svm_'+str(i)+'.jpg', segmented, cmap ='jet')
