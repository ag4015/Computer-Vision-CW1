import os
import sys
import cv2
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn import cluster
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import csv
import pickle
import scipy.io
import matplotlib.pyplot as plt

def bag_of_words_histogram(desc, kmeans):
    #this function creates all bags of words for the entire input set (either training or test set)
    bags_of_words = []
    for class_row in desc:
        class_histogram = []
        for image_descriptors in class_row:
            descriptors_nearest_clusters = kmeans.predict(image_descriptors.T)
            histogram = np.bincount(descriptors_nearest_clusters)
            if len(histogram) < 256:
                print(len(histogram))
            class_histogram.append(histogram)
        bags_of_words.append(class_histogram)
    return np.array(bags_of_words) #converts list of lists into a numpy array


#load variables from matlab data
desc_tr = scipy.io.loadmat('matlab_data/desc_tr.mat')['desc_tr']
desc_sel = scipy.io.loadmat('matlab_data/desc_sel.mat')['desc_sel'].T
desc_te = scipy.io.loadmat('matlab_data/desc_te.mat')['desc_te'].T
imgIdx = scipy.io.loadmat('matlab_data/imgIdx.mat')['imgIdx'].T
imgIdx_tr = scipy.io.loadmat('matlab_data/imgIdx_tr.mat')['imgIdx_tr'].T
imgIdx_te = scipy.io.loadmat('matlab_data/imgIdx_te.mat')['imgIdx_te'].T

if len(sys.argv) > 1:  
    compute_kmeans = int(sys.argv[1])
else:  #if no arguments were given (too lazy), use the following ones
    compute_kmeans = 0


if compute_kmeans:
    print('Computing K-Means...')
    num_clusters = 256
    kmeans = cluster.KMeans(n_clusters=num_clusters, random_state=0, n_jobs = 3).fit(desc_sel)
    pickle_out = open('kmeans.pickle', 'wb')
    pickle.dump(kmeans,pickle_out) 
    pickle_out.close()
else:
    pickle_in = open('kmeans.pickle', 'rb')
    kmeans = pickle.load(pickle_in)

codewords = kmeans.cluster_centers_.copy()

#classes = ['tick', 'trilobite', 'umbrella', 'watch', 'water_lilly', 'wheelchair', 'wild_cat', 'windsor_chair', 'wrench', 'yin_yang']

data_train = bag_of_words_histogram(desc_tr, kmeans)
data_test = bag_of_words_histogram(desc_te, kmeans)

#data_train contains the bags of words of the entire training set
#data_test contains the bags of words of the entire test set

#all code up to here is what would be in the getData.m file 

#an example of a training set image histogram would be
# plt.plot(data_train[8][6]) #training image class 9 image 7
# plt.show()

print("done")

# Construct the lables from the training data 1 lable == 1 class
train_lables = np.zeros(15)
training_data = np.zeros((10*15,256))
for n in range(1,10):
    train_lables = np.r_[train_lables, n*np.ones(15)]

# Format train_data
for i in range(0,10):
    try:
        for n in range(0,15):
            training_data[i*15+n] = data_train[i][n]
    except Exception:
        print(i,n)


import pdb; pdb.set_trace()

clf = RandomForestClassifier(n_estimators=10, max_depth=5,random_state=0)
clf.fit(data_train, train_lables)





