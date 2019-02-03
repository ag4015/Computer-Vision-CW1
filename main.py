import os
import sys
import cv2
from sklearn import cluster
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
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
            histogram = np.bincount(descriptors_nearest_clusters, minlength=256)
            class_histogram.append(histogram)
        bags_of_words.append(class_histogram)
    return np.array(bags_of_words) #converts list of lists into a numpy array

def print_all_CV_scores(clf):
    for (params, score, mean_fit_time, mean_score_time) in zip(clf.cv_results_['params'], clf.cv_results_['mean_test_score'], clf.cv_results_['mean_fit_time'], clf.cv_results_['mean_score_time']):
        print('params:', params,
            ' score:', score,
            ' mean_fit_time:', mean_fit_time,  #given in seconds
            ' mean_score_time:', mean_score_time)  #given in seconds

#load variables from matlab data
desc_tr = scipy.io.loadmat('matlab_data/desc_tr.mat')['desc_tr']
desc_sel = scipy.io.loadmat('matlab_data/desc_sel.mat')['desc_sel'].T
desc_te = scipy.io.loadmat('matlab_data/desc_te.mat')['desc_te']
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
    kmeans = cluster.KMeans(n_clusters=num_clusters, random_state=0, n_jobs = -1).fit(desc_sel)
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

# Construct training data labels
train_labels = [i//15 for i in range(150)]
test_labels = train_labels #in this case, since both are 10x15 images

# Flatten out train and test data (preparing them for RF classifier)
training_data = data_train.reshape(150, 256)
test_data = data_test.reshape(150, 256)


RFC = RandomForestClassifier(n_estimators=100, max_depth=5,random_state=0)
parameters = {'n_estimators':[10, 100], 'max_depth':[5, 10, 20]}
clf = GridSearchCV(RFC, param_grid=parameters, cv=3, return_train_score=False)
clf.fit(training_data, train_labels)
predictions = clf.predict(test_data)
reshaped_preds = predictions.reshape(10, 15)
print(reshaped_preds)

print('score:', clf.score(test_data, test_labels))

#print scores and times for all parameters tested
print_all_CV_scores(clf)

print("done")
