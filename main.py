import os
import sys
import cv2
from sklearn import cluster
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
import pickle
import scipy.io
import matplotlib.pyplot as plt

def bag_of_words_histogram(desc, kmeans, num_clusters):
    #this function creates all bags of words for the entire input set (either training or test set)
    bags_of_words = []
    for class_row in desc:
        class_histogram = []
        for image_descriptors in class_row:
            descriptors_nearest_clusters = kmeans.predict(image_descriptors.T)
            histogram = np.bincount(descriptors_nearest_clusters, minlength=num_clusters)
            class_histogram.append(histogram)
        bags_of_words.append(class_histogram)
    return np.array(bags_of_words) #converts list of lists into a numpy array

def kmeans_codebook(desc_sel, num_clusters, compute):

    if compute:
        print('Computing K-Means...')
        kmeans = cluster.KMeans(n_clusters=num_clusters, random_state=0, n_jobs = -1).fit(desc_sel)
        pickle_out = open('kmeans.pickle', 'wb')
        pickle.dump(kmeans,pickle_out) 
        pickle_out.close()
    else:
        pickle_in = open('kmeans.pickle', 'rb')
        kmeans = pickle.load(pickle_in)

    codewords = kmeans.cluster_centers_.copy()

    histogram_train = bag_of_words_histogram(desc_tr, kmeans, num_clusters)
    histogram_test = bag_of_words_histogram(desc_te, kmeans, num_clusters)

    # data_train contains the bags of words of the entire training set
    # data_test contains the bags of words of the entire test set

    #an example of a training set image histogram would be
    # plt.plot(data_train[8][6]) #training image class 9 image 7
    # plt.show()

    print("Done")
    
    return histogram_train, histogram_test

def print_all_CV_scores(clf):
    for (params, score, mean_fit_time, mean_score_time) in zip(clf.cv_results_['params'], clf.cv_results_['mean_test_score'], clf.cv_results_['mean_fit_time'], clf.cv_results_['mean_score_time']):
        print('params:', params,
            ' score:', score,
            ' mean_fit_time:', mean_fit_time,  #given in seconds
            ' mean_score_time:', mean_score_time)  #given in seconds

def test_vocabulary(desc_sel, desc_tr, desc_te):
    vocabulary_sizes = [50,100,200,300,400]
    score_list = []
    for num_clusters in vocabulary_sizes:
        print('Computing codebook for a vocabulary of', num_clusters) 
        kmeans = cluster.KMeans(n_clusters=num_clusters, random_state=0, n_jobs = 3).fit(desc_sel)
        
        # Construct training data lables
        train_lables = [i//15 for i in range(150)]
        test_lables = train_lables #in this case, since both are 10x15 images

        # Calculate the bag of words for training and test data
        data_train = bag_of_words_histogram(desc_tr, kmeans, num_clusters).reshape(150,num_clusters)
        data_test = bag_of_words_histogram(desc_te, kmeans, num_clusters).reshape(150,num_clusters)

        print('Computing RF for a vocabulary of', num_clusters) 
        # Use best performing parameters for RF
        RFC = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=0).fit(data_train, train_lables)
        score = RFC.score(data_test, test_lables)
        score_list.append(score)
        print('score:', score)
    pickle_out = open('vocabulary_scores.pickle', 'wb')
    pickle.dump(score_list, pickle_out)
    pickle_out.close()

def rf_codebook(desc_tr, desc_te, desc_sizes, compute):
    
    print('Computing RF Codebook...')

    # One lable corresponds to one image
    train_lables = np.hstack([np.ones(desc_sizes[0][i])*i for i in range(150)]).astype(int)
    # Reformat the training and testing data
    for i in range(10):
        for n in range(15):
            if i == 0 and n == 0:
                data_train = desc_tr[i][n]
                data_test = desc_te[i][n]
            else:
                data_train = np.hstack((data_train, desc_tr[i][n]))
                data_test = np.hstack((data_test, desc_te[i][n]))
    data_train = data_train.T
    data_test = data_test.T

    # Compute the random forest
    if compute:
        RFC = RandomForestClassifier(n_estimators=100, max_depth=10,random_state=0, n_jobs=3)
        RFC.fit(data_train, train_lables.ravel())
        
        # Predict the lables from the training and testing data
        prediction_train = RFC.predict(data_train)
        prediction_test = RFC.predict(data_test)
        
        # Compute the bag of words for each of the predictions
        histogram_train = np.bincount(prediction_train, minlength=150)
        histogram_test = np.bincount(prediction_test, minlength=150)
        pickle_out = open('rf_codebook.pickle', 'wb')
        pickle.dump([RFC,histogram_train, histogram_test],pickle_out) 
        pickle_out.close()
    else:
        pickle_in = open('rf_codebook.pickle', 'rb')
        RFC, histogram_train, histogram_test = pickle.load(pickle_in)
        pickle_in.close()

    
    print('Done')

    return histogram_train, histogram_test
    

#load variables from matlab data
desc_tr = scipy.io.loadmat('matlab_data/desc_tr.mat')['desc_tr']
desc_sel = scipy.io.loadmat('matlab_data/desc_sel.mat')['desc_sel'].T
desc_te = scipy.io.loadmat('matlab_data/desc_te.mat')['desc_te']
desc_sizes = scipy.io.loadmat('matlab_data/desc_sizes.mat')['desc_sizes']
imgIdx = scipy.io.loadmat('matlab_data/imgIdx.mat')['imgIdx'].T
imgIdx_tr = scipy.io.loadmat('matlab_data/imgIdx_tr.mat')['imgIdx_tr'].T
imgIdx_te = scipy.io.loadmat('matlab_data/imgIdx_te.mat')['imgIdx_te'].T

if len(sys.argv) ==  2:  
    compute_kmeans = int(sys.argv[1])
if len(sys.argv) ==  3:
    compute_search = int(sys.argv[2])
else:  #if no arguments were given (too lazy), use the following ones
    compute_kmeans = 0
    compute_search = 0


num_clusters = 256

# Construct training data lables
train_lables = [i//15 for i in range(150)]
test_lables = train_lables #in this case, since both are 10x15 images

data_train, data_test = rf_codebook(desc_tr,desc_te, desc_sizes, compute=False)
import pdb; pdb.set_trace()

# Flatten out train and test data (preparing them for RF classifier)
# training_data = data_train.reshape(150, num_clusters)
# test_data = data_test.reshape(150, num_clusters)


# RFC = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=0)
# parameters = {'n_estimators':[10, 100], 'max_depth':[5, 10, 20]}
#
# # Compute GridSearchCV or load it from file
# compute_search = 1
# if compute_search:
#     clf = GridSearchCV(RFC, param_grid=parameters, cv=3, return_train_score=False,n_jobs=3)
#     clf.fit(data_train, np.array(train_lables).reshape(1,-1))
#     pickle_out = open('clf.pickle', 'wb')
#     pickle.dump(clf, pickle_out)
#     pickle_out.close()
# else:
#     pickle_in = open('clf.pickle', 'rb')
#     clf = pickle.load(pickle_in)
#     pickle_in.close()
#
# predictions = clf.predict(data_test)
# reshaped_preds = predictions.reshape(10, 15)
# print(reshaped_preds)
#
# print('score:', clf.score(data_test, test_lables))
#
# #print scores and times for all parameters tested
# # print_all_CV_scores(clf)
#
# print("done")
# import pdb; pdb.set_trace()
# test_vocabulary(desc_sel, desc_tr, desc_te)







