import os
import sys
import cv2
from sklearn import cluster
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
import pickle
import scipy.io
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import time
from scipy.interpolate import make_interp_spline, BSpline

def pickle_save(x, file_name):
    #custom function to save variables locally
    pickle_out = open(file_name, 'wb')
    pickle.dump(x, pickle_out)
    pickle_out.close()

def pickle_load(file_name):
    #custom function to load variables
    pickle_in = open(file_name, 'rb')
    x = pickle.load(pickle_in)
    pickle_in.close()
    return x

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
        pickle_save(kmeans, 'kmeans.pickle')
    else:
        kmeans = pickle_load('kmeans.pickle')

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

def test_RF_classifier_params(parameters, training_data, train_labels, test_data, test_labels):
    X = [] #param1,
    Y = [] #param2
    Z = [] #accuracy scores
    all_times = [] #train and test time
    for n_estimators in parameters['n_estimators']:
        for max_depth in parameters['max_depth']:
            for max_features in parameters['max_features']:
                RFC = ExtraTreesClassifier(n_estimators=n_estimators, criterion= 'entropy', bootstrap=False, max_features=max_features, max_depth=max_depth, random_state=0)
                preds, score, RFC_fit, time_list = fit_and_predict(RFC, training_data, train_labels, test_data, test_labels)
                X.append(max_depth)
                Y.append(n_estimators)
                Z.append(score)
                print('params: ', 'n_estimators:', n_estimators, 'max_depth', max_depth, 'max_features', max_features, ' score:', score)
                all_times.append(time_list)
    max_scores_indices = np.argsort(Z)[-6:-1]
    print('max scores:')
    for index in max_scores_indices:
        print(X[index], Y[index], Z[index])
    return X, Y, Z, np.array(all_times)

def load_or_compute_pickle(num_clusters):
    pickle_filename = 'kmeans_' + str(num_clusters) + '.pickle'
    if os.path.isfile(pickle_filename):
        #the file exists
        print('File', pickle_filename, 'found')
        kmeans = pickle_load(pickle_filename)
    else:
        print('Computing codebook for a vocabulary of', num_clusters)
        start = time.time()
        kmeans = cluster.KMeans(n_clusters=num_clusters, random_state=0, n_jobs = 3).fit(desc_sel)
        time_taken = time.time() - start
        print('time taken:', time_taken)
        pickle_save(kmeans, pickle_filename)
    return kmeans

def test_vocabulary(vocabulary_sizes, desc_sel, desc_tr, desc_te):
    score_list = []
    for num_clusters in vocabulary_sizes:
        kmeans = load_or_compute_pickle(num_clusters)

        # Construct training data labels
        train_labels = [i//15 for i in range(150)]
        test_labels = train_labels #in this case, since both are 10x15 images

        # Calculate the bag of words for training and test data
        data_train = bag_of_words_histogram(desc_tr, kmeans, num_clusters).reshape(150,num_clusters)
        data_test = bag_of_words_histogram(desc_te, kmeans, num_clusters).reshape(150,num_clusters)

        print('Computing RF for a vocabulary of', num_clusters)
        # Use best performing parameters for RF
        RFC = ExtraTreesClassifier(n_estimators=100, max_depth=10, bootstrap=False, random_state=0).fit(data_train, train_labels)
        score = RFC.score(data_test, test_labels)
        score_list.append(score)
        print('score:', score)
    return score_list
    #pickle_out = open('vocabulary_scores.pickle', 'wb')
    #pickle.dump(score_list, pickle_out)
    #pickle_out.close()

def rf_codebook(desc_tr, desc_te, desc_sizes, compute):
    
    print('Computing RF Codebook...')

    # One lable corresponds to one image
    train_labels = np.hstack([np.ones(desc_sizes[0][i])*i for i in range(150)]).astype(int)
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
        RFC.fit(data_train, train_labels.ravel())
        
        # Predict the labels from the training and testing data
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
    
def fit_and_predict(clf, training_data, train_labels, test_data, test_labels):
    #works for both GridSearchCV and ExtraTreesClassifier
    start = time.time()
    clf.fit(training_data, train_labels)
    fit_time = time.time() - start
    #predictions = clf.predict(test_data)
    reshaped_preds = 0
    #reshaped_preds = predictions.reshape(10, 15)
    start = time.time()
    score = clf.score(test_data, test_labels)
    test_time = time.time() - start
    return reshaped_preds, score, clf, [fit_time, test_time]

def do_bsplines(var1, var2, num_points):
    spl = make_interp_spline(var1, var2, k=3)
    xnew = np.linspace(var1.min(), var1.max(), num_points) #new x axis
    return spl, xnew

def plot_acc_times(num_trees, accuracy, train_times, test_times):
    fig, ax1 = plt.subplots()
    ax1.plot(num_trees, accuracy, color='b')
    ax1.set_xlabel('depth of trees')
    ax1.set_ylabel('accuracy (%)')
    #yticks = [35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    yticks = [60, 65, 70, 75, 80]
    ax1.set_yticks(yticks)
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.plot(num_trees, train_times, color='r')
    ax2.plot(num_trees, test_times, color='g')
    ax2.set_ylabel('time (s)')
    ax2.legend(['train time', 'test time'], loc=1)
    ax1.legend(['accuracy (%)'], loc=2)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig('acc_depth_trees.png', bbox_inches='tight', dpi=300)
    plt.show()

def plot_3d(X, Y, Z):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    Z = np.array(Z)*100
    surf = ax.plot_trisurf(X, Y, Z, cmap=cm.coolwarm, linewidth=10, antialiased=False)
    ax.set_xlabel('$Tree \ depth$', fontsize=12)
    ax.set_ylabel('$Number \ of \ trees$', fontsize=12)
    ax.set_zlabel('$Accuracy\ (\%)$', fontsize=11)
    #xticks = np.array(list(range(10)))*2+1
    #yticks = np.array(list(range(6)))*50
    #yticks[0] = 1
    #yticks[-1] = 256
    #ax.set_xticks(xticks)
    #ax.set_yticks(yticks)
    ax.view_init(elev=30, azim=-110)
    plt.savefig('3dplot.png', bbox_inches='tight', dpi=300)
    plt.show()






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
    compute_search = 1


num_clusters = 256

# Construct training data labels
train_labels = [i//15 for i in range(150)]
test_labels = train_labels #in this case, since both are 10x15 images

#For rf codebook, uncomment the following line:
#data_train, data_test = rf_codebook(desc_tr,desc_te, desc_sizes, compute=False)

#For kmeans codebook, uncomment the following lines:
data_train, data_test = kmeans_codebook(desc_sel, num_clusters, compute_kmeans)

# Flatten out train and test data (preparing them for RF classifier)
training_data = data_train.reshape(150, num_clusters)
test_data = data_test.reshape(150, num_clusters)


vocabulary_sizes = [50,100,200,300,400, 500, 750, 1000, 3000]
score_list = test_vocabulary(vocabulary_sizes, desc_sel, desc_tr, desc_te)
print(score_list)


num_clusters = 256
max_features = num_clusters #max_features controls the randomness parameter (assuming bootstrap=False)

parameters = {'n_estimators':[100], 'max_depth':[5], 'max_features':[10]}
#clf = GridSearchCV(RFC, param_grid=parameters, cv=5, return_train_score=False)

#parameters['max_depth'] = np.linspace(1, 20, num=10, dtype=int)
parameters['max_depth'] = np.linspace(1, 40, num=5, dtype=int)

X, Y, Z, all_times = test_RF_classifier_params(parameters, training_data, train_labels, test_data, test_labels) #using for loops on test data
#plot_3d(X, Y, Z)

#pickle_save([X, Y, Z, all_times], 'tree_depth_time.pickle')

vars_list = pickle_load('tree_depth_time.pickle')

vars_list[0] = np.array(vars_list[0])

spl_acc, xnew = do_bsplines(vars_list[0], vars_list[2], 80)
spl_tr, xnew = do_bsplines(vars_list[0], vars_list[3][:,0], 80)
spl_te, xnew = do_bsplines(vars_list[0], vars_list[3][:,1], 80)

plot_acc_times(xnew, 100*np.array(spl_acc(xnew)), spl_tr(xnew), spl_te(xnew))
#plot_acc_times(vars_list[0], 100*np.array(vars_list[2]), vars_list[3][:,0], vars_list[3][:,1])

print(vars_list[3][:,1])
print(spl_te(xnew))
print(all_times)
print('train times:', all_times[:,0])
print('test times:', all_times[:,1])

#print scores and times for all parameters tested
#print_all_CV_scores(clf)


print("done")

#test_vocabulary(desc_sel, desc_tr, desc_te)