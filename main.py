import os
import sys
from sklearn.cluster import KMeans
from skimage.util.shape import view_as_windows
import numpy as np
import csv
import pickle

if len(sys.argv) > 1:  
    desc_filename = str(sys.argv[1])
    get_descriptors = int(sys.argv[2])
else:  #if no arguments were given (too lazy), use the following ones
    desc_filename = 'desc_sel.csv'
    get_descriptors = 0


# Get data descriptors from Caltech using MATLAB
if get_descriptors:
    os.system("cd RF_2019; matlab -nodesktop -nosplash -r \"run(\'init.m\');clc;get_descriptors(\'CalTech\', \'" + desc_filename + "\', 0);exit;\"")

# Load descriptors from csv into python variable
with open(str("RF_2019/" + desc_filename), 'rt') as csvfile:
    descriptors = np.zeros((100000,128)) 
    csv.field_size_limit(150000)
    f = csv.reader(csvfile, delimiter=',')
    i = 0
    for row in f:
        descriptors[:,i] = row  
        i = i + 1

with open("RF_2019/imgIdx_te.csv", 'rt') as csvfile:
    imgIdx_te = []
    f = csv.reader(csvfile, delimiter=',')
    for row in f:
        imgIdx_te.append(row)

# import pdb; pdb.set_trace()
#
num_clusters = 256


if get_descriptors:
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_jobs = 3).fit(descriptors)
    pickle_out = open('kmeans.pickle', 'wb')
    pickle.dump(kmeans,pickle_out) 
    pickle_out.close()
else:
    pickle_in = open('kmeans.pickle', 'rb')
    kmeans = pickle.load(pickle_in)

# Get all the descriptors in the same cluster and average them
codewords = np.zeros((num_clusters,128))
for n in range(0, max(kmeans.labels_)):
    cluster = np.where(kmeans.labels_==n)
    for i in range(0,128):
        codewords[n,i] = np.mean(descriptors[cluster[0],i])






