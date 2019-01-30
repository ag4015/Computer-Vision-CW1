import os
import sys
from sklearn.cluster import KMeans
import numpy as np
import csv

desc_filename = str(sys.argv[1])
get_descriptors = int(sys.argv[2])

# Get data descriptors from Caltech using MATLAB
if get_descriptors:
    os.system("cd RF_2019; matlab -nodesktop -nosplash -r \"run(\'init.m\');clc;get_descriptors(\'CalTech\', \'" + desc_filename + "\', 0);exit;\"")

# Load descriptors from csv into python variable
with open(str("RF_2019/" + desc_filename), 'rt') as csvfile:
    descriptors = np.zeros((100000,128)) 
    csv.field_size_limit(sys.maxsize)
    f = csv.reader(csvfile, delimiter=',')
    i = 0
    for row in f:
        descriptors[:,i] = row  
        i = i + 1

num_clusters = 256
kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_jobs = 3).fit(descriptors)
