
# Code source: Gaël Varoquaux
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause

import numpy as np
import cv2
import scipy as sp
import matplotlib.pyplot as plt
import scipy.io
from sklearn import cluster


imgIdx = scipy.io.loadmat('matlab_data/imgIdx.mat')['imgIdx'].T
imgIdx_tr = scipy.io.loadmat('matlab_data/imgIdx_tr.mat')['imgIdx_tr'].T
imgIdx_te = scipy.io.loadmat('matlab_data/imgIdx_te.mat')['imgIdx_te'].T

image_name = 'RF_2019/Caltech_101/101_ObjectCategories/tick/image_0002.jpg'
img = cv2.imread('RF_2019/Caltech_101/101_ObjectCategories/wild_cat/image_0001.jpg',cv2.IMREAD_GRAYSCALE)
n_clusters = 5
np.random.seed(0)

# X = img.reshape((-1, 1))  # We need an (n_sample, n_feature) array
# k_means = cluster.KMeans(n_clusters=n_clusters, n_init=4)
# k_means.fit(X)
# X_compressed = k_means.transform(X)
# import pdb; pdb.set_trace()
# values = k_means.cluster_centers_.squeeze()
# labels = k_means.labels_

face = img
n_clusters = 2 
k_means = cluster.KMeans(n_clusters=n_clusters)
# X_1 = face[:,:,0].reshape((-1, 1))  # We need an (n_sample, n_feature) array
# X_2 = face[:,:,1].reshape((-1, 1))  # We need an (n_sample, n_feature) array
# X_3 = face[:,:,2].reshape((-1, 1))  # We need an (n_sample, n_feature) array
# k_means_1 = cluster.KMeans(n_clusters=n_clusters)
# k_means_1.fit(X_1)
# k_means_2 = cluster.KMeans(n_clusters=n_clusters)
# k_means_2.fit(X_2)
# k_means_3 = cluster.KMeans(n_clusters=n_clusters)
# k_means_3.fit(X_3)
# face_compressed_1 = k_means_1.transform(X_1)[:,0].reshape(face[:,:,0].shape)
# face_compressed_2 = k_means_2.transform(X_2)[:,0].reshape(face[:,:,1].shape)
# face_compressed_3 = k_means_3.transform(X_3)[:,0].reshape(face[:,:,2].shape)
# face_compressed = np.zeros((face.shape))
# face_compressed[:,:,0] = face_compressed_1
# face_compressed[:,:,1] = face_compressed_2
# face_compressed[:,:,2] = face_compressed_3
face_compressed = k_means.fit_transform(face.reshape((-1, 1)))[:,0].reshape(face.shape)
# values = k_means.cluster_centers_.squeeze()
# labels = k_means.labels_

# create an array from labels and values
# b = np.array(values)
# face_compressed = b[range(len(labels)), labels]
# face_compressed = np.choose(labels, values)
# face_compressed.shape = face.shape

vmin = face.min()
vmax = face.max()
# original face
plt.figure()
# plt.subplot(122,frameon=False)
plt.axis('off')
plt.imshow(face, cmap=plt.cm.gray, vmin=vmin, vmax=256)
plt.savefig('wild_cat_original.jpg', dpi=300, bbox_inches='tight')

# compressed face
# plt.subplot(121,frameon=False)
plt.figure()
plt.axis('off')
plt.imshow(face_compressed, cmap=plt.cm.gray, vmin=vmin, vmax=vmax)
plt.savefig('wild_cat_compressed.jpg', dpi=300, bbox_inches='tight')
plt.show()
