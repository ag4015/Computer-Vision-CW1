
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

img = cv2.imread('RF_2019/Caltech_101/101_ObjectCategories/wild_cat/image_0009.jpg',cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('RF_2019/Caltech_101/101_ObjectCategories/wild_cat/image_0014.jpg',cv2.IMREAD_GRAYSCALE)

plt.subplot(121)
plt.imshow(img, cmap=plt.cm.gray)
plt.axis('off')
plt.subplot(122)
plt.imshow(img2, cmap=plt.cm.gray)
plt.axis('off')
plt.savefig('example_success.jpg', dpi=300, bbox_inches='tight')
plt.show()
import pdb; pdb.set_trace()


n_clusters = 5
np.random.seed(0)

face = img
n_clusters = 2 
k_means = cluster.KMeans(n_clusters=n_clusters)
face_compressed = k_means.fit_transform(face.reshape((-1, 1)))[:,0].reshape(face.shape)

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
