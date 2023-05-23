# -*- coding: utf-8 -*-
# 
# Dado que se usa SIFT, se requiere instalar lo siguiente:
#      pip install opencv-contrib-python
#

import cv2
from matplotlib import pyplot as plt

I1 = cv2.imread('images/gauguin_entre_les_lys.jpg', cv2.IMREAD_GRAYSCALE)
I2 = cv2.imread('images/gauguin_paintings.png', cv2.IMREAD_GRAYSCALE)

# Atributos importantes con el descritor SIFT
sift = cv2.xfeatures2d.SIFT_create()
keypts1, descriptores1 = sift.detectAndCompute(I1, None)
keypts2, descriptores2 = sift.detectAndCompute(I2, None)

# Parámetros de FLANN para ser usados con SIFT
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50) # A más chequeos, más exactitud (más lento)

# Correspondencias con FLANN
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(descriptores1, descriptores2, k=2)

# Máscara vacía para dibujar
mask_matches = [[0, 0] for i in range(len(matches))]
# Llenar valores usando el ratio
NNDR = 0.6
for i, (m, n) in enumerate(matches):
    if m.distance < NNDR * n.distance:
        mask_matches[i]=[1, 0]

# Dibujo de las correspondencias
img_matches = cv2.drawMatchesKnn(I1, keypts1, I2, keypts2, matches, None,
                                 matchColor=(0, 255, 0), 
                                 singlePointColor=(255, 0, 0),
                                 matchesMask=mask_matches, flags=0)

# Mostrar las correspondencias
plt.figure(figsize=(16,16))
plt.imshow(img_matches)
plt.show()