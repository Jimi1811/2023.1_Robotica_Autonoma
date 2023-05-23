# -*- coding: utf-8 -*-
#
# Basado en:
# https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_matcher/py_matcher.html#matcher

import cv2
from matplotlib import pyplot as plt

I1 = cv2.imread('images/box.jpg',0)          # queryImage
I2 = cv2.imread('images/box_in_scene.jpg',0) # trainImage

# Descriptor ORB
orb = cv2.ORB_create()

# 'Keypoints' y descriptores para cada imagen (usando ORB)
keypts1, descript1 = orb.detectAndCompute(I1, None)
keypts2, descript2 = orb.detectAndCompute(I2, None)

# ===============================================================
# Matching usando fuerza bruta
# ===============================================================

# "Matcher" de fuerza bruta (Brute Force)
matcherbf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# Match entre descriptores usando fuerza bruta
matches = matcherbf.match(descript1, descript2)
# Ordenar los "matches" según la distancia (de menor a mayor)
matches = sorted(matches, key = lambda x:x.distance)

# Dibujar los primeros 10 "matches" (con menor distancia)
Iout = cv2.drawMatches(I1, keypts1, I2, keypts2, matches[:10], None, flags=2)
plt.figure(figsize=(12,12))
plt.imshow(Iout); plt.axis('off'); plt.title('Usando fuerza bruta')
plt.show()



# ===============================================================
# Matching usando Vecinos más cercanos: FLANN
#    FLANN (Fast Library for Approximate Nearest Neighbors)
#    https://github.com/flann-lib/flann
# ===============================================================
 
FLANN_INDEX_LSH = 6 
# Parámetros sugeridos para FLANN con ORB
index_params= dict(algorithm = FLANN_INDEX_LSH, table_number = 6,
                   key_size = 12, multi_probe_level = 1)
# Número de veces que el índice se debe buscar en el árbol (mayores valores 
# dan mayor precisión, pero demoran más)
search_params = dict(checks = 50)

# Instancia de FLANN
flann = cv2.FlannBasedMatcher(index_params, search_params)
# Usando K-NN para match
matches = flann.knnMatch(descript1, descript2, k=2)

# Máscara para quedarse con los buenos "matches" 
matchesMask = [[0,0] for i in range(len(matches))]
NNDR = 0.7
# Test de "ratio"
for i, match in enumerate(matches):
    # Descartar "matches" que tienen menos de 2 componentes (por "bug"?)
    if (len(match)<2): continue
    # Separación de elementos del match
    (m, n) = match
    if (m.distance < NNDR*n.distance):
        matchesMask[i]=[1,0]

# Dibujar los "matches"
draw_params = dict(matchColor = (0,255,0), singlePointColor = (255,0,0),
                   matchesMask = matchesMask, flags = 0)
Iflann = cv2.drawMatchesKnn(I1, keypts1, I2, keypts2, matches, 
                            None, **draw_params)

plt.figure(figsize=(12,12))
plt.imshow(Iflann); plt.axis('off'); plt.title('Usando k-nearest neighbors')
plt.show()
