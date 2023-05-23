# -*- coding: utf-8 -*-
# Adaptado de:
# https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_feature_homography/py_feature_homography.html#py-feature-homography

import numpy as np
import cv2
from matplotlib import pyplot as plt

# Número mínimo de puntos para "match"
MIN_MATCH_COUNT = 10
# lectura de las imágenes
I1 = cv2.imread('images/box.jpg',0)          # queryImage
I2 = cv2.imread('images/box_in_scene.jpg',0) # trainImage


# Inicializar el descriptor ORB
orb = cv2.ORB_create()
# 'Keypoints' y descriptores para cada imagen (usando ORB)
keypts1, descript1 = orb.detectAndCompute(I1, None)
keypts2, descript2 = orb.detectAndCompute(I2, None)


# Matching usando FLANN
FLANN_INDEX_LSH = 6 
# Parámetros sugeridos para FLANN con ORB
index_params= dict(algorithm = FLANN_INDEX_LSH, table_number = 6,
                   key_size = 12, multi_probe_level = 1)
# Número de veces que el índice se debe buscar en el árbol
search_params = dict(checks = 50)
# Instancia de FLANN
flann = cv2.FlannBasedMatcher(index_params, search_params)
# Usando K-NN para match
matches = flann.knnMatch(descript1, descript2, k=2)

# Almacenamiento de los buenos "matches"
good_matches = []
for match in matches:
    # Descartar "matches" que tienen menos de 2 componentes (por "bug"?)
    if (len(match)<2): continue
    m,n = match
    if m.distance < (0.7*n.distance):
        good_matches.append(m)
                    
# Convertir imagen de escala de grises a 3 canales (para color del cuadro)
I2 = cv2.cvtColor(I2,cv2.COLOR_GRAY2RGB)
# Homografía entre los mejores "matches"
if (len(good_matches) > MIN_MATCH_COUNT):
    src_pts = np.float32([ keypts1[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
    dst_pts = np.float32([ keypts2[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)
    # Homografía entre pares de puntos 
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()
    # Borde del elemento detectado
    h,w = I1.shape
    pts = np.float32([[0,0], [0,h-1], [w-1,h-1], [w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts, M)
    # Gráfico sobre la imagen
    I2 = cv2.polylines(I2, [np.int32(dst)], True, (255,0,0), 3, cv2.LINE_AA)
else:
    print("No hay suficientes correspondencias - {}/{}".format(len(good_matches),MIN_MATCH_COUNT))
    matchesMask = None

# Parámetros para dibujar las correspondencias
draw_params = dict(matchColor = (0,255,0),    # Matches en verde
                   singlePointColor = None,
                   matchesMask = matchesMask, # Dibujar solo inliners (según RANSAC)
                   flags = 2)
# Imagen de salida (I1 con I2 con matches)
Iout = cv2.drawMatches(I1, keypts1, I2, keypts2, good_matches, None, **draw_params)

plt.figure(figsize=(12,12))
plt.imshow(Iout); plt.axis('off'); plt.title('Homografia')
plt.show()
