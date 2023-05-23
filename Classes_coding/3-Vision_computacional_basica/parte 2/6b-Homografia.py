# -*- coding: utf-8 -*-
import numpy as np
import cv2
from matplotlib import pyplot as plt

MIN_NUM_GOOD_MATCHES = 10

I1 = cv2.imread('images/query.png', cv2.IMREAD_GRAYSCALE)
I2 = cv2.imread('images/anchor-man.png', cv2.IMREAD_GRAYSCALE)

# Descriptores usando SIFT
sift = cv2.xfeatures2d.SIFT_create()
keypts1, descriptores1 = sift.detectAndCompute(I1, None)
keypts2, descriptores2 = sift.detectAndCompute(I2, None)

# Parámetros de correspondencia usando FLANN
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
# Correspondencia usando FLANN
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(descriptores1, descriptores2, k=2)

# Correspondencias adecuadas según el ratio
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

if len(good_matches) >= MIN_NUM_GOOD_MATCHES:
    # Coordenadas 2D de los correspondientes keypoints 
    src_pts = np.float32( [keypts1[m.queryIdx].pt for m in good_matches] ).reshape(-1, 1, 2)
    dst_pts = np.float32( [keypts2[m.trainIdx].pt for m in good_matches] ).reshape(-1, 1, 2)

    # Homografía
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    mask_matches = mask.ravel().tolist()

    # Transformación de perspectiva: proyectar los bordes rectangulares
    # en la escena para dibujar un borde
    h, w = I1.shape
    src_corners = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
    dst_corners = cv2.perspectiveTransform(src_corners, M)
    dst_corners = dst_corners.astype(np.int32)

    # Dibujar los bordes de la región correspondiente
    num_corners = len(dst_corners)
    for i in range(num_corners):
        x0, y0 = dst_corners[i][0]
        if i == num_corners - 1:
            next_i = 0
        else:
            next_i = i + 1
        x1, y1 = dst_corners[next_i][0]
        cv2.line(I2, (x0, y0), (x1, y1), 255, 3, cv2.LINE_AA)

    # Dibujar correspondencias que pasaron el test de ratio
    img_matches = cv2.drawMatches(I1, keypts1, I2, keypts2, good_matches, 
                                  None, matchColor=(0, 255, 0), 
                                  singlePointColor=None,
                                  matchesMask=mask_matches, flags=2)

    # Mostrar la homografía y los matches
    plt.figure(figsize=(12,12))
    plt.imshow(img_matches)
    plt.show()
else:
    print("No hay suficientes correspondencias")


# -----------------------------------------------------------------------------
# Mismo resultado pero solo el objeto detectado
I1 = cv2.cvtColor(cv2.imread('images/query.png'), cv2.COLOR_BGR2RGB)
I2 = cv2.cvtColor(cv2.imread('images/anchor-man.png'), cv2.COLOR_BGR2RGB)

# Bordes detectados
num_corners = len(dst_corners)
for i in range(num_corners):
    x0, y0 = dst_corners[i][0]
    if i == num_corners - 1:
        next_i = 0
    else:
        next_i = i + 1
    x1, y1 = dst_corners[next_i][0]
    cv2.line(I2, (x0, y0), (x1, y1), (0, 0, 255), 3, cv2.LINE_AA)

plt.figure(figsize=(12,12))
plt.imshow(I2)
plt.show()