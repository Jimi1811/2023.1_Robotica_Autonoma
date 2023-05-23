# -*- coding: utf-8 -*-

import cv2
import numpy as np

I1 = cv2.imread('images/pan1.jpg')
I2 = cv2.imread('images/pan2.jpg')

I1gr = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
I2gr = cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY)

# Inicializar el descriptor ORB
orb = cv2.ORB_create()
# 'Keypoints' y descriptores para cada imagen (usando ORB)
keypts1, descript1 = orb.detectAndCompute(I1gr, None)
keypts2, descript2 = orb.detectAndCompute(I2gr, None)

# Matcher
match = cv2.BFMatcher()
matches = match.knnMatch(descript1, descript2, k=2)
good_matches = []
for m,n in matches:
    if m.distance < 0.6*n.distance:
        good_matches.append(m)

# Dibujar los "matches" (las correspondencias)
draw_params = dict(matchColor=(0,255,0), singlePointColor = None, flags = 2)
Imatch = cv2.drawMatches(I1, keypts1, I2, keypts2, good_matches, None, **draw_params)
cv2.imshow("Correspondencias originales", Imatch)

# Encontrar la homografía dados los puntos correspondientes
MIN_MATCH_COUNT = 10
if len(good_matches) > MIN_MATCH_COUNT:
    src_pts = np.float32([ keypts1[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
    dst_pts = np.float32([ keypts2[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    h,w = I1gr.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts, M)
    img2 = cv2.polylines(I2gr, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
    cv2.imshow("Area de superposicion", I2gr)
else:
    print ("No se encuentran suficientes correspondencias - {}/{}".
           format(len(good_matches),MIN_MATCH_COUNT))

# Panorama usando la homografía
dst = cv2.warpPerspective(I1, M, (I2.shape[1] + I1.shape[1], I2.shape[0]))
dst[0:I2.shape[0],0:I2.shape[1]] = I2 
cv2.imshow("Panorama (con partes negras)", dst)


# Eliminar la parte negra del panorama
def trim(frame):
    # Cortar la imagen
    if not np.sum(frame[0]): return trim(frame[1:])
    if not np.sum(frame[-1]): return trim(frame[:-2])
    if not np.sum(frame[:,0]): return trim(frame[:,1:])
    if not np.sum(frame[:,-1]): return trim(frame[:,:-2])
    return frame
# Mostrar la imagen
cv2.imshow("Panorama", trim(dst))
cv2.waitKey(0)
cv2.destroyAllWindows() 
