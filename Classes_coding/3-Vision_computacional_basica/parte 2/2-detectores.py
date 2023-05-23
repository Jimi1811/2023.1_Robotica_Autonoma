# -*- coding: utf-8 -*-
# Adaptado de:
# https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_features_harris/py_features_harris.html

import cv2
import numpy as np

# Imagen de entrada
I = cv2.imread('images/jirafa.png')

Igray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
# Forzar a float dado que harris necesita valores grandes
Igray = np.float32(Igray)

# ===============================================================
# Detector usando Harris
# ===============================================================
nsize = 2    # Tamaño del vecindario (neighborhood)
ksize = 3    # Tamaño del Kernel (para el filtro de Sobel)
k = 0.06     # Valor de k en Harris
Iharris = cv2.cornerHarris(Igray, nsize, ksize, k)

# Dilatar el resultado solo para observar mejor las esquinas (corners)
Iharris_dil = cv2.dilate(Iharris, None)
# Cambiar el mapa de color solo para mostrar
Iharrish = cv2.applyColorMap(cv2.convertScaleAbs(Iharris_dil), cv2.COLORMAP_JET)

# Mostrar las esquinas (corners) si son mayores a th*max (como rojo)
th = 0.01
I[Iharris_dil > th*Iharris_dil.max()] = [0, 0, 255]

cv2.imshow('Matriz M de Harris', Iharrish)
cv2.imshow('Resultado Harris', I)
cv2.waitKey(0)
cv2.destroyAllWindows()


# ===============================================================
# Shi-Tomasi: goodFeaturesToTrack
# ===============================================================

I = cv2.imread('images/blocks.jpg')

Igray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
Igray = np.float32(Igray)

numcorners = 25  # Mejores esquinas a mantener
quality = 0.01   # Debajo de este valor se rechaza
mindist = 10     # Mínima distancia euclideana entre esquinas

Icorners = cv2.goodFeaturesToTrack(Igray, numcorners, quality, mindist)
Icorners = np.int0(Icorners)

for i in Icorners:
    x,y = i.ravel()
    cv2.circle(I, (x,y), 3, 255, -1)

cv2.imshow('Resultado Shi-Tomasi', I)
cv2.waitKey(0)
cv2.destroyAllWindows()


