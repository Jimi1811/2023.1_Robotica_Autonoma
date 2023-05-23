# -*- coding: utf-8 -*-
import cv2
from matplotlib import pyplot as plt

# Lectura de una imagen (el 0 al final indica que se lee como escala de gris)
Igray = cv2.imread('images/building.jpg', 0)


# Forma 1: histograma usando cv2
# ===============================
# Crear histograma: (I, canal 0, mask=None, tamaño, rango)
hist = cv2.calcHist([Igray], [0], None, [256], [0,256])
# Mostrar histograma
plt.plot(hist)
plt.show()


# Forma 2: usando numpy
# =====================
import numpy as np
# Crear histograma con 128 barras en el rango 0 a 256
hist = np.histogram(Igray, bins=128, range=(0, 256))
# Bordes de las barras
bin_edges = hist[1]
# Centro de la barra: promedio entre los bordes
bin_centers = (bin_edges[1:]  + bin_edges[0:len(bin_edges)-1])/2
# Gráfico del histograma (rhist[0] contiene las frecuencias)
plt.bar(bin_centers, hist[0]);
plt.show()


# Histograma de color
# ====================
# Leer imagen de color
I = cv2.imread('images/building.jpg')
# Histograma de cada canal RGB
bhist = np.histogram(I[:,:,0], bins=32, range=(0, 256))
ghist = np.histogram(I[:,:,1], bins=32, range=(0, 256))
rhist = np.histogram(I[:,:,2], bins=32, range=(0, 256))
# Concatenación de histogramas ("features")
hist = np.concatenate((rhist[0], ghist[0], bhist[0])).astype(np.float64)
# Vector del tamaño de hist 
n = np.arange(len(hist))
# Gráfico de los 3 histogramas concatenados ("features")
plt.figure(figsize=(12,3))
plt.bar(n, hist)


# Ecualización de histograma (opcional)
# =====================================
ecualizacion = False
if (ecualizacion):
    I = cv2.imread('images/dark.jpg',0)
    equ = cv2.equalizeHist(I)
    cv2.imshow('original', I)
    cv2.imshow('equalized', I)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
