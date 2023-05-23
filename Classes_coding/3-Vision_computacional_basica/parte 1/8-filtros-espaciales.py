# -*- coding: utf-8 -*-
# Adaptado de:
# https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_filtering/py_filtering.html
# https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_gradients/py_gradients.html

from __future__ import print_function
import numpy as np
import cv2
from matplotlib import pyplot as plt

# Imagen inicial
I = cv2.imread('images/building.jpg')

# ==========================================
# SUAVIZADO
# ==========================================

# Especificación manual de un filtro (kernel)
kernel = np.ones((5,5), np.float32)/25.0
Imedia = cv2.filter2D(I, -1, kernel)

#plt.figure(figsize=(12,12))
plt.subplot(121)
plt.imshow(I), plt.title('Original'); plt.axis('off')
plt.subplot(122)
plt.imshow(Imedia), plt.title('Filtro de media'); plt.axis('off')
plt.show()


# Uso de funciones que implementan filtros
I_media = cv2.blur(I, (15,15))              # Filtro de media
I_gauss = cv2.GaussianBlur(I, (5,5), 0)   # Filtro Gaussiano
I_mediana = cv2.medianBlur(I, 5)          # Filtro de mediana
I_bilat = cv2.bilateralFilter(I, 9, 75, 75)  # Filtro bilateral
# Filtro bilateral: http://people.csail.mit.edu/sparis/bf_course/

# Mostrar los resultados
cv2.imshow('Filtro de media', I_media)
cv2.imshow('Filtro Gaussiano', I_gauss)
cv2.imshow('Filtro de mediana', I_mediana)
cv2.imshow('Filtro bilateral', I_bilat)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Nota: para crear un kernel Gaussiano: cv2.getGaussianKernel()


# ==========================================
# GRADIENTES
# ==========================================

Igray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)

I_laplacian = cv2.Laplacian(Igray, cv2.CV_64F)
I_sobelx = cv2.Sobel(Igray, cv2.CV_64F, 1, 0, ksize=5)
I_sobely = cv2.Sobel(Igray, cv2.CV_64F, 0, 1, ksize=5)

print("\nGradientes de la imagen:")
#plt.figure(figsize=(10,10))
plt.subplot(2,2,1)
plt.imshow(Igray,cmap='gray'); plt.title('Original'); plt.axis('off')
plt.subplot(2,2,2)
plt.imshow(I_laplacian,cmap='gray'); plt.title('Laplaciano'); plt.axis('off')
plt.subplot(2,2,3) 
plt.imshow(I_sobelx,cmap='gray'); plt.title('Sobel X'); plt.axis('off')
plt.subplot(2,2,4) 
plt.imshow(I_sobely,cmap='gray'); plt.title('Sobel Y'); plt.axis('off')
plt.show()

# Magnitud del gradiente
Igrad_mag = np.sqrt(I_sobelx**2+I_sobely**2)
# Orientación del gradiente
Igrad_or  = np.arctan2(I_sobely,I_sobelx)
# Mostrar con OpenCV
plt.imshow(Igrad_mag,cmap='gray'); plt.title('Magnitud del gradiente')
plt.axis('off'); plt.show()
plt.imshow(Igrad_or,cmap='gray'); plt.title('Orientación del gradiente') 
plt.axis('off'); plt.show()
