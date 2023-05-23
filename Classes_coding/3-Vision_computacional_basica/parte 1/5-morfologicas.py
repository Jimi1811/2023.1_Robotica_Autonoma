# -*- coding: utf-8 -*-
# Adaptado de:
# https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html

from __future__ import print_function
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Leer la imagen como escala de grises
I = cv2.imread('images/bw.png', 0)

# Elemento estructurante
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
# También se puede definir un elemento estructurante como un array
kernel2 = np.ones((5,5), np.uint8)

# Operaciones Morfológicas
Ierosion    = cv2.erode(I, kernel, iterations=1)
Idilatacion = cv2.dilate(I, kernel, iterations=1)
Iapertura   = cv2.morphologyEx(I, cv2.MORPH_OPEN, kernel)
Icierre     = cv2.morphologyEx(I, cv2.MORPH_CLOSE, kernel)

# Mostrar la imagen original
plt.imshow(I, cmap='gray'); plt.axis('off'); plt.title("Imagen original")
plt.show()
# Mostrar las imágenes resultantes
plt.figure(figsize=(10,10))
plt.subplot(2,2,1); plt.imshow(Ierosion, cmap='gray'); plt.axis('off'); 
plt.title("Imagen erosionada")
plt.subplot(2,2,2); plt.imshow(Idilatacion, cmap='gray'); plt.axis('off'); 
plt.title("Imagen dilatada")
plt.subplot(2,2,3); plt.imshow(Iapertura, cmap='gray'); plt.axis('off'); 
plt.title("Imagen con apertura")
plt.subplot(2,2,4); plt.imshow(Icierre, cmap='gray'); plt.axis('off'); 
plt.title("Imagen con cierre")
plt.show()


# Otros elementos estructurantes
k2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))     # Rectangular
k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))  # Elíptico
k4 = cv2.getStructuringElement(cv2.MORPH_CROSS, (5,5))    # Forma de cruz


# Algunas otras operaciones morfológicas
otro = False
if (otro):
    # Diferencia entre la dilatación y la erosión (gradiente)
    gradient = cv2.morphologyEx(I, cv2.MORPH_GRADIENT, kernel)
    # Diferencia entre una imagen y su apertura (tophat)
    tophat = cv2.morphologyEx(I, cv2.MORPH_TOPHAT, kernel)
    # Diferencia entre el el cierre y la imagen de entrada (blackhat)
    blackhat = cv2.morphologyEx(I, cv2.MORPH_BLACKHAT, kernel)
    # Mostrar con OpenCV
    print("\nIMPORTANTE: Presionar una tecla para cerrar las imágenes")
    cv2.imshow('gradiente', gradient)
    cv2.imshow('tophat', tophat)
    cv2.imshow('blackhat', blackhat)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

