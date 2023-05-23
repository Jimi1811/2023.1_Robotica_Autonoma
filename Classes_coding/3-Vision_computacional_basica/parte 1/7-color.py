# -*- coding: utf-8 -*-
# Adaptado de:
# https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_colorspaces/py_colorspaces.html

import cv2
import numpy as np

# Todas las transformaciones entre espacios de color están en:
# https://docs.opencv.org/master/d8/d01/group__imgproc__color__conversions.html

# Leer la imagen
I = cv2.imread("images/kittens.jpg")
cv2.imshow("Imagen original", I); cv2.waitKey(0)

# Convertir a escala de grises
Igray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
cv2.imshow("Imagen en escala de grises", Igray); cv2.waitKey(0)

# Convertir BGR a HSV
Ihsv = cv2.cvtColor(I, cv2.COLOR_BGR2HSV)
# Mostrar el componente H de HSV
cv2.imshow('Hue de HSV', Ihsv[:,:,0]); cv2.waitKey(0)
# Nota, los rangos son
#  Hue: [0, 179], Saturation/Value: [0, 255]


# ========================================================================
# Segmentación basada en color
# ========================================================================

# Definir el rango de de amarillo que se quedará en la imagen
lower_yellow = np.array([20,50,50])
upper_yellow = np.array([40,255,255])

# Generar una máscara para los valores que satisfacen el rango de color
Imask = cv2.inRange(Ihsv, lower_yellow, upper_yellow)
cv2.imshow('Mascara', Imask); cv2.waitKey(0)

# Aplicar la máscara a la imagen original (mantener los puntos dentro del rango
# de color establecido)
Ifiltcolor = cv2.bitwise_and(I, I, mask=Imask)
cv2.imshow('Imagen filtrada por color', Ifiltcolor); cv2.waitKey(0)

cv2.destroyAllWindows()
