# -*- coding: utf-8 -*-
# 
# Adaptado de:
# https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_canny/py_canny.html
# https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Lectura de la imagen original
I = cv2.imread('images/building2.jpg')

# ======================================================
# Canny
# ======================================================

T_low = 50     # Umbral bajo
T_high = 150    # Umbral alto
ksize = (5,5)   # Tamaño de Kernel (sobel) para gradientes (por defecto 3)
Icanny = cv2.Canny(I, T_low, T_high, ksize)

plt.figure(figsize=(14,14))
plt.subplot(121)
plt.imshow(I, cmap='gray'); plt.title('Imagen original'); plt.axis('off')
plt.subplot(122)
plt.imshow(Icanny, cmap='gray'); plt.title('Bordes con Canny'); plt.axis('off')
plt.show()

# Ejemplos con otros umbrales
Icanny2 = cv2.Canny(I, 5, 50)
Icanny3 = cv2.Canny(I, 150, 250)
plt.figure(figsize=(14,14))
plt.subplot(121)
plt.imshow(Icanny2,cmap='gray'); plt.title('Canny Th=5-50'); plt.axis('off')
plt.subplot(122)
plt.imshow(Icanny3,cmap='gray'); plt.title('Canny Th=150-200'); plt.axis('off')
plt.show()


# ======================================================
# HOUGH para Líneas
# ======================================================

# Bordes de la imagen en escala de grises
Igray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
Iedges = cv2.Canny(Igray, 50, 150, apertureSize=3)

# Parámetros de Hough
votos = 210
lineas = cv2.HoughLines(Iedges, 1, 1*np.pi/180, votos)

# Imagen donde se dibujará las líneas
Ihough = I.copy()
for linea in lineas:
    for rho,theta in linea:
        a = np.cos(theta); b = np.sin(theta)
        x0 = a*rho; y0 = b*rho
        x1 = int(x0 + 1000*(-b)); y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b)); y2 = int(y0 - 1000*(a))
        cv2.line(Ihough, (x1,y1), (x2,y2), (0,0,255), 2)
    
cv2.imshow('Bordes', Iedges)
cv2.imshow('Transf de Hough', Ihough)
cv2.waitKey(0)
cv2.destroyAllWindows()


# ======================================================
# Hough probabilístico (para líneas)
# ======================================================

# Parámetros de hough probabilístico
minLineLength = 100
maxLineGap = 10
thvotes = 180
lines = cv2.HoughLinesP(Iedges, 1, np.pi/180, thvotes, minLineLength, maxLineGap)

# Imagen donde se dibujará las líneas
Ihoughp = I.copy()
for line in lines:
    for x1,y1,x2,y2 in line:
        cv2.line(Ihoughp, (x1,y1), (x2,y2), (0,255,0), 2)

cv2.imshow('Transf Hough Probabilistico', Ihoughp)
cv2.waitKey(0)
cv2.destroyAllWindows()


# ======================================================
# Hough para círculos
# ======================================================

I2 = cv2.imread('images/coins.jpg',0)
Imedian = cv2.medianBlur(I2, 5)

# Hough
circles = cv2.HoughCircles(Imedian, cv2.HOUGH_GRADIENT, 1, 30, param1=150,
                           param2=100, minRadius=10, maxRadius=200)
circles = np.uint16(np.around(circles))

# Imagen donde se dibujará los círculos
Ihough = cv2.cvtColor(I2, cv2.COLOR_GRAY2BGR)
for c in circles[0,:]:
    # Círculo externo
    cv2.circle(Ihough, (c[0],c[1]), c[2], (0,255,0), 2)
    # centro del círculo
    cv2.circle(Ihough, (c[0], c[1]), 2, (0,0,255), 3)

cv2.imshow('Círculos detectados', Ihough)
cv2.waitKey(0)
cv2.destroyAllWindows()
