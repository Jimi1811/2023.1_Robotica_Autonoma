# -*- coding: utf-8 -*-

from __future__ import print_function
import cv2
import matplotlib.pyplot as plt

# Cargar la imagen en modo gris
Igray = cv2.imread('images/building.jpg', 0)

# Valores para umbralizar
ths = [100, 150, 200]

# Threshold (umbralización)
for th in ths: 
    print("\n Con valor de umbralización {}: ".format(th))
    retval, Ibw = cv2.threshold(Igray, th, 255, cv2.THRESH_BINARY)
    plt.imshow(Ibw, cmap='gray')
    plt.show()
