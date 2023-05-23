# -*- coding: utf-8 -*-
import cv2
from matplotlib import pyplot as plt

# ---------------------------------------------
# Intento 1: Haciendo matching por fuerza bruta
# ---------------------------------------------

# Cargar las imágenes
I1 = cv2.imread('images/nasa_logo.png', cv2.IMREAD_GRAYSCALE)
I2 = cv2.imread('images/kennedy_space_center.jpg', cv2.IMREAD_GRAYSCALE)

# Usar descriptores ORB en ambas imágenes
orb = cv2.ORB_create()
keypts1, descriptores1 = orb.detectAndCompute(I1, None)
keypts2, descriptores2 = orb.detectAndCompute(I2, None)

# Matching (correspondencia) por fuerza bruta
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(descriptores1, descriptores2)

# Ordenar las correspondencias ("matches") por distancia
matches = sorted(matches, key=lambda x:x.distance)

# Dibujar las mejores 25 correspondencias
Imatches = cv2.drawMatches(I1, keypts1, I2, keypts2, matches[:25], I2,
                           flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

# Mostrar las correspondencias
plt.figure(figsize=(12,12))
plt.imshow(Imatches)
plt.show()


# ----------------------------------------------------------
# Intento 2: Haciendo matching usando k-vecinos más cercanos
# ----------------------------------------------------------

# Uso de fuerza bruta con k-NN para k=2 (2 mejores correspondencias por punto)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
pares_de_matches = bf.knnMatch(descriptores1, descriptores2, k=2)

# Ordenar los pares de puntos por distancia
pares_de_matches = sorted(pares_de_matches, key=lambda x:x[0].distance)

# Escoger los mejores 25 pares
Imatches2 = cv2.drawMatchesKnn(I1, keypts1, I2, keypts2, 
                               pares_de_matches[:25], I2,
                               flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

# Mostrar las corresponencias
plt.figure(figsize=(12,12))
plt.imshow(Imatches2)
plt.show()


# ----------------------------------------------------------
# Intento 3: Añadiendo la razón de distancia NNDR
# ----------------------------------------------------------

# Realizar la prueba
NNDR = 0.75
matches = [x[0] for x in pares_de_matches
           if len(x) > 1 and x[0].distance < NNDR * x[1].distance]

# Escoger los mejores 25 pares
Imatches3 = cv2.drawMatches(I1, keypts1, I2, keypts2, matches[:25], I2,
                            flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)

# Mostrar las corresponencias
plt.figure(figsize=(12,12))
plt.imshow(Imatches3)
plt.show()