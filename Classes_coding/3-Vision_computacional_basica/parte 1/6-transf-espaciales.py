# -*- coding: utf-8 -*-
# Adaptado de:
# https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html#geometric-transformations

import cv2
import numpy as np

# Leer una imagen
I = cv2.imread('images/cat1.jpg')
# Tamaño de la imagen
nrows, ncols, nch = I.shape
print("\nIMPORTANTE: Para cerra la imagen, presionar cualquier tecla.")


# Escalar el tamaño de la imagen: cv2.resize()
# ============================================
# Interpolaciones: cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LINEAR (default)
Iescalada = cv2.resize(I, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
# También se puede escalar de la siguiente manera (solo con enteros):
# height, width = I.shape[:2]
# result = cv2.resize(I, (2*width, 2*height), interpolation=cv2.INTER_CUBIC)

cv2.imshow('Imagen escalada', Iescalada)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Traslación de la imagen
# =======================
# Se puede escoger la matriz M = [1 0 tx
#                                 0 1 ty]
# Traslación
tx = 100; ty = 50
M = np.float32([[1, 0, tx], [0, 1, ty]])
Itrasl = cv2.warpAffine(I, M, (ncols,nrows))
# Mostrar el resultado
cv2.imshow('Imagen trasladada', Itrasl)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Rotación de la imagen
# =====================
# Centro de la rotación y ángulo de rotación (en grados)
centro = (ncols/2, nrows/2)
angulo = 45 
M = cv2.getRotationMatrix2D(centro, angulo, 1)
Irot = cv2.warpAffine(I, M, (ncols, nrows))
cv2.imshow('Imagen rotada', Irot)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Rotación continua
# =================
rotcontinua = False
if(rotcontinua):
    ang = 0
    for i in range(360):
        M = cv2.getRotationMatrix2D((ncols/2, nrows/2), ang, 1)
        Irot = cv2.warpAffine(I, M, (ncols, nrows))
        cv2.imshow('Imagen rotada', Irot)
        cv2.waitKey(2)
        ang += 1
    cv2.destroyAllWindows()

# "Shearing"
# ===========
# Se puede usar los respectivos componentes en la matriz M
sx = 0.4
sy = 0.4
M_sx = np.array([[1.0,  sx, 0.],
                 [0.0, 1.0, 0.]])
M_sy = np.array([[1.0, 0.0, 0.],
                 [ sy, 1.0, 0.]])
# Aplicar 2 shearings
Ishear_x = cv2.warpAffine(I, M_sx, (ncols,nrows))
Ishear_y = cv2.warpAffine(I, M_sy, (ncols,nrows))
# Mostrar las imágenes resultantes
cv2.imshow('Shear en x', Ishear_x); cv2.waitKey(0)
cv2.imshow('Shear en y', Ishear_y); cv2.waitKey(0)
cv2.destroyAllWindows()

# Nota: si se desea encontrar M que representa la transformación afín entre 
# puntos, utilizar: M = cv2.getAffineTransform(pts1,pts2), donde cada fila de
# pts contiene un punto (es un arreglo de numpy)

# Perspectiva
# ============
p1 = 0.002
p2 = 0.002
M1 = np.array([[1.0, 0.0, 0.0],
               [0.0, 1.0, 0.0],
               [ p1, 0.0, 1.0]])
M2 = np.array([[1.0, 0.0, 0.0],
               [0.0, 1.0, 0.0],
               [0.0,  p2, 1.0]])
# Aplicar las perspectivas
Ipersp1 = cv2.warpPerspective(I, M1, (ncols,nrows))
Ipersp2 = cv2.warpPerspective(I, M2, (ncols,nrows))
# Mostrar las imágenes resultantes
cv2.imshow('Perspectiva 1', Ipersp1); cv2.waitKey(0)
cv2.imshow('Perspectiva 2', Ipersp2); cv2.waitKey(0)
cv2.destroyAllWindows()
