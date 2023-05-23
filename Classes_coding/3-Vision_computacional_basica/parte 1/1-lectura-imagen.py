# -*- coding: utf-8 -*-

from __future__ import print_function
import cv2

"""
 NOTA: 
   * Usualmente cuando se muestra la imagen con cv2.imshow se abrirá una nueva 
     ventana (pero a veces por defecto no se cambia a la nueva ventana y 
     manualmente hay que buscarla). Para cerrar la imagen, presionar cualquier
     tecla.
   * Cuando se muestra la imagen con matplotlib.pyplot.imshow, la imagen se
     carga "inline"

""" 

# ============================================================================
# Usando cv2.imshow
# ============================================================================
print("\nUsando cv2.imshow")
print("\nIMPORTANTE: Para cerrar la imagen, presionar cualquier tecla\n")

# Leer una imagen
I = cv2.imread('images/cat1.jpg')
print("Tamaño de la imagen BGR: ", I.shape)

# Mostrar la imagen
# Argumentos de imshow: (Nombre de la ventana, imagen)
cv2.imshow('image', I)
cv2.waitKey(1000)
cv2.destroyAllWindows()


# Leer una imagen como si estuviese en escala de grises
# Nota: se puede usar '0' en lugar de cv2.IMREAD_GRAYSCALE
Igray = cv2.imread('images/cat1.jpg', cv2.IMREAD_GRAYSCALE)
print("Tamaño de la imagen en gris: ", Igray.shape)

# Mostrar la imagen 
cv2.imshow('image gray', Igray)
k = cv2.waitKey(0)
# Si se presiona 's' se grabará la imagen usando imwrite
if k==ord('s'):
    cv2.imwrite('catgray.png', Igray)
cv2.destroyAllWindows()


# ============================================================================
# Usando matplotlib.pyplot.imshow
# ============================================================================
print("\nUsando matplotlib.pyplot.imshow:")

# Mostrar con matplotlib
from matplotlib import pyplot as plt
# Convertir de BGR (defecto de opencv) en RGB
Irgb = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
# Mostrar la imagen
plt.imshow(Irgb, interpolation='bicubic')
# Eliminar las escalas en los ejes (también se puede usar .axis('off'))
plt.xticks([]), plt.yticks([])
plt.show()

# Separar componentes R G B
Ir = Irgb[:,:,0]; Ig = Irgb[:,:,1]; Ib = Irgb[:,:,2]
# Mostrar imágenes RGB separadas
plt.figure(figsize=(15,15))
plt.subplot(1,3,1)
plt.imshow(Ir, cmap='Reds'); plt.axis('off')
plt.subplot(1,3,2)
plt.imshow(Ir, cmap='Greens'); plt.axis('off')
plt.subplot(1,3,3)
plt.imshow(Ir, cmap='Blues'); plt.axis('off')