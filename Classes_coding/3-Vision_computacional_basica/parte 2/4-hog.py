# -*- coding: utf-8 -*-
# 
# Más información en:
# https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_hog.html
# https://scikit-image.org/docs/dev/api/skimage.feature.html?highlight=hog#skimage.feature.hog

import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure


I = cv2.imread('images/cat1.jpg')
# Convertir de BGR en RGB
I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)

# Extraer HoG (Histogram of Gradients)
# descr_hog: descriptor de HoG
# Ihog: Imagen con características de HoG 
descr_hog, Ihog = hog(I, orientations=8, pixels_per_cell=(16, 16),
                      cells_per_block=(1, 1), visualize=True, multichannel=True)

# Mejorar el nivel de gris para mostrar la imagen
Ihog = exposure.rescale_intensity(Ihog, in_range=(0, 10))

plt.figure(figsize=(12,12))
# Imagen original
plt.subplot(1,2,1); plt.imshow(I, cmap='gray')
plt.title('Image de entrada'); plt.axis('off')
# Imagen HoG
plt.subplot(1,2,2); plt.imshow(Ihog, cmap='gray')
plt.title('HoG'); plt.axis('off')

plt.show()