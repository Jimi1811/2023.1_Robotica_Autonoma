# -*- coding: utf-8 -*-
import cv2

def is_inside(i, o):
    # Determinar si un rectángulo se encuentra dentro de otro rectángulo
    #   i - posible rectángulo interior
    #   o - posible rectángulo exterior
    ix, iy, iw, ih = i
    ox, oy, ow, oh = o
    return ix > ox and ix + iw < ox + ow and iy > oy and iy + ih < oy + oh


I = cv2.imread('images/campo.jpg')

# Crear una instancia del descriptor HoG
hog = cv2.HOGDescriptor()
# Especificar que se utilizará el detector de personas pre-entrenado
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Detectar personas en la imagen: retorna rectángulos y certezas
rectangs, pesos = hog.detectMultiScale(I, winStride=(4, 4),
                                       scale=1.02, finalThreshold=1.9)

# Filtrar para remover rectángulos anidados
rects_filt = []
pesos_filt = []
for ri, r in enumerate(rectangs):
    for qi, q in enumerate(rectangs):
        if ri != qi and is_inside(r, q):
            break
    else:
        rects_filt.append(r)
        pesos_filt.append(pesos[ri])

# Dibujar los rectángulos restantes
for ri, r in enumerate(rects_filt):
    x, y, w, h = r
    cv2.rectangle(I, (x, y), (x + w, y + h), (0, 255, 255), 2)
    text = '%.2f' % pesos_filt[ri]
    cv2.putText(I, text, (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

cv2.imshow('Personas detectadas', I)
cv2.waitKey(0)
cv2.destroyAllWindows()