# -*- coding: utf-8 -*-
#
# Adaptado de:
# https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_video_display/py_video_display.html

import cv2

# Para grabar un video, cambiar esta variable a: True
generar_video = False

# Instancia que capturará el video
cam = cv2.VideoCapture(0)

if (generar_video):
    # Define el codec y crea el objeto VideoWriter que almacenará el video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

# Variable para almacenar el tamaño de la imagen
Isize = None

# Mientras la cámara esté "abierta"
print("\nIMPORTANTE: Para detener la cámara presionar \"q\"")
while(cam.isOpened()):
    # Leer la imagen de la cámara: 
    #   retval=1 si hay una imagen válida
    #   frame es la imagen (en caso que retval sea 1)
    retval, frame = cam.read()
    
    # Solo si se tiene una imagen válida
    if retval==True:
        # Mostrar el tamaño de la imagen solo 1 vez
        if (Isize==None):
            Isize = frame.shape
            print("Tamaño de la imagen: ", Isize)
        # Voltear horizontalmente (opcional)
        #frame = cv2.flip(frame, 1)
        # Si se desea grabar el video, se almacena en "out"
        if (generar_video):
            out.write(frame)
        # Mostrar la imagen de la cámara
        cv2.imshow('My camera', frame)
        # Esperar 30ms entre imágenes
        # Si se presiona 'q' se termina el bucle
        if (cv2.waitKey(30) & 0xFF) == ord('q'):
            break
    else:
        break
    
# Cuando termine de usarse la cámara, eliminar los recursos
cam.release()
cv2.destroyAllWindows()
# Detener la grabación
if (generar_video):
    out.release()

