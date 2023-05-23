# -*- coding: utf-8 -*-
#
# Adaptado de
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_brief/py_brief.html#brief
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_orb/py_orb.html
# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_fast/py_fast.html
# https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_sift_intro/py_sift_intro.html
# https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_surf_intro/py_surf_intro.html

import cv2

I = cv2.imread('images/blocks.jpg')
Igray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
print("\nIMPORTANTE: Para cerrar las imagenes, presionar cualquier tecla")


# =======================================================
#  Descriptor: ORB (Oriented FAST and Rotated BRIEF)
#  Más información en: https://docs.opencv.org/3.4/db/d95/classcv_1_1ORB.html
# =======================================================

# Inicializar detector ORB
orb = cv2.ORB_create()

# "Keypoints" con ORB
keypts = orb.detect(I,None)
# Calcular los descriptores con ORB
keypts, descript_orb = orb.compute(I, keypts)

# Dibuja los "keypoints" (sin incluir tamaño ni orientación)
Iorb = I.copy()
cv2.drawKeypoints(I, keypts, Iorb, color=(0,255,0), flags=0)

cv2.imshow('Usando ORB', Iorb)
cv2.waitKey(0)
cv2.destroyAllWindows()



# ==============================================================
#  Descriptor BRISK con detector FAST: 
#   FAST (Features from Accelerated Segment Test)
#   BRISK (Binary Robust Invariant Scalable Keypoints)
# ==============================================================

# Iniciar del detector FAST con valores por defecto
fast = cv2.FastFeatureDetector_create()
# "Keypoints" usando FAST
keypts = fast.detect(Igray, None)

# Descriptores usando BRISK (a partir de puntos dados por FAST)
br = cv2.BRISK_create();
keypts, descript_brisk = br.compute(I, keypts)

# Dibujar y mostrar los keypoints en la imagen
Ifast = I.copy()
cv2.drawKeypoints(I, keypts, Ifast, color=(255,0,0))
cv2.imshow('FAST', Ifast)

# Mostrar los parámetros por defecto
print("Threshold: ", fast.getThreshold())
print("nonmaxSuppression: ", fast.getNonmaxSuppression())
print("neighborhood: ", fast.getType())
print("Keypoints con supresión de no máximos: ", len(keypts))


# Detección sin supresión de no máximos (non-max suppression)
# ----------------------------------------------------------
fast.setNonmaxSuppression(0)
keypts2 = fast.detect(Igray, None)

print("Keypoints sin supresión de no máximos: ", len(keypts2))
Ifast2 = I.copy()
cv2.drawKeypoints(I, keypts2, Ifast2, color=(255,0,0))

cv2.imshow('FAST sin non-max sup', Ifast2)
cv2.waitKey(0)
cv2.destroyAllWindows()





# =====================================================================
# Descriptor BRIEF con detector STAR
#  BRIEF (Binary Robust Independent Elementary Features)
#  STAR (STAR Detector, llamado CenSurE: Center Surround Extremas)
# =====================================================================

# Necesita: opencv-contrib-python (se puede instalar con pip install) 
use_brief = False

if (use_brief):
    # Inicializar el detector STAR
    star = cv2.xfeatures2d.StarDetector_create()
    # Inicializar el descriptor BRIEF
    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
    
    # Encontrar "keypoints" usando STAR (CenSurE)
    keypts = star.detect(I, None)
    # Calcular los descriptores usando BRIEF
    keypts, descript_brief = brief.compute(I, keypts)
    
    # Dibuja los "keypoints" (sin incluir tamaño ni orientación)
    Ibrief = I.copy()
    cv2.drawKeypoints(I, keypts, Ibrief, color=(0,255,0), flags=0)
    
    cv2.imshow('Usando BRIEF', Ibrief)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ==========================================
# SIFT (Scale-Invariant Feature Transform)
# ==========================================

# Solo se puede usar sift con versiones antiguas de opencv por patentes
# Necesita opencv-contrib-python
use_sift = False

if (use_sift):
    sift = cv2.xfeatures2d.SIFT_create()
    keypts = sift.detect(Igray, None)

    Isift = I.copy()
    cv2.drawKeypoints(Igray, keypts, Isift, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    cv2.imshow('Usando SIFT', Isift)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    keypts, descriptors = sift.detectAndCompute(Igray, None)
    # descriptors será: Nkeypoints x 128


# ======================================
# SURF (Speeded-Up Robust Features)
# ======================================

# Solo se puede usar surf con versiones antiguas de opencv por patentes
# Necesita opencv-contrib-python-nonfree
use_surf = False
if (use_surf):
    hessian_threshold = 4000
    surf = cv2.xfeatures2d.SURF_create(hessian_threshold)
    keypoints, descriptors = surf.detectAndCompute(Igray, None)
    
    Isurf = I.copy()
    cv2.drawKeypoints(Igray, keypoints, Isurf, (255,0,0), 4)
    
    cv2.imshow('Usando SURF', Isurf)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

