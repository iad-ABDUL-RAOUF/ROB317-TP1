import numpy as np
import cv2

from matplotlib import pyplot as plt

import sys
if len(sys.argv) != 2:
  print ("Usage :",sys.argv[0],"detector(= orb ou kaze)")
  sys.exit(2)
if sys.argv[1].lower() == "orb":
  detector = 1
elif sys.argv[1].lower() == "kaze":
  detector = 2
else:
  print ("Usage :",sys.argv[0],"detector(= orb ou kaze)")
  sys.exit(2)

#Lecture de la paire d'images
img1 = cv2.imread('./Image_Pairs/torb_small1.png')
print("Dimension de l'image 1 :",img1.shape[0],"lignes x",img1.shape[1],"colonnes")
print("Type de l'image 1 :",img1.dtype)
img2 = cv2.imread('./Image_Pairs/torb_small2.png')
print("Dimension de l'image 2 :",img2.shape[0],"lignes x",img2.shape[1],"colonnes")
print("Type de l'image 2 :",img2.dtype)


# #Affichage des images originales
# plt.subplot(121)
# plt.imshow(img1,cmap = 'gray',vmin = 0.0,vmax = 255.0)
# plt.title('Image 1 originale')
# plt.subplot(122)
# plt.imshow(img2,cmap = 'gray',vmin = 0.0,vmax = 255.0)
# plt.title('Image 2 originale')
# plt.show()


#Début du calcul
t1 = cv2.getTickCount()
#Création des objets "keypoints"
if detector == 1:
  kp1 = cv2.ORB_create(nfeatures = 250,#Par défaut : 500, prof 250 ; nombre de features detectees
                       scaleFactor = 1.2,#Par défaut : 1.2, prof 2 ; plus il est grand, plus les cercles sont petits ? En tout cas il est sensible, selon les valeurs ca fait planter tout le programme.
                       nlevels = 3)#Par défaut : 8, prof 3 ; J'ai regarde la doc pour savoir a quoi correspondent les parametres, et je comprends rien... :'(
  kp2 = cv2.ORB_create(nfeatures=250, #prof = 250
                        scaleFactor = 2.5, # prof = 2.5
                        nlevels = 3) #prof = 3
  print("Détecteur : ORB")
else:
  kp1 = cv2.KAZE_create(upright = False,#Par défaut : false ; Je n'ai vu aucune différence.
    		        threshold = 0.001,#Par défaut : 0.001 ; Plus il est bas, plus il y a de points detectes.
  		        nOctaves = 4,#Par défaut : 4; 12 -> Points detectes sont plus aleatoires, cercles plus grands ? / 1-> peu de points, cerles tout petits.
		        nOctaveLayers = 4,#Par défaut : 4 ; 1-> peu de points, petits / 12 -> plus de points que defaut, un peu plus gros.
		        diffusivity = 2)#Par défaut : 2 ; 1-> petits cercles ; 3-> beaucoup petits cercles; 4-> bcp points
  kp2 = cv2.KAZE_create(upright = False,#Par défaut : false
	  	        threshold = 0.001,#Par défaut : 0.001
		        nOctaves = 4,#Par défaut : 4
		        nOctaveLayers = 4,#Par défaut : 4
		        diffusivity = 2)#Par défaut : 2
  print("Détecteur : KAZE")
#Conversion en niveau de gris
gray1 =  cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
gray2 =  cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
#Détection des keypoints
pts1 = kp1.detect(gray1,None)
pts2 = kp2.detect(gray2,None)
t2 = cv2.getTickCount()
time = (t2 - t1)/ cv2.getTickFrequency()
print("Détection des points d'intérêt :",time,"s")

#Affichage des keypoints
img1 = cv2.drawKeypoints(gray1, pts1, None, flags=4)
# flags définit le niveau d'information sur les points d'intérêt
# 0 : position seule ; 4 : position + échelle + direction
img2 = cv2.drawKeypoints(gray2, pts2, None, flags=4)


plt.subplot(121)
plt.imshow(img1)
plt.title('Image n°1')

plt.subplot(122)
plt.imshow(img2)
plt.title('Image n°2')

plt.show()
