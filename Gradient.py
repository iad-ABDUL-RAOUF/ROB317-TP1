import numpy as np
import cv2

from matplotlib import pyplot as plt

#Lecture image en niveau de gris et conversion en float64
img=np.float64(cv2.imread('../Image_Pairs/FlowerGarden2.png',0))
(h,w) = img.shape
print("Dimension de l'image :",h,"lignes x",w,"colonnes")

#Méthode directe
t1 = cv2.getTickCount()
img2 = cv2.copyMakeBorder(img,0,0,0,0,cv2.BORDER_REPLICATE)
img3 = cv2.copyMakeBorder(img,0,0,0,0,cv2.BORDER_REPLICATE)
for y in range(1,h-1):
  for x in range(1,w-1):
    valy = img[y, x] - img[y-1, x]
    valx = img[y, x] - img[y, x-1]
    val = np.sqrt(valx**2 + valy**2)
    img2[y,x] = min(max(val,0),255)
    if(valx == 0):
        img3[y,x] = np.pi/2
    else:
        img3[y,x] = np.arctan2(valy,valx)
t2 = cv2.getTickCount()
time = (t2 - t1)/ cv2.getTickFrequency()
print("Méthode directe :",time,"s")

# plt.subplot(221)
# plt.imshow(img2,cmap = 'gray')
# plt.title('Gradient horizontal - direct')

# plt.subplot(222)
# plt.imshow(img3,cmap = 'gray')
# plt.title('Gradient vertical - direct')

plt.subplot(221)
plt.imshow(img2,cmap = 'gray')
plt.title('Norme du gradient - direct')

plt.subplot(222)
plt.imshow(img3,cmap = 'gray')
plt.title('Orientation du gradient - direct')

#Méthode filter2D
t1 = cv2.getTickCount()
kernel = np.array([[0, -1, 0],[0, 1, 0],[0, 0, 0]])
img4 = cv2.filter2D(img,-1,kernel) #-1, profondeur = nombre de canaux (i.e. couleurs)
kernel2 = np.array([[0, 0, 0],[-1, 1, 0],[0, 0, 0]])
img5 = cv2.filter2D(img,-1,kernel2) #-1, profondeur = nombre de canaux (i.e. couleurs)
img6 = np.sqrt(img4**2 + img5**2)
img7 = np.arctan2(img4,img5)
t2 = cv2.getTickCount()
time = (t2 - t1)/ cv2.getTickFrequency()
print("Méthode filter2D :",time,"s")

# plt.subplot(223)
# plt.imshow(img4,cmap = 'gray',vmin = 0.0,vmax = 255.0)
# plt.title('Gradient horizontal - filter2D')

# plt.subplot(224)
# plt.imshow(img5,cmap = 'gray',vmin = 0.0,vmax = 255.0)
# plt.title('Gradient vertical - filter2D')

plt.subplot(223)
plt.imshow(img6,cmap = 'gray')
plt.title('Norme du gradient - filter2D')
plt.subplot(224)
plt.imshow(img7,cmap = 'gray')
plt.title('Orientation du gradient - filter2D')

plt.show()