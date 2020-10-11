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
for y in range(1,h-1):
  for x in range(1,w-1):
    val = 5*img[y, x] - img[y-1, x] - img[y, x-1] - img[y+1, x] - img[y, x+1] 
    img2[y,x] = min(max(val,0),255)
t2 = cv2.getTickCount()
time = (t2 - t1)/ cv2.getTickFrequency()
print("Méthode directe :",time,"s")

plt.subplot(121)
plt.imshow(img2,cmap = 'gray')
plt.title('Convolution - Méthode Directe')

#Méthode filter2D
t1 = cv2.getTickCount()
kernel = np.array([[0, -1, 0],[-1, 5, -1],[0, -1, 0]])
img3 = cv2.filter2D(img,-1,kernel)
t2 = cv2.getTickCount()
time = (t2 - t1)/ cv2.getTickFrequency()
print("Méthode filter2D :",time,"s")

plt.subplot(122)
plt.imshow(img3,cmap = 'gray',vmin = 0.0,vmax = 255.0)
plt.title('Convolution - filter2D')
plt.show()

'''
plt.imshow(img2,cmap = 'gray')
plt.title('Convolution - Méthode Directe')
plt.show()

plt.imshow(img3,cmap = 'gray',vmin = 0.0,vmax = 255.0)
plt.title('Convolution - filter2D')
plt.show()
'''
plt.imshow(img,cmap = 'gray',vmin = 0.0,vmax = 255.0)
plt.title('image originale')
plt.show()


# derivé partielle selon x (méthode directe)
img4 = cv2.copyMakeBorder(img,0,0,0,0,cv2.BORDER_REPLICATE)
for y in range(0,h):
  for x in range(1,w-1):
    val = img[y, x+1] - img[y, x-1]
    #img4[y,x] = min(max(val,0),255)
    img4[y,x] = val

# derivé partielle selon x (méthode filter2D)
kernel = np.array([[-1, 0, 1]])
img5 = cv2.filter2D(img,-1,kernel)

# derivé partielle selon y (méthode directe)
img6 = cv2.copyMakeBorder(img,0,0,0,0,cv2.BORDER_REPLICATE)
for y in range(1,h-1):
  for x in range(0,w):
    val = img[y+1, x] - img[y-1, x]
    img6[y,x] = val

# derivé partielle selon y (méthode filter2D)
kernel = np.array([[-1, 0, 1]]).T
img7 = cv2.filter2D(img,-1,kernel)

plt.subplot(221)
plt.imshow(img4,cmap = 'gray')
plt.title('Ix direct')

plt.subplot(222)
plt.imshow(img5,cmap = 'gray')
plt.title('Ix filter2D')

plt.subplot(223)
plt.imshow(img6,cmap = 'gray')
plt.title('Iy direct')

plt.subplot(224)
plt.imshow(img7,cmap = 'gray')
plt.title('Iy filter2D')
plt.show()

'''
max(max(x) for x in img4)
min(min(x) for x in img4)

max(max(x) for x in img5)
min(min(x) for x in img5)
'''

module_grad = np.sqrt(img5*img5+img7*img7)
plt.imshow(module_grad,cmap = 'gray')
plt.title('module gradient')
plt.show()

orientation = np.arctan2(img7,img5)
plt.imshow(orientation,cmap = 'gray')
plt.title('argument gradient')
plt.show()

