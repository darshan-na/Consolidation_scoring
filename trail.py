import cv2
import math
import numpy as np
import math
import matplotlib.pyplot as plt
import PIL
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage import data
from skimage.filters import gaussian
from skimage.segmentation import active_contour
img = cv2.imread(r'C:\Users\Dinesh M S\Downloads\archive\chest_xray\train\PNEUMONIA\person8_bacteria_37.jpeg')
print(img.shape) # 512x512
width = 500
height = 500
dim = (width, height)
 
# resize image
resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
 
print('Resized Dimensions : ',resized.shape)

cv2.imshow("Resized image", resized)

gray_img=cv2.cvtColor(resized,cv2.COLOR_BGR2GRAY)

hist=cv2.calcHist(gray_img,[0],None,[256],[0,256])

plt.subplot(121)
plt.title("Image1")
plt.xlabel('bins')
plt.ylabel("No of pixels")
plt.plot(hist)
plt.show()
gray_img_eqhist=cv2.equalizeHist(gray_img)

hist=cv2.calcHist(gray_img_eqhist,[0],None,[256],[0,256])

plt.subplot(121)
plt.plot(hist)

plt.show()
#cv2.imshow("Img3",gray_img_eqhist)

clahe=cv2.createCLAHE(clipLimit=20)
gray_img_clahe=clahe.apply(gray_img_eqhist)

th, dst = cv2.threshold(gray_img_clahe, 150, 255,   cv2.THRESH_BINARY | cv2.THRESH_OTSU);

img_neg1 = 1.0 -  dst
cv2.imshow('negation of threshold', img_neg1)


m4=cv2.imshow("threshold",dst)
norm_image = cv2.normalize(img_neg1, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(norm_image)
cv2.circle(norm_image, maxLoc, 5, (255, 0, 0), 2)
cv2.imshow(' normalised Image', norm_image)
print( maxVal)

#when printing while negating things are not the same pixel value but need to get on with the same pixel value.




result_image = cv2.multiply( norm_image.astype(np.uint8),gray_img)
cv2.imshow('Multiply Image', result_image)




s = np.linspace(0, 2*np.pi, 400)
r = 100 + 100*np.sin(s)
c = 220 + 100*np.cos(s)
init = np.array([r, c]).T

snake = active_contour(gaussian(img, 3, preserve_range=False),
                       init, alpha=0.015, beta=9, gamma=0.001)

fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(img, cmap=plt.cm.gray)
ax.plot(init[:, 1], init[:, 0], '--r', lw=3)
ax.plot(snake[:, 1], snake[:, 0], '-b', lw=3)
ax.set_xticks([]), ax.set_yticks([])
ax.axis([0, img.shape[1], img.shape[0], 0])

plt.show()


#m=cv2.imshow("m",gray_img_clahe)
sum = cv2.addWeighted(gray_img, 0.4, gray_img_clahe, 0.8, 0)

#sub = cv2.subtract(gray_img, sum)
img_neg = 1 -  sum 

#sub = gray_img - sum
cv2.imshow('negation of sum ', img_neg)# rib compress
img3 = img_neg + gray_img
#sum1 = cv2.addWeighted(gray_img, 0.4, img3, 0.8, 0)
#cv2.imshow('sub', img3)
#cv2.imshow('sub1', sum)
#cv2.imshow('sub2', sum1)
cv2.imshow('addition of gray_img and img_neg', img3)
cv2.waitKey(0)
cv2.destroyAllWindows()


