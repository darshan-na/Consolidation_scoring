import cv2
import math
import numpy as np
import math
import matplotlib.pyplot as plt
import PIL
from PIL import Image


img = cv2.imread(r'C:\Users\Dinesh M S\Downloads\archive\chest_xray\train\PNEUMONIA\person8_bacteria_37.jpeg')
print(img.shape) # 512x512
width = 500
height = 500
dim = (width, height)
 
# resize image
resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
 
print('Resized Dimensions : ',resized.shape)
 
cv2.imshow("Resized image", resized)
img_2=cv2.cvtColor(resized,cv2.COLOR_BGR2GRAY)
windowsize_r = 255
windowsize_c = 255
for r in range(0,windowsize_r,512):
    for c in range(0,windowsize_c,512):
        LL = resized[r:r+windowsize_r,c:c+windowsize_c]
        gray_img=cv2.cvtColor(LL,cv2.COLOR_BGR2GRAY)
print(LL)
#tile=cv2.imshow("B",LL)

for r in range(0,windowsize_r,512):
    for c in range(256,512,256):
        LR = resized[r:r+windowsize_r,c:c+windowsize_c]
print(LR)
#tileb = cv2.imshow("C",LR)


for r in range(256,512,256):
    for c in range(0,windowsize_c,512):
        RL = resized[r:r+windowsize_r,c:c+windowsize_c]
print(RL)
#cv2.imshow("p",RL)

for r in range(256,512,256):
    for c in range(256,512,256):
        RR = resized[r:r+windowsize_r,c:c+windowsize_c]
print(RR)
#cv2.imshow("r",RR)


gray_img=cv2.cvtColor(LL,cv2.COLOR_BGR2GRAY)

hist=cv2.calcHist(gray_img,[0],None,[256],[0,256])

plt.subplot(121)
plt.title("Image1")
plt.xlabel('bins')
plt.ylabel("No of pixels")
plt.plot(hist)
#plt.show()
gray_img_eqhist=cv2.equalizeHist(gray_img)

hist=cv2.calcHist(gray_img_eqhist,[0],None,[256],[0,256])

plt.subplot(121)
plt.plot(hist)

#plt.show()
#cv2.imshow("Img3",gray_img_eqhist)

clahe=cv2.createCLAHE(clipLimit=20)
gray_img_clahe=clahe.apply(gray_img_eqhist)




#m=cv2.imshow("m",gray_img_clahe)



gray_img1=cv2.cvtColor(LR,cv2.COLOR_BGR2GRAY)

hist1=cv2.calcHist(gray_img1,[0],None,[256],[0,256])

plt.subplot(122)
plt.title("Image1")
plt.xlabel('bins')
plt.ylabel("No of pixels")
plt.plot(hist1)
#plt.show()
gray_img_eqhist1=cv2.equalizeHist(gray_img1)

hist1=cv2.calcHist(gray_img_eqhist1,[0],None,[256],[0,256])

plt.subplot(121)
plt.plot(hist1)

#plt.show()
#Img8=cv2.imshow("Img8",gray_img_eqhist1)

clahe=cv2.createCLAHE(clipLimit=20)
gray_img_clahe1=clahe.apply(gray_img_eqhist1)

#m1=cv2.imshow("m1",gray_img_clahe1)


gray_img2=cv2.cvtColor(RL,cv2.COLOR_BGR2GRAY)

hist2=cv2.calcHist(gray_img2,[0],None,[256],[0,256])

plt.subplot(122)
plt.title("Image1")
plt.xlabel('bins')
plt.ylabel("No of pixels")
plt.plot(hist1)
#plt.show()
gray_img_eqhist2=cv2.equalizeHist(gray_img2)

hist1=cv2.calcHist(gray_img_eqhist2,[0],None,[256],[0,256])

plt.subplot(122)
plt.plot(hist1)

#plt.show()
#cv2.imshow("Img2",gray_img_eqhist2)

clahe=cv2.createCLAHE(clipLimit=20)
gray_img_clahe2=clahe.apply(gray_img_eqhist2)

#m2=cv2.imshow("m2",gray_img_clahe2)



gray_img3=cv2.cvtColor(RR,cv2.COLOR_BGR2GRAY)

hist3=cv2.calcHist(gray_img3,[0],None,[256],[0,256])

plt.subplot(121)
plt.title("Image1")
plt.xlabel('bins')
plt.ylabel("No of pixels")
plt.plot(hist1)
#plt.show()
gray_img_eqhist3=cv2.equalizeHist(gray_img3)

hist3=cv2.calcHist(gray_img_eqhist3,[0],None,[256],[0,256])

plt.subplot(122)
plt.plot(hist1)

#plt.show()
#cv2.imshow("Img1",gray_img_eqhist3)

clahe=cv2.createCLAHE(clipLimit=20)
gray_img_clahe3=clahe.apply(gray_img_eqhist3)

#m4=cv2.imshow("m4",gray_img_clahe3)

ksize = 1#Use size that makes sense to the image and fetaure size. Large may not be good. 
#On the synthetic image it is clear how ksize affects imgae (try 5 and 50)
sigma = 5#Large sigma on small features will fully miss the features. 
theta = 1*np.pi/4 #/4 shows horizontal 3/4 shows other horizontal. Try other contributions
lamda = 1*np.pi/2#1/4 works best for angled. 
gamma=0.9  #Value of 1 defines spherical. Calue close to 0 has high aspect ratio
#Value of 1, spherical may not be ideal as it picks up features from other regions.
phi = 1 #Phase offset. I leave it to 0. (For hidden pic use 0.8)


kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, phi, ktype=cv2.CV_32F)

#plt.imshow(kernel)

#img = cv2.imread('images/synthetic.jpg')
#img = cv2.imread('images/zebra.jpg')  #Image source wikipedia: https://en.wikipedia.org/wiki/Plains_zebra
#img = cv2.imread('images/hidden.jpg') #USe ksize:15, s:5, q:pi/2, l:pi/4, g:0.9, phi:0.8
#plt.imshow(gray_img_eqhist1, cmap='gray')
#plt.imshow(gray_img_eqhist2, cmap='gray')
#plt.imshow(gray_img_eqhist3, cmap='gray')
#plt.imshow(gray_img_eqhist, cmap='gray')


fimg1= cv2.filter2D(gray_img_clahe1, cv2.CV_8UC3, kernel)
fimg2 = cv2.filter2D(gray_img_clahe2, cv2.CV_8UC3, kernel)
fimg3= cv2.filter2D(gray_img_clahe3, cv2.CV_8UC3, kernel)
fimg = cv2.filter2D(gray_img_clahe, cv2.CV_8UC3, kernel)

#kernel_resized = cv2.resize(kernel, (400, 400))                    # Resize image
#fimg4 = cv2.filter2D(fimg, cv2.CV_8UC3, kernel)

#plt.imshow(kernel_resized)
#plt.imshow(fimg, cmap='gray')
#plt.imshow(fimg1, cmap='gray')
#plt.imshow(fimg2, cmap='gray')
#plt.imshow(fimg3, cmap='gray')

#edges = cv2.Canny(fimg2,50,50)
#cv2.imshow('Kernel', kernel_resized)
#cv2.imshow('Original Img.', img)
#cv2.imshow('Filtered', fimg1)
#cv2.imshow('F3', fimg)
#cv2.imshow('F2', fimg2)
#cv2.imshow('F', edges)
#cv2.imshow('F', fimg2)
#cv2.imshow('F4', fimg3)
 #rejoining 3072 blocks into 1 image

# importing the libraries

  
# creating an array using np.full 
# 255 is code for white color
array_created = np.full((500, 500, 3),
                        255, dtype = np.uint8)
  
# displaying the image
#cv2.imshow("image", array_created)
img_1=cv2.cvtColor(array_created,cv2.COLOR_BGR2GRAY)


#ray_img_clahe1.shape
#img2_resized = cv2.resize(gray_img_clahe1, (arr.shape[1], arr.shape[0]))
 

for r in range(0,windowsize_r,512):
	for c in range(0,windowsize_c,512):
		x_offset = 0
		y_offset = 0
		x_end = x_offset + gray_img_clahe.shape[1]
		y_end = y_offset + gray_img_clahe.shape[0]
		img_1[y_offset:y_end,x_offset:x_end] = gray_img_clahe
		
for r in range(0,windowsize_r,512):
	for c in range(256,512,256):
		x_offset = 245
		y_offset = 0
		x_end = x_offset + gray_img_clahe1.shape[1]
		y_end = y_offset + gray_img_clahe1.shape[0]
		img_1[y_offset:y_end,x_offset:x_end] = gray_img_clahe1
	    

for r in range(256,512,256):
	for c in range(0,windowsize_c,512):
		x_offset = 0
		y_offset = 245
		x_end = x_offset + gray_img_clahe2.shape[1]
		y_end = y_offset + gray_img_clahe2.shape[0]
		img_1[y_offset:y_end,x_offset:x_end] = gray_img_clahe2

for r in range(256,512,256):
	for c in range(256,512,256):
	 
		x_offset = 245
		y_offset = 245
		x_end = x_offset + gray_img_clahe3.shape[1]
		y_end = y_offset + gray_img_clahe3.shape[0]
		img_1[y_offset:y_end,x_offset:x_end] = gray_img_clahe3

cv2.imshow('image_merged', img_1)


#sub = cv2.subtract(gray_img, gray_img_clahe )
th, dst = cv2.threshold(img_1, 150, 255,   cv2.THRESH_BINARY | cv2.THRESH_OTSU);

img_neg1 = 1.0 -  img_1
cv2.imshow('image negation for threshold', img_neg1)



norm_image = cv2.normalize(img_neg1, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(norm_image)
cv2.circle(norm_image, maxLoc, 5, (255, 0, 0), 2)
cv2.imshow(' normalised Image', norm_image)
print( maxVal)




result_image = cv2.multiply( norm_image.astype(np.uint8),img_2)
cv2.imshow('Multiply Image', result_image)

#m=cv2.imshow("m",gray_img_clahe)
sum = cv2.addWeighted(img_2, 0.4, img_1, 0.8, 0)

#sub = cv2.subtract(gray_img, sum)
img_neg = 1 -  sum 

#sub = gray_img - sum
cv2.imshow('negation of sum ', img_neg)# rib supress
img3 = img_neg + img_2

#cv2.imshow('sub', sub)
cv2.imshow('addition of gray_img and img_neg', img3)
#cv2.imshow('add', img3)
cv2.waitKey(0)
cv2.destroyAllWindows()