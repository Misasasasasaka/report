from PIL import Image,ImageEnhance
img_original = Image.open("example.jpg")
img_original.show("Original Image")
img = ImageEnhance.Contrast(img_original)
img.enhance(3.8).show("Image With More Contrast")

input()

from PIL import Image
import numpy as np
img = np.array(Image.open('example.jpg'))
img_red = img.copy()
img_red[:, :, (1, 2)] = 0
img_green = img.copy()
img_green[:, :, (0, 2)] = 0
img_blue = img.copy()
img_blue[:, :, (0, 1)] = 0
img_ORGB = np.concatenate((img,img_red, img_green, img_blue), axis=1)
img_converted = Image.fromarray(img_ORGB)
img_converted.show()

input()

import cv2
img = cv2.imread("example.jpg")
imgCropped = img[150:383,125:290]
shape = imgCropped.shape
print(shape[0])
imgCropped = cv2.resize(imgCropped,(shape[0]*12//10,shape[1]*2))
cv2.imshow("Image cropped",imgCropped)
cv2.imshow("Image",img)
cv2.waitKey(0)


