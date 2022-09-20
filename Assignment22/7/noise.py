import random

import cv2
img = cv2.imread('L.jfif', 0 )
w,h = img.shape
for k in range(90):
    i = random.randint(0,w)
    j = random.randint(0,h)
    img[i,j] = 255
cv2.imshow('result', img)
cv2.waitKey()
cv2.imwrite('result.jpg', img)