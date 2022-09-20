import cv2
import numpy as np
img1 = cv2.imread('R.jfif',0)
img2 = cv2.imread('L.jfif',0)

w,h = img1.shape
img2 = cv2.resize(img2,(h//4,w//4))
img1 = cv2.resize(img1,(h//4,w//4))

img3 = img2 + img1
img4 = img2+ img1//2
img5 = img2 + img1//4
img6 = img2 + img1//8

final = cv2.hconcat([img1, img3, img4, img5, img6, img2])
cv2.imshow('final',final)
cv2.waitKey()
cv2.imwrite('result.jpg', final)
