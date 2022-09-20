import cv2
import numpy as np

image = []
img1 = cv2.imread('highway/h0.jpg', 0 )
w,h = img1.shape
ave = np.zeros((w,h),dtype = 'uint8')
ave[:] = 255

for i in range(15):
    image = cv2.imread(f'highway/h{i}.jpg',0)
    ave += image // 15


cv2.imshow('result.jpg', ave)
cv2.waitKey()
cv2.imwrite('result.jpg', ave)