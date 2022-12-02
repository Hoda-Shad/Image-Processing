import cv2
import numpy
import numpy as np

img = cv2.imread("building.tif",0)
rows , cols = img.shape
result = np.zeros(img.shape)
mask = np.array([[-1,-1,-1],[-0,0,0],[1,1,1]])

for i in range (1,rows-1):
    for j in range(1, cols-1):
        small_img = img[i-1:i+2, j-1:j+2]
        result[i,j] = np.sum(mask*small_img)

cv2.imwrite('result3_2.jpg',result)