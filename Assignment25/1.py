import cv2
import numpy as np
img = cv2.imread("flower_input.jpg",0)
result = np.zeros(img.shape)
rows,cols = img.shape
mask = np.ones((21, 21)) / 400

for i in range (10,rows-10):
    for j in range (10,cols-10):
        if img[i][j] < 180:
            small_img = img[i-10:i+11, j-10:j+11]
            result[i, j] = np.sum(small_img * mask)
        else:
            result[i][j] = img[i][j]

cv2.imwrite("flo_output_me.jpg", result)
