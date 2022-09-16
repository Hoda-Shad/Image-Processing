import cv2
import numpy as np
img = np.zeros((200,200),np.uint8)
w,h = 200,200
for i in range (w-1,0,-1):
        img[i,:] = img[i,:] - i

cv2.imwrite('result.jpg', img)