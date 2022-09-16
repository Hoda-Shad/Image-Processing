import cv2
import numpy as np
img8 = np. zeros((300, 300) )
img8[:,:]=100
cv2.line(img8, (100,100), (100,200), (0,0,255), 30)
cv2.line(img8, (200,100), (200,200), (0,0,255), 30)
cv2.line(img8, (100,150), (200,150), (0,0,255), 30)
cv2.imwrite('name.jpg', img8)