import cv2
a = cv2.imread('img/a.tif', 0)
b = cv2.imread('img/b.tif', 0)
c = cv2.subtract(b,a)
cv2.imwrite("result.jpg",c)
