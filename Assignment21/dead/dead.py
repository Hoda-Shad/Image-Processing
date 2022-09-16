import cv2
img = cv2.imread('5.jpg', 0)
w,h  = img.shape
cv2.line(img, (300,0), (0,300), (0,0,0), 40)
cv2.imwrite('dead.jpg', img)


