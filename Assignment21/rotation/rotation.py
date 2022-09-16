import cv2
img1 = cv2.imread('img/3.jpg')
image2 = cv2.rotate(img1,cv2.ROTATE_180)
cv2.imwrite('img.jpg',image2)
cv2.imshow('rotate 180',image2)
cv2.waitKey()
