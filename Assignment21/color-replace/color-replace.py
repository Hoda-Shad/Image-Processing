import cv2
img1 = cv2.imread('img/1.jpg')
img2 = cv2.imread('img/2.jpg')
img11 = 255 - img1[:]
img22 = 255 - img2[:]
cv2.imshow('girl', img11)
cv2.waitKey()
cv2.imshow('boy', img22)
cv2.waitKey()
cv2.imwrite('girl.jpg', img11)
cv2.imwrite('boy.jpg', img22)