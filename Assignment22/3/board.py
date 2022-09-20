import cv2
img1 = cv2.imread('hw2/board - origin.bmp', 0 )
img2 = cv2.imread('hw2/board - test.bmp', 0 )
img2 = cv2.flip(img2, 1)
img3 = cv2.subtract(img1, img2)

cv2.imwrite('result.jpg', img3)