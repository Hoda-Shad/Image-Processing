import cv2
img = cv2.imread('4.jpg', 0)
w, h = img.shape
for i in range(w):
    for j in range(h):
        if img[i,j] <= 150:
            img[i,j] = 0


cv2.imwrite('result.jpg',img )
