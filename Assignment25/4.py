import cv2
import numpy as np

img = cv2.imread("image/girl.png", 0)
result = np.zeros(img.shape)
rows, cols = img.shape
filter_size = int(input("Enter size of filter: "))


def filtering(k,img):
    mask1 = np.ones((k,k)) / (k * k)
    # print(mask1)
    for i in range(k//2 , rows - k//2):
        for j in range(k//2 , cols - k//2):
            print(i,j,k)
            small_img = img[i - k // 2 : i + k // 2 + 1 , j - k // 2 : j + k // 2 + 1]
            result[i, j] = np.sum(small_img * mask1)
    return (result)


result = filtering(filter_size, img)
cv2.imwrite("image/res.jpg", result)




