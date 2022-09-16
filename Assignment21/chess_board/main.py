import cv2
import numpy as np
rows = 800
cols = 800
arr = np.zeros((rows,cols), dtype=np.uint8)
count = 0
for i in range(0,rows,100):
    count += 1
    for j in range(0,cols,100):
        if count % 2 == 0 :
            arr[i: i + 100, j: j + 100] = 255
        else:
            pass
        count +=1

cv2.imshow('ChessBoard',arr)
cv2.waitKey()
cv2.imwrite('chessBoard.jpg', arr)


