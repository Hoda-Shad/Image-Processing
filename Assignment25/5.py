import cv2
import numpy as np

def filtering(k,img):
    # print(img)
    rows, cols = frame.shape
    result = np.zeros(frame.shape, dtype = int)
    mask1 = (1/k*k) * np.ones((k,k), dtype = int)
    for i in range(k//2 , rows - k//2):
        for j in range(k//2 , cols - k//2):
            # print(i,j,k)
            small_img = img[i - k // 2 : i + k // 2 + 1 , j - k // 2 : j + k // 2 + 1]
            result[i, j] = np.sum(small_img * mask1)
    return (result)




cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if ret == False:
        break

    filter_frame = filtering(15,frame)
    filter_frame = filter_frame.astype(np.uint8)
    target = np.average(filter_frame[180:300, 250:350])
    if target <= 85:
        cv2.putText(filter_frame, "Black", (10, 60), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 3)
    elif target > 85 and target <= 150:
        cv2.putText(filter_frame, "Gray", (10, 60), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 3)
    elif target > 150:
        cv2.putText(filter_frame, "White", (10, 60), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 3)

    cv2.rectangle(frame, (250, 180), (350, 300), (0, 255, 0), 2)
    cv2.imshow('output', filter_frame)

    key = cv2.waitKey(10)
    if key == 27:  # esc
        break