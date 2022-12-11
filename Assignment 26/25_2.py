import cv2
import numpy as np


cam = cv2.VideoCapture(0)
while True:
    ret, frame = cam.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if ret == False:
        break

    id_kernel = np.ones((25, 25)) / 625
    filter_frame = cv2.filter2D(frame, -1, id_kernel)
    alpha = 3
    beta = 0
    filter_frame = cv2.convertScaleAbs(filter_frame, alpha=alpha, beta=beta)
    target = np.average(filter_frame[180:300, 250:350])

    if target <= 85:
        cv2.putText(filter_frame, "Black", (10, 60), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 3)
    elif target > 85 and target <= 150:
        cv2.putText(filter_frame, "Gray", (10, 60), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 3)
    elif target > 150:
        cv2.putText(filter_frame, "White", (10, 60), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 3)

    cv2.rectangle(filter_frame, (250, 180), (350, 300), (0, 255, 0), 2)
    cv2.imshow('output', filter_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    cv2.waitKey(10)

