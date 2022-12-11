import cv2
import keyboard
import numpy as np

video_cap = cv2.VideoCapture(0)
face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_detector = cv2.CascadeClassifier("haarcascade_eye.xml")
lip_detector = cv2.CascadeClassifier("haarcascade_smile.xml")
img = cv2.imread('img/eyeee.png')
liiip = cv2.imread('img/liiip.jfif')


def Add_face_emoji(frame):
    emoji = cv2.imread('emoji2.png')
    faces = face_detector.detectMultiScale(frame, 1.3)
    for face in faces:
        x, y, w, h = face
        emoji = cv2.resize(emoji, (w, h))
        frame[y:y + h, x:x + w] = emoji
        for i in range(y, y + h):
            for j in range(x, x + w):
                pixel = frame[i, j]
                if all(pixel < 100):
                    frame[i, j] = f[i, j]
    return frame

def Add_eye_lips_emoji(frame):
    eyes = eye_detector.detectMultiScale(frame, 1.3, minSize=(50, 50))
    lips = lip_detector.detectMultiScale(frame,3.8, minSize=(35, 35))
    for eye in eyes:
        x, y, w, h = eye
        emoji = cv2.resize(img, (w, h))
        frame[y:y + h, x:x + w] = emoji
        for i in range(y, y + h):
            for j in range(x, x + w):
                pixel = frame[i, j]
                if all(pixel > 200):
                    frame[i, j] = f[i, j]
    for lip in lips:
        x, y, w, h = lip
        emoji = cv2.resize(liiip, (w, h))
        frame[y:y + h, x:x + w] = emoji
        # #
        for i in range(y, y + h):
            for j in range(x, x + w):
                pixel = frame[i, j]
                if all(pixel > 200):
                    frame[i, j] = f[i, j]
    return frame

def Checkered_face(frame):

    faces = face_detector.detectMultiScale(frame, 1.3)
    for face in faces:
        x, y, w, h = face
    low_quality_img = cv2.resize(frame[y:y+ h, x:x+ w], (0, 0), fx=0.1, fy=0.1)
    frame[y:y + h, x:x + w] = cv2.resize(low_quality_img, (w, h),interpolation=cv2.INTER_NEAREST)
    return frame


def Flipped_horizental(frame):
    faces = face_detector.detectMultiScale(frame, 1.3)
    for face in faces:
        x, y, w, h = face
        frame[y:y + h, x:x + w] = cv2.rotate(frame[y:y + h, x:x + w], cv2.ROTATE_180)
    return frame


def Blur(frame):

    faces = face_detector.detectMultiScale(frame, 1.3)
    kernel = np.ones((15,15) , dtype=np.uint8) / 225
    for face in faces:
        x,y,w,h = face
        frame[y:y + h, x:x + w] = cv2.filter2D(frame[y:y + h, x:x + w],-1,kernel)
        return (frame)

while True:
    ret, frame = video_cap.read()
    if not ret:
        break
    f = frame.copy()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if keyboard.is_pressed('1'):
        frame = Add_face_emoji(frame)
        cv2.waitKey(0)

    if keyboard.is_pressed('2'):
        frame = Add_eye_lips_emoji(frame)

    if keyboard.is_pressed('3'):
        frame = Checkered_face(frame)

    if keyboard.is_pressed('4'):
        frame = Flipped_horizental(frame)

    if keyboard.is_pressed('5'):
        frame = Blur(frame)

    if keyboard.is_pressed('ESC'):
        exit()

    cv2.imshow('output', frame)
    cv2.waitKey(10)
