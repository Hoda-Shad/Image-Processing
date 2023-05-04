import cv2
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description = "Hoda Sudoku detector, v = 1.0 ")
parser.add_argument("input", type = str ,  help = "The path of Input image")
parser.add_argument("output", type= str ,  help = "The path of Output image")

args = parser.parse_args()


img = cv2.imread(args.input)
Grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_blured = cv2.GaussianBlur(Grayimg, (7,7), 3)
plt.imshow(img_blured, cmap = "gray")
thresh = cv2.adaptiveThreshold(img_blured,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
plt.imshow(thresh , cmap="gray")
thresh = cv2.bitwise_not(thresh)
plt.imshow(thresh, cmap="gray")
contours = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0]
contours = sorted(contours, key = cv2.contourArea, reverse = True)
Sudoku_contour = None
for contour in contours:
    epsilon = 0.02 * (cv2.arcLength(contour, True))
    approx = cv2.approxPolyDP(contour, epsilon , True)
    if len(approx) == 4 : 
        Sudoku_contour = approx
        print(Sudoku_contour)
        break
if Sudoku_contour is None:
    print("Not Found!")
result = cv2.drawContours(img, [Sudoku_contour], -1 , (0,255,0), 20)
plt.imshow(result)
cv2.imwrite(args.output, result)
