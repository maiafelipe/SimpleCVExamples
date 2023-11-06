import cv2
import numpy as np

img = cv2.imread("data/lenna.png")

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_classifier.detectMultiScale(img_gray, 1.1, 5, minSize=(40, 40))
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)

cv2.namedWindow("Lenna Original", cv2.WINDOW_NORMAL)
cv2.imshow("Lenna Original", img)
cv2.waitKey(0)

