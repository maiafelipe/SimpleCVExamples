import cv2
import numpy as np

def detect_bounding_box(vid):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
    for (x, y, w, h) in faces:
        cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)
    return faces

vcap = cv2.VideoCapture(0)

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

while True:
    r, frame = vcap.read() 
    if r is False:
        break 

    faces = detect_bounding_box(frame)

    cv2.imshow("Faces", frame) 

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

vcap.release()
cv2.destroyAllWindows()

