import cv2
import numpy as np

vcap = cv2.VideoCapture("data/pedestrian.mp4")

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

while True:
    r, frame = vcap.read() 
    if r is False:
        break 

    boxes, weights = hog.detectMultiScale(frame, winStride=(8,8) )

    boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

    for (xA, yA, xB, yB) in boxes:
        # display the detected boxes in the colour picture
        cv2.rectangle(frame, (xA, yA), (xB, yB),
                          (0, 255, 0), 2)
    
    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

vcap.release()
cv2.destroyAllWindows()