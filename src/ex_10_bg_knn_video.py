import cv2
import numpy as np


vcap = cv2.VideoCapture(0)

backSub = cv2.createBackgroundSubtractorKNN()

while True:
    r, frame = vcap.read() 
    if r is False:
        break 

    fgMask = backSub.apply(frame)

    cv2.imshow('Frame', frame)
    cv2.imshow('FG Mask', fgMask)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

vcap.release()
cv2.destroyAllWindows()

