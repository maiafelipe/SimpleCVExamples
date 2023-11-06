import cv2

img_ori = cv2.imread("data/lenna.png", cv2.IMREAD_GRAYSCALE)

(T,img_bin) = cv2.threshold(img_ori, maxval= 255, thresh= 128, type=cv2.THRESH_BINARY)

cv2.namedWindow("Lenna Original", cv2.WINDOW_NORMAL)
cv2.imshow("Lenna Original", img_ori)

cv2.namedWindow("Lenna Binary", cv2.WINDOW_NORMAL)
cv2.imshow("Lenna Binary", img_bin)

cv2.waitKey(0)



