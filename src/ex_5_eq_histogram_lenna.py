import cv2
import numpy as np
from matplotlib import pyplot as plt

img_ori = cv2.imread("data/lenna.png", cv2.IMREAD_GRAYSCALE)
img_equ = cv2.equalizeHist(img_ori)

hist,bins = np.histogram(img_ori.flatten(),256,[0,256])
hist2,bins2 = np.histogram(img_ori.flatten(),256,[0,256])

cv2.namedWindow("Lenna Original", cv2.WINDOW_NORMAL)
cv2.imshow("Lenna Original", img_ori)

cv2.namedWindow("Lenna Equalizada", cv2.WINDOW_NORMAL)
cv2.imshow("Lenna Equalizada", img_equ)

plt.hist(img_ori.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.legend(('histogram'), loc = 'upper left')
plt.show()

plt.hist(img_equ.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.legend(('histogram'), loc = 'upper left')
plt.show()

cv2.waitKey(0)



