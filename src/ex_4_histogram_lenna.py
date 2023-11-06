import cv2
import numpy as np
from matplotlib import pyplot as plt

img_ori = cv2.imread("data/lenna.png", cv2.IMREAD_GRAYSCALE)

hist,bins = np.histogram(img_ori.flatten(),256,[0,256])

#cdf = hist.cumsum()
#cdf_normalized = cdf * float(hist.max()) / cdf.max()

cv2.namedWindow("Lenna Original", cv2.WINDOW_NORMAL)
cv2.imshow("Lenna Original", img_ori)


#plt.plot(cdf_normalized, color = 'b')
plt.hist(img_ori.flatten(),256,[0,256], color = 'r')
plt.xlim([0,256])
plt.legend(('histogram'), loc = 'upper left')
plt.show()

cv2.waitKey(0)



