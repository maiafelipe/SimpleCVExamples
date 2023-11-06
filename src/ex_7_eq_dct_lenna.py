import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("data/lenna.png", cv2.IMREAD_GRAYSCALE)

imf = np.float32(img) # float conversion

# find discrete cosine transform
dct = cv2.dct(imf, cv2.DCT_ROWS)

redux_size = 150
dct_redux = np.zeros((512, 512))

dct_redux[0:redux_size, 0:redux_size] = dct[0:redux_size, 0:redux_size]

# apply inverse discrete cosine transform
img1 = cv2.idct(dct_redux)

# convert to uint8
img1 = np.uint8(img1)


cv2.namedWindow("Lenna Original", cv2.WINDOW_NORMAL)
cv2.imshow("Lenna Original", img)
cv2.namedWindow("DCT", cv2.WINDOW_NORMAL)
cv2.imshow("DCT", dct)

cv2.namedWindow("DCT Redux", cv2.WINDOW_NORMAL)
cv2.imshow("DCT Redux", dct_redux)

cv2.namedWindow("Lenna de Volta", cv2.WINDOW_NORMAL)
cv2.imshow("Lenna de Volta", img1)
cv2.waitKey(0)
cv2.destroyAllWindows()