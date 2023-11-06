import cv2

img = cv2.imread("data/samus.png", cv2.IMREAD_GRAYSCALE)

print("Exibindo a matriz da imagem:")
print(img)

cv2.namedWindow("Samus", cv2.WINDOW_NORMAL)
cv2.imshow("Samus", img)
cv2.waitKey(10)

