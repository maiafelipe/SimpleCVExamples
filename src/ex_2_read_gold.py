import cv2

img = cv2.imread("data/gold.png", cv2.IMREAD_COLOR)
pos_y = 10
pos_x = 155

print(f"Exibindo pixel em ({pos_x}, {pos_y}):")
print(img[pos_y][pos_x])
img[pos_y][pos_x] = [0,0,0]

cv2.namedWindow("Gold", cv2.WINDOW_NORMAL)
cv2.imshow("Gold", img)
cv2.waitKey(0)



