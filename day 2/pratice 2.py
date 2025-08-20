import cv2
import matplotlib.pyplot as plt
img = cv2.imread("beautiful-natural-image-1844362_1280.jpg")
(h,w)= img.shape[:2]
RM=cv2.getRotationMatrix2D((w//2,h//2),45,1.0)
rotated = cv2.warpAffine(img,RM,(w,h))
plt.imshow(cv2.cvtColor(rotated,cv2.COLOR_BGR2RGB))
plt.show()
plt.axis("off")
print(img.shape)