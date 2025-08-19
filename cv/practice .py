import cv2
import matplotlib.pyplot as plt

# Read the image
img = cv2.imread("image.jpeg")
print("Original shape:", img.shape)

# Draw a rectangle (top-left to bottom-right)
cv2.rectangle(img, (50, 50), (120, 120), (255, 0, 0), 10)

# Draw a line
cv2.line(img, (100, 100), (100, 150), (0, 255, 0), 3)

# Draw a circle (center, radius)
cv2.circle(img, (100, 100), 50, (255, 0, 0), 2)

# Resize image
resize = cv2.resize(img, (200, 200))

# Display using matplotlib
plt.imshow(cv2.cvtColor(resize, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()
