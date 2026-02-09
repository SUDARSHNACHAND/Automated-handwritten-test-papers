import cv2
import os

IMAGE_PATH = r"C:\Users\Sudarshan chand\OneDrive\Desktop\TRAINED PROCEESS\exam write.jpeg"

# -----------------------------
# Load image (ORIGINAL)
# -----------------------------
image = cv2.imread(IMAGE_PATH)

if image is None:
    print("❌ Error: Image not found")
    exit()

h, w = image.shape[:2]
print("Original Image Size:", image.shape)

# -----------------------------
# Preprocessing (ORIGINAL SIZE)
# -----------------------------
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)

_, thresh = cv2.threshold(
    blur, 0, 255,
    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)

# -----------------------------
# Resize ONLY for display
# -----------------------------
screen_w, screen_h = 1366, 768
scale = min(screen_w / w, screen_h / h)

new_w, new_h = int(w * scale), int(h * scale)

image_disp = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
gray_disp = cv2.resize(gray, (new_w, new_h))
thresh_disp = cv2.resize(thresh, (new_w, new_h))

print("Displayed Image Size:", image_disp.shape)

# -----------------------------
# Display
# -----------------------------
cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
cv2.namedWindow("Grayscale", cv2.WINDOW_NORMAL)
cv2.namedWindow("Threshold", cv2.WINDOW_NORMAL)

cv2.imshow("Original", image_disp)
cv2.imshow("Grayscale", gray_disp)
cv2.imshow("Threshold", thresh_disp)

cv2.waitKey(0)
cv2.destroyAllWindows()
