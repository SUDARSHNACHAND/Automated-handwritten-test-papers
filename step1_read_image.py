import cv2
import os

# Absolute path (Windows safe)
IMAGE_PATH = r"C:\Users\Sudarshan chand\OneDrive\Desktop\TRAINED PROCEESS\exam write.jpeg"

print("Looking for image at:")
print(IMAGE_PATH)

# Read image
image = cv2.imread(IMAGE_PATH)

if image is None:
    print("❌ Error: Image not found or unreadable")
    exit()

print("✅ Image loaded successfully")
print("Original Image Size:", image.shape)

# -----------------------------
# Resize for better viewing
# -----------------------------
screen_width = 1366   # typical laptop width
screen_height = 768   # typical laptop height

h, w = image.shape[:2]

scale = min(screen_width / w, screen_height / h)
new_w = int(w * scale)
new_h = int(h * scale)

# High-quality resize
resized = cv2.resize(
    image,
    (new_w, new_h),
    interpolation=cv2.INTER_CUBIC
)

print("Displayed Image Size:", resized.shape)

# -----------------------------
# Display FULL image
# -----------------------------
cv2.namedWindow("Test Paper Image - Full View", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Test Paper Image - Full View", new_w, new_h)

cv2.imshow("Test Paper Image - Full View", resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
