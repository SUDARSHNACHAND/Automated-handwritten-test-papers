import cv2
import os

IMAGE_PATH = r"C:\Users\Sudarshan chand\OneDrive\Desktop\TRAINED PROCEESS\exam write.jpeg"

# -----------------------------
# Load image
# -----------------------------
image = cv2.imread(IMAGE_PATH)
if image is None:
    print("❌ Error: Image not found")
    exit()

h, w = image.shape[:2]
print("Original Image Size:", image.shape)

# -----------------------------
# Preprocessing
# -----------------------------
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)

_, thresh = cv2.threshold(
    blur, 0, 255,
    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)

# -----------------------------
# Segmentation (ORIGINAL)
# -----------------------------
contours, _ = cv2.findContours(
    thresh,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

output = image.copy()
region_id = 0

for cnt in contours:
    x, y, cw, ch = cv2.boundingRect(cnt)

    if cw > 60 and ch > 40:
        region_id += 1

        cv2.rectangle(
            output,
            (x, y),
            (x + cw, y + ch),
            (0, 255, 0),
            2
        )

        cv2.putText(
            output,
            f"R{region_id}",
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )

print("Detected Regions:", region_id)

# -----------------------------
# Resize for display
# -----------------------------
screen_w, screen_h = 1366, 768
scale = min(screen_w / w, screen_h / h)
new_w, new_h = int(w * scale), int(h * scale)

output_disp = cv2.resize(output, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
thresh_disp = cv2.resize(thresh, (new_w, new_h))

print("Displayed Image Size:", output_disp.shape)

# -----------------------------
# Display
# -----------------------------
cv2.namedWindow("Threshold", cv2.WINDOW_NORMAL)
cv2.namedWindow("Segmented Output", cv2.WINDOW_NORMAL)

cv2.imshow("Threshold", thresh_disp)
cv2.imshow("Segmented Output", output_disp)

cv2.waitKey(0)
cv2.destroyAllWindows()
