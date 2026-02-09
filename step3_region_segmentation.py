import cv2
import os

# Image path
IMAGE_PATH = r"C:\Users\Sudarshan chand\OneDrive\Desktop\Automated-scoring-of-handwritten-test-papers\exam write.jpeg"

# Read image
image = cv2.imread(IMAGE_PATH)
if image is None:
    print("❌ Image not found")
    exit()

# ---------- PREPROCESSING (same as Step-2) ----------
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
_, thresh = cv2.threshold(
    blur, 0, 255,
    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)

# ---------- FIND CONTOURS ----------
contours, _ = cv2.findContours(
    thresh,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

regions = []

# ---------- FILTER LARGE REGIONS ----------
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)

    # filter small noise
    if w > 200 and h > 80:
        regions.append((x, y, w, h))

# ---------- SORT REGIONS TOP → BOTTOM ----------
regions = sorted(regions, key=lambda r: r[1])

# ---------- DRAW REGIONS ----------
output = image.copy()

for i, (x, y, w, h) in enumerate(regions):
    label = f"R{i+1}"
    cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(
        output,
        label,
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )

# ---------- CAPTURE BOTTOM BOX (R4) ----------
if len(regions) >= 4:
    x, y, w, h = regions[-1]   # LAST region = bottom
    r4 = image[y:y+h, x:x+w]

    cv2.imshow("R4 - Bottom Box", r4)
    cv2.imwrite("R4_bottom_box.png", r4)
    print("✅ R4 bottom box captured and saved")

else:
    print("⚠️ Less than 4 regions detected")

# ---------- DISPLAY ----------
cv2.imshow("Segmented Regions", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
