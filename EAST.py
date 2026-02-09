import cv2
import numpy as np
import os

# ============================================
# PATHS
# ============================================
IMAGE_PATH = r"C:\Users\Sudarshan chand\OneDrive\Desktop\TRAINED PROCEESS\exam write.jpeg"
EAST_MODEL = "frozen_east_text_detection.pb"  # must be in same folder
OUTPUT_DIR = "east_output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================
# LOAD IMAGE
# ============================================
image = cv2.imread(IMAGE_PATH)
orig = image.copy()
H, W = image.shape[:2]

# ============================================
# DEFINE ZONES (RELATIVE COORDINATES)
# ============================================
ZONES = {
    "REGISTER_NO":  (0.10, 0.20, 0.05, 0.65),   # y1,y2,x1,x2
    "COURSE_INFO":  (0.20, 0.32, 0.05, 0.95),
    "INTERNAL_ASSESS": (0.30, 0.38, 0.60, 0.95),
    "CO_TABLE":     (0.75, 0.95, 0.05, 0.95)
}

def in_zone(box, zone):
    (x, y, w, h) = box
    cx = x + w // 2
    cy = y + h // 2

    y1, y2, x1, x2 = zone
    return (y1*H <= cy <= y2*H) and (x1*W <= cx <= x2*W)

# ============================================
# PREPARE IMAGE FOR EAST
# ============================================
newW, newH = (320, 320)
rW = W / float(newW)
rH = H / float(newH)

image_resized = cv2.resize(image, (newW, newH))
blob = cv2.dnn.blobFromImage(
    image_resized, 1.0, (newW, newH),
    (123.68, 116.78, 103.94), swapRB=True, crop=False
)

# ============================================
# LOAD EAST MODEL
# ============================================
net = cv2.dnn.readNet(EAST_MODEL)
net.setInput(blob)

(scores, geometry) = net.forward([
    "feature_fusion/Conv_7/Sigmoid",
    "feature_fusion/concat_3"
])

# ============================================
# DECODE EAST OUTPUT
# ============================================
boxes = []
conf_threshold = 0.5

for y in range(scores.shape[2]):
    for x in range(scores.shape[3]):
        if scores[0, 0, y, x] < conf_threshold:
            continue

        offsetX, offsetY = x * 4.0, y * 4.0
        angle = geometry[0, 4, y, x]

        cos = np.cos(angle)
        sin = np.sin(angle)

        h = geometry[0, 0, y, x] + geometry[0, 2, y, x]
        w = geometry[0, 1, y, x] + geometry[0, 3, y, x]

        endX = int(offsetX + (cos * geometry[0, 1, y, x]) + (sin * geometry[0, 2, y, x]))
        endY = int(offsetY - (sin * geometry[0, 1, y, x]) + (cos * geometry[0, 2, y, x]))
        startX = int(endX - w)
        startY = int(endY - h)

        boxes.append((
            int(startX * rW),
            int(startY * rH),
            int(w * rW),
            int(h * rH)
        ))

# ============================================
# FILTER & SAVE REGIONS
# ============================================
zone_boxes = {k: [] for k in ZONES.keys()}

for box in boxes:
    for zone_name, zone in ZONES.items():
        if in_zone(box, zone):
            zone_boxes[zone_name].append(box)

# ============================================
# DRAW & SAVE
# ============================================
for zone_name, bxs in zone_boxes.items():
    zone_img = orig.copy()
    for (x, y, w, h) in bxs:
        cv2.rectangle(zone_img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    out_path = os.path.join(OUTPUT_DIR, f"{zone_name}.jpg")
    cv2.imwrite(out_path, zone_img)
    print(f"✅ Saved zone: {zone_name} → {out_path}")

print("🎉 EAST region detection completed")

