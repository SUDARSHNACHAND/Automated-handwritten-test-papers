import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

# =====================================
# PATHS
# =====================================
MODEL_PATH = r"C:\Users\Sudarshan chand\OneDrive\Desktop\TRAINED PROCEESS\model.h5"
IMAGE_PATH = r"C:\Users\Sudarshan chand\OneDrive\Desktop\TRAINED PROCEESS\east_output\REGISTER_NO.jpg"

OUTPUT_RESULT = r"C:\Users\Sudarshan chand\OneDrive\Desktop\TRAINED PROCEESS\east_output\REGISTER_NO_RESULT.jpg"
DEBUG_BOXES = r"C:\Users\Sudarshan chand\OneDrive\Desktop\TRAINED PROCEESS\east_output\DEBUG_BOXES.jpg"

# =====================================
# LOAD MODEL
# =====================================
model = load_model(MODEL_PATH)
print("✅ model.h5 loaded")

# =====================================
# 1️⃣ READ IMAGE
# =====================================
img = cv2.imread(IMAGE_PATH)
if img is None:
    raise FileNotFoundError("❌ REGISTER_NO.jpg not found")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# =====================================
# 2️⃣ NOISE REMOVAL
# =====================================
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# =====================================
# 3️⃣ MNIST-STYLE THRESHOLD
# =====================================
thresh = cv2.adaptiveThreshold(
    blur, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    11, 2
)

# =====================================
# 4️⃣ FIND DIGIT CONTOURS (FIXED)
# =====================================
contours, _ = cv2.findContours(
    thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

digit_boxes = []
for c in contours:
    x, y, w, h = cv2.boundingRect(c)

    # ✅ LOOSE FILTER (IMPORTANT FIX)
    if w > 10 and h > 20 and h > w:
        digit_boxes.append((x, y, w, h))

if len(digit_boxes) == 0:
    print("❌ No digit contours detected – check cropping or threshold")

# SORT LEFT → RIGHT
digit_boxes = sorted(digit_boxes, key=lambda b: b[0])

# =====================================
# 5️⃣ PREDICT DIGITS
# =====================================
register_number = ""
debug_img = img.copy()

for (x, y, w, h) in digit_boxes:
    pad = 5
    roi = thresh[
        max(0, y-pad):y+h+pad,
        max(0, x-pad):x+w+pad
    ]

    roi = cv2.resize(roi, (28, 28))
    roi = roi.astype("float32") / 255.0
    roi = roi.reshape(1, 28, 28, 1)

    pred = model.predict(roi, verbose=0)
    digit = np.argmax(pred)
    register_number += str(digit)

    # draw results
    cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(
        debug_img, str(digit), (x, y-5),
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2
    )

# =====================================
# 6️⃣ SAVE OUTPUT IMAGES
# =====================================
cv2.imwrite(OUTPUT_RESULT, debug_img)
cv2.imwrite(DEBUG_BOXES, debug_img)

print("\n🟢 REGISTER NUMBER:", register_number)
print("🖼 Result image saved at:")
print(OUTPUT_RESULT)
print("🧪 Debug image saved at:")
print(DEBUG_BOXES)
