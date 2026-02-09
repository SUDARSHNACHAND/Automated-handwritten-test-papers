import cv2
import os

IMAGE_PATH = r"C:\Users\Sudarshan chand\OneDrive\Desktop\TRAINED PROCEESS\exam write.jpeg"
OUTPUT_DIR = "east_output"

os.makedirs(OUTPUT_DIR, exist_ok=True)

img = cv2.imread(IMAGE_PATH)
H, W = img.shape[:2]

def crop_and_save(name, y1, y2, x1, x2):
    crop = img[int(y1*H):int(y2*H), int(x1*W):int(x2*W)]
    cv2.imwrite(f"{OUTPUT_DIR}/{name}.jpg", crop)
    print(f"✅ Saved {name}")

# =========================
# DEFINE REGIONS (TUNED FOR YOUR SHEET)
# =========================

# 1️⃣ Register Number (12 boxes)
crop_and_save("REGISTER_NO", 0.12, 0.20, 0.05, 0.65)

# 2️⃣ Course Code / Name
crop_and_save("COURSE_INFO", 0.20, 0.30, 0.05, 0.95)

# 3️⃣ Internal Assessment (I / II / III)
crop_and_save("INTERNAL_ASSESS", 0.30, 0.38, 0.60, 0.95)

# 4️⃣ Course Outcomes + Marks Table
crop_and_save("CO_TABLE", 0.74, 0.95, 0.05, 0.95)

print("🎉 REGION CROPPING COMPLETED (NO EAST NEEDED)")
