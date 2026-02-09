import cv2
import os
import numpy as np

# ======================================
# 🔹 YOUR DATASET PATH (FIXED)
# ======================================
INPUT_DIR = r"C:\dataset\digits 0-9"     # Normal camera images
OUTPUT_DIR = r"C:\dataset\mnist_images" # MNIST-style output

os.makedirs(OUTPUT_DIR, exist_ok=True)

for digit in range(10):
    input_digit_folder = os.path.join(INPUT_DIR, str(digit))
    output_digit_folder = os.path.join(OUTPUT_DIR, str(digit))
    os.makedirs(output_digit_folder, exist_ok=True)

    for file in os.listdir(input_digit_folder):
        img_path = os.path.join(input_digit_folder, file)

        # 1️⃣ Read image
        img = cv2.imread(img_path)
        if img is None:
            continue

        # 2️⃣ Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 3️⃣ Noise removal
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # 4️⃣ Threshold (MNIST style)
        _, thresh = cv2.threshold(
            blur, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        # 5️⃣ Digit bounding box
        coords = cv2.findNonZero(thresh)
        if coords is None:
            continue

        x, y, w, h = cv2.boundingRect(coords)
        digit_crop = thresh[y:y+h, x:x+w]

        # 6️⃣ Resize to 20x20
        resized = cv2.resize(digit_crop, (20, 20))

        # 7️⃣ Pad to 28x28
        mnist_img = np.zeros((28, 28), dtype=np.uint8)
        mnist_img[4:24, 4:24] = resized

        # 8️⃣ Save output
        save_path = os.path.join(output_digit_folder, file)
        cv2.imwrite(save_path, mnist_img)

print("✅ Conversion complete! MNIST images saved to:")
print(OUTPUT_DIR)
