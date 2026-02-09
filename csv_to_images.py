import os
import pandas as pd
import numpy as np
import cv2
import string

# ==================================================
# 🔹 PATH SETTINGS (YOUR EXACT PATH)
# ==================================================
CSV_FILE = r"C:\Users\Sudarshan chand\OneDrive\Desktop\datamnist\A_Z Handwritten Data.csv"
BASE_DIR = r"C:\Users\Sudarshan chand\OneDrive\Desktop\datamnist"
OUTPUT_DIR = os.path.join(BASE_DIR, "images")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Create A–Z folders
alphabet = list(string.ascii_uppercase)
for letter in alphabet:
    os.makedirs(os.path.join(OUTPUT_DIR, letter), exist_ok=True)

# ==================================================
# 🔹 READ CSV IN CHUNKS (FIX MEMORY ERROR)
# ==================================================
CHUNK_SIZE = 5000   # safe for low RAM systems

print("🚀 Starting CSV to image conversion (chunk-wise)...")

img_count = 0

for chunk in pd.read_csv(CSV_FILE, chunksize=CHUNK_SIZE):

    for idx, row in chunk.iterrows():

        label = int(row.iloc[0])               # 0–25
        pixels = row.iloc[1:].values.astype("uint8")

        letter = alphabet[label]

        # Reshape to 28x28
        img = pixels.reshape(28, 28)

        # Optional noise reduction
        img = cv2.GaussianBlur(img, (3, 3), 0)

        # Save image
        img_name = f"{letter}_{img_count}.png"
        save_path = os.path.join(OUTPUT_DIR, letter, img_name)
        cv2.imwrite(save_path, img)

        img_count += 1

    print(f"✅ Processed {img_count} images so far...")

print("🎉 COMPLETED!")
print("Total images saved:", img_count)
print("Output folder:", OUTPUT_DIR)
