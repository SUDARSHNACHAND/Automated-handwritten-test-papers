import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# ======================================
# 🔹 DATASET PATH (YOUR PATH)
# ======================================
DATASET_DIR = r"C:\Users\Sudarshan chand\OneDrive\Desktop\TRAINED PROCEESS\mnist_images"

# ======================================
# 🔹 LOAD DATA
# ======================================
X = []
y = []

for digit in range(10):
    digit_folder = os.path.join(DATASET_DIR, str(digit))
    for file in os.listdir(digit_folder):
        img_path = os.path.join(digit_folder, file)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        img = cv2.resize(img, (28, 28))
        img = img.astype("float32") / 255.0

        X.append(img)
        y.append(digit)

X = np.array(X).reshape(-1, 28, 28, 1)
y = to_categorical(y, 10)

print("✅ Total images loaded:", X.shape[0])

# ======================================
# 🔹 TRAIN / TEST SPLIT
# ======================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ======================================
# 🔹 BUILD CNN MODEL
# ======================================
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),

    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.3),
    Dense(10, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ======================================
# 🔹 TRAIN MODEL
# ======================================
model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=128,
    validation_data=(X_test, y_test)
)

# ======================================
# 🔹 SAVE MODEL
# ======================================
model.save("model.h5")
print("🎉 model.h5 saved successfully!")
