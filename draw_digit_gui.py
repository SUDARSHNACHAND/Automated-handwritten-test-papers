import tkinter as tk
import cv2
import numpy as np
from PIL import Image, ImageDraw
from tensorflow.keras.models import load_model

# =====================================
# LOAD MODEL
# =====================================
MODEL_PATH = r"C:\Users\Sudarshan chand\OneDrive\Desktop\TRAINED PROCEESS\model.h5"
model = load_model(MODEL_PATH)
print("✅ model.h5 loaded")

# =====================================
# TKINTER WINDOW
# =====================================
window = tk.Tk()
window.title("Draw Digit - MNIST Test")
window.resizable(False, False)

CANVAS_SIZE = 280  # big canvas (10x MNIST)
canvas = tk.Canvas(window, width=CANVAS_SIZE, height=CANVAS_SIZE, bg="black")
canvas.pack()

# PIL image to draw on
image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
draw = ImageDraw.Draw(image)

# =====================================
# DRAW FUNCTION
# =====================================
def paint(event):
    x1, y1 = event.x - 8, event.y - 8
    x2, y2 = event.x + 8, event.y + 8
    canvas.create_oval(x1, y1, x2, y2, fill="white", outline="white")
    draw.ellipse([x1, y1, x2, y2], fill=255)

canvas.bind("<B1-Motion>", paint)

# =====================================
# PREDICT FUNCTION
# =====================================
def predict_digit():
    img = image.resize((28, 28))
    img = np.array(img)
    img = img.astype("float32") / 255.0
    img = img.reshape(1, 28, 28, 1)

    pred = model.predict(img, verbose=0)
    digit = np.argmax(pred)

    result_label.config(text=f"Predicted Digit: {digit}")

# =====================================
# CLEAR FUNCTION
# =====================================
def clear_canvas():
    canvas.delete("all")
    draw.rectangle([0, 0, CANVAS_SIZE, CANVAS_SIZE], fill=0)
    result_label.config(text="Draw a digit")

# =====================================
# BUTTONS
# =====================================
btn_frame = tk.Frame(window)
btn_frame.pack()

predict_btn = tk.Button(btn_frame, text="PREDICT", command=predict_digit, width=15)
predict_btn.grid(row=0, column=0, padx=5, pady=5)

clear_btn = tk.Button(btn_frame, text="CLEAR", command=clear_canvas, width=15)
clear_btn.grid(row=0, column=1, padx=5, pady=5)

# =====================================
# RESULT LABEL
# =====================================
result_label = tk.Label(window, text="Draw a digit", font=("Arial", 16))
result_label.pack(pady=10)

window.mainloop()
