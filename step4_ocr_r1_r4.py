import cv2
import pytesseract
import re

# -------------------------------------------------
# If tesseract is not in PATH, uncomment & set this
# -------------------------------------------------
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Image paths
R1_PATH = "R1_Header.png"
R4_PATH = "R4_Course_Outcomes_Full.png"

# -------------------------------------------------
# OCR helper function
# -------------------------------------------------
def ocr_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ Cannot read {img_path}")
        return ""

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    text = pytesseract.image_to_string(
        gray,
        config="--psm 6"
    )
    return text


# -------------------------------------------------
# OCR R1 – HEADER
# -------------------------------------------------
r1_text = ocr_image(R1_PATH)
print("\n========== R1 HEADER RAW OCR ==========")
print(r1_text)

# Simple structured extraction
header_data = {
    "Register Number": re.findall(r"\b\d{10,12}\b", r1_text),
    "Course Code": re.findall(r"[A-Z]{2,}\d+", r1_text),
    "Degree": "BE" if "BE" in r1_text else "",
    "Course Name": "Human Values and Ethics" if "Human" in r1_text else ""
}

print("\n========== R1 HEADER STRUCTURED ==========")
for k, v in header_data.items():
    print(f"{k}: {v}")


# -------------------------------------------------
# OCR R4 – COURSE OUTCOMES
# -------------------------------------------------
r4_text = ocr_image(R4_PATH)
print("\n========== R4 COURSE OUTCOMES RAW OCR ==========")
print(r4_text)

# Extract numbers (marks)
numbers = re.findall(r"\d+", r4_text)

r4_data = {
    "CO1": numbers[0] if len(numbers) > 0 else "",
    "CO2": numbers[1] if len(numbers) > 1 else "",
    "CO3": numbers[2] if len(numbers) > 2 else "",
    "CO4": numbers[3] if len(numbers) > 3 else "",
    "CO5": numbers[4] if len(numbers) > 4 else "",
    "CO6": numbers[5] if len(numbers) > 5 else "",
    "Total": numbers[-1] if len(numbers) > 0 else ""
}

print("\n========== R4 COURSE OUTCOMES STRUCTURED ==========")
for k, v in r4_data.items():
    print(f"{k}: {v}")
