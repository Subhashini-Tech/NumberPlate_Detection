from ultralytics import YOLO
import cv2
import pytesseract
import numpy as np
import re
import os
import csv
from datetime import datetime

# -----------------------------
# Functions for OCR correction
# -----------------------------
def correct_plate(text):
    """Correct common OCR mistakes in Indian number plates."""
    text = text.upper()
    replacements = {
        'O': '0',
        'I': '1',
        'L': '1',
        'Z': '2',
        'S': '5',
        'B': '8',
        'G': '6',
        'T': '7'
    }
    corrected = ''.join(replacements.get(c, c) for c in text)
    return corrected

def validate_indian_plate(text):
    """Validate text matches Indian vehicle number plate pattern."""
    pattern = r'^[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}$'
    return re.match(pattern, text)

# -----------------------------
# Load YOLO model
# -----------------------------
model = YOLO("D:/vscode/violation/best.pt")
print("Class Names:", model.names)

# Tesseract path (Windows)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Class IDs
PLATE = 0
RIDER = 1
HELMET = 2
NO_HELMET = 3

# -----------------------------
# Load input image
# -----------------------------
image_path = "D:\\vscode\\violation\\input7.JPG"
image = cv2.imread(image_path)
results = model(image)

no_helmet_boxes = []
helmet_present = False
plate_boxes = []

# -----------------------------
# CSV SETUP
# -----------------------------
csv_file = "D:/vscode/violation/violations.csv"
if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Plate Number"])

# -----------------------------
# Detection
# -----------------------------
for r in results:
    for box in r.boxes:
        cls = int(box.cls[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = model.names[cls]
        print("Detected:", label, x1, y1, x2, y2)

        if cls == NO_HELMET:
            no_helmet_boxes.append((x1, y1, x2, y2))
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(image, "No Helmet", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        if cls == HELMET:
            helmet_present = True
            cv2.putText(image, "Helmet", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        if cls == PLATE:
            plate_boxes.append((x1, y1, x2, y2))
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(image, "Plate", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

        if cls == RIDER:
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, "Rider", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

# -----------------------------
# Process plates
# -----------------------------
output_dir = "D:/vscode/violation/"
os.makedirs(output_dir, exist_ok=True)

for idx, plate_box in enumerate(plate_boxes):
    print(f"\nProcessing plate {idx}")

    x1, y1, x2, y2 = plate_box
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image.shape[1]-1, x2), min(image.shape[0]-1, y2)

    plate_crop = image[y1:y2, x1:x2]
    if plate_crop.size == 0:
        print(f"Skipping zero-size crop {idx}")
        continue

    # Save raw plate
    raw_path = os.path.join(output_dir, f"raw_plate_{idx}.jpg")
    cv2.imwrite(raw_path, plate_crop)
    print("Saved raw plate:", raw_path)

    # Preprocess for OCR
    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Save debug image
    debug_path = os.path.join(output_dir, f"debug_plate_{idx}.jpg")
    cv2.imwrite(debug_path, thresh)
    print("Saved debug plate:", debug_path)

    # OCR
    config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    raw_text = pytesseract.image_to_string(thresh, config=config)
    print("RAW OCR:", raw_text)

    text = re.sub(r'[^A-Z0-9]', '', raw_text.upper())
    corrected_text = correct_plate(text)
    print("CORRECTED OCR:", corrected_text)

    # -----------------------------
    # Save to CSV or display "No Violation"
    # -----------------------------
    if helmet_present:
        print(f"✅ No Violation: Rider wearing helmet")
        cv2.putText(image, "No Violation", (x1, y1-30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
    elif validate_indian_plate(corrected_text):
        final_text = corrected_text
        print(f"🚨 Violation Plate {idx}:", final_text)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, final_text])
        print("Saved to CSV:", final_text)

        # Draw plate number
        cv2.putText(image, final_text, (x1, y1-30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
    else:
        print(f"❌ OCR failed or invalid plate: {corrected_text}")

# -----------------------------
# Save output image
# -----------------------------
output_img_path = os.path.join(output_dir, "output.jpg")
cv2.imwrite(output_img_path, image)
print("\nSaved output image with bounding boxes:", output_img_path)