import streamlit as st
from ultralytics import YOLO
import cv2
import pytesseract
import numpy as np
import re
import os
import csv
from datetime import datetime

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "D:/vscode/violation/best.pt"
CSV_FILE = "D:/vscode/violation/violations.csv"
OUTPUT_DIR = "D:/vscode/violation/"

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load model
model = YOLO(MODEL_PATH)

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def correct_plate(text):
    """Correct common OCR mistakes in Indian number plates"""
    replacements = {
        'O': '0', 'I': '1', 'L': '1', 'Z': '2',
        'S': '5', 'B': '8', 'G': '6', 'T': '7'
    }
    return ''.join(replacements.get(c, c) for c in text.upper())

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("🚨 Helmet Violation Detection System")
st.write("Upload an image to detect helmet violations and extract number plate")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

# Create CSV if not exists
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Plate Number"])

# -----------------------------
# PROCESS IMAGE
# -----------------------------
if uploaded_file is not None:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    st.image(image, caption="Uploaded Image", channels="BGR")

    results = model(image)

    plate_boxes = []
    helmet_boxes = []
    no_helmet_boxes = []

    # -----------------------------
    # DETECTION
    # -----------------------------
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = model.names[cls]

            if label == "with helmet":
                helmet_boxes.append((x1,y1,x2,y2))
                cv2.rectangle(image, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(image, "Helmet", (x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            elif label == "without helmet":
                no_helmet_boxes.append((x1,y1,x2,y2))
                cv2.rectangle(image, (x1,y1), (x2,y2), (0,0,255), 2)
                cv2.putText(image, "No Helmet", (x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

            elif label == "number plate":
                plate_boxes.append((x1,y1,x2,y2))
                cv2.rectangle(image, (x1,y1), (x2,y2), (255,0,0), 2)
                cv2.putText(image, "Plate", (x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

            elif label == "rider":
                cv2.rectangle(image, (x1,y1), (x2,y2), (255,255,0), 2)
                cv2.putText(image, "Rider", (x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

    # -----------------------------
    # VIOLATION LOGIC
    # -----------------------------
    violation = False
    if len(no_helmet_boxes) > 0:
        violation = True

    detected_plates = []

    # -----------------------------
    # OCR ONLY IF VIOLATION
    # -----------------------------
    if violation:

        st.subheader("OCR Results for Detected Plates")
        for idx, (x1,y1,x2,y2) in enumerate(plate_boxes):

            plate_crop = image[y1:y2, x1:x2]

            if plate_crop.size == 0:
                continue

            # Preprocess
            gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(2.0, (8,8))
            gray = clahe.apply(gray)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # OCR
            config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            raw_text = pytesseract.image_to_string(thresh, config=config)
            after_clean = re.sub(r'[^A-Z0-9]', '', raw_text.upper())
            after_correction = correct_plate(after_clean)

            # Display OCR stages
            # st.write(f"Plate {idx}:")
            # st.write(f"- RAW OCR: {raw_text.strip()}")
            # st.write(f"- After Clean: {after_clean}")
            # st.write(f"- After Correction: {after_correction}")

            if len(after_correction) >= 5:
                detected_plates.append(after_correction)

                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                with open(CSV_FILE, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([timestamp, after_correction])

                # Draw plate text
                cv2.putText(image, after_correction, (x1, y1-30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

    # -----------------------------
    # SHOW OUTPUT IMAGE
    # -----------------------------
    st.image(image, caption="Processed Output", channels="BGR")

    # -----------------------------
    # SHOW RESULT MESSAGE
    # -----------------------------
    if violation:
        if detected_plates:
            st.error("🚨 Violation Detected!")
            for p in detected_plates:
                st.write(f"🚗 Plate: {p}")
        else:
            st.warning("⚠️ No helmet detected but plate not readable")
    else:
        st.success("✅ No Violation (Helmet detected)")