Intelligent License Plate Reader & Vehicle Insights Dashboard
Summary:
Road safety is a major concern, especially in countries like India, where two-wheeler riders often neglect wearing helmets. This project proposes an automated system that detects helmet violations and extracts vehicle number plates using computer vision and deep learning techniques.
The system uses object detection to identify riders, helmets, and number plates, followed by Optical Character Recognition (OCR) to extract the plate number. The detected violations are logged with timestamps for further action.
Objectives:
•	Detect whether a rider is wearing a helmet
•	Identify number plates of violating vehicles
•	Extract plate numbers using OCR
•	Store violation records with timestamp
•	Provide a user-friendly interface using a web application
System Overview:
The system consists of three main components:
1.	Object Detection (YOLO Model)
2.	OCR (Tesseract)
3.	User Interface (Streamlit App)
Workflow:
1.	Input image uploaded
2.	YOLO detects:
o	Rider
o	Helmet / No Helmet
o	Number Plate
3.	If "No Helmet" detected:
o	Crop number plate
o	Apply pre-processing
o	Extract text using OCR
4.	Store result in CSV file
5.	Display output with bounding boxes


Technologies Used:
•	Programming Language: Python
•	Deep Learning Framework: Ultralytics YOLO
•	Image Processing: OpenCV
•	OCR Engine: Tesseract OCR
•	Web Framework: Streamlit
•	Libraries:
•	NumPy
•	CSV
•	Regex
Dataset Description:
The dataset contains annotated images with the following classes:
•	Number Plate
•	Rider
•	With Helmet
•	Without Helmet
Annotations were prepared in YOLO format using bounding boxes.
Model Training:
•	Model: YOLO (custom trained)
•	Input size: 640x640
•	Classes: 4
•	Output: best.pt
Training includes:
•	Data labeling
•	Model training
•	Validation
•	Exporting trained weights
Methodology:
1.Object Detection:
YOLO model detects objects in the image:
•	Rider
•	Helmet status
•	Number plate
Bounding boxes are drawn around detected objects.
2.Violation Detection Logic
IF "without helmet" detected:
    violation = True
ELSE:
    No violation
3.Number Plate Extraction
1.	Crop number plate region
2.	Convert to grayscale
3.	Apply CLAHE (contrast enhancement)
4.	Apply thresholding
5.	Perform OCR
4.OCR Post-processing
OCR errors are corrected using character mapping:
Example:
O → 0
S → 5
B → 8
G → 6
Example Correction:
Raw OCR: UPGSAESTICS
Cleaned: UPGSAESTICS
Corrected: UP65AE7169
5.Data Logging
Detected violations are stored in a csv file.


System Architecture
Input (Image)
        ↓
YOLO Detection
        ↓
Helmet Check
        ↓
[If Violation]
        ↓
Plate Extraction
        ↓
OCR (Tesseract)
        ↓
Text Cleaning & Correction
        ↓
CSV Storage + Display
Implementation
•	Image upload
•	Image processing
•	Real-time detection
•	OCR extraction
•	CSV logging
•	Streamlit UI
Results
•  Successfully detects:
•	Riders
•	Helmet usage
•	Number plates
•  OCR accuracy improved using:
•	Image preprocessing
•	Character correction logic
•  Outputs:
•	Annotated image
•	Extracted plate numbers
•	CSV log file
Advantages
•	Fully automated system
•	Real-time processing capability
•	Easy deployment using Streamlit
•	Scalable for traffic monitoring
Limitations
•	OCR accuracy depends on image quality
•	Small or blurred plates may fail
•	Night-time detection may be less accurate
•	Multiple vehicles may cause ambiguity
Conclusion
This project demonstrates an efficient system for detecting helmet violations and extracting number plate information using deep learning and OCR. It can be used by traffic authorities to automate monitoring and improve road safety enforcement.












