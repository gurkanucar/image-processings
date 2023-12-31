import cv2
from ultralytics import YOLO
import cv2
import math


def process_frame(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Adaptive threshold
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 9)

    # Fill rectangular contours
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(thresh, [c], -1, (255, 255, 255), -1)

    # Morph open
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=4)

    # Draw rectangles
    cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    area_threshold = 4000

    cropped_frames = []

    for c in cnts:
        if cv2.contourArea(c) > area_threshold:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (36, 255, 12), 3)

            # Crop the rectangle area
            cropped = frame[y:y+h, x:x+w]
            cropped_frames.append(cropped)

    return frame, cropped_frames

# change path to working dir
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

model = YOLO("best.pt")

classes_to_detect = ["phone_back_camera"]

max_cropped_frames = 0
open_windows = set()


# Start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

while True:

    ret, frame = cap.read()
    if not ret:
        break

    processed_frame, cropped_frames = process_frame(frame)

    # Create a dictionary to store the highest-confidence detection for each class
    highest_confidence = {}

    # Iterate over cropped frames
    # Iterate over cropped frames
    for cropped_frame in cropped_frames:
        results = model(cropped_frame)

        # Access the prediction results
        pred = results.pred[0]

        # Iterate over detected objects
        for r in pred:
            # Bounding box
            x1, y1, x2, y2 = r.xyxy

            # Convert confidence to float
            confidence = float(r.conf)

            # Class name
            cls = int(r.cls)
            detected_class = model.names[cls]

            # Check if the detected class is in classes_to_detect
            if detected_class in classes_to_detect:
                # If the class is not in the dictionary or has higher confidence, update it
                if detected_class not in highest_confidence or confidence > highest_confidence[detected_class][0]:
                    highest_confidence[detected_class] = (confidence, (x1, y1, x2, y2))


    # Iterate over the highest-confidence detections and draw rectangles and labels on the processed frame
    for detected_class, (confidence, (x1, y1, x2, y2)) in highest_confidence.items():
        cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
        cv2.putText(processed_frame, f"{detected_class}: {confidence:.2f}",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Display the processed frame with bounding boxes and labels
    cv2.imshow('Webcam', processed_frame)
    if cv2.waitKey(1) == ord('q'):
        break

# Release the capture and close all windows
cap.release()
for window in open_windows:
    cv2.destroyWindow(window)
cv2.destroyAllWindows()