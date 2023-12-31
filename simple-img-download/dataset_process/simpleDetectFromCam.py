import cv2
from ultralytics import YOLO
import numpy as np
import os

# change path to working dir
os.chdir(os.path.dirname(os.path.abspath(__file__)))



def process_frame(frame):
    # Convert to grayscale and apply adaptive threshold
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 51, 9)

    # Find contours and fill them
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        cv2.drawContours(thresh, [c], -1, (255, 255, 255), -1)

    # Morph open
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=4)

    # Find contours for bounding boxes
    cnts, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area_threshold = 4000

    cropped_frames = []
    if cnts:  # Check if there are any contours
        largest_contour = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > area_threshold:
            x, y, w, h = cv2.boundingRect(largest_contour)
            # Apply offset
            offset = 5
            x, y, w, h = x - offset, y - offset, w + 2 * offset, h + 2 * offset
            # Ensure coordinates are within frame boundaries
            x, y, w, h = max(0, x), max(0, y), min(w, frame.shape[1] - x), min(h, frame.shape[0] - y)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (36, 255, 12), 3)
            cropped = frame[y:y+h, x:x+w]
            cropped_frames.append(cropped)

    return frame, cropped_frames


model = YOLO("best.pt")
classes_to_detect = ["phone_back_camera"]



# Initialize webcam
cap = cv2.VideoCapture(0)
cropped_window_open = False

# ... [previous code, including YOLO model initialization] ...

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Process the frame to get cropped frame
    processed_frame, cropped_frames = process_frame(frame)

    # If cropped frame is available, run model detection
    if cropped_frames:
        cropped_img = cropped_frames[0]
        results = model(cropped_img, stream=True)

        for r in results:
            boxes = r.boxes

            for box in boxes:
                # Bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Class name
                cls = int(box.cls[0])
                detected_class = model.names[cls]

                # Check if the detected class is in classes_to_detect
                if detected_class in classes_to_detect:
                    # Draw bounding box on cropped image
                    cv2.rectangle(cropped_img, (x1, y1), (x2, y2), (0, 0, 255), 3)

                    # Draw bounding box on original image
                    # Adjust coordinates to fit the original frame
                    x, y, w, h = cv2.boundingRect(cropped_frames[0])  # Bounding box of cropped frame
                    cv2.rectangle(processed_frame, (x + x1, y + y1), (x + x2, y + y2), (0, 0, 255), 3)
                    cv2.putText(processed_frame, f"{detected_class}", (x + x1, y + y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.imshow('Cropped Frame', cropped_img)
        cropped_window_open = True
    elif cropped_window_open:
        cv2.destroyWindow('Cropped Frame')
        cropped_window_open = False

    cv2.imshow('Processed Frame', processed_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
