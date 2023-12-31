import cv2
import numpy as np

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

# Webcam capture
cap = cv2.VideoCapture(0)
cropped_window_open = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    processed_frame, cropped_frames = process_frame(frame)
    cv2.imshow('Processed Frame', processed_frame)

    if cropped_frames:
        cv2.imshow('Cropped Frame', cropped_frames[0])
        cropped_window_open = True
    elif cropped_window_open:
        cv2.destroyWindow('Cropped Frame')
        cropped_window_open = False

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
