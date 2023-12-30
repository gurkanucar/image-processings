import cv2
import numpy as np

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


# Start capturing from webcam
cap = cv2.VideoCapture(0)
max_cropped_frames = 0
open_windows = set()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Process the frame
    processed_frame, cropped_frames = process_frame(frame)

    # Display the processed frame
    cv2.imshow('Processed Frame', processed_frame)

    # Display each cropped frame
    for i, cropped in enumerate(cropped_frames):
        window_name = f'Cropped Frame {i+1}'
        cv2.imshow(window_name, cropped)
        open_windows.add(window_name)
    
    # Close extra windows if the number of cropped frames is less than before
    for i in range(len(cropped_frames), max_cropped_frames):
        window_name = f'Cropped Frame {i+1}'
        if window_name in open_windows:
            cv2.destroyWindow(window_name)
            open_windows.remove(window_name)
    
    # Update the max number of cropped frames
    max_cropped_frames = max(max_cropped_frames, len(cropped_frames))

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cap.release()
for window in open_windows:
    cv2.destroyWindow(window)
cv2.destroyAllWindows()
