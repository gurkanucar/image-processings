
import cv2
from ultralytics import YOLO
import numpy as np
import os
import socket
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
    x, y, w, h = 0, 0, 0, 0  # Initialize variables
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

    return frame, cropped_frames, (x, y, w, h)


model = YOLO("best.pt")
classes_to_detect = ["phone_back_camera"]

# Initialize webcam
# cap = cv2.VideoCapture(0)


UDP_IP = "192.168.0.16"  # Local IP address
UDP_PORT = 2451         # The port you choose to use

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
sock.setblocking(0)  # Set the socket to non-blocking

def receive_image(sock):
    try:
        data, addr = sock.recvfrom(65507)
        if len(data) == 4:
            # This is a message error sent back by the server
            return None
        if len(data) > 4:
            img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), 1)
            # Rotate the image by 180 degrees
            rotated_img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            mirrored_img = cv2.flip(rotated_img, -1)
            return mirrored_img
    except BlockingIOError:
        return None


while True:
    # ret, frame = cap.read()
    # if not ret:
    #     break

    frame = receive_image(sock)
    if frame is not None:

        processed_frame, cropped_frames, (crop_x, crop_y, crop_w, crop_h) = process_frame(frame)

        highest_confidence = 0
        highest_conf_box = None

        if cropped_frames:
            cropped_img = cropped_frames[0]
            results = model(cropped_img, stream=True)

            for r in results:
                boxes = r.boxes

                for box in boxes:
                    cls = int(box.cls[0])
                    detected_class = model.names[cls]

                    if detected_class in classes_to_detect:
                        confidence = box.conf[0]
                        if confidence > highest_confidence:
                            highest_confidence = confidence
                            highest_conf_box = map(int, box.xyxy[0])

            if highest_conf_box:
                x1, y1, x2, y2 = highest_conf_box
                # Calculate phone camera position in the original frame
                phone_pos_y = crop_y + (y1 + y2) / 2

                # Determine if the phone is top to bottom or bottom to top
                phone_position = "Top to Bottom" if phone_pos_y < crop_y + crop_h * 0.6 else "Bottom to Top"
                cv2.putText(processed_frame, phone_position, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.rectangle(processed_frame, (crop_x + x1, crop_y + y1), (crop_x + x2, crop_y + y2), (0, 0, 255), 3)
                cv2.putText(processed_frame, "Phone camera detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Processed Frame', processed_frame)
        
        if cropped_frames:
            cv2.imshow('Cropped Frame', cropped_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# cap.release()
cv2.destroyAllWindows()