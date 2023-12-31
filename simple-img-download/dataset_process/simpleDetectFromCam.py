import cv2
from ultralytics import YOLO
import cv2
import math


# change path to working dir
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

model = YOLO("best.pt")

classes_to_detect = ["phone_back_camera"]

# Start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

while True:
    success, img = cap.read()

    # Get the dimensions of the image
    height, width = img.shape[:2]

    # Define the margins to crop from each side of the image
    offsetY = 20
    offsetX = 120
    left, top = offsetX, offsetY
    right, bottom = width - offsetX, height - offsetY

    # Crop the image
    cropped_img = img[top:bottom, left:right]

    results = model(cropped_img, stream=True)

    # Create a dictionary to store the highest-confidence detection for each class
    highest_confidence = {}

    # Iterate over detected objects
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # Bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Confidence
            confidence = math.ceil((box.conf[0] * 100)) / 100

            # Class name
            cls = int(box.cls[0])
            detected_class = model.names[cls]

            # Check if the detected class is in classes_to_detect
            if detected_class in classes_to_detect:
                # If the class is not in the dictionary or has higher confidence, update it
                if detected_class not in highest_confidence or confidence > highest_confidence[detected_class][0]:
                    highest_confidence[detected_class] = (confidence, (x1, y1, x2, y2))

    # Iterate over the highest-confidence detections and draw rectangles and labels
    for detected_class, (confidence, (x1, y1, x2, y2)) in highest_confidence.items():
        cv2.rectangle(cropped_img, (x1, y1), (x2, y2), (255, 0, 255), 3)
        cv2.putText(cropped_img, f"{detected_class}: {confidence:.2f}",
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Display the image
    cv2.imshow('Webcam', cropped_img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
