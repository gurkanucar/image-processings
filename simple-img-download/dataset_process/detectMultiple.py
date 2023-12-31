
from ultralytics import YOLO
import cv2
import os

# Set working directory to script location
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Initialize webcam and set resolution
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Load models
model1 = YOLO("yolo-Weights/yolov8n.pt")
model2 = YOLO("best.pt")

# Object classes for detection
classes_model1 = {"cell phone", "person"}
classes_model2 = {"phone_back_camera"}

def detect_highest_confidence(model, image, classes_to_detect):
    highest_confidence_detections = []
    results = model(image, stream=True)
    for r in results:
        for box in r.boxes:
            confidence = float(box.conf[0])
            detected_class = model.names[int(box.cls[0])]
            if detected_class in classes_to_detect:
                highest_confidence_detections.append((confidence, box.xyxy[0], detected_class))
    if highest_confidence_detections:
        return max(highest_confidence_detections, key=lambda x: x[0])
    return None

def draw_boxes(image, detections):
    for detection in detections:
        if detection:
            confidence, box, detected_class = detection
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.putText(image, f"{detected_class}: {round(confidence, 2):.2f}",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

while True:
    success, img = cap.read()
    if not success:
        break

    # Crop image with fixed margins
    cropped_img = img[20:-20, 120:-120]

    # Detect highest confidence boxes using both models
    detection1 = detect_highest_confidence(model1, cropped_img, classes_model1)
    detection2 = detect_highest_confidence(model2, cropped_img, classes_model2)

    # Draw boxes for highest confidence detections
    draw_boxes(cropped_img, [detection1, detection2])

    # Display the image
    cv2.imshow('Webcam', cropped_img)
    if cv2.waitKey(1) == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
