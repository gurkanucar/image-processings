
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
def get_center(box):
    x1, y1, x2, y2 = map(int, box)
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    return center_x, center_y

def annotate_image(image, phone_box, camera_box):
    # Check if both detections are valid and not empty
    if phone_box is not None and camera_box is not None and phone_box.numel() > 0 and camera_box.numel() > 0:
        _, phone_center_y = get_center(phone_box)
        _, camera_center_y = get_center(camera_box)

        phone_height = abs(int(phone_box[3]) - int(phone_box[1]))
        threshold = phone_center_y + 0.6 * phone_height

        text = "Bottom -> Top" if camera_center_y > threshold else "Top -> Bottom"
        cv2.putText(image, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        # Handle case where detections are None or empty
        print("No valid phone or camera box found for annotation.")



while True:
    success, img = cap.read()
    if not success:
        break

    # Crop image with fixed margins
    cropped_img = img[20:-20, 120:-120]

    # Detect highest confidence boxes using both models
    detection1 = detect_highest_confidence(model1, cropped_img, classes_model1)
    detection2 = detect_highest_confidence(model2, cropped_img, classes_model2)

    if detection1 and detection1[2] == "cell phone":
        phone_box = detection1[1]
    else:
        phone_box = None

    if detection2 and detection2[2] == "phone_back_camera":
        camera_box = detection2[1]
    else:
        camera_box = None

    # Annotate the image based on the position of the camera
    annotate_image(cropped_img, phone_box, camera_box)

    # Draw boxes for highest confidence detections
    draw_boxes(cropped_img, [detection1, detection2])

    # Display the image
    cv2.imshow('Webcam', cropped_img)
    if cv2.waitKey(1) == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
