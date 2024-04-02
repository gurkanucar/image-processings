import numpy as np
import cv2
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))


class CircleDetector:
    def __init__(self, image_path, color_ranges, labels, output_dir='./label_result_custom/'):
        self.image_path = image_path
        self.color_ranges = color_ranges
        self.labels = labels
        self.output_dir = output_dir
        self.bbox_list = []
        self.load_image()
        self.preprocess_image()

    def load_image(self):
        self.org_img = cv2.imread(self.image_path, cv2.IMREAD_COLOR)
        self.height, self.width = self.org_img.shape[:2]

    def preprocess_image(self):
        self.img = self.org_img[340:self.height-350, 170:self.width-170]
        self.hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)

    def detect_circles(self):
        self.filtered_imgs = {}
        for color_range, label in zip(self.color_ranges, self.labels):
            _, filtered_img = self._detect_circles(self.img.copy(), self.hsv, color_range, label)
            self.filtered_imgs[label] = filtered_img

    def _detect_circles(self, imageCopy, hsv, color_range, label):
        mask = None
        for lower, upper in color_range:
            lower = np.array(lower)
            upper = np.array(upper)
            if mask is None:
                mask = cv2.inRange(hsv, lower, upper)
            else:
                mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))
        filtered_img = cv2.bitwise_and(imageCopy, imageCopy, mask=mask)
        gray = cv2.cvtColor(filtered_img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 5)
        detected_circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=1, maxRadius=40)
        if detected_circles is not None:
            detected_circles = np.uint16(np.around(detected_circles))
            for x, y, r in detected_circles[0, :]:
                if r > 15:
                    cv2.circle(self.img, (x, y), r, (0, 0, 0), 15)
                    cv2.circle(self.img, (x, y), 1, (0, 255, 0), 6)
                    bbox = [x - r, y - r, x + r, y + r]
                    self.bbox_list.append((label, bbox))
        return detected_circles, filtered_img

    def save_labels(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        with open(os.path.join(self.output_dir, self.image_path.split(".")[0]+'.txt'), 'w') as f:
            for label, bbox in self.bbox_list:
                label_index = self.labels.index(label)  # Get the index of the label

                x_min = bbox[0] + 170
                y_min = bbox[1] + 340
                x_max = bbox[2] + 170
                y_max = bbox[3] + 340

                x_center = (x_min + x_max) / 2 / self.width
                y_center = (y_min + y_max) / 2 / self.height
                bbox_width = (x_max - x_min) / self.width
                bbox_height = (y_max - y_min) / self.height
                f.write(f'{label_index} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n')      

    def save_classes(self):
        if not os.path.exists(os.path.join(self.output_dir, 'classes.txt')):
            with open(os.path.join(self.output_dir, 'classes.txt'), 'w') as f:
                for label_name in self.labels:
                    f.write(f'{label_name}\n')


    def display_results(self):
        scale_percent = 50
        cv2.imshow("Detected Circle", self.resize_image(self.img, scale_percent))
        for label, filtered_img in self.filtered_imgs.items():
            cv2.imshow(f"filtered_{label}", self.resize_image(filtered_img, scale_percent))

    @staticmethod
    def resize_image(image, scale_percent):
        """Resize the given image by the specified scale percentage."""
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dimensions = (width, height)
        return cv2.resize(image, dimensions, interpolation=cv2.INTER_AREA)


def main():
    # Define color ranges and labels
    color_ranges = [
        [([0, 120, 70], [10, 255, 255])],  # Red range
        [([40, 40, 40], [90, 255, 255])]   # Green range
    ]
    labels = ['wrong', 'correct']

    # Initialize the CircleDetector
    detector = CircleDetector('jig.png', color_ranges, labels)

    # Detect circles
    detector.detect_circles()

    # Display results
    detector.display_results()

    # Save labels and classes
    detector.save_labels()
    detector.save_classes()

    # Wait for a key press to close the windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
