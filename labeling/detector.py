import numpy as np
import cv2
import os


def resize_image(image, scale_percent):
    """Resize the given image by the specified scale percentage."""
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dimensions = (width, height)
    return cv2.resize(image, dimensions, interpolation=cv2.INTER_AREA)


def detect_circles(imageCopy, image, hsv, color_range, min_radius, threshold):
    """Detect circles of a specific color in the given image."""
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
        blurred, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=min_radius, maxRadius=40)
    circles = []
    if detected_circles is not None:
        detected_circles = np.uint16(np.around(detected_circles))
        for x, y, r in detected_circles[0, :]:
            if r > threshold:
                circles.append({'a': x, 'b': y, 'r': r})
                cv2.circle(image, (x, y), r, (0, 0, 0), 15)
                cv2.circle(image, (x, y), 1, (0, 255, 0), 6)
    return circles, filtered_img


def main():
    # Change path to working directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Define color ranges
    red_range = [([0, 120, 70], [10, 255, 255]),
                 ([160, 120, 70], [179, 255, 255])]
    green_range = [([40, 40, 40], [90, 255, 255])]

    # Load and preprocess the image
    org_img = cv2.imread('jig.png', cv2.IMREAD_COLOR)
    height, width = org_img.shape[:2]
    img = org_img[340:height-350, 170:width-170]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Detect circles
    red_circles, filtered_img_red = detect_circles(
        img.copy(), img, hsv, red_range, 1, 15)
    green_circles, filtered_img_green = detect_circles(
        img.copy(), img, hsv, green_range, 1, 15)

    # Display results
    scale_percent = 50
    cv2.putText(filtered_img_red, f'count: {len(red_circles)}', (
        50, 500), cv2.FONT_HERSHEY_COMPLEX, 2, (102, 255, 255), 2)
    cv2.putText(filtered_img_green, f'count: {len(green_circles)}', (
        50, 500), cv2.FONT_HERSHEY_COMPLEX, 2, (102, 255, 255), 2)
    cv2.imshow("Detected Circle", resize_image(img, scale_percent))
    cv2.imshow("filtered_red", resize_image(filtered_img_red, scale_percent))
    cv2.imshow("filtered_green", resize_image(
        filtered_img_green, scale_percent))
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
