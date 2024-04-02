# change path to working dir
import numpy as np
import cv2
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))


scale_percent = 50

# Resize function


def resize_image(image, scale_percent):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)


# change path to working dir
os.chdir(os.path.dirname(os.path.abspath(__file__)))


# Define the HSV range for red color (covering both high and low range of hue)
lower_red1 = np.array([0, 120, 70])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 120, 70])
upper_red2 = np.array([179, 255, 255])

# Define the HSV range for green color
lower_green = np.array([40, 40, 40])
upper_green = np.array([90, 255, 255])

# Read image.
img = cv2.imread('img.png', cv2.IMREAD_COLOR)

# Convert image to HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Create masks for red and green colors
mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
mask_green = cv2.inRange(hsv, lower_green, upper_green)

# Combine masks
mask = cv2.bitwise_or(mask_red1, mask_red2)
mask = cv2.bitwise_or(mask, mask_green)

# Bitwise-AND mask and original image
filtered_img = cv2.bitwise_and(img, img, mask=mask)

# Convert to grayscale.
gray = cv2.cvtColor(filtered_img, cv2.COLOR_BGR2GRAY)

# Blur using 3 * 3 kernel.
gray_blurred = cv2.blur(gray, (3, 3))

# Apply Hough transform on the blurred image.
detected_circles = cv2.HoughCircles(gray_blurred,
                                    cv2.HOUGH_GRADIENT, 1, 20, param1=50,
                                    param2=30, minRadius=1, maxRadius=40)

# Draw circles that are detected.
if detected_circles is not None:

    # Convert the circle parameters a, b and r to integers.
    detected_circles = np.uint16(np.around(detected_circles))

    for pt in detected_circles[0, :]:
        a, b, r = pt[0], pt[1], pt[2]

        # Only draw circles with radius greater than the threshold.
        if r > 15:
            # Draw the circumference of the circle in black.
            cv2.circle(img, (a, b), r, (0, 0, 0), 2)

            # Draw a small circle (of radius 1) to show the center in black.
            cv2.circle(img, (a, b), 1, (0, 0, 0), 3)

# Display the image with all circles drawn.
cv2.imshow("Detected Circle", resize_image(img, scale_percent))
cv2.imshow("filtered", resize_image(filtered_img, scale_percent))

cv2.waitKey(0)