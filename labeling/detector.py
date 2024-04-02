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

redCircles = []
greenCircles = []


# Define the HSV range for red color (covering both high and low range of hue)
lower_red1 = np.array([0, 120, 70])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 120, 70])
upper_red2 = np.array([179, 255, 255])

# Define the HSV range for green color
lower_green = np.array([40, 40, 40])
upper_green = np.array([90, 255, 255])

# Read image.
org_img = cv2.imread('jig.png', cv2.IMREAD_COLOR)

height, width = org_img.shape[:2]
# Define the margins to crop from each side of the image
left = 170
top = 340
right = width - 170
bottom = height - 350

# Crop the image
img = org_img[top:bottom, left:right]
img2 = org_img[top:bottom, left:right]
# Convert image to HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)

# Create masks for red and green colors
mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
mask_green = cv2.inRange(hsv2, lower_green, upper_green)

# Combine masks
mask_red = cv2.bitwise_or(mask_red1, mask_red2)

# Bitwise-AND mask and original image
filtered_img_red = cv2.bitwise_and(img, img, mask=mask_red)
filtered_img_green = cv2.bitwise_and(img2, img2, mask=mask_green)

# Convert to grayscale.
gray_red = cv2.cvtColor(filtered_img_red, cv2.COLOR_BGR2GRAY)
gray_green = cv2.cvtColor(filtered_img_green, cv2.COLOR_BGR2GRAY)


# Blur using 3 * 3 kernel.
# gray_blurred = cv2.blur(gray, (3, 3))
gray_blurred_red = cv2.GaussianBlur(gray_red, (5, 5), 5)
gray_blurred_green = cv2.GaussianBlur(gray_green, (5, 5), 5)

# Apply Hough transform on the blurred image.
detected_circles_red = cv2.HoughCircles(gray_blurred_red,
                                        cv2.HOUGH_GRADIENT, 1, 20, param1=50,
                                        param2=30, minRadius=1, maxRadius=40)
# Apply Hough transform on the blurred image.
detected_circles_green = cv2.HoughCircles(gray_blurred_green,
                                          cv2.HOUGH_GRADIENT, 1, 20, param1=50,
                                          param2=30, minRadius=1, maxRadius=40)

# Draw circles that are detected.
if detected_circles_red is not None:
    # Convert the circle parameters a, b and r to integers.
    detected_circles_red = np.uint16(np.around(detected_circles_red))
    for pt in detected_circles_red[0, :]:
        a, b, r = pt[0], pt[1], pt[2]
        # Only draw circles with radius greater than the threshold.
        if r > 15:
            redCircles.append({a: a, b: b, r: r})
            # Draw the circumference of the circle in black.
            cv2.circle(img, (a, b), r, (0, 0, 0), 15)
            # Draw a small circle (of radius 1) to show the center in black.
            cv2.circle(img, (a, b), 1, (0, 255, 0), 6)

if detected_circles_green is not None:
    # Convert the circle parameters a, b and r to integers.
    detected_circles_green = np.uint16(np.around(detected_circles_green))
    for pt in detected_circles_green[0, :]:
        a, b, r = pt[0], pt[1], pt[2]
        # Only draw circles with radius greater than the threshold.
        if r > 15:
            greenCircles.append({a: a, b: b, r: r})
            # Draw the circumference of the circle in black.
            cv2.circle(img, (a, b), r, (0, 0, 0), 15)
            # Draw a small circle (of radius 1) to show the center in black.
            cv2.circle(img, (a, b), 1, (0, 255, 0), 6)


cv2.putText(filtered_img_red, 'count: {}'.format(len(redCircles)), (50, 500),
            cv2.FONT_HERSHEY_COMPLEX, 2, (102, 255, 255), 2)

cv2.putText(filtered_img_green, 'count: {}'.format(len(greenCircles)), (50, 500),
            cv2.FONT_HERSHEY_COMPLEX, 2, (102, 255, 255), 2)

# Display the image with all circles drawn.
cv2.imshow("Detected Circle", resize_image(img, scale_percent))
cv2.imshow("filtered_red", resize_image(filtered_img_red, scale_percent))
cv2.imshow("filtered_green", resize_image(filtered_img_green, scale_percent))

cv2.waitKey(0)
