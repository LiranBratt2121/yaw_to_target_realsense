import numpy as np
import cv2

FOCAL_LENGTH = 600  # Adjust this based on your camera calibration
HORIZONAL_PIXEL_SIZE = 640  # Adjust this based on your camera calibration
CENTER_PIXEL = HORIZONAL_PIXEL_SIZE / 2


def calculate_yaw_angle(depth_frame, target_pixel_coordinates):
    depth_value = depth_frame.get_distance(
        target_pixel_coordinates[0], target_pixel_coordinates[1])

    # Calculate horizontal angle
    horizontal_angle = np.arctan2(
        target_pixel_coordinates[0] - CENTER_PIXEL, FOCAL_LENGTH)

    # Calculate yaw angle
    yaw_angle = np.degrees(horizontal_angle)

    return yaw_angle


def detect_target(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_bound = np.array([1, 152, 93])  # Tomato detector!
    upper_bound = np.array([21, 229, 195])

    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_contour = max(contours, key=cv2.contourArea, default=None)

    if max_contour is not None:
        x, y, w, h = cv2.boundingRect(max_contour)

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        target_coordinates = (x + w // 2, y + h // 2)

        return target_coordinates

    return None
