import cv2
import numpy as np
import pyrealsense2 as rs
from utils import *

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)

try:
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        target_coordinates = detect_target(
            np.asanyarray(color_frame.get_data()))

        if target_coordinates:
            yaw_angle = calculate_yaw_angle(depth_frame, target_coordinates)
            print(f"Yaw Angle: {yaw_angle} degrees")

        color_image = np.asanyarray(color_frame.get_data())
        cv2.imshow('Color Frame', color_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
