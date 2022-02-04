import os
import sys
import random
import math
from unittest import result
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

import pyrealsense2 as rs
import cv2

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)
device = profile.get_device()
color_sensor = device.first_color_sensor()

try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())

        # Show images        
        cv2.imshow('RealSense', color_image)
        # cv2.imshow('RealSense', color_image)
        key = cv2.waitKey(1)

        if key == 27: # press ESC to close the streaming
            cv2.destroyAllWindows()
            break

finally:
    # Stop streaming
    pipeline.stop()