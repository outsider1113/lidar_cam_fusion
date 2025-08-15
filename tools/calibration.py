#!/usr/bin/env python3
# Computes the Intrinsic Parameters of the camera
# Maps Camera frame 3D points onto Image pixel frame 2D positions
# x_pixel = K @ X_camera

# modified from https://github.com/Triton-AI/sick_lidar_fusion_project/blob/main/src/cam_lidar_fusion/cam_lidar_fusion/camera_calibration.py


import cv2
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- ROS 2 additions (only change in data source) ---
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class ImageBuffer(Node):
    """Minimal ROS 2 subscriber that buffers the latest BGR frame from /camera/color/image_raw."""
    def __init__(self, topic="/camera/color/image_raw"):
        super().__init__("camera_calibration_image_buffer")
        self.bridge = CvBridge()
        self.latest_frame = None
        self.sub = self.create_subscription(Image, topic, self._cb, 10)

    def _cb(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.latest_frame = frame
        except CvBridgeError as e:
            self.get_logger().warn(f"CvBridge error: {e}")

# Chessboard configuration
rows = 9    # Number of corners (not cells) in row
cols = 6    # Number of corners (not cells) in column
size = 20   # Physical size of a cell (mm) (the distance between neighboring corners).

# Input images capturing the chessboard above
# input_files = '../data/chessboard/*.jpg'

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (size,0,0), (2*size,0,0) ....,((cols-1)*size,(rows-1)*size,0)
# cols and rows flipped because cols represent x and rows represent y
objp = np.zeros((rows * cols, 3), np.float32)
objp[:, :2] = (np.mgrid[0:cols, 0:rows] * size).T.reshape(-1, 2)
# print(objp)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

# --- Replace cv2.VideoCapture with ROS2 image topic ---
rclpy.init()
node = ImageBuffer(topic="/camera/color/image_raw")

print("Please move the camera slowly when you are collecting sample images")
print("Please press 'q' to quit once you are done collecting sample images for calibration")

gray_frame = None  # will be set once we process a frame

try:
    while rclpy.ok():
        # spin the node briefly to receive frames
        rclpy.spin_once(node, timeout_sec=0.1)

        # UI key handling
        key = cv2.waitKey(100) & 0xFF
        if key == ord("q"):
            break

        # If we haven't received any frame yet, keep waiting
        if node.latest_frame is None:
            continue

        frame = node.latest_frame
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        found, corners = cv2.findChessboardCorners(gray_frame, (cols, rows), None)

        # If found, add object points, image points (after refining them)
        if found:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray_frame, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            # Draw and display the corners
            cv2.drawChessboardCorners(frame, (cols, rows), corners2, found)
            cv2.imshow('frame', frame)

finally:
    cv2.destroyAllWindows()
    rclpy.shutdown()

print("# of Images taken (want at least 10): ", len(objpoints))

if gray_frame is None or len(objpoints) == 0:
    raise RuntimeError("No frames/samples collected. Make sure /camera/color/image_raw is publishing and the board is visible.")

ret, K, d, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray_frame.shape[::-1], None, None)

print('K: ', K)
print('d: ', d)
print('img size: ', gray_frame.shape)

# # To undistort an image: ##########################################################################################
# img = cv2.imread('your_img.jpg')
# h, w = img.shape[:2]
# newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, d, (w, h), 1, (w, h))
#
# # undistort
# dst = cv2.undistort(img, K, d, None, newcameramtx)
# # crop the image
# x, y, w, h = roi
# dst = dst[y:y+h, x:x+w]
# cv2.imwrite('calibresult.png', dst)

# ####################################################################################################################

# Visualize undistort results ######################################################################################
# (If you want to visualize from the live topic instead, you can adapt similarly)
# cap = cv2.VideoCapture(0)
# while True:
#     key = cv2.waitKey(300) & 0xFF
#     if key == ord("q"):
#         break
#     ret, frame = cap.read()
#
#     h, w = frame.shape[:2]
#     newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, d, (w, h), 1, (w, h))
#     undistort_frame = cv2.undistort(frame, K, d, None, newcameramtx)
#     x, y, w, h = roi
#     undistort_frame = undistort_frame[y:y+h, x:x+w]
#     cv2.imshow('Original', frame)
#     cv2.imshow('Undistorted', undistort_frame)
#
# cv2.destroyAllWindows()
# cap.release()
