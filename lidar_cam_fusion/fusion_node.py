#!/usr/bin/env python3
"""
fusion_node.py  –  LiDAR × Camera fusion (GPU optimized with fast overlay)
"""

import os
import cv2
import rclpy
import numpy as np
import sensor_msgs_py.point_cloud2 as pc2
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, PointCloud2, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
from ultralytics import YOLO
from ament_index_python.packages import get_package_share_directory
import torch
import yaml
from scipy.spatial.transform import Rotation as Rot
from lidar_cam_fusion.msg import FusionDetections, DetectedObject  # Import custom messages
from geometry_msgs.msg import Point  # For center point

# Enable CUDA optimizations if available
torch.backends.cudnn.benchmark = True

try:
    TORCH_CUDA = torch.cuda.is_available()
except ImportError:
    TORCH_CUDA = False

# Define QoS profile for reliable communication
qos_profile_reliable = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10
)

class FusionNode(Node):
    def __init__(self):
        super().__init__('fusion_node')

        # Load configuration from YAML file
        config_path = os.path.join(
            get_package_share_directory("lidar_cam_fusion"),
            "config", "fusion_config.yaml"
        )
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            if config is None:
                raise ValueError("YAML configuration is empty")
        except (FileNotFoundError, ValueError) as e:
            self.get_logger().error(f"Configuration error: {e}")
            raise
        except yaml.YAMLError as e:
            self.get_logger().error(f"Error parsing YAML file: {e}")
            raise

        # Extract parameters from YAML config
        camera_config = config['camera']
        extrinsics_config = config['extrinsics']
        topics_config = config['topics']
        display_config = config['display']
        debug_config = config.get('debug', {'enabled': False})  # Default to False if not present
        self.debug = debug_config.get('enabled', False)

        # Camera intrinsics
        self.K = np.array(camera_config['K'], dtype=np.float32).reshape(3, 3)
        self.dist_coeffs = np.array(camera_config['dist_coeffs'], dtype=np.float32)
        self.img_size = None  # Set dynamically

        # Initial extrinsics (will be adjustable in debug mode)
        self.base_R = np.array(extrinsics_config['R'], dtype=np.float32).reshape(3, 3)
        self.base_T = np.array(extrinsics_config['T'], dtype=np.float32).reshape(3, 1)
        self.R = self.base_R.copy()
        self.T = self.base_T.copy()

        # Debug adjustments for rotation (Euler angles in degrees)
        self.debug_r_adjust = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # pitch, yaw, roll

        # ROS 2 topic names
        self.img_topic_name = topics_config['camera_image']
        self.lidar_topic_name = topics_config['lidar_points']
        self.fusion_output_topic = topics_config['fusion_output']

        # Display flags
        self.show_lidar_projections = display_config['show_lidar_projections']
        self.show_fusion_result_opencv = display_config['show_fusion_result_opencv']

        # Initialize YOLO model
        model_path = os.path.join(
            get_package_share_directory("lidar_cam_fusion"),
            "config", "sim_box.pt"
        )
        self.yolo_model = YOLO(model_path)
        if TORCH_CUDA:
            self.yolo_model.to(device='cuda')
            self.get_logger().info("YOLO running on CUDA")
        else:
            self.yolo_model.to(device='cpu')
            self.get_logger().info("YOLO running on CPU")

        # Set up device and transformation matrices
        self.device = torch.device('cuda' if TORCH_CUDA else 'cpu')
        self.update_transformation_matrices()

        # Initialize overlay and debug state
        self.overlay = None
        self.frame = None
        self.frame_counter = 0
        self.last_detections = []
        self.last_show_time = 0

        # Initialize CV Bridge
        self.bridge = CvBridge()

        # Subscriptions and publishers
        if self.img_topic_name.endswith("/compressed"):
            self.get_logger().info(f"Subscribing to compressed image topic: {self.img_topic_name}")
            self.image_subscription = self.create_subscription(
                CompressedImage, self.img_topic_name, self.camera_cb_compressed, qos_profile_reliable
            )
        else:
            self.get_logger().info(f"Subscribing to raw image topic: {self.img_topic_name}")
            self.image_subscription = self.create_subscription(
                Image, self.img_topic_name, self.camera_cb_raw, qos_profile_reliable
            )
        self.create_subscription(PointCloud2, self.lidar_topic_name, self.lidar_cb, qos_profile_reliable)
        self.fusion_img_pub = self.create_publisher(Image, self.fusion_output_topic, 10)
        self.fusion_detections_pub = self.create_publisher(FusionDetections, '/fusion/detections', 10)  # New topic for metadata

        # Setup window and debug trackbars if enabled
        if self.show_fusion_result_opencv:
            cv2.namedWindow("Fusion")
            if self.debug:
                self.setup_debug_trackbars()

    def setup_debug_trackbars(self):
        """Set up trackbars for adjusting T and rotation (Euler angles) in debug mode with defaults from config."""
        # Extended T ranges: ±5.0 for all, scale factor 1000 for 0.001 precision (0 to 10000)
        t_x_default = int((self.base_T[0, 0] + 5.0) * 1000)  # Map base to position
        t_y_default = int((self.base_T[1, 0] + 5.0) * 1000)
        t_z_default = int((self.base_T[2, 0] + 5.0) * 1000)

        cv2.createTrackbar("T_x (x0.001)", "Fusion", t_x_default, 10000, self.on_trackbar_change)
        cv2.createTrackbar("T_y (x0.001)", "Fusion", t_y_default, 10000, self.on_trackbar_change)
        cv2.createTrackbar("T_z (x0.001)", "Fusion", t_z_default, 10000, self.on_trackbar_change)

        # R adjustments: -5 to 5 degrees, scale factor 100 for precision (0 to 1000, offset 500)
        r_pitch_default = 500  # 0 degrees
        r_yaw_default = 500
        r_roll_default = 500
        cv2.createTrackbar("R_pitch (x0.01 deg)", "Fusion", r_pitch_default, 1000, self.on_trackbar_change)
        cv2.createTrackbar("R_yaw (x0.01 deg)", "Fusion", r_yaw_default, 1000, self.on_trackbar_change)
        cv2.createTrackbar("R_roll (x0.01 deg)", "Fusion", r_roll_default, 1000, self.on_trackbar_change)
        self.get_logger().info("Debug mode enabled with trackbars for T and rotation adjustments attached to Fusion window")

        # Trigger initial update with default values
        self.on_trackbar_change(0)  # Dummy call to update

    def on_trackbar_change(self, value):
        """Callback for trackbar changes to update T and R."""
        if not self.debug:
            return
        # Convert trackbar values to actual T values (±5.0 range)
        t_x = (cv2.getTrackbarPos("T_x (x0.001)", "Fusion") / 1000.0) - 5.0  # 0 to 10000 -> -5.0 to 5.0
        t_y = (cv2.getTrackbarPos("T_y (x0.001)", "Fusion") / 1000.0) - 5.0
        t_z = (cv2.getTrackbarPos("T_z (x0.001)", "Fusion") / 1000.0) - 5.0

        self.T = np.array([t_x, t_y, t_z], dtype=np.float32).reshape(3, 1)

        # Convert trackbar values to rotation adjustments (degrees)
        r_pitch = (cv2.getTrackbarPos("R_pitch (x0.01 deg)", "Fusion") / 100.0) - 5.0  # 0 to 1000 -> -5.0 to 5.0
        r_yaw = (cv2.getTrackbarPos("R_yaw (x0.01 deg)", "Fusion") / 100.0) - 5.0
        r_roll = (cv2.getTrackbarPos("R_roll (x0.01 deg)", "Fusion") / 100.0) - 5.0

        self.debug_r_adjust = np.array([r_pitch, r_yaw, r_roll], dtype=np.float32)

        # Log adjusted values for easy copying to YAML
        self.get_logger().info(f"Adjusted T: [{t_x:.3f}, {t_y:.3f}, {t_z:.3f}]")
        self.get_logger().info(f"Adjusted R angles (pitch, yaw, roll deg): [{r_pitch:.2f}, {r_yaw:.2f}, {r_roll:.2f}]")

        # Update transformation matrices
        self.update_transformation_matrices()

    def update_transformation_matrices(self):
        """Update M_t and M with debug adjustments for T and R."""
        # Apply small rotation adjustments
        adjust_rot = Rot.from_euler('xyz', self.debug_r_adjust, degrees=True).as_matrix().astype(np.float32)
        self.R = adjust_rot @ self.base_R  # Pre-multiply for local adjustment in camera frame; swap to self.base_R @ adjust_rot if needed

        if TORCH_CUDA:
            self.M_t = torch.cat((
                torch.cat((
                    torch.as_tensor(self.R, dtype=torch.float32, device=self.device),
                    torch.as_tensor(self.T, dtype=torch.float32, device=self.device)
                ), dim=1),
                torch.tensor([[0, 0, 0, 1]], dtype=torch.float32, device=self.device)
            ), dim=0)
            self.K_t = torch.as_tensor(self.K, dtype=torch.float32, device=self.device)
        else:
            self.M = np.vstack((np.hstack((self.R, self.T)), [0, 0, 0, 1]))

    def camera_cb_compressed(self, msg: CompressedImage):
        """Callback for compressed camera image messages."""
        try:
            self.frame = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
            if self.img_size is None:
                self.img_size = (self.frame.shape[0], self.frame.shape[1])
                self.overlay = np.zeros((self.img_size[0], self.img_size[1], 3), dtype=np.uint8)
                self.get_logger().info(f"Initialized img_size: {self.img_size}")
        except CvBridgeError as e:
            self.get_logger().error(f"CV Bridge error (compressed): {e}")

    def camera_cb_raw(self, msg: Image):
        """Callback for raw camera image messages."""
        try:
            self.frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            if self.img_size is None:
                self.img_size = (self.frame.shape[0], self.frame.shape[1])
                self.overlay = np.zeros((self.img_size[0], self.img_size[1], 3), dtype=np.uint8)
                self.get_logger().info(f"Initialized img_size: {self.img_size}")
        except CvBridgeError as e:
            self.get_logger().error(f"CV Bridge error (raw): {e}")

    def lidar_cb(self, msg: PointCloud2):
        """Callback for LiDAR point cloud messages."""
        if self.frame is None or self.img_size is None:
            return

        # Extract points from PointCloud2 message
        pts = list(pc2.read_points(msg, field_names=('x', 'y', 'z', 'intensity'), skip_nans=True))
        if not pts:
            return

        xs, ys, zs, intensities = zip(*pts)
        xs = np.array(xs, dtype=np.float32)
        ys = np.array(ys, dtype=np.float32)
        zs = np.array(zs, dtype=np.float32)
        intensities = np.array(intensities, dtype=np.float32)

        # Project LiDAR points to image plane
        if TORCH_CUDA:
            px, py, depths, proj_intens = lidar2pixel_cuda(xs, ys, zs, intensities, self.M_t, self.K_t, self.device)
            px_i = px.round().to(dtype=torch.int32).cpu().numpy()
            py_i = py.round().to(dtype=torch.int32).cpu().numpy()
            depths = depths.cpu().numpy()
            proj_intens = proj_intens.cpu().numpy()
        else:
            px_i, py_i, depths, proj_intens = lidar2pixel_cpu(xs, ys, zs, intensities, self.M, self.K)

        # Filter points within image bounds
        h, w = self.img_size
        mask = (px_i >= 0) & (px_i < w) & (py_i >= 0) & (py_i < h) & (depths > 0)
        px_i, py_i, depths, proj_intens = px_i[mask], py_i[mask], depths[mask], proj_intens[mask]

        # Create depth image for YOLO bounding box depth calculation
        depth_img = np.full((h, w), np.inf, dtype=np.float32)
        np.minimum.at(depth_img, (py_i, px_i), depths)

        # YOLO detection (every 3rd frame for performance)
        self.frame_counter += 1
        if self.frame_counter % 3 == 0:
            with torch.cuda.amp.autocast(enabled=TORCH_CUDA):
                results = self.yolo_model.predict(source=self.frame, verbose=False)[0]
            self.last_detections = results.boxes.cpu().xyxyn.numpy()

        # Draw YOLO detections with depth and prepare detections data
        frame = np.array(self.frame, copy=True)
        detections = []  # List for DetectedObject msgs
        for xmin, ymin, xmax, ymax in self.last_detections:
            x1 = int(xmin * w)
            y1 = int(ymin * h)
            x2 = int(xmax * w)
            y2 = int(ymax * h)
            x1 = np.clip(x1, 0, w-1)
            x2 = np.clip(x2, 0, w-1)
            y1 = np.clip(y1, 0, h-1)
            y2 = np.clip(y2, 0, h-1)
            box_depth = np.min(depth_img[y1:y2, x1:x2]) if (y2 > y1 and x2 > x1) else np.nan
            if np.isnan(box_depth):
                continue  # Skip invalid depths
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"{box_depth:.2f} m", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Compute center (x, y in pixels; z=0). Left/right relative to w/2.
            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0
            center_point = Point()
            center_point.x = center_x
            center_point.y = center_y
            center_point.z = 0.0

            # Add to detections list
            det_obj = DetectedObject()
            det_obj.depth = float(box_depth)
            det_obj.center = center_point
            detections.append(det_obj)

        # Overlay LiDAR points if enabled
        if self.show_lidar_projections:
            self.overlay.fill(0)
            if len(px_i) > 0:
                max_dist_thresh = 10.0
                color_intensities = np.clip((depths / max_dist_thresh * 255), 0, 255).astype(np.uint8)
                colors = np.column_stack([np.zeros_like(color_intensities), color_intensities, 255 - color_intensities])
                self.overlay[py_i, px_i] = colors
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        py_offset = np.clip(py_i + dy, 0, h-1)
                        px_offset = np.clip(px_i + dx, 0, w-1)
                        self.overlay[py_offset, px_offset] = colors
            mask = (self.overlay > 0).any(axis=2)
            frame[mask] = self.overlay[mask]

        # Display fusion result if enabled
        now = rclpy.clock.Clock().now().nanoseconds
        if self.show_fusion_result_opencv and now - self.last_show_time > 100_000_000:
            cv2.imshow("Fusion", frame)
            cv2.waitKey(1)
            self.last_show_time = now

        # Publish fused image (standard Image msg)
        out_img_msg = self.bridge.cv2_to_imgmsg(frame, 'bgr8')
        out_img_msg.header.stamp = self.get_clock().now().to_msg()
        out_img_msg.header.frame_id = 'camera_link'
        self.fusion_img_pub.publish(out_img_msg)

        # Publish detections metadata separately
        detections_msg = FusionDetections()
        detections_msg.header.stamp = out_img_msg.header.stamp
        detections_msg.header.frame_id = 'camera_link'
        detections_msg.object_detected = len(detections) > 0
        detections_msg.detections = detections
        self.fusion_detections_pub.publish(detections_msg)

def lidar2pixel_cuda(xs, ys, zs, intensities, M_t, K_t, device):
    """Project LiDAR points to image plane using CUDA."""
    xyz = torch.from_numpy(np.stack([xs, ys, zs])).to(device, dtype=torch.float32)
    intens = torch.tensor(intensities, device=device, dtype=torch.float32)
    ones = torch.ones((1, xyz.shape[1]), dtype=torch.float32, device=device)
    hom = torch.cat((xyz, ones), dim=0)
    cam = M_t @ hom
    z_cam = cam[2]
    mask = z_cam > 0
    if torch.sum(mask) == 0:
        return torch.zeros_like(xyz[0]), torch.zeros_like(xyz[0]), torch.zeros_like(xyz[0]), torch.zeros_like(intens)
    cam_filtered = cam[:, mask]
    z_filtered = z_cam[mask]
    xyz_filtered = xyz[:, mask]
    intens_filtered = intens[mask]
    xy_pixel = K_t @ (cam_filtered[:3] / z_filtered)
    depths = torch.linalg.norm(xyz_filtered, dim=0)
    return xy_pixel[0], xy_pixel[1], depths, intens_filtered

def lidar2pixel_cpu(xs, ys, zs, intensities, M, K):
    """Project LiDAR points to image plane using CPU."""
    xyz = np.vstack((xs, ys, zs))
    depths = np.linalg.norm(xyz, axis=0)
    hom = np.vstack((xyz, np.ones_like(xs)))
    cam = M @ hom
    z_cam = cam[2]
    mask = z_cam > 0
    if np.sum(mask) == 0:
        return np.zeros_like(xyz[0]), np.zeros_like(xyz[0]), np.zeros_like(depths), np.zeros_like(intensities)
    cam_filtered = cam[:, mask]
    z_filtered = z_cam[mask]
    xyz_filtered = xyz[:, mask]
    intens_filtered = intensities[mask]
    xy_pixel = (K @ (cam_filtered[:3] / z_filtered)).astype(int)
    depths = np.linalg.norm(xyz_filtered, axis=0)
    return xy_pixel[0], xy_pixel[1], depths, intens_filtered

def main(args=None):
    rclpy.init(args=args)
    node = FusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()