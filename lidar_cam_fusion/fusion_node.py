#!/usr/bin/env python3
"""
fusion_node.py – LiDAR × Camera fusion (GPU optimized)
Color overlay is based ONLY on distance (range), not intensity.
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
from lidar_cam_fusion.msg import FusionDetections, DetectedObject
from geometry_msgs.msg import Point

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
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file) or {}
        camera_config = config['camera']
        extrinsics_config = config['extrinsics']
        topics_config = config['topics']
        display_config = config['display']
        debug_config = config.get('debug', {'enabled': False})
        self.debug = debug_config.get('enabled', False)

        # Camera intrinsics
        self.K = np.array(camera_config['K'], dtype=np.float32).reshape(3, 3)
        self.dist_coeffs = np.array(camera_config['dist_coeffs'], dtype=np.float32)
        self.img_size = None  # Will be set at runtime

        # Extrinsics (base rotation and translation)
        self.base_R = np.array(extrinsics_config['R'], dtype=np.float32).reshape(3, 3)
        self.base_T = np.array(extrinsics_config['T'], dtype=np.float32).reshape(3, 1)
        self.R = self.base_R.copy()
        self.T = self.base_T.copy()
        self.debug_r_adjust = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # pitch, yaw, roll

        # Topics
        self.img_topic_name = topics_config['camera_image']
        self.lidar_topic_name = topics_config['lidar_points']
        self.fusion_output_topic = topics_config['fusion_output']

        # Display flags
        self.show_lidar_projections = display_config['show_lidar_projections']
        self.show_fusion_result_opencv = display_config['show_fusion_result_opencv']

        # Initialize YOLO model
        model_path = os.path.join(
            get_package_share_directory("lidar_cam_fusion"),
            "config", "cones_best.pt"
        )
        self.yolo_model = YOLO(model_path)
        # Fuse conv and BN for speed
        try:
            self.yolo_model.fuse()
        except Exception:
            pass
        # Use half precision on GPU if available
        if TORCH_CUDA:
            try:
                self.yolo_model.half()
            except Exception:
                pass
            self.yolo_model.to(device='cuda')
            self.get_logger().info("YOLO running on CUDA")
        else:
            self.yolo_model.to(device='cpu')
            self.get_logger().info("YOLO running on CPU")
        self.yolo_imgsz = config.get('yolo_imgsz', 416)

        # Device tensors (for GPU)
        self.device = torch.device('cuda' if TORCH_CUDA else 'cpu')
        self.update_transformation_matrices()

        # State variables
        self.overlay = None
        self.frame = None
        self.depth_img = None
        self.dilate_kernel = np.ones((3, 3), np.uint8)
        self.frame_counter = 0
        self.last_detections = []
        self.last_show_time = 0
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
        self.fusion_detections_pub = self.create_publisher(FusionDetections, '/fusion/detections', 10)

        # Debug UI
        if self.show_fusion_result_opencv:
            cv2.namedWindow("Fusion")
            if self.debug:
                self.setup_debug_trackbars()

    def setup_debug_trackbars(self):
        """Set up trackbars for adjusting T and rotation (Euler angles) in debug mode."""
        t_x_default = int((self.base_T[0, 0] + 5.0) * 1000)
        t_y_default = int((self.base_T[1, 0] + 5.0) * 1000)
        t_z_default = int((self.base_T[2, 0] + 5.0) * 1000)

        cv2.createTrackbar("T_x (x0.001)", "Fusion", t_x_default, 10000, self.on_trackbar_change)
        cv2.createTrackbar("T_y (x0.001)", "Fusion", t_y_default, 10000, self.on_trackbar_change)
        cv2.createTrackbar("T_z (x0.001)", "Fusion", t_z_default, 10000, self.on_trackbar_change)

        r_pitch_default = r_yaw_default = r_roll_default = 500
        cv2.createTrackbar("R_pitch (x0.01 deg)", "Fusion", r_pitch_default, 1000, self.on_trackbar_change)
        cv2.createTrackbar("R_yaw (x0.01 deg)", "Fusion", r_yaw_default, 1000, self.on_trackbar_change)
        cv2.createTrackbar("R_roll (x0.01 deg)", "Fusion", r_roll_default, 1000, self.on_trackbar_change)
        self.get_logger().info("Debug mode enabled with trackbars for T and rotation adjustments.")
        self.on_trackbar_change(0)

    def on_trackbar_change(self, value):
        """Update T and R when trackbars move."""
        if not self.debug:
            return
        t_x = (cv2.getTrackbarPos("T_x (x0.001)", "Fusion") / 1000.0) - 5.0
        t_y = (cv2.getTrackbarPos("T_y (x0.001)", "Fusion") / 1000.0) - 5.0
        t_z = (cv2.getTrackbarPos("T_z (x0.001)", "Fusion") / 1000.0) - 5.0
        self.T = np.array([t_x, t_y, t_z], dtype=np.float32).reshape(3, 1)

        r_pitch = (cv2.getTrackbarPos("R_pitch (x0.01 deg)", "Fusion") / 100.0) - 5.0
        r_yaw   = (cv2.getTrackbarPos("R_yaw (x0.01 deg)", "Fusion") / 100.0) - 5.0
        r_roll  = (cv2.getTrackbarPos("R_roll (x0.01 deg)", "Fusion") / 100.0) - 5.0
        self.debug_r_adjust = np.array([r_pitch, r_yaw, r_roll], dtype=np.float32)
        self.get_logger().info(
            f"Adjusted T: [{t_x:.3f}, {t_y:.3f}, {t_z:.3f}] | "
            f"Adjusted R: [{r_pitch:.2f}, {r_yaw:.2f}, {r_roll:.2f}]"
        )
        self.update_transformation_matrices()

    def update_transformation_matrices(self):
        """Recompute R, M/M_t with debug adjustments."""
        adjust_rot = Rot.from_euler('xyz', self.debug_r_adjust, degrees=True).as_matrix().astype(np.float32)
        self.R = adjust_rot @ self.base_R
        if TORCH_CUDA:
            R_torch = torch.as_tensor(self.R, dtype=torch.float32, device=self.device)
            T_torch = torch.as_tensor(self.T, dtype=torch.float32, device=self.device)
            bottom = torch.tensor([[0, 0, 0, 1]], dtype=torch.float32, device=self.device)
            self.M_t = torch.cat((torch.cat((R_torch, T_torch), dim=1), bottom), dim=0)
            self.K_t = torch.as_tensor(self.K, dtype=torch.float32, device=self.device)
        else:
            self.M = np.vstack((np.hstack((self.R, self.T)), [0, 0, 0, 1]))

    # Camera callbacks
    def camera_cb_compressed(self, msg: CompressedImage):
        try:
            self.frame = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
            if self.img_size is None:
                self.img_size = (self.frame.shape[0], self.frame.shape[1])
                self.overlay   = np.zeros((self.img_size[0], self.img_size[1], 3), dtype=np.uint8)
                self.depth_img = np.full((self.img_size[0], self.img_size[1]), np.inf, dtype=np.float32)
                self.get_logger().info(f"Initialized img_size: {self.img_size}")
        except CvBridgeError as e:
            self.get_logger().error(f"CV Bridge error (compressed): {e}")

    def camera_cb_raw(self, msg: Image):
        try:
            self.frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            if self.img_size is None:
                self.img_size = (self.frame.shape[0], self.frame.shape[1])
                self.overlay   = np.zeros((self.img_size[0], self.img_size[1], 3), dtype=np.uint8)
                self.depth_img = np.full((self.img_size[0], self.img_size[1]), np.inf, dtype=np.float32)
                self.get_logger().info(f"Initialized img_size: {self.img_size}")
        except CvBridgeError as e:
            self.get_logger().error(f"CV Bridge error (raw): {e}")

    # LiDAR callback
    def lidar_cb(self, msg: PointCloud2):
        if self.frame is None or self.img_size is None:
            return

        # Convert point cloud to numpy array
        try:
            pts = pc2.read_points_numpy(msg, field_names=('x', 'y', 'z'), skip_nans=True)
            xs, ys, zs = pts['x'], pts['y'], pts['z']
        except Exception:
            # Fallback: read via list then convert
            lst = list(pc2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True))
            if not lst:
                return
            arr = np.array(lst, dtype=np.float32)
            xs, ys, zs = arr[:,0], arr[:,1], arr[:,2]

        # Project LiDAR points
        if TORCH_CUDA:
            px, py, depths = lidar2pixel_cuda(xs, ys, zs, self.M_t, self.K_t, self.device)
            px_i = px.round().to(dtype=torch.int32).cpu().numpy()
            py_i = py.round().to(dtype=torch.int32).cpu().numpy()
            depths = depths.cpu().numpy()
        else:
            px_i, py_i, depths = lidar2pixel_cpu(xs, ys, zs, self.M, self.K)

        h, w = self.img_size
        mask = (px_i >= 0) & (px_i < w) & (py_i >= 0) & (py_i < h) & (depths > 0)
        px_i, py_i, depths = px_i[mask], py_i[mask], depths[mask]

        # Reset depth image
        self.depth_img.fill(np.inf)
        np.minimum.at(self.depth_img, (py_i, px_i), depths)

        # YOLO detection every 3rd frame
        self.frame_counter += 1
        if self.frame_counter % 3 == 0:
            with torch.cuda.amp.autocast(enabled=TORCH_CUDA):
                results = self.yolo_model.predict(source=self.frame, imgsz=self.yolo_imgsz, verbose=False)[0]
            self.last_detections = results.boxes.cpu().xyxyn.numpy()

        # Draw detections and collect metadata
        frame_drawn = np.array(self.frame, copy=True)
        detections = []
        for xmin, ymin, xmax, ymax in self.last_detections:
            x1 = int(xmin * w)
            y1 = int(ymin * h)
            x2 = int(xmax * w)
            y2 = int(ymax * h)
            x1 = np.clip(x1, 0, w-1); x2 = np.clip(x2, 0, w-1)
            y1 = np.clip(y1, 0, h-1); y2 = np.clip(y2, 0, h-1)
            box_depth = np.min(self.depth_img[y1:y2, x1:x2]) if (y2 > y1 and x2 > x1) else np.nan
            if np.isnan(box_depth):
                continue
            cv2.rectangle(frame_drawn, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                frame_drawn, f"{box_depth:.2f} m", (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )
            # Prepare detection message
            center = Point()
            center.x = (x1 + x2) / 2.0
            center.y = (y1 + y2) / 2.0
            center.z = 0.0
            det_obj = DetectedObject()
            det_obj.depth = float(box_depth)
            det_obj.center = center
            detections.append(det_obj)

        # Overlay LiDAR points by distance
        if self.show_lidar_projections:
            self.overlay.fill(0)
            if len(px_i) > 0:
                # Map distance [0, max] → [0, 255]
                max_dist_thresh = 10.0  # meters
                dist_scaled = np.clip((depths / max_dist_thresh) * 255.0, 0, 255).astype(np.uint8)
                # Near = blue, Far = green
                colors = np.column_stack([
                    255 - dist_scaled,          # Blue channel (near)
                    dist_scaled,                # Green channel (far)
                    np.zeros_like(dist_scaled)  # Red channel
                ])
                self.overlay[py_i, px_i] = colors
                # Dilate to fill small gaps
                self.overlay = cv2.dilate(self.overlay, self.dilate_kernel)
            mask = (self.overlay > 0).any(axis=2)
            frame_drawn[mask] = self.overlay[mask]

        # Show window at ~10 Hz
        now = rclpy.clock.Clock().now().nanoseconds
        if self.show_fusion_result_opencv and now - self.last_show_time > 100_000_000:
            cv2.imshow("Fusion", frame_drawn)
            cv2.waitKey(1)
            self.last_show_time = now

        # Publish fused image
        out_img = self.bridge.cv2_to_imgmsg(frame_drawn, 'bgr8')
        out_img.header.stamp = self.get_clock().now().to_msg()
        out_img.header.frame_id = 'camera_link'
        self.fusion_img_pub.publish(out_img)

        # Publish detection metadata
        det_msg = FusionDetections()
        det_msg.header.stamp = out_img.header.stamp
        det_msg.header.frame_id = 'camera_link'
        det_msg.object_detected = bool(detections)
        det_msg.detections = detections
        self.fusion_detections_pub.publish(det_msg)

def lidar2pixel_cuda(xs, ys, zs, M_t, K_t, device):
    """Project LiDAR points to image plane using CUDA; returns pixels and Euclidean depths."""
    xyz = torch.from_numpy(np.stack([xs, ys, zs])).to(device, dtype=torch.float32, non_blocking=True)
    ones = torch.ones((1, xyz.shape[1]), dtype=torch.float32, device=device)
    hom = torch.cat((xyz, ones), dim=0)
    cam = M_t @ hom
    z_cam = cam[2]
    mask = z_cam > 0
    if torch.sum(mask) == 0:
        z = torch.zeros_like(xyz[0], device=device)
        return z, z, z
    cam_filtered = cam[:, mask]
    z_filtered   = z_cam[mask]
    xyz_filtered = xyz[:, mask]
    xy = K_t @ (cam_filtered[:3] / z_filtered)
    depths = torch.linalg.norm(xyz_filtered, dim=0)
    return xy[0], xy[1], depths

def lidar2pixel_cpu(xs, ys, zs, M, K):
    """Project LiDAR points to image plane using CPU; returns pixels and Euclidean depths."""
    xyz = np.vstack((xs, ys, zs))
    depths = np.linalg.norm(xyz, axis=0)
    hom = np.vstack((xyz, np.ones_like(xs)))
    cam = M @ hom
    z_cam = cam[2]
    mask = z_cam > 0
    if np.sum(mask) == 0:
        z = np.zeros_like(xyz[0])
        return z, z, z
    cam_f = cam[:, mask]
    z_f   = z_cam[mask]
    xyz_f = xyz[:, mask]
    xy = (K @ (cam_f[:3] / z_f)).astype(np.float32)
    depths = np.linalg.norm(xyz_f, axis=0)
    return xy[0], xy[1], depths

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
