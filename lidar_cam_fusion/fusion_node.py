#!/usr/bin/env python3
"""
fusion_node_updated.py – Simplified LiDAR × Camera fusion with CUDA acceleration

Key features:
* Projects LiDAR points into the camera frame using calibration parameters
  read from a YAML file.
* Colours each valid LiDAR point according to its Euclidean distance: near
  points appear red and far points blue.
* Runs a YOLO model (from Ultralytics) on every third frame to detect
  objects. For each bounding box it computes the nearest LiDAR depth inside
  that box and publishes the depth along with the pixel center.
* Publishes the fused image and a custom `FusionDetections` message.
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

# Enable cuDNN autotune if CUDA exists
try:
    torch.backends.cudnn.benchmark = True
except Exception:
    pass
try:
    TORCH_CUDA = torch.cuda.is_available()
except Exception:
    TORCH_CUDA = False

# Default reliable QoS for image and point cloud topics
qos_profile_reliable = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10
)


class FusionNode(Node):
    """ROS 2 node performing LiDAR–camera fusion with YOLO object detection."""

    def __init__(self):
        super().__init__('fusion_node')

        # ---- Load configuration ----
        cfg_path = os.path.join(
            get_package_share_directory("lidar_cam_fusion"),
            "config", "fusion_config_updated.yaml"
        )
        if not os.path.exists(cfg_path):
            # Fall back to original file name if updated one is absent
            cfg_path = os.path.join(
                get_package_share_directory("lidar_cam_fusion"),
                "config", "fusion_config.yaml"
            )
        try:
            with open(cfg_path, 'r') as f:
                config = yaml.safe_load(f) or {}
        except Exception as e:
            self.get_logger().error(f"Failed loading YAML: {e}")
            raise

        camera_cfg  = config.get('camera', {})
        extr_cfg    = config.get('extrinsics', {})
        topics_cfg  = config.get('topics', {})
        display_cfg = config.get('display', {})

        # ---- Camera intrinsics ----
        # Intrinsic matrix K (3×3) and distortion coefficients
        self.K = np.array(camera_cfg.get('K', []), dtype=np.float32).reshape(3, 3)
        self.dist_coeffs = np.array(camera_cfg.get('dist_coeffs', []), dtype=np.float32)
        # Image size will be set on receipt of first image
        self.img_size = None  # (H, W)

        # ---- Extrinsics (lidar -> camera) ----
        self.base_R = np.array(extr_cfg.get('R', []), dtype=np.float32).reshape(3, 3)
        self.base_T = np.array(extr_cfg.get('T', []), dtype=np.float32).reshape(3, 1)
        # Final rotation/translation used for projection
        self.R = self.base_R.copy()
        self.T = self.base_T.copy()

        # ---- Topics ----
        self.img_topic_name   = topics_cfg.get('camera_image', '/camera/color/image_raw')
        self.lidar_topic_name = topics_cfg.get('lidar_points', '/livox/lidar')
        self.fusion_output_topic = topics_cfg.get('fusion_output', '/fusion/output')
        # Detection topic; default to '/fusion/detections'
        self.detections_topic = topics_cfg.get('detection_output', '/fusion/detections')

        # ---- Display / colour mapping params ----
        # Colour coding uses near→red to far→blue mapping within [range_min, range_max]
        self.range_min = float(display_cfg.get('range_min', 0.3))
        self.range_max = float(display_cfg.get('range_max', 6.0))

        # Preallocate overlay buffer once image size is known
        self.overlay = None  # np.uint8 H×W×3
        self.frame = None    # np.uint8 H×W×3
        self.frame_counter = 0
        self.last_detections = []

        # ---- YOLO model ----
        model_path = os.path.join(
            get_package_share_directory("lidar_cam_fusion"),
            "config", camera_cfg.get('yolo_model', 'sim_box.pt')
        )
        self.yolo_model = YOLO(model_path)
        if TORCH_CUDA:
            self.yolo_model.to('cuda')
            self.get_logger().info("YOLO running on CUDA")
        else:
            self.yolo_model.to('cpu')
            self.get_logger().info("YOLO running on CPU")
        self.device = torch.device('cuda' if TORCH_CUDA else 'cpu')
        # Build transformation matrices for projection
        self.update_transformation_matrices()

        # ---- ROS wiring ----
        self.bridge = CvBridge()

        # Subscribe to camera topic (compressed or raw)
        if self.img_topic_name.endswith('/compressed'):
            self.get_logger().info(f"Subscribing to compressed image topic: {self.img_topic_name}")
            self.image_subscription = self.create_subscription(
                CompressedImage, self.img_topic_name, self.camera_cb_compressed, qos_profile_reliable
            )
        else:
            self.get_logger().info(f"Subscribing to raw image topic: {self.img_topic_name}")
            self.image_subscription = self.create_subscription(
                Image, self.img_topic_name, self.camera_cb_raw, qos_profile_reliable
            )

        # Subscribe to LiDAR points
        self.create_subscription(PointCloud2, self.lidar_topic_name, self.lidar_cb, qos_profile_reliable)
        # Publishers: fused image and detections
        self.fusion_img_pub = self.create_publisher(Image, self.fusion_output_topic, 10)
        self.fusion_detections_pub = self.create_publisher(FusionDetections, self.detections_topic, 10)

    # ---------- Matrix preparation ----------
    def update_transformation_matrices(self):
        """Compute homogeneous transformation and intrinsic tensors for CUDA/CPU."""
        # For this simplified version, no additional rotation adjustments; use base values
        if TORCH_CUDA:
            self.M_t = torch.cat((
                torch.cat((torch.as_tensor(self.R, dtype=torch.float32, device=self.device),
                            torch.as_tensor(self.T, dtype=torch.float32, device=self.device)), dim=1),
                torch.tensor([[0, 0, 0, 1]], dtype=torch.float32, device=self.device)
            ), dim=0)
            self.K_t = torch.as_tensor(self.K, dtype=torch.float32, device=self.device)
        else:
            self.M = np.vstack((np.hstack((self.R, self.T)), [0, 0, 0, 1]))

    # ---------- Camera callbacks ----------
    def camera_cb_compressed(self, msg: CompressedImage):
        try:
            self.frame = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
            if self.img_size is None:
                h, w = self.frame.shape[:2]
                self.img_size = (h, w)
                # Allocate overlay buffer with zeros (same size as frame)
                self.overlay = np.zeros((h, w, 3), dtype=np.uint8)
                self.get_logger().info(f"Initialized image size: {self.img_size}")
        except CvBridgeError as e:
            self.get_logger().error(f"CV Bridge error (compressed): {e}")

    def camera_cb_raw(self, msg: Image):
        try:
            self.frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            if self.img_size is None:
                h, w = self.frame.shape[:2]
                self.img_size = (h, w)
                self.overlay = np.zeros((h, w, 3), dtype=np.uint8)
                self.get_logger().info(f"Initialized image size: {self.img_size}")
        except CvBridgeError as e:
            self.get_logger().error(f"CV Bridge error (raw): {e}")

    # ---------- LiDAR callback ----------
    def lidar_cb(self, msg: PointCloud2):
        """Handle incoming LiDAR point cloud, project points and overlay onto image."""
        # Ensure we have a current frame to fuse with
        if self.frame is None or self.img_size is None:
            return
        # Convert point cloud to list of (x, y, z, intensity) tuples
        pts = list(pc2.read_points(msg, field_names=('x', 'y', 'z', 'intensity'), skip_nans=True))
        if not pts:
            return
        xs, ys, zs, intens = zip(*pts)
        xs = np.asarray(xs, dtype=np.float32)
        ys = np.asarray(ys, dtype=np.float32)
        zs = np.asarray(zs, dtype=np.float32)
        intens = np.asarray(intens, dtype=np.float32)
        # Project to image plane using CUDA or CPU
        if TORCH_CUDA:
            px, py, depths, _ = lidar2pixel_cuda(xs, ys, zs, intens, self.M_t, self.K_t, self.device)
            px_i = px.round().to(torch.int32).cpu().numpy()
            py_i = py.round().to(torch.int32).cpu().numpy()
            depths = depths.cpu().numpy()
        else:
            px_i, py_i, depths, _ = lidar2pixel_cpu(xs, ys, zs, intens, self.M, self.K)
        # Filter valid points: inside image bounds and in front of camera
        h, w = self.img_size
        mask = (px_i >= 0) & (px_i < w) & (py_i >= 0) & (py_i < h) & (depths > 0)
        if not np.any(mask):
            return
        px_i = px_i[mask]
        py_i = py_i[mask]
        depths = depths[mask]
        # Create a fresh overlay (zeros) for this frame and colour each valid point
        self.overlay.fill(0)
        colors = self._depths_to_bgr(depths)
        # Vectorised assignment: assign each point's colour to its pixel location
        self.overlay[py_i, px_i] = colors
        # Prepare a depth image for nearest depth lookup inside bounding boxes
        depth_img = np.full((h, w), np.inf, dtype=np.float32)
        # Use numpy's minimum-at to fill nearest depth per pixel
        np.minimum.at(depth_img, (py_i, px_i), depths)
        # Perform YOLO detection every third frame to reduce load
        self.frame_counter += 1
        if self.frame_counter % 3 == 0:
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=TORCH_CUDA):
                    results = self.yolo_model.predict(source=self.frame, verbose=False)[0]
            # Convert to normalised xyxy (0–1) bounding boxes in numpy
            self.last_detections = results.boxes.cpu().xyxyn.numpy()
        # Draw bounding boxes and compute detection message
        frame = self.frame.copy()
        detections = []
        # For each detection, compute pixel coords and approximate depth
        for xmin, ymin, xmax, ymax in self.last_detections:
            x1 = int(np.clip(xmin * w, 0, w - 1))
            y1 = int(np.clip(ymin * h, 0, h - 1))
            x2 = int(np.clip(xmax * w, 0, w - 1))
            y2 = int(np.clip(ymax * h, 0, h - 1))
            if x2 <= x1 or y2 <= y1:
                continue
            # Calculate nearest depth inside bounding box
            box_depth = np.min(depth_img[y1:y2, x1:x2])
            if not np.isfinite(box_depth):
                continue
            # Draw bounding box on frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"{box_depth:.2f} m", (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            # Create DetectedObject message
            center = Point()
            center.x = (x1 + x2) / 2.0
            center.y = (y1 + y2) / 2.0
            center.z = 0.0
            det = DetectedObject()
            det.depth = float(box_depth)
            det.center = center
            detections.append(det)
        # Blend overlay onto frame: where overlay is non‑zero, use that colour
        mask_nonzero = (self.overlay > 0).any(axis=2)
        frame[mask_nonzero] = self.overlay[mask_nonzero]
        # Publish fused image
        out_img = self.bridge.cv2_to_imgmsg(frame, 'bgr8')
        out_img.header.stamp = self.get_clock().now().to_msg()
        out_img.header.frame_id = 'camera_link'
        self.fusion_img_pub.publish(out_img)
        # Publish detection message
        det_msg = FusionDetections()
        det_msg.header.stamp = out_img.header.stamp
        det_msg.header.frame_id = 'camera_link'
        det_msg.object_detected = len(detections) > 0
        det_msg.detections = detections
        self.fusion_detections_pub.publish(det_msg)

    # ---------- Colour mapping ----------
    def _depths_to_bgr(self, depths: np.ndarray) -> np.ndarray:
        """Map depth values to bright BGR colours for visualisation.

        Near distances within [range_min, range_max] map to red; far distances to blue.
        """
        rng = max(1e-6, (self.range_max - self.range_min))
        t = np.clip((depths - self.range_min) / rng, 0.0, 1.0)  # 0 near .. 1 far
        # Map to OpenCV HSV hue range [0, 120]: 0=red, 60=green, 120=blue
        H = (t * 120.0).astype(np.uint8)
        S = np.full_like(H, 255, dtype=np.uint8)
        V = np.full_like(H, 255, dtype=np.uint8)
        hsv = np.stack([H, S, V], axis=-1).reshape(-1, 1, 3)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).reshape(-1, 3)
        return bgr


# ---------- Projection helpers ----------
def lidar2pixel_cuda(xs, ys, zs, intensities, M_t, K_t, device):
    """Project LiDAR points to image plane using CUDA (Torch).

    Returns pixel x, y, Euclidean depth and intensities for symmetry.
    """
    xyz = torch.from_numpy(np.stack([xs, ys, zs])).to(device, dtype=torch.float32)
    ones = torch.ones((1, xyz.shape[1]), dtype=torch.float32, device=device)
    hom = torch.cat((xyz, ones), dim=0)
    cam = M_t @ hom
    z_cam = cam[2]
    mask = z_cam > 0
    if not torch.any(mask):
        z = torch.zeros_like(xyz[0])
        return z, z, z, z
    cam_f = cam[:, mask]
    z_f = z_cam[mask]
    xyz_f = xyz[:, mask]
    # pixel coordinates (homogeneous division)
    xy_pix = K_t @ (cam_f[:3] / z_f)
    depths = torch.linalg.norm(xyz_f, dim=0)
    return xy_pix[0], xy_pix[1], depths, intensities  # intensities kept for API symmetry

def lidar2pixel_cpu(xs, ys, zs, intensities, M, K):
    """Project LiDAR points to image plane on CPU.

    Returns integer pixel x, y, Euclidean depth and filtered intensities.
    """
    xyz = np.vstack((xs, ys, zs))
    hom = np.vstack((xyz, np.ones_like(xs)))
    cam = M @ hom
    z_cam = cam[2]
    mask = z_cam > 0
    if not np.any(mask):
        z = np.zeros_like(xs, dtype=np.float32)
        return z, z, z, np.zeros_like(xs, dtype=np.float32)
    cam_f = cam[:, mask]
    z_f = z_cam[mask]
    xyz_f = xyz[:, mask]
    xy_pix = (K @ (cam_f[:3] / z_f))
    depths = np.linalg.norm(xyz_f, axis=0)
    # Pixel indices as ints
    return xy_pix[0].astype(int), xy_pix[1].astype(int), depths, intensities[mask]


# ---------- Main entry ----------
def main(args=None):
    rclpy.init(args=args)
    node = FusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
