#!/usr/bin/env python3
"""
fusion_node.py – LiDAR × Camera fusion (CUDA-capable) with improved line-like projections
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
torch.backends.cudnn.benchmark = True
try:
    TORCH_CUDA = torch.cuda.is_available()
except Exception:
    TORCH_CUDA = False

qos_profile_reliable = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10
)

class FusionNode(Node):
    def __init__(self):
        super().__init__('fusion_node')

        # ---- Load configuration ----
        config_path = os.path.join(
            get_package_share_directory("lidar_cam_fusion"),
            "config", "fusion_config.yaml"
        )
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f) or {}
        except Exception as e:
            self.get_logger().error(f"Failed loading YAML: {e}")
            raise

        camera_config    = config['camera']
        extrinsics_config= config['extrinsics']
        topics_config    = config['topics']
        display_config   = config.get('display', {})
        debug_config     = config.get('debug', {'enabled': False})
        self.debug = bool(debug_config.get('enabled', False))

        # ---- Camera intrinsics ----
        self.K = np.array(camera_config['K'], dtype=np.float32).reshape(3, 3)
        self.dist_coeffs = np.array(camera_config['dist_coeffs'], dtype=np.float32)
        self.img_size = None  # (H, W) set on first image

        # ---- Extrinsics (can be adjusted in debug) ----
        self.base_R = np.array(extrinsics_config['R'], dtype=np.float32).reshape(3, 3)
        self.base_T = np.array(extrinsics_config['T'], dtype=np.float32).reshape(3, 1)
        self.R = self.base_R.copy()
        self.T = self.base_T.copy()
        self.debug_r_adjust = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # pitch, yaw, roll in deg

        # ---- Topics ----
        self.img_topic_name     = topics_config['camera_image']
        self.lidar_topic_name   = topics_config['lidar_points']
        self.fusion_output_topic= topics_config['fusion_output']

        # ---- Display / visualization params ----
        self.show_lidar_projections   = bool(display_config.get('show_lidar_projections', True))
        self.show_fusion_result_opencv= bool(display_config.get('show_fusion_result_opencv', True))

        # Indoor-friendly defaults; override via YAML if you like
        self.range_min   = float(display_config.get('range_min', 0.3))   # meters
        self.range_max   = float(display_config.get('range_max', 6.0))   # meters
        splat_size_cfg   = int(display_config.get('splat_size', 5))      # 3,5,7...
        if splat_size_cfg < 1: splat_size_cfg = 1
        if splat_size_cfg % 2 == 0: splat_size_cfg += 1
        self.splat_radius = splat_size_cfg // 2

        self.temporal_accum_frames = int(display_config.get('temporal_accum_frames', 3))
        self.temporal_decay        = float(display_config.get('temporal_decay', 0.6))
        self.trails = None  # float32 H×W×3 for temporal smoothing

        # Precompute neighborhood offsets for fast "splat"
        self._offsets = [(dy, dx)
                         for dy in range(-self.splat_radius, self.splat_radius + 1)
                         for dx in range(-self.splat_radius, self.splat_radius + 1)]

        # ---- YOLO ----
        model_path = os.path.join(
            get_package_share_directory("lidar_cam_fusion"),
            "config", "sim_box.pt"
        )
        self.yolo_model = YOLO(model_path)
        if TORCH_CUDA:
            self.yolo_model.to('cuda')
            self.get_logger().info("YOLO running on CUDA")
        else:
            self.yolo_model.to('cpu')
            self.get_logger().info("YOLO running on CPU")

        self.device = torch.device('cuda' if TORCH_CUDA else 'cpu')
        self.update_transformation_matrices()

        # ---- State ----
        self.overlay = None          # uint8 H×W×3 (current frame)
        self.frame = None            # latest BGR frame
        self.frame_counter = 0
        self.last_detections = []
        self.last_show_time = 0

        # ---- ROS wiring ----
        self.bridge = CvBridge()

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

        if self.show_fusion_result_opencv:
            cv2.namedWindow("Fusion")
            if self.debug:
                self.setup_debug_trackbars()

    # ---------- Debug UI ----------
    def setup_debug_trackbars(self):
        """Trackbars for T and small Euler tweaks."""
        t_x_default = int((self.base_T[0, 0] + 5.0) * 1000)
        t_y_default = int((self.base_T[1, 0] + 5.0) * 1000)
        t_z_default = int((self.base_T[2, 0] + 5.0) * 1000)

        cv2.createTrackbar("T_x (x0.001)", "Fusion", t_x_default, 10000, self.on_trackbar_change)
        cv2.createTrackbar("T_y (x0.001)", "Fusion", t_y_default, 10000, self.on_trackbar_change)
        cv2.createTrackbar("T_z (x0.001)", "Fusion", t_z_default, 10000, self.on_trackbar_change)

        cv2.createTrackbar("R_pitch (x0.01 deg)", "Fusion", 500, 1000, self.on_trackbar_change)
        cv2.createTrackbar("R_yaw (x0.01 deg)", "Fusion", 500, 1000, self.on_trackbar_change)
        cv2.createTrackbar("R_roll (x0.01 deg)", "Fusion", 500, 1000, self.on_trackbar_change)

        self.get_logger().info("Debug trackbars enabled (T and Euler adjustments)")
        self.on_trackbar_change(0)

    def on_trackbar_change(self, _):
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

        self.get_logger().info(f"Adjusted T: [{t_x:.3f}, {t_y:.3f}, {t_z:.3f}]")
        self.get_logger().info(f"Adjusted R (deg): [{r_pitch:.2f}, {r_yaw:.2f}, {r_roll:.2f}]")

        self.update_transformation_matrices()

    # ---------- Math / transforms ----------
    def update_transformation_matrices(self):
        """Build 4×4 [R|T] (CPU) or torch tensors (GPU) with debug rotation tweak."""
        adjust_rot = Rot.from_euler('xyz', self.debug_r_adjust, degrees=True).as_matrix().astype(np.float32)
        self.R = adjust_rot @ self.base_R  # local tweak in camera frame

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

    # ---------- Camera callbacks ----------
    def camera_cb_compressed(self, msg: CompressedImage):
        try:
            self.frame = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
            if self.img_size is None:
                h, w = self.frame.shape[:2]
                self.img_size = (h, w)
                self.overlay = np.zeros((h, w, 3), dtype=np.uint8)
                self.trails  = np.zeros((h, w, 3), dtype=np.float32)
                self.get_logger().info(f"Initialized img_size: {self.img_size}")
        except CvBridgeError as e:
            self.get_logger().error(f"CV Bridge error (compressed): {e}")

    def camera_cb_raw(self, msg: Image):
        try:
            self.frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            if self.img_size is None:
                h, w = self.frame.shape[:2]
                self.img_size = (h, w)
                self.overlay = np.zeros((h, w, 3), dtype=np.uint8)
                self.trails  = np.zeros((h, w, 3), dtype=np.float32)
                self.get_logger().info(f"Initialized img_size: {self.img_size}")
        except CvBridgeError as e:
            self.get_logger().error(f"CV Bridge error (raw): {e}")

    # ---------- LiDAR / fusion ----------
    def lidar_cb(self, msg: PointCloud2):
        if self.frame is None or self.img_size is None:
            return

        # (x,y,z,intensity); intensity optional in color mapping
        pts = list(pc2.read_points(msg, field_names=('x', 'y', 'z', 'intensity'), skip_nans=True))
        if not pts:
            return

        xs, ys, zs, intens = zip(*pts)
        xs = np.asarray(xs, dtype=np.float32)
        ys = np.asarray(ys, dtype=np.float32)
        zs = np.asarray(zs, dtype=np.float32)
        intens = np.asarray(intens, dtype=np.float32)

        # Project to pixels; compute Euclidean distance (range)
        if TORCH_CUDA:
            px, py, depths, _ = lidar2pixel_cuda(xs, ys, zs, intens, self.M_t, self.K_t, self.device)
            px_i = px.round().to(torch.int32).cpu().numpy()
            py_i = py.round().to(torch.int32).cpu().numpy()
            depths = depths.cpu().numpy()
        else:
            px_i, py_i, depths, _ = lidar2pixel_cpu(xs, ys, zs, intens, self.M, self.K)

        # Keep only valid pixels in bounds and in front of camera
        h, w = self.img_size
        mask = (px_i >= 0) & (px_i < w) & (py_i >= 0) & (py_i < h) & (depths > 0)
        if not np.any(mask):
            return
        px_i = px_i[mask]
        py_i = py_i[mask]
        depths = depths[mask]

        # Depth buffer (nearest) for YOLO box depth query
        depth_img = np.full((h, w), np.inf, dtype=np.float32)
        np.minimum.at(depth_img, (py_i, px_i), depths)

        # Run YOLO sparsely for speed
        self.frame_counter += 1
        if self.frame_counter % 3 == 0:
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=TORCH_CUDA):
                    results = self.yolo_model.predict(source=self.frame, verbose=False)[0]
            self.last_detections = results.boxes.cpu().xyxyn.numpy()

        # Draw detections (with nearest depth in bbox)
        frame = np.array(self.frame, copy=True)
        detections = []
        for xmin, ymin, xmax, ymax in self.last_detections:
            x1 = int(np.clip(xmin * w, 0, w - 1))
            y1 = int(np.clip(ymin * h, 0, h - 1))
            x2 = int(np.clip(xmax * w, 0, w - 1))
            y2 = int(np.clip(ymax * h, 0, h - 1))
            if x2 <= x1 or y2 <= y1:
                continue
            box_depth = np.min(depth_img[y1:y2, x1:x2])
            if not np.isfinite(box_depth):
                continue
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"{box_depth:.2f} m", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            center_point = Point()
            center_point.x = (x1 + x2) / 2.0
            center_point.y = (y1 + y2) / 2.0
            center_point.z = 0.0

            det = DetectedObject()
            det.depth = float(box_depth)
            det.center = center_point
            detections.append(det)

        # ----------- Improved line-like LiDAR overlay -----------
        if self.show_lidar_projections:
            # Fresh overlay for this frame
            self.overlay.fill(0)

            # 1) Color by indoor range using HSV (OpenCV H∈[0,179] ≈ degrees/2)
            colors = self._depths_to_bgr(depths)

            # 2) Fast "splat" → draw a (2R+1)×(2R+1) block per point
            #    Small outer loops over offsets; vectorized indexing over all points.
            for dy, dx in self._offsets:
                yy = np.clip(py_i + dy, 0, h - 1)
                xx = np.clip(px_i + dx, 0, w - 1)
                self.overlay[yy, xx] = colors

            # 3) Optional temporal blending for solid strokes
            if self.temporal_accum_frames > 0 and self.trails is not None:
                # Decay history and add current overlay (keep max for crispness)
                self.trails *= self.temporal_decay
                # Use max to preserve bright lines; cheap and effective
                np.maximum(self.trails, self.overlay.astype(np.float32), out=self.trails)
                overlay_to_draw = self.trails.astype(np.uint8)
            else:
                overlay_to_draw = self.overlay

            mask_any = (overlay_to_draw > 0).any(axis=2)
            frame[mask_any] = overlay_to_draw[mask_any]

        # Show and publish
        now = rclpy.clock.Clock().now().nanoseconds
        if self.show_fusion_result_opencv and now - self.last_show_time > 100_000_000:  # ~10 Hz cap
            cv2.imshow("Fusion", frame)
            cv2.waitKey(1)
            self.last_show_time = now

        out_img = self.bridge.cv2_to_imgmsg(frame, 'bgr8')
        out_img.header.stamp = self.get_clock().now().to_msg()
        out_img.header.frame_id = 'camera_link'
        self.fusion_img_pub.publish(out_img)

        det_msg = FusionDetections()
        det_msg.header.stamp = out_img.header.stamp
        det_msg.header.frame_id = 'camera_link'
        det_msg.object_detected = len(detections) > 0
        det_msg.detections = detections
        self.fusion_detections_pub.publish(det_msg)

    # ---------- Color mapping ----------
    def _depths_to_bgr(self, depths: np.ndarray) -> np.ndarray:
        """
        Map metric depth to bright, distinct BGR colors using HSV ramp:
        near→red → yellow → green → cyan → blue, tuned for indoor ranges.
        """
        rng = max(1e-6, (self.range_max - self.range_min))
        t = np.clip((depths - self.range_min) / rng, 0.0, 1.0)  # 0 near .. 1 far

        # OpenCV HSV: H∈[0,179] ~ degrees/2. We'll map 0..120 (red..blue).
        H = (t * 120.0).astype(np.uint8)     # 0=red, 60=green, 120=blue
        S = np.full_like(H, 255, dtype=np.uint8)
        V = np.full_like(H, 255, dtype=np.uint8)

        hsv = np.stack([H, S, V], axis=-1).reshape(-1, 1, 3)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).reshape(-1, 3)
        return bgr

# ---------- Projection helpers ----------
def lidar2pixel_cuda(xs, ys, zs, intensities, M_t, K_t, device):
    """Project LiDAR points to image plane using CUDA; returns pixel coords and Euclidean depth."""
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
    xy_pix = K_t @ (cam_f[:3] / z_f)
    depths = torch.linalg.norm(xyz_f, dim=0)
    return xy_pix[0], xy_pix[1], depths, intensities  # intensities kept for API symmetry

def lidar2pixel_cpu(xs, ys, zs, intensities, M, K):
    """CPU projection; returns pixel coords and Euclidean depth."""
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
    return xy_pix[0].astype(int), xy_pix[1].astype(int), depths, intensities[mask]

# ---------- Main ----------
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
