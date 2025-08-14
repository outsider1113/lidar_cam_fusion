#!/usr/bin/env python3
"""
fusion_node.py  –  LiDAR × Camera fusion
- Publishes fused image overlay (optional visualization)
- Publishes RGB-D depth image (32FC1, meters) using LiDAR Z-depth aligned to camera
- No object detection
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
from ament_index_python.packages import get_package_share_directory
import yaml
from scipy.spatial.transform import Rotation as Rot

# --- Optional CUDA acceleration via PyTorch (auto-disabled if not installed) ---
try:
    import torch
    TORCH_CUDA = torch.cuda.is_available()
except Exception:
    torch = None  # type: ignore
    TORCH_CUDA = False

# QoS for reliable communication
qos_profile_reliable = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10
)

class FusionNode(Node):
    def __init__(self):
        super().__init__('fusion_node')

        # load config
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

        # param file
        camera_config     = config['camera']
        extrinsics_config = config['extrinsics']
        topics_config     = config['topics']
        display_config    = config['display']
        debug_config      = config.get('debug', {'enabled': False})
        depth_cfg         = config.get('depth', {})  # optional

        self.debug        = bool(debug_config.get('enabled', False))

        # coloring for normalization
        self.depth_min = float(depth_cfg.get('min', 0.5))    # meters
        self.depth_max = float(depth_cfg.get('max', 30.0))   # meters

        # intrinsics
        self.K           = np.array(camera_config['K'], dtype=np.float32).reshape(3, 3)
        self.dist_coeffs = np.array(camera_config['dist_coeffs'], dtype=np.float32)
        self.img_size    = camera_config['img_size']  # (H, W) or set on first frame

        # extrinsics
        self.base_R = np.array(extrinsics_config['R'], dtype=np.float32).reshape(3, 3)
        self.base_T = np.array(extrinsics_config['T'], dtype=np.float32).reshape(3, 1)
        self.R = self.base_R.copy()
        self.T = self.base_T.copy()

        # Debug rotation adjustments (pitch,yaw,roll in deg)
        self.debug_r_adjust = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        # Topics
        self.img_topic_name       = topics_config['camera_image']
        self.lidar_topic_name     = topics_config['lidar_points']
        self.fusion_output_topic  = topics_config['fusion_output']
        self.depth_output_topic   = topics_config.get('depth_output', '/fusion/depth')

        # Display flags
        self.show_lidar_projections    = bool(display_config['show_lidar_projections'])
        self.show_fusion_result_opencv = bool(display_config['show_fusion_result_opencv'])

        # Device (only if CUDA + torch available)
        self.device = torch.device('cuda') if TORCH_CUDA else None
        self.update_transformation_matrices()

        # State
        self.overlay        = None        # for color overlay
        self.frame          = None        # latest RGB frame (BGR for cv2)
        self.last_show_time = 0

        # Pre-alloc for depth image (allocated on first frame)
        self.depth_img = None  # 32FC1 (meters), np.inf for invalid internally

        # Bridge
        self.bridge = CvBridge()

        # Subscriptions / publishers
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
        self.fusion_img_pub  = self.create_publisher(Image, self.fusion_output_topic, 10)   # visualization (BGR8)
        self.depth_img_pub   = self.create_publisher(Image, self.depth_output_topic, 10)    # RGB-D depth (32FC1)

        # Optional window + debug trackbars
        if self.show_fusion_result_opencv:
            cv2.namedWindow("Fusion")
            if self.debug:
                self.setup_debug_trackbars()

        # LUT for depth overlay coloring (Turbo)
        lut_src = np.arange(256, dtype=np.uint8).reshape(-1, 1)
        self.depth_lut = cv2.applyColorMap(lut_src, cv2.COLORMAP_TURBO).reshape(256, 3)

        self.get_logger().info("Fusion node ready (publishing fused image and depth).")

    # ---------------- Debug controls ----------------
    def setup_debug_trackbars(self):
        """Trackbars for adjusting T and small Euler rotations in debug mode."""
        # map base_T into [0,10000] for ±5.0 m range at 0.001 steps
        t_x_default = int((self.base_T[0, 0] + 5.0) * 1000)
        t_y_default = int((self.base_T[1, 0] + 5.0) * 1000)
        t_z_default = int((self.base_T[2, 0] + 5.0) * 1000)
        cv2.createTrackbar("T_x (x0.001)", "Fusion", t_x_default, 10000, self.on_trackbar_change)
        cv2.createTrackbar("T_y (x0.001)", "Fusion", t_y_default, 10000, self.on_trackbar_change)
        cv2.createTrackbar("T_z (x0.001)", "Fusion", t_z_default, 10000, self.on_trackbar_change)

        # rotations: -5..+5 deg at 0.01 precision (0..1000 with 500 offset)
        cv2.createTrackbar("R_pitch (x0.01 deg)", "Fusion", 500, 1000, self.on_trackbar_change)
        cv2.createTrackbar("R_yaw (x0.01 deg)",   "Fusion", 500, 1000, self.on_trackbar_change)
        cv2.createTrackbar("R_roll (x0.01 deg)",  "Fusion", 500, 1000, self.on_trackbar_change)
        self.get_logger().info("Debug trackbars ready (T and R).")
        self.on_trackbar_change(0)

    def on_trackbar_change(self, _):
        if not self.debug:
            return
        t_x = (cv2.getTrackbarPos("T_x (x0.001)", "Fusion") / 1000.0) - 5.0
        t_y = (cv2.getTrackbarPos("T_y (x0.001)", "Fusion") / 1000.0) - 5.0
        t_z = (cv2.getTrackbarPos("T_z (x0.001)", "Fusion") / 1000.0) - 5.0
        self.T = np.array([t_x, t_y, t_z], dtype=np.float32).reshape(3, 1)

        r_pitch = (cv2.getTrackbarPos("R_pitch (x0.01 deg)", "Fusion") / 100.0) - 5.0
        r_yaw   = (cv2.getTrackbarPos("R_yaw (x0.01 deg)",   "Fusion") / 100.0) - 5.0
        r_roll  = (cv2.getTrackbarPos("R_roll (x0.01 deg)",  "Fusion") / 100.0) - 5.0
        self.debug_r_adjust = np.array([r_pitch, r_yaw, r_roll], dtype=np.float32)

        self.get_logger().info(f"Adjusted T: [{t_x:.3f}, {t_y:.3f}, {t_z:.3f}]")
        self.get_logger().info(f"Adjusted R (deg): [{r_pitch:.2f}, {r_yaw:.2f}, {r_roll:.2f}]")

        self.update_transformation_matrices()

    def update_transformation_matrices(self):
        """Update 4×4 extrinsic and (optionally) torch tensors."""
        adjust_rot = Rot.from_euler('xyz', self.debug_r_adjust, degrees=True).as_matrix().astype(np.float32)
        self.R = adjust_rot @ self.base_R
        if TORCH_CUDA and self.device is not None:
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

    # ---------------- Callbacks ----------------
    def camera_cb_compressed(self, msg: CompressedImage):
        try:
            self.frame = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
            if self.img_size is None:
                self.img_size = (self.frame.shape[0], self.frame.shape[1])  # (H, W)
                self.overlay = np.zeros((self.img_size[0], self.img_size[1], 3), dtype=np.uint8)
                self.depth_img = np.full(self.img_size, np.inf, dtype=np.float32)
                self.get_logger().info(f"Initialized img_size: {self.img_size}")
        except CvBridgeError as e:
            self.get_logger().error(f"CV Bridge error (compressed): {e}")

    def camera_cb_raw(self, msg: Image):
        try:
            self.frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            if self.img_size is None:
                self.img_size = (self.frame.shape[0], self.frame.shape[1])  # (H, W)
                self.overlay = np.zeros((self.img_size[0], self.img_size[1], 3), dtype=np.uint8)
                self.depth_img = np.full(self.img_size, np.inf, dtype=np.float32)
                self.get_logger().info(f"Initialized img_size: {self.img_size}")
        except CvBridgeError as e:
            self.get_logger().error(f"CV Bridge error (raw): {e}")

    def lidar_cb(self, msg: PointCloud2):
        """Project LiDAR points to the current frame and publish fused image + depth."""
        if self.frame is None or self.img_size is None:
            return

        # Extract points (x,y,z,intensity)
        pts = list(pc2.read_points(msg, field_names=('x', 'y', 'z', 'intensity'), skip_nans=True))
        if not pts:
            return

        xs, ys, zs, intensities = zip(*pts)
        xs = np.asarray(xs, dtype=np.float32)
        ys = np.asarray(ys, dtype=np.float32)
        zs = np.asarray(zs, dtype=np.float32)
        intensities = np.asarray(intensities, dtype=np.float32)

        #  use camera-axis Z as depth
        if TORCH_CUDA and self.device is not None:
            px, py, depth_z, _ = lidar2pixel_cuda(xs, ys, zs, intensities, self.M_t, self.K_t, self.device)
            px_i = px.round().to(dtype=torch.int32).cpu().numpy()
            py_i = py.round().to(dtype=torch.int32).cpu().numpy()
            depth_z = depth_z.cpu().numpy()
        else:
            px_i, py_i, depth_z, _ = lidar2pixel_cpu(xs, ys, zs, intensities, self.M, self.K)

        # Bounds + positive Z filter
        h, w = self.img_size
        mask = (px_i >= 0) & (px_i < w) & (py_i >= 0) & (py_i < h) & (depth_z > 0.0)
        if not np.any(mask):
            return
        px_i = px_i[mask]; py_i = py_i[mask]; depth_z = depth_z[mask]

        # Z buffer
        if (self.depth_img is None) or (self.depth_img.shape != (h, w)):
            self.depth_img = np.full((h, w), np.inf, dtype=np.float32)
        else:
            self.depth_img.fill(np.inf)  # reset per LiDAR frame

        # Z-buffer: keep nearest Z per pixel
        np.minimum.at(self.depth_img, (py_i, px_i), depth_z)

        # optional overlay
        frame = np.array(self.frame, copy=True)
        if self.show_lidar_projections:
            if (self.overlay is None) or (self.overlay.shape[:2] != (h, w)):
                self.overlay = np.zeros((h, w, 3), dtype=np.uint8)
            self.overlay.fill(0)

            # Map depth to [0,255] for coloring (near -> warm)
            z_norm = np.clip((self.depth_img - self.depth_min) / (self.depth_max - self.depth_min), 0.0, 1.0)
            z_uint8 = np.round((1.0 - z_norm) * 255.0).astype(np.uint8)

            # Only color finite pixels
            finite_mask = np.isfinite(self.depth_img)
            # Create per-pixel color from LUT
            color_img = self.depth_lut[z_uint8]    # (H, W, 3) BGR

            # Thicken: draw on neighbors for visible overlay of sparse samples
            self.overlay[finite_mask] = color_img[finite_mask]
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    y_off = np.clip(py_i + dy, 0, h - 1)
                    x_off = np.clip(px_i + dx, 0, w - 1)
                    self.overlay[y_off, x_off] = color_img[y_off, x_off]

            mask_overlay = (self.overlay > 0).any(axis=2)
            frame[mask_overlay] = self.overlay[mask_overlay]

        # Publish depth image (32FC1, meters) 
        depth_to_pub = self.depth_img.copy()
        depth_to_pub[~np.isfinite(depth_to_pub)] = 0.0
        out_depth_msg = self.bridge.cv2_to_imgmsg(depth_to_pub, encoding='32FC1')
        # Use the node clock stamp to keep it monotonic and insync with image
        stamp = self.get_clock().now().to_msg()
        out_depth_msg.header.stamp = stamp
        out_depth_msg.header.frame_id = 'camera_link'
        self.depth_img_pub.publish(out_depth_msg)

        # ublish fused image 
        out_img_msg = self.bridge.cv2_to_imgmsg(frame, 'bgr8')
        out_img_msg.header.stamp = stamp
        out_img_msg.header.frame_id = 'camera_link'
        self.fusion_img_pub.publish(out_img_msg)

# Projection helpers 
def lidar2pixel_cuda(xs, ys, zs, intensities, M_t, K_t, device):
    """Project LiDAR points to image plane using CUDA (PyTorch). Returns (u,v,Z)."""
    xyz   = torch.from_numpy(np.stack([xs, ys, zs])).to(device, dtype=torch.float32)
    intens= torch.tensor(intensities, device=device, dtype=torch.float32)
    ones  = torch.ones((1, xyz.shape[1]), dtype=torch.float32, device=device)
    hom   = torch.cat((xyz, ones), dim=0)
    cam   = M_t @ hom                    # camera frame
    z_cam = cam[2]
    mask  = z_cam > 0
    if mask.sum() == 0:
        return (torch.zeros_like(xyz[0]), torch.zeros_like(xyz[0]),
                torch.zeros_like(xyz[0]), torch.zeros_like(intens))
    cam_f = cam[:, mask]
    z_f   = z_cam[mask]                  # camera-axis Z depth
    xy_pix = K_t @ (cam_f[:3] / z_f)
    return xy_pix[0], xy_pix[1], z_f, intens[mask]

def lidar2pixel_cpu(xs, ys, zs, intensities, M, K):
    """Project LiDAR points to image plane using CPU (NumPy). Returns (u,v,Z)."""
    xyz     = np.vstack((xs, ys, zs))
    hom     = np.vstack((xyz, np.ones_like(xs)))
    cam     = M @ hom                    # camera frame
    z_cam   = cam[2]
    mask    = z_cam > 0
    if mask.sum() == 0:
        zeros = np.zeros_like(xs, dtype=np.float32)
        return zeros, zeros, zeros, np.zeros_like(intensities, dtype=np.float32)
    cam_f   = cam[:, mask]
    z_f     = z_cam[mask]                # camera-axis Z depth
    xy_pix  = (K @ (cam_f[:3] / z_f)).astype(int)
    return xy_pix[0], xy_pix[1], z_f, intensities[mask]


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
