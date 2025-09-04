#!/usr/bin/env python3
"""
fusion_node_ptp_sync.py – LiDAR × Camera fusion with PTP-timestamp pairing and CUDA acceleration.

Changes vs. your last version:
- Use message_filters.ApproximateTimeSynchronizer to PAIR Image+PointCloud2 by header.stamp (PTP-synced).
- Run fusion per paired callback; YOLO still runs every 3rd pair to save compute.
- Publish fused outputs stamped with the image header time (PTP domain).
- No debug UI, no temporal trails, no point "splat": one pixel per point, color by depth.

Dependencies: rclpy, message_filters, cv_bridge, ultralytics, torch, numpy, OpenCV, sensor_msgs_py.
"""

import os
import cv2
import yaml
import torch
import numpy as np
import rclpy
import sensor_msgs_py.point_cloud2 as pc2

from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge, CvBridgeError
from ultralytics import YOLO
from ament_index_python.packages import get_package_share_directory
from scipy.spatial.transform import Rotation as Rot  # kept if you later want to tweak extrinsics
from geometry_msgs.msg import Point
from lidar_cam_fusion.msg import FusionDetections, DetectedObject

from message_filters import Subscriber, ApproximateTimeSynchronizer

# cuDNN autotune (safe to skip if not available)
try:
    torch.backends.cudnn.benchmark = True
except Exception:
    pass

try:
    TORCH_CUDA = torch.cuda.is_available()
except Exception:
    TORCH_CUDA = False


class FusionNode(Node):
    def __init__(self):
        super().__init__('fusion_node')

        # ---------- Load configuration ----------
        cfg_path = os.path.join(
            get_package_share_directory("lidar_cam_fusion"),
            "config", "fusion_config_updated.yaml"
        )
        if not os.path.exists(cfg_path):
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
        sync_cfg    = config.get('sync', {'slop_ms': 100, 'queue': 30})

        # ---------- Camera intrinsics ----------
        self.K = np.array(camera_cfg['K'], dtype=np.float32).reshape(3, 3)
        self.dist_coeffs = np.array(camera_cfg.get('dist_coeffs', []), dtype=np.float32)
        self.img_size = None  # set on first image

        # ---------- Extrinsics (LiDAR -> camera) ----------
        self.base_R = np.array(extr_cfg['R'], dtype=np.float32).reshape(3, 3)
        self.base_T = np.array(extr_cfg['T'], dtype=np.float32).reshape(3, 1)
        self.R = self.base_R.copy()
        self.T = self.base_T.copy()

        # ---------- Topics ----------
        self.img_topic   = topics_cfg.get('camera_image', '/camera/color/image_raw')
        self.lidar_topic = topics_cfg.get('lidar_points', '/livox/lidar')
        self.fused_topic = topics_cfg.get('fusion_output', '/fusion/output')
        self.det_topic   = topics_cfg.get('detection_output', '/fusion/detections')

        # ---------- Display / colour coding ----------
        self.range_min = float(display_cfg.get('range_min', 0.3))
        self.range_max = float(display_cfg.get('range_max', 6.0))

        # ---------- YOLO ----------
        model_path = os.path.join(
            get_package_share_directory("lidar_cam_fusion"),
            "config", camera_cfg.get('yolo_model', 'sim_box.pt')
        )
        self.yolo_model = YOLO(model_path)
        if TORCH_CUDA:
            self.yolo_model.to('cuda')
            self.get_logger().info("YOLO on CUDA")
        else:
            self.yolo_model.to('cpu')
            self.get_logger().info("YOLO on CPU")
        self.device = torch.device('cuda' if TORCH_CUDA else 'cpu')

        # ---------- Build projection matrices ----------
        self._update_mats()

        # ---------- ROS I/O ----------
        self.bridge = CvBridge()

        # message_filters subscribers (SensorData QoS)
        self.sub_img  = Subscriber(self, Image,      self.img_topic,  qos_profile=qos_profile_sensor_data)
        self.sub_lidar= Subscriber(self, PointCloud2, self.lidar_topic, qos_profile=qos_profile_sensor_data)

        slop_sec = float(sync_cfg.get('slop_ms', 100)) / 1000.0
        queue_sz = int(sync_cfg.get('queue', 30))

        self.sync = ApproximateTimeSynchronizer([self.sub_img, self.sub_lidar], queue_size=queue_sz, slop=slop_sec)
        self.sync.registerCallback(self._paired_cb)

        self.pub_fused = self.create_publisher(Image, self.fused_topic, 10)
        self.pub_det   = self.create_publisher(FusionDetections, self.det_topic, 10)

        # state
        self.frame_counter = 0
        self.last_detections = []

        self.get_logger().info(
            f"Fusion (PTP-paired) ready:\n"
            f"  img={self.img_topic}\n  lidar={self.lidar_topic}\n"
            f"  slop={slop_sec:.3f}s queue={queue_sz}\n"
            f"  CUDA={TORCH_CUDA}\n"
        )

    # ---------- Matrices ----------
    def _update_mats(self):
        if TORCH_CUDA:
            self.M_t = torch.cat((
                torch.cat((
                    torch.as_tensor(self.R, dtype=torch.float32, device=self.device),
                    torch.as_tensor(self.T, dtype=torch.float32, device=self.device)), dim=1),
                torch.tensor([[0, 0, 0, 1]], dtype=torch.float32, device=self.device)
            ), dim=0)
            self.K_t = torch.as_tensor(self.K, dtype=torch.float32, device=self.device)
        else:
            self.M = np.vstack((np.hstack((self.R, self.T)), [0, 0, 0, 1]))

    # ---------- Paired callback (Image + PointCloud2) ----------
    def _paired_cb(self, img_msg: Image, lidar_msg: PointCloud2):
        # Decode image
        try:
            frame = self.bridge.imgmsg_to_cv2(img_msg, 'bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f"cv_bridge: {e}")
            return

        if self.img_size is None:
            h, w = frame.shape[:2]
            self.img_size = (h, w)
            self.get_logger().info(f"Initialized image size: {self.img_size}")

        # Read LiDAR points (x,y,z,intensity)
        pts = list(pc2.read_points(lidar_msg, field_names=('x', 'y', 'z', 'intensity'), skip_nans=True))
        if not pts:
            return
        xs, ys, zs, intens = zip(*pts)
        xs = np.asarray(xs, dtype=np.float32)
        ys = np.asarray(ys, dtype=np.float32)
        zs = np.asarray(zs, dtype=np.float32)
        intens = np.asarray(intens, dtype=np.float32)

        # Project to pixels & compute Euclidean depth
        if TORCH_CUDA:
            px, py, depths, _ = lidar2pixel_cuda(xs, ys, zs, intens, self.M_t, self.K_t, self.device)
            px_i = px.round().to(torch.int32).cpu().numpy()
            py_i = py.round().to(torch.int32).cpu().numpy()
            depths = depths.cpu().numpy()
        else:
            px_i, py_i, depths, _ = lidar2pixel_cpu(xs, ys, zs, intens, self.M, self.K)

        h, w = self.img_size
        mask = (px_i >= 0) & (px_i < w) & (py_i >= 0) & (py_i < h) & (depths > 0)
        if not np.any(mask):
            return
        px_i = px_i[mask]
        py_i = py_i[mask]
        depths = depths[mask]

        # Build depth image for bbox queries (nearest depth per pixel)
        depth_img = np.full((h, w), np.inf, dtype=np.float32)
        np.minimum.at(depth_img, (py_i, px_i), depths)

        # YOLO every 3rd paired frame
        self.frame_counter += 1
        if self.frame_counter % 3 == 0:
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=TORCH_CUDA):
                    results = self.yolo_model.predict(source=frame, verbose=False)[0]
            self.last_detections = results.boxes.cpu().xyxyn.numpy()

        # Draw detections & build Detection message
        out = frame.copy()

        detections = []
        for xmin, ymin, xmax, ymax in self.last_detections:
            x1 = int(np.clip(xmin * w, 0, w - 1))
            y1 = int(np.clip(ymin * h, 0, h - 1))
            x2 = int(np.clip(xmax * w, 0, w - 1))
            y2 = int(np.clip(ymax * h, 0, h - 1))
            if x2 <= x1 or y2 <= y1:
                continue

            roi = depth_img[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            box_depth = np.min(roi)
            if not np.isfinite(box_depth):
                continue

            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(out, f"{box_depth:.2f} m", (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            c = Point()
            c.x = (x1 + x2) / 2.0
            c.y = (y1 + y2) / 2.0
            c.z = 0.0

            det = DetectedObject()
            det.depth = float(box_depth)
            det.center = c
            detections.append(det)

        # Colorize LiDAR points (near bright → far darker). One pixel per point (no splat/trails).
        overlay = np.zeros_like(out)
        colors = self._depths_to_bgr(depths)
        overlay[py_i, px_i] = colors
        mask_nonzero = (overlay > 0).any(axis=2)
        out[mask_nonzero] = overlay[mask_nonzero]

        # Publish fused image stamped with the *camera* time (PTP domain)
        img_out = self.bridge.cv2_to_imgmsg(out, 'bgr8')
        img_out.header.stamp = img_msg.header.stamp
        img_out.header.frame_id = 'camera_link'
        self.pub_fused.publish(img_out)

        # Publish detections with same stamp
        det_msg = FusionDetections()
        det_msg.header.stamp = img_out.header.stamp
        det_msg.header.frame_id = 'camera_link'
        det_msg.object_detected = len(detections) > 0
        det_msg.detections = detections
        self.pub_det.publish(det_msg)

    # ---------- Colour mapping ----------
    def _depths_to_bgr(self, depths: np.ndarray) -> np.ndarray:
        rng = max(1e-6, (self.range_max - self.range_min))
        t = np.clip((depths - self.range_min) / rng, 0.0, 1.0)
        # HSV: 0 (red) .. 120 (blue); full S,V
        H = (t * 120.0).astype(np.uint8)
        S = np.full_like(H, 255, dtype=np.uint8)
        V = np.full_like(H, 255, dtype=np.uint8)
        hsv = np.stack([H, S, V], axis=-1).reshape(-1, 1, 3)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).reshape(-1, 3)
        return bgr


# ---------- Projection helpers ----------
def lidar2pixel_cuda(xs, ys, zs, intensities, M_t, K_t, device):
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
    return xy_pix[0], xy_pix[1], depths, intensities

def lidar2pixel_cpu(xs, ys, zs, intensities, M, K):
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
