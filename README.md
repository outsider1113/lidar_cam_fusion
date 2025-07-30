# LiDAR-Camera Fusion Node README

Hey there! Welcome to this ROS2 package for fusing LiDAR points with camera images. It's designed to project LiDAR data onto camera frames, run YOLO for object detection, and add depth info to those detections. The whole thing is optimized for GPU if you've got CUDA.
## Config File: fusion_config.yaml

The `fusion_config.yaml` file is your main setup, it's loaded at runtime to configure the node without hardcoding stuff in the Python script. It's located in the `config/` directory and gets installed to the share path so the node can find it easily. Here's a breakdown of each section and what the options are for:

- **camera**: This handles your camera's internal parameters.
  - `K`: The 3x3 intrinsics matrix (focal lengths, principal point). You need to calibrate your camera beforehand to get this—tools like OpenCV's calibration functions or ROS camera_calibration package work great. No shortcuts here; inaccurate K means wonky projections.
  - `dist_coeffs`: Distortion coefficients (radial and tangential). Again, from calibration. If your camera has no distortion (like in some sims), keep it at zeros.
  - `img_size`: Image dimensions [height, width]. It's set dynamically in code but good to note for reference.

- **extrinsics**: The transformation between LiDAR and camera frames.
  - `R`: 3x3 rotation matrix to align LiDAR to camera coordinates.
  - `T`: 3x1 translation vector (in meters).

- **topics**: ROS2 topic names for input/output.
  - `camera_image`: The camera image topic (raw or compressed).
  - `lidar_points`: The LiDAR PointCloud2 topic.
  - `fusion_output`: Where the fused image gets published.

- **display**: Visualization options.
  - `show_lidar_projections`: If true, overlays projected LiDAR points on the image (colored by depth for cool effect).
  - `show_fusion_result_opencv`: If true, pops up an OpenCV window to show the fused image live.

- **debug**: 
  - `enabled`: If true, enables real-time tweaking of extrinsics using OpenCV trackbars in the "Fusion" window. This is super useful for calibrating on the fly—sliders for translation (T_x, T_y, T_z up to ±5m) and small rotations (pitch, yaw, roll up to ±5°). Adjustments update projections instantly, and values are logged so you can copy them back into the config for permanence. Great for dialing in alignment without restarting the node.

Just a note: The K matrix and dist_coeffs really need to be calibrated properly ahead of time. In a simulation like yours, you can derive K from known FOV and resolution (e.g., focal = width / (2 * tan(FOV/2))), but for real hardware, use a checkerboard pattern.

The YOLO model is specified in the code at `model_path = os.path.join(get_package_share_directory("lidar_cam_fusion"), "config", "sim_box.pt")`—so drop your .pt file in the config directory and name it `sim_box.pt`, or tweak the code if you want a different name.

## How the Code Works: The Math Behind Projections

The core idea is transforming 3D LiDAR points into 2D image pixels so they overlay correctly on the camera frame. This uses extrinsics (R, T) to move points from LiDAR frame to camera frame, then intrinsics (K) to project them onto the image plane.

Here's a simple ASCII visualization of the transformation:

```
LiDAR Point (X_l, Y_l, Z_l)  --[ Homogeneous ]-->  [X_l, Y_l, Z_l, 1]

Apply Extrinsics:
Camera Point = [ R | T ] * [X_l]
                   [Y_l]
                   [Z_l]
                   [ 1 ]

= [X_c, Y_c, Z_c, 1]

Project to Image (normalize by Z_c, apply K):
u = fx * (X_c / Z_c) + cx
v = fy * (Y_c / Z_c) + cy

Where K = | fx  0  cx |
          |  0 fy  cy |
          |  0  0   1 |
```

In code, this happens in `lidar2pixel_cuda` (GPU) or `lidar2pixel_cpu` (CPU): points are transformed via the 4x4 matrix [R|T; 0 1], projected with K, and filtered to image bounds. Depths are norms of original points for coloring.

## The Pipeline: Step by Step

The node runs as a ROS2 subscriber/publisher loop. Here's how it flows:

1. **Initialization**: Loads config, sets up YOLO (on GPU if available), creates subscribers for camera and LiDAR, publishers for fused image and detections, and debug trackbars if enabled.

2. **Camera Callback**: Grabs the image frame (raw or compressed), sets img_size if first time.

3. **LiDAR Callback** (main processing):
   - Extract and project LiDAR points to image pixels (using math above).
   - Filter points to image bounds and positive depth.
   - Build a depth map from projections.
   - Run YOLO every 3 frames to detect boxes (saves compute).
   - For each detection: Calculate min depth in the box area, draw box and depth text on image, compute center pixel (x for left/right relative to image center).
   - Overlay colored LiDAR points if enabled (green close, blue far).
   - Show in OpenCV window if display enabled.
   - Publish fused image as standard Image msg on /fusion/output.

4. **Detections Publishing**: Separately publish metadata on /fusion/detections (FusionDetections msg): header (sync with image), object_detected bool, array of DetectedObject (depth, center Point with x/y pixels, z=0).

5. **Debug Mode**: Trackbars adjust T and rotations live, updating matrices and projections on-the-fly.

The pipeline is efficient—projection on GPU if possible, YOLO throttled—and outputs both visual fused image and structured data for autonomy tasks.

## Custom Topic: /fusion/detections

This topic publishes `FusionDetections` messages, synced with the fused image timestamp. Purpose: To give downstream nodes (e.g., planners) quick access to object metadata without parsing the image. It's lightweight, so low overhead.

Message format (from msg files):
- `std_msgs/Header header`: Timestamp and frame_id ('camera_link') for sync.
- `bool object_detected`: True if at least one valid detection (with depth).
- `DetectedObject[] detections`: Array of objects, each with:
  - `float32 depth`: Min distance (m) from camera to object (from LiDAR in box).
  - `geometry_msgs/Point center`: Pixel center (x, y; z=0). For left/right: compare x to image width/2 (left if <, right if >, assuming camera centered on car).

That's the basics! If you're setting this up in a sim or real hardware, start with debug enabled to tweak alignments.
