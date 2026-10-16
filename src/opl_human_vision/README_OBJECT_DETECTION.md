# Wrist camera object and color detection

Build the interface and vision packages from this workspace, then source its
install overlay:

```bash
cd /home/furia/opl_ros/src/opl_ws
colcon build --packages-up-to opl_human_vision
source install/setup.bash
```

The existing head camera workflow remains available:

```bash
ros2 launch opl_human_vision object_detection.launch.py
```

Launch YOLO11m with the wrist RGB-D streams and publish structured objects:

```bash
ros2 launch opl_human_vision wrist_camera_detection.launch.py
```

The wrist launch reads `/wrist_cam/image`, `/wrist_cam/depth_image`, and
`/wrist_cam/camera_info`, then publishes `/wrist_cam/objects`. Set
`publish_annotated_image:=true` and/or `publish_markers:=true` to enable debug
outputs. Set `enable_visualizer:=true` to open the YOLO box window. `target_frame`
defaults to `base_footprint`; transforms are looked up at the RGB acquisition
time, and a missing transform or invalid synchronized sample produces no object
estimate.

The 3D value is a median-depth pinhole estimate of each detection-box center.
The message contains a stamped point and does not claim an object or grasp
orientation. Color labels use the inner image crop and return red, green, blue,
yellow, or unknown.
