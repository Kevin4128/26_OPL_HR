# Gemini 335 camera

This package uses the installed `orbbec_camera` driver's
`gemini_330_series.launch.py`. Install `ros-jazzy-orbbec-camera` with the
system package manager if the driver is missing; it is not a Python venv dependency.

After building the workspace and sourcing `install/setup.bash`:

```bash
# Camera only
ros2 launch opl_gemini335 gemini335.launch.py
# Complete pipeline with GUI (Gemini is the default camera)
ros2 launch opl_human_vision test_detector.launch.py camera_type:=gemini335 enable_visualizer:=true
# Previous camera, when connected
ros2 launch opl_human_vision test_detector.launch.py camera_type:=realsense enable_visualizer:=true
```

Default streams are 640x480 at 30 FPS. `image_width`, `image_height`, and
`framerate` configure both streams. The standalone camera launch also accepts
`serial_number` when multiple devices are attached.

Gemini publishes `/camera/color/image_raw` and `/camera/depth/image_raw`.
Depth registration and depth scaling are enabled: depth is `16UC1` in
millimetres in the colour optical frame, with colour intrinsics on
`/camera/depth/camera_info`. Point clouds and infrared streams are disabled.
The pipeline selects camera-specific input topics; output topics and the
enrollment service are unchanged. Only launch one camera owner at a time.
