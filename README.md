# OPL Human Vision

## Build

```bash
source /opt/ros/jazzy/setup.bash
source ~/orbbec_ws/install/setup.bash  # Gemini 335 driver overlay
cd ~/opl_ros/src/opl_hr
colcon build --symlink-install --packages-up-to opl_human_vision
source install/setup.bash
```

If using the Jazzy apt-installed Orbbec driver, omit the `orbbec_ws` source line.
In each new terminal, source ROS, the camera-driver overlay, and this workspace
before launching.

## Launch

Gemini 335 with the visualizer:

```bash
ros2 launch opl_human_vision test_detector.launch.py camera_type:=gemini335 enable_visualizer:=true
```

RealSense with the visualizer:

```bash
ros2 launch opl_human_vision test_detector.launch.py camera_type:=realsense enable_visualizer:=true
```

Launch only the Gemini camera driver:

```bash
ros2 launch opl_gemini335 gemini335.launch.py
```

## Export TensorRT engines on the target device

Build the engines on the Jetson or laptop that will run them; TensorRT engine
files are tied to the target GPU and TensorRT environment. From the workspace
root, run:

```bash
cd src/opl_human_vision/laptop_models
for model in yolov8n-pose scrfd_2.5g adaface_ir50 osnet_x1_0; do
  trtexec --onnx="${model}.onnx" --saveEngine="${model}.engine"
done
```

This writes the four engine files used by `test_detector.launch.py` beside
their ONNX files. It replaces any existing engine files with the same names.
