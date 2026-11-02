# OPL Human Vision — Raspberry Pi 5 + Hailo-8L (`Pi_5` branch)

ROS 2 Jazzy pipeline for person detection, tracking with appearance re-ID, and
face/body recognition with enrollment. This branch ports the Jetson/TensorRT
version (`jetson` branch) to a **Raspberry Pi 5 with a Hailo-8L AI accelerator**.
All neural networks run on the Hailo.

```
Orbbec Gemini 335 ──► human_detector_node ──► human_tracker_node ──► face_recognizer_node
  (MJPG 640x480@30)    yolov8s_pose (Hailo)    IoU + camera-motion     SCRFD + ArcFace + re-ID (Hailo)
                       raw_detections          + re-ID recovery (Hailo) recognized_humans, ~/enroll_person
                                               tracked_humans
```

## Target hardware (as measured)

| | |
|---|---|
| Board | Raspberry Pi 5 Model B, 4× Cortex-A76 @ 2.4 GHz, 8 GB, NVMe |
| OS | Ubuntu 24.04, ROS 2 Jazzy |
| Accelerator | Hailo-8L (13 TOPS), M.2 on PCIe Gen2 x1 behind a switch shared with the NVMe |
| Runtime | HailoRT 4.23 (`hailort` + `hailort-pcie-driver` .debs), `hailort.service` |
| Camera | Orbbec Gemini 335, `ros-jazzy-orbbec-camera` 2.9.3 |

## Models

Compiled HEFs for the Hailo-8L come from the Hailo Model Zoo (v2.17.0). The
Jetson models have no Hailo-8L build, so they are replaced:

| Role | Jetson (TensorRT) | Pi 5 (Hailo-8L HEF) | Standalone latency |
|---|---|---|---|
| Person + pose | yolov8n-pose | `yolov8s_pose` | 38 ms |
| Face + 5 landmarks | SCRFD 2.5g | `scrfd_2.5g` | 14 ms |
| Face embedding | AdaFace IR50 | `arcface_mobilefacenet` (default) / `arcface_r50` | 7 ms / 61 ms |
| Body re-ID | OSNet x1_0 | `repvgg_a0_person_reid_512` | 2 ms |
| Objects (optional) | YOLO11m | `yolov8s` (default) / `yolov8m`, on-chip NMS | 47 ms / 69 ms |

ArcFace and RepVGG produce **different embedding spaces** than AdaFace and OSNet:
re-enroll everyone, and re-check the similarity thresholds (see *Calibrating
thresholds*). Galleries are held in memory only, so nothing needs migrating.

To use the original models instead, compile them to HEF with the Hailo Dataflow
Compiler on an x86 Linux PC, then pass the paths as launch arguments.

## Setup

```bash
# 1. Build tools (one-off)
sudo apt install python3-colcon-common-extensions python3-pytest

# 2. HEFs: links the ones hailo-apps already installed under /usr/local/hailo
#    and downloads the rest from the Hailo Model Zoo into models/hailo8l/ (gitignored)
src/opl_human_vision/scripts/fetch_hailo_models.sh                  # required set
src/opl_human_vision/scripts/fetch_hailo_models.sh --with-optional  # + arcface_r50, yolov8m
```

## Build

```bash
deactivate 2>/dev/null || true   # an active venv (e.g. ~/fyp_env) breaks ROS message generation
source /opt/ros/jazzy/setup.bash
cd ~/26_OPL_HR
colcon build --packages-up-to opl_human_vision \
  --cmake-args -DCMAKE_BUILD_TYPE=Release -DPython3_EXECUTABLE=/usr/bin/python3
source install/setup.bash
```

The package links `HailoRT::libhailort` (from `/usr/lib/cmake/HailoRT`) and
builds with `-O3 -mcpu=cortex-a76`. Rebuild after fetching new HEFs so they are
installed into the package share directory.

## Launch

```bash
ros2 launch opl_human_vision pi5_hailo.launch.py                          # Gemini 335
ros2 launch opl_human_vision pi5_hailo.launch.py enable_visualizer:=true
ros2 launch opl_human_vision pi5_hailo.launch.py camera_type:=image \
    test_image:=/usr/local/hailo/resources/images/zidane.jpg              # no camera needed
```

`test_detector.launch.py` still works and forwards to `pi5_hailo.launch.py`.

The Gemini driver and all three vision nodes run in **one**
`component_container_mt` with intra-process communication. Frames are passed
as shared pointers, not serialised over DDS, and every model shares one Hailo
VDevice. The HailoRT round-robin scheduler time-slices that VDevice, with
pose at the highest priority.

Useful arguments (`ros2 launch opl_human_vision pi5_hailo.launch.py --show-args`
lists them all):

| Argument | Default | Notes |
|---|---|---|
| `camera_type` | `gemini335` | `gemini335`, `realsense`, `image`, `none` (bring your own `camera/color/image_raw`) |
| `detector_max_hz` | `20` | Hailo budget knob: `0` = uncapped (~22 Hz), `15` = faster recognition |
| `face_model_path` | `arcface_mobilefacenet.hef` | `arcface_r50.hef` for accuracy (~9× slower per face) |
| `gemini_color_format` | `MJPG` | Lower Pi CPU than `YUYV` (measured 89% vs 98% of one core) |
| `gemini_align_mode` | `HW` | Depth-to-colour alignment on the camera, not the Pi CPU |
| `hailo_multi_process` | `false` | Share the Hailo with other processes via `hailort.service` |
| `timing_debug` | `false` | Per-stage wall-time logs (`[timing] ...` on logger `vision_timing`) |

### Enrollment

```bash
ros2 service call /face_recognizer_node/enroll_person opl_interfaces/srv/EnrollPerson \
  "{raw_track_id: 0, name: 'Alice'}"     # 0 = first active track
```

The person must stay in view with their face visible for `min_confirm_frames`
(15) face embeddings and at least `min_confirm_duration_sec` (2 s).

## Performance (measured on this Pi 5 + Hailo-8L)

Two people continuously in view (`camera_type:=image`, `timing_debug:=true`):

| `detector_max_hz` | Detector | Tracker | Recognizer callbacks | Face pass per person* |
|---|---|---|---|---|
| 0 (uncapped) | 22.5 Hz | 21.1 Hz | 11.0 Hz | 156 ms |
| **20 (default)** | **20.0 Hz** | **19.3 Hz** | **12.8 Hz** | **125 ms** |
| 15 | 15.0 Hz | 15.1 Hz | 13.2 Hz | 78 ms |

\* SCRFD + re-ID + ArcFace for one person. Locked identities are only
re-validated every `identity_validation_interval_sec` (0.5 s).

* CPU, vision container without camera, 2 people: ~47% of one core. With the
  Gemini driver in the container and no people: ~89% of one core. Most of that
  is the driver's MJPEG decode and depth handling.
* With the camera and nobody in view, the whole chain runs at ~28 Hz.
* SoC temperature stayed around 48–55 °C, with no throttling (fan forced on in
  `config.txt`).
* The Hailo-8L is the bottleneck. yolov8s_pose takes ~38 ms per frame, so at
  20 Hz it uses ~75% of the chip. Each network-group switch also costs time:
  re-ID takes 2 ms on its own but 20–50 ms when shared. The tracker and the
  recognizer therefore share one configured re-ID network.

Optional, for steadier latency: `sudo cpupower frequency-set -g performance`.
The default governor is `ondemand`.

## Sharing the Pi with other Hailo / camera users

* **Hailo:** without `hailo_multi_process:=true`, whichever process opens
  `/dev/hailo0` first owns it, and the other fails with
  `HAILO_OUT_OF_PHYSICAL_DEVICES`. To run alongside another Hailo app (e.g.
  face-agent's `track_human.py`), both sides must open the device through
  `hailort.service` (multi-process mode).
* **Camera:** only one process can stream the Gemini 335. Stop other camera
  launches (e.g. face-agent's `start_stack.py`, which starts it as
  `head_camera`). Alternatively, use `camera_type:=none` and remap to their
  topics.

## Calibrating thresholds

The defaults are provisional for the new embedding spaces:

| Threshold | Default |
|---|---|
| `face_match_threshold` | 0.45 |
| `body_match_threshold` | 0.70 |
| `appearance_match_threshold` | 0.70 |

Values observed on the sample images:

* Same face, mirrored: 0.80–0.95.
* Different people's faces: about −0.13 to 0.0.
* Different people's bodies (RepVGG): 0.50–0.65.

So the body thresholds should stay well above 0.65. To calibrate on your own
data:

1. Enroll a person.
2. Walk out of view and back in, and watch `match_score` in `recognized_humans`.
3. Repeat with two people crossing paths.

## Tests

```bash
colcon test --packages-select opl_human_vision && colcon test-result --verbose
```

| Test | Hardware | Checks |
|---|---|---|
| `object_perception`, `pose_nms`, `tracker_occlusion` | none | Original logic tests (the object test now targets Hailo NMS output) |
| `hailo_decoders` | none | Dequantisation, YOLOv8-pose DFL decode, SCRFD decode, letterbox, using synthetic tensors |
| `hailo_models` | Hailo | Loads every HEF and runs `zidane.jpg` through pose → SCRFD → ArcFace / re-ID. Skips if the device is absent or busy |

## Object detection (optional)

```bash
ros2 launch opl_human_vision object_detection_sim.launch.py   # yolov8s.hef, 10 Hz
```

This uses the HEF's on-chip NMS (Hailo NMS-by-class output).

## Jetson / TensorRT

The TensorRT version lives on the `jetson` branch.
