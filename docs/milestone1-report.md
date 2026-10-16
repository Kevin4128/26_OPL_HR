# Milestone 1 — reliable RGB-D human perception

Scope: `/home/furia/opl_hr` only. Software verification is separate from physical
camera and moving-robot verification. No reference-workspace edits, remote changes,
commits, pushes, navigation changes, motion commands, or Milestone 2 work.

## Root causes and behavior

| Area | Before | After |
| --- | --- | --- |
| RGB tracking | Detections used whichever RGB frame arrived last. | Exact acquisition timestamp and normalized frame ID select the original RGB. Bounded queues tolerate delivery reordering. Missing original RGB expires explicitly. |
| RGB recognition | Nearest RGB within 100 ms could provide an unrelated crop. | Exact RGB matching and bounded pending observations. If original RGB is unavailable, publish tracked output without running recognition on another frame. |
| Depth timing | Only latest depth was retained; a 150 ms difference was accepted. | Select the nearest compatible buffered depth within configurable 20 ms. Reject equal-distance ambiguity, incompatible frames/dimensions, invalid/future/stale stamps, and duplicates. |
| Missing depth | Depth quality and failure were implicit. | A maximum 100 ms pending wait yields 2D output with `position_valid=false`. Reordering cannot extend a successor's deadline. Scheduling/inference can add execution latency; this is not a real-time deadline guarantee. |
| Calibration | First CameraInfo was frozen; missing focal lengths were invented. | Use current RGB-stream calibration, validate dimensions/frame/intrinsics/distortion, and invalidate queued samples on calibration/stream changes. Unchanged calibration is reusable independently of exposure timestamps, including zero-stamped static calibration. |
| Human position | One valid torso pixel sufficed; the bbox centre was projected using a torso depth. | Median depth from central 40% width and 20–60% height; at least 25 pixels and 25% valid coverage by default; range 0.2–10 m; reject excessive 10–90% spread; project the sampled region centre, correcting supported distortion. |
| Robot motion | Only camera-relative points were available. | Keep camera-relative points and add independently stamped fixed-frame points using TF at corrected depth acquisition time. No latest-transform fallback. Missing TF leaves visual processing and valid camera positions available. |

`depth_mad_m` reports median absolute deviation; acceptance uses the 10–90% depth
spread, limited by `max(depth_max_spread_m, median * depth_max_spread_ratio)`.
Defaults are 0.3 m and 0.15. A torso-region median is a surface estimate, not a
person's anatomical centre. Predicted-only track boxes do not produce new 3D.

Buffers default to 60 samples, with a maximum observation age of 1 second.
Waiting uses a steady clock, so a paused ROS clock cannot freeze pending work.
Backward ROS time resets the buffers. Processing and lifecycle transitions share
a mutually exclusive callback group; image receipt remains separate.

## Output contract

The original array header remains the RGB acquisition header. The original
`position_3d` remains camera-relative; it is never silently converted to `odom`.

- `position_valid`: camera point is usable; invalid points retain the legacy zero
  sentinel, which must never be interpreted as a valid position.
- `position_header`: corrected depth acquisition stamp and RGB optical frame.
- `depth_header`: original depth stamp/frame, before a configured clock offset.
- `depth_valid_pixels`, `depth_valid_fraction`, `depth_mad_m`: sampling quality.
- `fixed_position_valid`: fixed-frame point is available.
- `fixed_position`: independently stamped point in configurable `fixed_frame`
  (default `odom`). A false validity flag makes its numeric value unusable,
  including when TF fails.

All publishers/subscribers using `TrackedHuman` must rebuild because the ROS
message schema changed. Recognition carries all spatial fields forward when it
assigns identity/name fields. Existing display and target-state consumers now
check `position_valid`; target-state output still uses camera coordinates and
does not send motion commands.

## Camera and configuration boundary

Both launches retain registered depth: RealSense aligned depth and Gemini depth
registration. The pipeline now uses color CameraInfo for both. `camera_info_topic`
and `depth_topic` are overridable. Registered depth must share RGB pixel geometry,
dimensions, and metric interpretation: `16UC1` millimetres or `32FC1` metres.

`registered_frame_aliases` explicitly declares additional depth/calibration frame
IDs known to express **RGB optical coordinates**. It does not geometrically
register raw depth. Leading slashes are normalized. The launch default `['']`
declares no aliases while satisfying Jazzy's typed nonempty-list requirement.

`depth_timestamp_offset_sec` adds a known offset to depth stamps for matching and
TF; original depth stamps remain in `depth_header`. Zero is the default. RGB and
the corrected depth stamps must use the ROS clock domain; unknown offsets are
never inferred from receipt time. Calibration metadata is not an exposure clock.
Unregistered depth, incompatible geometry, unsupported distortion, and unsupported
cropped/binned calibration yield 2D-only observations. Set
`depth_registered_to_rgb:=false` for an explicitly 2D-only pipeline.

The reference `wrist_perception.py` informed timestamp, calibration and depth
validation only. No wrist-camera code was integrated or modified.

## Modified files

- `src/opl_interfaces/msg/TrackedHuman.msg`: additive spatial fields.
- `src/opl_human_vision/include/opl_human_vision/observation_buffer.hpp` (new):
  shared timestamp-aware buffering and matching.
- `src/opl_human_vision/include/opl_human_vision/human_position.hpp` (new):
  calibration checks, robust torso depth and timestamped TF.
- `src/opl_human_vision/include/opl_human_vision/human_tracker_node.hpp` and
  `src/opl_human_vision/src/human_tracker_node.cpp`: buffering and 3D integration.
- `src/opl_human_vision/include/opl_human_vision/face_recognizer_node.hpp` and
  `src/opl_human_vision/src/face_recognizer_node.cpp`: exact RGB crops, pending
  delivery, spatial-field preservation and validity-aware gate inputs.
- `src/opl_human_vision/src/human_detector_node.cpp`: reject missing/invalid
  observation stamps and frames; continue preserving original RGB headers.
- `src/opl_human_vision/scripts/following_target.py` and `scripts/visualizer.py`:
  honor explicit position validity.
- `src/opl_human_vision/launch/test_detector.launch.py`: color calibration topics
  and configurable timing, registered-frame and depth-quality parameters.
- `src/opl_human_vision/CMakeLists.txt` and `package.xml`: installed tf2
  dependencies, launch dependency declarations, and scoped tests.
- `src/opl_human_vision/test/test_rgbd_observation.cpp` (new): deterministic
  timestamp, depth, calibration, payload and TF tests.
- `src/opl_human_vision/test/test_rgbd_tracker_node.cpp` (new): synthetic ROS
  callback/lifecycle integration with no camera, navigation, or model inference.
- `src/opl_human_vision/test/test_camera_configuration.py` (new): resolve both
  launch configurations and parameter types without starting camera drivers.
- `docs/milestone1-plan.md` and this report.

The three requested `.cpp/.hpp` pairs were inspected. The detector header needs
no change. Tracker/FaceEngine association, recognition thresholds, enrollment,
and recognition models were not changed.

## Verification

Final results on the x86_64 laptop:

- Package-scoped build: **3 packages finished, exit 0** (`opl_interfaces`,
  `opl_gemini335`, `opl_human_vision`); TensorRT/CUDA support compiled.
- Package-scoped tests: **5 tests, 0 errors, 0 failures, 0 skipped**:
  `camera_configuration`, `pose_nms`, `tracker_occlusion`, `rgbd_observation`,
  `rgbd_tracker_node`.
- Python syntax checks: **4 files passed** (launch, both updated consumers, camera
  configuration test). Installed launch argument inspection also passed.
- `git diff --check`: passed.
- Deadline regression: failed before the queue fix, passed after it.
- Exact-RGB mutation: inverting stamp equality caused the expected assertion
  failure; restoring the condition passed.
- Focused independent code review found the deadline issue above; it was fixed
  with the failing/passing regression. The remaining diff was reviewed for scope
  and unintended changes.

Logs are local under `/home/furia/opl_hr/build/milestone1-verification/`:
`milestone1-build.log`, `milestone1-test.log`, `test-results.log`,
`milestone1-regression-red.log`, `milestone1-regression-green.log`, and
`mutation.log`. ROS-generated package test details are in
`build/opl_human_vision/Testing/`.

Commands:

```bash
cd /home/furia/opl_hr
source /opt/ros/jazzy/setup.bash
colcon build --packages-select opl_interfaces opl_gemini335 opl_human_vision \
  --cmake-args -DBUILD_TESTING=ON
source install/setup.bash
colcon test --packages-select opl_interfaces opl_gemini335 opl_human_vision \
  --event-handlers console_direct+
colcon test-result --verbose
git diff --check
```

Tests cover exact RGB matching, delivery reordering, late RGB, duplicate/zero/
malformed/future/stale/out-of-order timestamps, clock resets, queue capacity,
reordered deadlines, absent/ambiguous/wrong-frame/wrong-size depth, known clock
offsets, static/replaced calibration, intrinsics and distortion validation,
16-bit and floating depth, padded big-endian payloads and truncated data, sparse
pixels/coverage, range and foreground/background consistency, zero-invalid 3D,
missing/extrapolated TF, acquisition-time TF under synthetic translation and
rotation, and both camera launch configurations. Existing pose NMS and occlusion
tests remain registered.

The deadline regression was observed failing before the fix and passing after it.
The exact-RGB-match mutation check also failed as expected under mutation and passed after restoration. Software test coverage
does not establish camera alignment, GPU recognition quality, or live robot
performance.

## Physical verification commands — run later

Use the robot's calibrated camera mounting TF and real odometry. Replace
`ACTUAL_RGB_OPTICAL_FRAME` below with the camera's reported optical frame ID.
Run camera launch commands separately, stopping one before starting another.

```bash
cd /home/furia/opl_hr
source /opt/ros/jazzy/setup.bash
source install/setup.bash

# Confirm the chosen camera driver exists.
ros2 pkg prefix orbbec_camera
ros2 pkg prefix realsense2_camera

# Gemini, when its driver and camera are available.
ros2 launch opl_human_vision test_detector.launch.py \
  camera_type:=gemini335 enable_visualizer:=true fixed_frame:=odom

# Or RealSense.
ros2 launch opl_human_vision test_detector.launch.py \
  camera_type:=realsense enable_visualizer:=true fixed_frame:=odom
```

In another sourced terminal, for Gemini:

```bash
ros2 topic echo /camera/color/image_raw --field header --once
ros2 topic echo /camera/depth/image_raw --field header --once
ros2 topic echo /camera/color/camera_info --once
ros2 run tf2_ros tf2_echo odom ACTUAL_RGB_OPTICAL_FRAME
ros2 topic echo /recognized_humans --once
```

For RealSense, use `/camera/camera/color/image_raw`,
`/camera/camera/aligned_depth_to_color/image_raw`, and
`/camera/camera/color/camera_info`. Verify actual stamp domain, depth encoding,
registration, CameraInfo dimensions/intrinsics, optical frame conventions and TF.
Configure aliases only after confirming registration; configure offsets only
after measuring a stable known clock offset.

Observe a stationary person while the robot is translated and rotated by its
existing authorized controls. Camera-relative coordinates should change;
`fixed_position` should stay consistent within sensor/timing error when valid.
Record RGB/depth delta and all validity/quality fields alongside TF if assessing
accuracy. This milestone adds no robot-driving functionality.

To test depth loss with the camera connected, restart the selected pipeline with
the tracker listening to an intentionally absent topic:

```bash
ros2 launch opl_human_vision test_detector.launch.py \
  camera_type:=gemini335 depth_topic:=/milestone1/no_depth enable_visualizer:=true
```

Use `camera_type:=realsense` for RealSense. Expect visual detection/recognition to
continue after the bounded wait, with both position validity flags false.
For TF loss, restart with `fixed_frame:=milestone1_unavailable_frame`: expect
valid camera positions when depth is good and `fixed_position_valid=false`.

## Remaining risks and approval boundary

- Physical cameras, alignment/scaling/frame conventions, live recognition, and
  moving-robot behavior have not been tested. Jetson compilation and performance
  have not been tested; builds here use the x86_64 laptop toolchain.
- The Orbbec driver was unavailable in the checked Jazzy environment. Launch
  parameter resolution does not prove the external driver starts successfully.
- The existing camera-relative recognition spatial gate is unchanged by explicit
  agreement. Robot ego-motion can still affect identity gating even though the
  additive fixed-frame spatial output uses observation-time TF.
- Up to 20 ms RGB/depth exposure difference is allowed by default. Fast camera or
  person motion and real sensor noise can affect positional accuracy; physically
  validate or tighten this bound. Quality defaults are conservative initial
  settings, not hardware-calibrated guarantees. A torso crop dominated by a
  consistent background surface can still pass depth consistency checks.
- Timestamped TF must be available when processing the observation. Missing TF
  marks fixed position unavailable; there is no asynchronous TF retry/backfill.

Stop here for approval. No Milestone 2 work is authorized by this report.
