# Human identity tracking

Enrollment is explicit and starts only through the service, for a fresh observed,
unambiguous raw track:

```bash
ros2 service call /face_recognizer_node/enroll_person opl_interfaces/srv/EnrollPerson "{raw_track_id: 1, name: 'furia'}"
```

Keep the face visible while the authorized enrollment collects the configured
face samples. Authorization is cancelled if that trajectory disappears or becomes
ambiguous. No enrolled identity is created just by observing an unknown person.

After enrollment, an existing identity can be recognized automatically from three
consecutive distinct face observations. Body appearance alone cannot acquire or
transfer an enrolled ID. A continuous, unambiguous track retains its confirmed ID
when its face becomes unavailable. Crossings with indistinguishable associations
require face reacquisition; identical clothing cannot resolve complete overlap.

`TrackedHuman` exposes `observed`, `association_ambiguous`, `position_valid`, and
`identity_state`: `UNCONFIRMED`, `FACE_CONFIRMED`, `TRACKED`, or `UNCERTAIN`.
`match_score=0` on `TRACKED` means no fresh confirming face comparison, not an
unknown identity. Detector `confidence` retains its original meaning.

Person TF is disabled at startup. Select an enrolled person using the separate
following service using its enrolled `identity_id`. It must resolve to exactly one
freshly observed track with valid depth:

```bash
ros2 service call /select_following_target opl_interfaces/srv/SelectTarget "{identity_id: 1}"
```

Only that enrolled person's `opl_hr_person_<enrolled_id>` frame receives updates.
Cancel at any time, even without camera input:

```bash
ros2 service call /select_following_target opl_interfaces/srv/SelectTarget "{identity_id: 0}"
```

The selector is started by `test_detector.launch.py`. Selection stays bound to the
enrolled ID; another person cannot take over through a reused raw ID. TF stops on
loss, ambiguity, stale observations, or invalid depth. After loss or ambiguity,
a newer face confirmation of the same enrolled identity is required before updates
resume, including when that identity returns under a new raw ID. A temporary depth
failure alone does not require face reacquisition if tracking remains continuous.
`FollowingTarget.identity_id` identifies the requested person even while `LOST`;
only `valid=true` means its position is currently usable.

TF uses the camera optical frame and original observation timestamp, with a 150 ms
freshness limit. The quaternion is a frame convention, not measured body heading.
TF buffers retain historical samples after cancellation/loss; consumers must check
measurement age and target validity. This package does not command robot motion.

Tracker RGB is buffered for at most 30 frames and matched to the detection stamp
within 100 ms, including out-of-order arrivals. Missing RGB disables camera-motion
and appearance updates while geometric tracking continues. Valid positions require
matched RGB dimensions/frame metadata, registered depth, and valid calibration.
The RGB buffer clears on cleanup or a backward ROS clock jump.

`scrfd_conf_threshold` (default `0.45`) and `scrfd_nms_threshold` (default `0.30`)
are applied to SCRFD during configuration. Values must be finite and in `[0, 1]`;
parameter changes take effect after reconfiguration, not during active inference.

The optional webcam publisher reconnects at one-second intervals after open failure
or three consecutive failed reads (immediately detecting a closed device). It permits
five reconnect attempts, reapplies capture settings, and resets its retry budget only
after a frame is read successfully. Exhaustion requires restarting the node.

After rebuilding interfaces and the vision package, source this checkout's
`install/setup.bash` in every launch/consumer terminal and restart the old nodes.
Other overlays can still contain incompatible generated message definitions.

Verification: `colcon test --packages-select opl_human_vision`, then
`python3 -B -m unittest discover -s src/opl_human_vision/test -v`, from this workspace
with ROS and its local install overlay sourced. The recognizer-dropout test runs
the real node's lifecycle and subscriptions with GPU inference disabled; feature
and TF policy tests do not establish live model accuracy or Jetson performance.
