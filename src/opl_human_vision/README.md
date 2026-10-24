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

The recognizer publishes dynamic `opl_hr_person_<enrolled_id>` TF children in the
measurement's camera optical frame. Only fresh (at most 150 ms), observed, safely
associated enrolled identities with valid measured positions produce updates.
Positions require registered depth, matching RGB/depth/calibration frames and
resolution, and valid intrinsics. No guessed calibration is accepted as a valid
position. The quaternion is a frame convention, not measured body orientation.
On loss or uncertainty, updates stop; TF retains historical samples, so consumers
must check measurement age. No robot motion is commanded by this change.

After rebuilding interfaces and the vision package, source this checkout's
`install/setup.bash` in every launch/consumer terminal and restart the old nodes.
Other overlays can still contain incompatible generated message definitions.

Verification: `colcon test --packages-select opl_human_vision`, then
`python3 -B -m unittest discover -s src/opl_human_vision/test -v`, from this workspace
with ROS and its local install overlay sourced. The recognizer-dropout test runs
the real node's lifecycle and subscriptions with GPU inference disabled; feature
and TF policy tests do not establish live model accuracy or Jetson performance.
