# Milestone 2 — tracking, identity and enrollment

Work is limited to `/home/furia/opl_hr`. Existing Milestone 1 changes are retained.
No models, object detector logic, navigation, wrist code, remotes, or commits were
changed. No push was made. Milestone 3 requires the user's approval.

## Causes and resulting behavior

| Area | Previous behavior | Implemented behavior |
| --- | --- | --- |
| Association | Primarily IoU assignment could accept an ambiguous crossing before appearance recovery ran. | Preserve Hungarian, IoU, velocity, GMC and OSNet; check ambiguous row/column candidates using appearance and a rival margin. Refuse forced transfers when evidence is indistinguishable. |
| Time | Frame counts and short prediction clamps made retention/motion depend on processing rate. | Expire stale tracks before assignment using observation time, default 2 s; prediction and smoothing use elapsed time. Duplicate time is rejected; backwards clocks and stream changes reset temporal state. Legacy frame limits remain fallback for invalid timestamps. |
| Camera motion | A weak image-motion fit could shift tracks and appearance recovery could lack valid world evidence. | Vet GMC inliers and plausible affine motion; transform box corners, velocities and previous observed centres. M1 depth/TF enrichment happens before association. Compare spatial observations only when valid and expressed in the same fixed frame at their observation times. |
| Gallery | A bad association could become an appearance template. | Skip overlapping crops and reject incompatible templates; preserve existing gallery diversity and OSNet extraction. |
| Registered recognition | Relaxed locked-identity face/body comparisons and cached labels could retain a wrong identity. | Keep nominal face similarity 0.55, require current good face evidence, a 0.08 rival margin, one visible owner, and compatible valid fixed-frame evidence. Clothing alone cannot assign a registered name. Uncertain frames report unknown, then can recover. No relaxed locked thresholds. |
| Face quality | Geometric, lighting and sharpness checks were insufficient. | Check five finite in-crop landmarks, eye separation, frontal pose, aligned-image sharpness and exposure. Live evidence showed the initial new SCRFD confidence gate of 0.8 rejected all samples (scores approximately 0.5–0.76); use 0.65, still above SCRFD's existing 0.50 proposal gate. This is face-detection confidence, not identity similarity. |
| Enrollment | Request authorization, target choice, samples and final persistence were insufficiently separated. | Explicit operator request snapshots fresh observed targets, ranks person confidence, then scans one frozen raw target at a time. Require 20 consistent samples and at least 2.5 s, distinct sample times >=0.1 s. Reject bad/overlapping crops and inconsistent embeddings. Target missing >=2 s or session deadline 30 s fails, discards partial data and advances. |
| Repeated requests | Names or identities could be overwritten or duplicated. | Existing face returns its existing identity, conflicting names fail, generated names avoid custom-name collisions, replacement requires explicit deletion. Newcomers require another request. |
| Persistence | Runtime galleries disappeared on restart. | Store only explicitly enrolled face templates and labels with version/encoder metadata in workspace `.private/identities/`. Body galleries remain temporary. No face images are saved. |
| Ownership | Temporary/raw and registered IDs could overlap, particularly during scanning or RGB failure. | Registered IDs are below 1 billion; anonymous/scanning IDs start at 1 billion; unavailable identity is 0. `raw_track_id` always remains the tracker selector. Registered claims retire old anonymous sessions. |

Unknown anonymous re-identification also uses valid fixed-frame gates. Missing
TF/depth falls back to visual evidence; no historical camera-relative point is
compared as though it were in a stationary frame. Predicted-only boxes cannot
supply new facial enrollment samples or valid new 3D observations.

Registered templates are frozen after explicit enrollment. Conservative unknown
output can reduce recognition availability, especially at distance, while turning
away, at image edges, during overlap or when lighting is poor. The minimum eye
separation is 12 source pixels; the live enrollment was conducted near 1 m.

## Local biometric storage

Default directory: `/home/furia/opl_hr/.private/identities/`.
The directory is owned by the running account with mode 0700; the identity file
and writer lock are owner-only 0600. The node refuses paths outside the workspace
`.private` tree. It refuses symlink path components/files, unexpected ownership or
permissions, malformed/oversized data, invalid vectors and mismatched encoder
fingerprints. An exclusive writer lock prevents concurrent writers. File writes
are atomic, with file and directory flushes; a post-rename directory flush error
blocks further mutations until restart reloads the uncertain commit.

`.private/` is Git-ignored. Access control is ordinary Linux account/file access,
not encryption or authentication of the ROS graph. Enrollment and deletion
services assume the agreed trusted ROS graph. A different AdaFace TensorRT
engine fingerprint requires an explicit migration/re-enrollment decision; an
engine rebuild on another GPU may therefore reject this store. Do not silently
edit its model metadata to bypass compatibility checks.

## Modified files

Milestone 2 changes:

- Storage exclusion: [.gitignore](/home/furia/opl_hr/.gitignore).
- Tracker association and pre-assignment geometry: [tracker_engine.hpp](/home/furia/opl_hr/src/opl_human_vision/include/opl_human_vision/tracker_engine.hpp), [tracker_engine.cpp](/home/furia/opl_hr/src/opl_human_vision/src/tracker_engine.cpp), [human_tracker_node.hpp](/home/furia/opl_hr/src/opl_human_vision/include/opl_human_vision/human_tracker_node.hpp), [human_tracker_node.cpp](/home/furia/opl_hr/src/opl_human_vision/src/human_tracker_node.cpp).
- Face recognition and enrollment integration: [face_engine.hpp](/home/furia/opl_hr/src/opl_human_vision/include/opl_human_vision/face_engine.hpp), [face_engine.cpp](/home/furia/opl_hr/src/opl_human_vision/src/face_engine.cpp), [face_recognizer_node.hpp](/home/furia/opl_hr/src/opl_human_vision/include/opl_human_vision/face_recognizer_node.hpp), [face_recognizer_node.cpp](/home/furia/opl_hr/src/opl_human_vision/src/face_recognizer_node.cpp).
- New storage and enrollment state: [identity_store.hpp](/home/furia/opl_hr/src/opl_human_vision/include/opl_human_vision/identity_store.hpp), [identity_store.cpp](/home/furia/opl_hr/src/opl_human_vision/src/identity_store.cpp), [enrollment_queue.hpp](/home/furia/opl_hr/src/opl_human_vision/include/opl_human_vision/enrollment_queue.hpp).
- Stream epoch for temporal/calibration resets: [observation_buffer.hpp](/home/furia/opl_hr/src/opl_human_vision/include/opl_human_vision/observation_buffer.hpp).
- Interfaces; retain M1 spatial fields and add observed flag: [TrackedHuman.msg](/home/furia/opl_hr/src/opl_interfaces/msg/TrackedHuman.msg), [EnrollPerson.srv](/home/furia/opl_hr/src/opl_interfaces/srv/EnrollPerson.srv), [EnrollmentStatus.msg](/home/furia/opl_hr/src/opl_interfaces/msg/EnrollmentStatus.msg), [DeleteIdentity.srv](/home/furia/opl_hr/src/opl_interfaces/srv/DeleteIdentity.srv), [CMakeLists.txt](/home/furia/opl_hr/src/opl_interfaces/CMakeLists.txt).
- Build and elapsed-time configuration: [CMakeLists.txt](/home/furia/opl_hr/src/opl_human_vision/CMakeLists.txt), [package.xml](/home/furia/opl_hr/src/opl_human_vision/package.xml), [test_detector.launch.py](/home/furia/opl_hr/src/opl_human_vision/launch/test_detector.launch.py).
- New deterministic tests: [test_identity_policy.cpp](/home/furia/opl_hr/src/opl_human_vision/test/test_identity_policy.cpp), [test_tracking_sequences.cpp](/home/furia/opl_hr/src/opl_human_vision/test/test_tracking_sequences.cpp), [test_identity_regressions.cpp](/home/furia/opl_hr/src/opl_human_vision/test/test_identity_regressions.cpp), [test_identity_commit_failure.cpp](/home/furia/opl_hr/src/opl_human_vision/test/test_identity_commit_failure.cpp), [test_directory_sync_failure.cpp](/home/furia/opl_hr/src/opl_human_vision/test/test_directory_sync_failure.cpp).
- Plan and results: [milestone2-plan.md](/home/furia/opl_hr/docs/milestone2-plan.md), [milestone2-report.md](/home/furia/opl_hr/docs/milestone2-report.md).

M1-only detector, depth helpers, following-target validity and visualizer changes
remain present in the working tree. See `docs/milestone1-report.md` for that file
list. Rebuild all consumers of the changed ROS interfaces.

## Deterministic verification

Package-scoped build: `opl_interfaces`, `opl_human_vision`, `opl_gemini335`.
All 11 registered tests passed. Evidence is under
`build/milestone2-verification/{build.log,tests.log}`. A final package-scoped
`colcon test` reported 11 tests, zero errors/failures/skips; `colcon test-result`
confirmed those results. The restored build libraries also passed all 11 tests
in `tests-final-build-library.log`. `git diff --check` passed.

| Test | Coverage |
| --- | --- |
| identity_policy | Quality rejection, distinct scans, duration, known/unknown recognition, nominal-threshold rejection, occupied identity, temporary failure/recovery, restart, deletion, model/permission/symlink/corruption checks, queue ordering and deadlines. |
| tracking_sequences | Synthetic two-person crossing; partial/full occlusion; return and expiry at 5/15/30 processed Hz; duplicate/backwards timestamps; similar clothing ambiguity; actual image GMC on a synthetic rotated textured background. |
| identity_namespace_regression | Scanning IDs cannot collide with registered/raw IDs. |
| identity_geometry_regression | Anonymous recovery refuses impossible valid fixed-frame displacement. |
| identity_names_regression | A custom `Person_2` cannot block generated-name enrollment. |
| identity_commit_failure | Inject a post-rename directory fsync error; later writes block and restart reloads the committed file. |
| camera_configuration | RealSense/Gemini registered depth, color calibration and launch parameter types. |
| pose_nms | Existing detector postprocessing regression test. |
| tracker_occlusion | Existing velocity, occlusion and gallery behavior. |
| rgbd_observation | M1 timestamp errors, buffering, calibration, depth validity and timestamp-correct TF/failures. |
| rgbd_tracker_node | Synthetic ROS delivery ordering, 2D-only tracking, calibrated 3D and TF failure. |

The four review regressions failed before their fixes and passed afterwards.
A final review also preserved enrollment timeouts/session numbering when a
stream reset cancels all queued targets; cancellation and independent engine
ownership checks were added to `identity_policy`. That reset path was checked
synthetically after the live captures; the ordinary enrollment/recognition paths
used in the captures were unchanged.

A deliberate mutation preventing all registered matches made `identity_policy`
fail; the source was restored. The mutation run explicitly preferred the build
library, avoiding the sourced installation's library precedence.

Synthetic metrics: crossing ID changes 0 in the fixture; partial/full occlusion
recovery 2/2; known face matches 20/20 and unknown false registered matches 0/20.
These axis-vector/image fixtures verify decisions, not real-world error rates.
Similar-clothing ambiguity deliberately fragments tracks instead of guessing.

## Physical verification

Hardware: connected RealSense D455, laptop RTX 4060. The camera kept its existing
640x480/30 Hz settings. Isolated ROS domain 72 avoided unrelated robot processes;
software tests used domain 73. No odom transform was available on this isolated
laptop, so fixed positions were explicitly unavailable throughout the real tests.
No invented odom transform was used.

M1 recheck: 313/337 camera positions valid in the initial ten-second check, with
zero selected RGB/depth time difference for valid observations. M2's first stable
single-person capture had 891/893 valid camera positions and one raw tracker ID.
Probe startup/teardown can omit image stamps from the probe's own inventory; those
edges do not establish an association error. Internal matching is independently
covered by deterministic tests.

| Live check | Observed result |
| --- | --- |
| Two-person sequential enrollment | Snapshot selected raw 12 before raw 15 (nearby person confidences 0.8896 and 0.8622). Raw 12 failed after 17 inconsistent samples; partial data was discarded and the queue advanced. Raw 15 completed 20 samples as Person_1. A later explicit retry completed raw 12 as Person_2. |
| Known plus unknown, standing apart | Raw 12 remained unknown in 296/296 observations. Enrolled raw 15 was Person_1 in 231/296 observations; the other 65 were unknown. No duplicate nonzero IDs. |
| Unknown alone | After the initial movement interval, 211 single-person observations had zero registered-name matches. This is a small instructed test, not a population false-match estimate. |
| Crossing, brief occlusion, leaving and returning | 1,188 recognized arrays and 1,759 person observations in 40 s. Person_1 appeared 505 times, Person_2 550, unknown 704. Both names appeared again during the final 5 s; no duplicate nonzero IDs. **13 distinct raw tracker IDs appeared**, showing fragmentation. The metadata has no frame-by-frame person ground truth, so this is not an ID-switch count or proof that no identities ever switched. |
| Application restart | Without enrolling again, the new process emitted Person_1 368 times and Person_2 141 times in 622 observations; the remaining 113 were unknown. No duplicate nonzero IDs. |
| Depth withheld from tracker; TF also absent | 444 person observations: 397 Person_1, 47 unknown; all 444 had `position_valid=false` and `fixed_position_valid=false`. The camera continued publishing depth, but the tracker subscribed to an absent depth topic. Visual recognition continued. |

All valid spatial observations in these captures had zero corrected position-time
versus original depth-time difference (default clock offset zero). All captures
reported zero duplicate nonzero output IDs. RGB inventory misses at capture edges
were 0–7 per stage; the probe was independent of the nodes' buffers. These counts
are not evidence of an internal RGB mismatch.

Two identities remain saved in the private store with one normalized template
each. Directory/file modes were verified as 0700/0600, with no tracked `.private`
files. Live deletion was not applied to the enrolled people; durable deletion and
restart are covered by the synthetic storage/engine tests.

Metadata only is written under ignored `build/milestone2-verification/`; biometric
templates stay in the private store. See `physical-metrics.json`, each `live-*.json`
and the launch logs there. The test pipeline was stopped after the checks.

The live tests demonstrate enrollment, explicit failure, unknown rejection in the
instructed scenario, restart persistence and recognition without depth/TF. They
also expose a remaining tracking limit: conservative association fragments under
crossing/occlusion. There is no equivalent labeled pre-change recording, so
real-world tracking non-regression and ID-switch/recovery rates remain unproven.

## Repeat the physical check

From `/home/furia/opl_hr`, in a dedicated terminal:

```bash
source /opt/ros/jazzy/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=72
export ROS_LOG_DIR="$PWD/build/milestone2-verification/roslogs"
ros2 launch opl_human_vision test_detector.launch.py camera_type:=realsense \
  enable_visualizer:=true \
  scrfd_model_path:="$PWD/src/opl_human_vision/laptop_models/scrfd_2.5g.engine"
```

Use the same environment in a second terminal. Observe status before requesting
enrollment, so authorization cannot be mistaken for completion:

```bash
ros2 topic echo /face_recognizer_node/enrollment_status
ros2 service call /face_recognizer_node/enroll_person opl_interfaces/srv/EnrollPerson \
  '{raw_track_id: 0, name: ""}'
```

Have two people stand apart, face the camera around 1 m away, and keep their heads
steady until each target reaches `completed` or `already_enrolled`. Check the
session's completed/failed counts. A failed target must be retried explicitly.
For a named individual, read `raw_track_id` from `/recognized_humans`, then request
that exact track and name. Group requests use generated names.

Before enrolling the second person, let the enrolled person leave and confirm the
other remains unknown. After both enroll, cross slowly, briefly occlude one
person, separate and face the camera, then leave and return after >2 s. Registered
names should recover with face evidence; raw tracks may expire and get new IDs.
No frame may assign the same nonzero identity to two people.

Restart the pipeline without deleting `.private`, then verify the names return
without another enrollment request. To check depth loss, restart the same launch
with `depth_topic:=/milestone2/no_depth`; expect 2D tracking and recognition with
`position_valid=false`. Missing odom must leave `fixed_position_valid=false`.

Deletion is explicit and durable; replace ID with the actual identity to delete:

```bash
ros2 service call /face_recognizer_node/delete_identity opl_interfaces/srv/DeleteIdentity \
  '{persistent_id: 1}'
```

Deletion is rejected while enrollment is busy. Re-enrollment gets a new persistent
ID. Only stop the launch started for this check; do not stop unrelated robot nodes.

For Gemini hardware use `camera_type:=gemini335`, retaining registered depth and
color CameraInfo. For moving-robot proof, supply the actual observation-time
camera-to-odom TF and confirm that a stationary person's fixed-frame position
remains stable while camera-relative coordinates change. This laptop check cannot
establish that result.

## Remaining limits

Gemini hardware, Jetson runtime and a moving robot with real odometry were not
available. There is no labeled recorded benchmark, so real ID-switch/recovery
rates and population-level false-match rates are not claimed. Quality, similarity,
world-speed and ambiguity gates need a representative deployment dataset. Missing
TF, near-identical clothing and full overlap can cause conservative fragmentation.
No resolution/FPS tuning, navigation, wrist integration or Milestone 3 work was done.
