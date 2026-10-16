# Milestone 1: RGB-D human perception

Agreed scope: software build/tests now; physical cameras deferred. Preserve
camera-relative positions, association, recognition thresholds, enrollment, and
models. Add timestamp-correct observations in configurable `odom`.

1. Add a shared, bounded observation buffer: exact RGB stamp/frame matches,
   20 ms configurable depth tolerance, 100 ms pending wait, timestamp/clock
   checks, explicit registered-frame aliases and known depth clock offset.
   Test delay/reordering, stale/invalid/duplicate stamps and clock resets.
2. Replace tracker depth projection with calibrated torso median sampling,
   minimum count/fraction and consistency checks. Add validity/quality and
   independently stamped camera/fixed-frame fields to `TrackedHuman`.
   Test sparse/invalid/inconsistent depth, both encodings, calibration, TF
   failure, and simulated translation/rotation at acquisition timestamps.
3. Wire shared buffering into tracker and recognition, retain original detector
   headers, and make existing consumers honor validity. Keep registered-depth
   camera launches supported and expose timing/frame configuration.
4. Build affected packages, run relevant deterministic tests, review the diff,
   and record results plus physical verification commands in the final report.

Known boundary: existing camera-relative recognition spatial gating remains
unchanged and may reject identities during robot motion. Registered depth is
required; unregistered streams and unknown clock offsets must not produce 3D.
All message consumers must rebuild. No hardware/Jetson proof is claimed.

Software verification complete: all three packages built; five registered tests
passed with no errors, failures or skips. The deadline review finding was fixed
and verified with a failing/passing regression. Physical checks remain deferred.
See `milestone1-report.md` for evidence, risks and hardware commands.
