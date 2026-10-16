# Corrective Milestone 2.1

Design confirmed through two grilling rounds and the operator's Q7 clarification.
All work remains in `/home/furia/opl_hr`; preserve existing uncommitted changes.
No commits, pushes, remotes, reference workspace, object detector, navigation,
manipulation, arm cameras, resolution or FPS tuning changes.

## Agreed behavior

- Keep verified identity while continuous body tracking remains reliable, including
  hidden faces and 45/90/180 degree turns. No face-age-only expiry. Recheck usable
  faces every 0.5 s. Contradiction/ambiguity suspends identity immediately; recovery
  needs three consistent face observations spanning >=0.2 s.
- Recover previous verified lineage within 2 s only with unique body/motion/history
  and valid available fixed-frame evidence. Clothing cannot establish a new identity.
- Predict boxes for <=0.5 s with observed=false and invalid fresh 3D; retain internal
  tracks for 2 s. Low-confidence detections may recover confirmed tracks, never
  create tracks or enrollment/gallery samples.
- Frontal-only enrollment remains available (20 good samples, >=2.5 s). Learn actual
  usable side views later, never synthetic inferred biometric views. Use a separate
  learning routine in the existing recognition node with one storage writer.
- Keep original templates as anchors, bounded gallery 12; only redundant adaptive
  templates may be replaced after strong repeated isolated face verification.
- Preserve existing TrackedHuman wire fields; use additive identity-status output.
- Original `.private/identities/identities.json` stays byte-for-byte untouched during
  ALL testing. Use owner-only `.private/milestone2.1-test/` copy and SHA256 guards.
- One operator is available. Use object occlusion and turns physically; two-person
  crossing/ownership safety remains synthetic, explicitly unverified on hardware.
- Reproducible >5% FPS loss or >10% P95 latency increase triggers investigation.
  Target >=95% valid unobstructed torso depth and median absolute Z error <=0.10 m
  at operator-measured distances 1–3 m; reject ambiguous/occluded depth explicitly.

## Sequential work and gates

1. Baseline unchanged algorithms with isolated build/ROS domain74 and test store.
   Phase1: preserve YOLO boxes unless reliable whole-pose refinement, elapsed-time
   prediction publication, high/low confidence association, conservative ambiguity
   retention, bbox stabilization. Reuse existing pose/tracker helpers and tests.
   Run scoped build/tests, physical single-person object occlusion; report results.
2. Phase2: VERIFIED/TRACKED/LOST/UNKNOWN/AMBIGUOUS continuity, unique ownership,
   bounded recovery and current-face contradiction checks. Add companion status
   interface, preserve actual face-score semantics, gate learning on real high
   confidence isolated observations. Test states, turns, losses and conflict cases.
3. Phase3: distinct anchored galleries and trusted adaptive memory; preserve legacy
   identities and validate any metadata migration on test copies. Test contamination,
   bounded replacement, persistence, restart and I/O errors.
4. Phase4: exact depth diagnostics, coherent torso cluster selection including its
   representative pixel, confidence/consistency rejection and unchanged timestamped
   TF. Test background mixtures, occluders, bad depths, TF and timing errors. Diagnose
   compressed-image errors independently from raw aligned depth.
5. Phase5: full scoped build/test, D455 launch, repeated single-person physical
   sequences, measured distances, restart/gallery safety, depth loss, FPS/latency and
   timestamp comparison. Review diff, confirm protected hash, stop test processes.

Build with `colcon build --symlink-install --packages-up-to opl_human_vision` using
workspace-local isolated build/install bases to preserve the existing installation.
Use only relevant package tests after each phase. Fix failed gates before moving
on; distinguish synthetic, physical and unverified claims in the phase report.
No original range/FPS optimization work before operator approval.
