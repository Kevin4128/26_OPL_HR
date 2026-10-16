# Milestone 2 — tracking, identity and enrollment

Design confirmed through grilling. Preserve M1 edits and detector code. Preserve
Hungarian/IoU, velocity prediction, image camera-motion compensation, SCRFD,
AdaFace and OSNet. No camera resolution/FPS tuning, navigation, wrist or pushes.

1. Move M1 spatial enrichment before tracker assignment. Add same-fixed-frame
   gates, conservative appearance ambiguity checks, clean gallery updates,
   elapsed-time expiry (2s), and temporal/GMC resets. Preserve 2D fallback.
2. Separate extraction from deterministic identity decisions. Validate face
   quality; require face evidence plus ambiguity/ownership checks for registered
   names. Remove relaxed locked-identity thresholds. Never compare historical
   camera-relative coordinates. Keep transient body-based anonymous tracking.
3. Persist only explicitly enrolled face templates in owner-only files under
   `.private/identities/`, ignored by Git. Version/model validation, bounded
   parsing, symlink refusal, exclusive writer lock and atomic durable saves.
   Existing faces return existing IDs; conflicting names fail; deletion is explicit.
4. Explicit request snapshots visible tracks, sorted by person confidence. Scan
   one stable target at a time. Group names are unique Person_ID; explicit targets
   may have custom names. Require 20 consistent good samples spanning >=2.5s.
   Target missing 2s or scanning >30s fails/discards partial data and advances.
   Structured progress separates authorization, scanning, completion and failure;
   completion requires durable storage. Newcomers need a new session.
5. Deterministic tests cover crossings, occlusions, return, camera rotation,
   similar appearance, unknowns, ownership conflicts, enrollment/restart/deletion,
   invalid storage and depth/TF loss. Run package build/tests and focused review.
6. Use connected D455 for M1 checks, then staged M2 physical checks with user
   instructions. Quantify only evidence actually observed; synthetic tests do not
   establish real-world error rates. Stop before Milestone 3 for approval.

Baseline evidence: build/milestone2-verification/milestone1-ready-check.json.
The 10s live check recorded 313/337 valid camera positions, zero timestamp delta
for valid samples, and no fixed positions (no odom TF in isolated ROS domain72).
All five existing package tests passed again in isolated domain73.
