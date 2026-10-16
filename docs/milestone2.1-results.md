# Corrective Milestone 2.1 verification

## Phase 1 — tracking

Preserved original YOLO boxes for partial/invalid poses; reliable full poses apply
only a small correction. Added high/low confidence recovery, elapsed-time predicted
publication and prediction-residual box smoothing. Ambiguous associations defer new
IDs instead of fragmenting immediately. Weak detections cannot create tracks or
seed/update appearance, enrollment or anonymous galleries.

Build: three packages succeeded in the isolated symlink build. All 11 package tests
passed. Added gallery-contamination regression also passed. Tests cover low-score
birth prevention/recovery, crossing/occlusion/return, similar-clothing ambiguity,
world-frame rotation, 5/15/30 Hz prediction hold, box jitter, partial-pose fallback,
and predicted observations without fresh 3D. Existing timestamp/depth/TF tests pass.

D455 stationary capture: one raw ID, 598 observed samples, 100% valid depth,
29.99 recognized outputs/s, P95 message age 59.38 ms. Partial-object capture:
one raw ID, 1,189 observed and seven predicted samples, 89.23% valid depth across
the mixed occluded/unoccluded sequence, 29.94 outputs/s, P95 age 59.33 ms.
No duplicate output identities were observed. Identity names still disappeared
through turns; this is the Phase 2 defect.

Baseline stationary output was 18.25/s and P95 age 251.19 ms; however baseline
RGB input was 23.38/s versus 29.93/s later and shared machine load varied. These
captures show no observed slowdown, but do not establish a causal speedup.
Raw ID was also stable in the baseline sequence, so no measured physical reduction
in fragmentation is claimed. Strong/weak association was exercised synthetically;
no weak real detections occurred in the partial-object capture.

Only partial hiding is physically possible. Full-body occlusion and two-person
crossing remain synthetic checks. The protected original identity file stayed
byte-for-byte unchanged; physical runs used an owner-only private test copy.

## Phase 2 — identity continuity

Removed duplicate node-level identity caches and mandatory per-frame face validation.
The shared engine now distinguishes VERIFIED/TRACKED/LOST/UNKNOWN/AMBIGUOUS, retains
verified observed body lineages without a face-age expiry, requires three consistent
face observations over >=0.2 s after contradiction, and recovers only recent verified
lineages using unique body similarity plus box-motion and available fixed-frame gates.
A new identity still requires a usable face. Simultaneous competing face claims are
resolved before per-track decisions; one active owner is enforced. Added companion
IdentityStatus/IdentityStatusArray output; existing TrackedHuman is unchanged by this
phase. Original RGB matching remains exact; missing RGB permits continuity without
face inference or learning. Prediction reports LOST and never refreshes measured state.

Scoped build succeeded; all 11 tests passed (3.00 s). Added deterministic checks for
>10 s face absence with observed body, predicted LOST age, new-raw short-gap recovery,
contradiction suspension and three-face restoration, stale recovery rejection,
identical simultaneous face claims, strongest unambiguous claim independent of order,
and fixed-world spatial contradiction with missing-TF recovery. Existing storage,
ownership, crossing, synchronization, depth and TF tests still pass.

D455 instructed turn/partial-object sequence: original raw track 2 retained Person_2
in 1,160/1,163 observations (99.74%, including three initial unknown observations).
A second raw track 3 appeared late and remained unknown in all 367 observations;
operator feedback is pending to identify that extra box. No duplicate registered
ownership. 29.73 outputs/s; P95 message age 58.48 ms; matching depth timestamp delta0.
This is one operator-run sequence, not proof of angle-specific or real-world zero
error rates. No full-body occlusion or two-person physical test was possible.
Original identity SHA256 unchanged; live recognizer parameter confirmed private copy.

Repeat-turn check was less stable: raw IDs 3/4/5/6, 623 named and 566 unknown
person observations, 64 AMBIGUOUS status entries; depth validity 37.95%, 29.67/s,
P95 message age55.66ms. Several boxes coexisted; their physical targets remain
unconfirmed. This repeat prevents claiming complete physical turn robustness.
A regression additionally reproduced recovery being reset by feature-free frames
between scheduled face checks; the counter now survives those frames and counts
only distinct face observations. Physical verification must be repeated after this
fix and depth diagnostics. First-capture continuity is not a general success claim.

Operator clarified the repeat actually involved TWO people, using the second as an
obstacle. The enrolled name briefly appeared on the unenrolled person's box. This
is a physical false assignment and a failed safety check, despite later rejection;
metadata-only duplicate-ownership counts did not detect it. The second person is
no longer available for a corrective physical repeat. The engine's body recovery
could promote an already-visible anonymous track, and continued raw-ID trust through
overlap was too permissive. Added deterministic reproductions for both conditions;
these must pass before progressing. Physical multi-person safety remains unverified
after the correction, and the earlier live false assignment remains in the record.

Safety correction: the new regression first failed because an already-visible
anonymous person inherited the absent owner's name. Recovery now excludes established
anonymous tracks. Person-person overlap suspends registered output as AMBIGUOUS and
requires three distinct, consistent usable faces over >=0.2 s after separation.
This deliberately accepts temporary unknown labels rather than trusting a possibly
switched raw ID. Face-free continuity remains enabled for reliable isolated body
tracks and short new-raw recovery. Both new safety regressions and the interleaved
face-check recovery regression pass; a two-person physical repeat is still needed.

## Phase 3 — anchored face galleries

Enrollment now saves actual distinct observations (central medoid plus up to three
other distinct good samples), retaining frontal-only scan completion requirements.
Adaptive learning is a separate routine in the existing engine/node writer: observed,
isolated high-confidence bodies; strong unambiguous usable faces; immutable-anchor
similarity; >=3 repeated distinct-time compatible observations over >=0.5 s; duplicate
rejection; >=5 s between durable writes. New galleries cap at12. At capacity only a
redundant adaptive template can be replaced. Legacy anchors are never removed;
legacy galleries larger than12 are retained without growth rather than destroyed.
Version2 records anchor counts; version1 loads without rewriting and migrates only
on a successful authorized enrollment/adaptive durable write. Existing secure path,
permissions, locks, atomic save and uncertain-commit protection are preserved.

Scoped build passed; all11 tests passed, including synthetic v1-to-v2 migration,
anchor equality after restart, durable adaptive updates, repeated duplicate/body/weak
observations, cap12 and existing I/O-failure protections. D455 single-person instructed
angle capture:29.92 outputs/s, P95 age52.77ms, raw IDs3/4,511 named and328 unknown
observations, only3 VERIFIED status observations. No adaptive template was accepted:
test copy remains version1 with galleries[1,1]; anchors exactly unchanged. This does
NOT verify successful live adaptive learning or angle robustness; repeat at a closer,
measured position is required. Protected original identity data remains untouched.
