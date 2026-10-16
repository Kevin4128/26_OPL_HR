# Companion output; TrackedHuman wire format and match_score semantics stay unchanged.
uint64 raw_track_id
uint64 persistent_id
string name
# VERIFIED, TRACKED, LOST, UNKNOWN or AMBIGUOUS
string state
string reason
float32 last_face_score
# -1 means no verified face/body history.
float64 face_age_sec
float64 body_age_sec
