std_msgs/Header header
uint64 session_id
uint64 raw_track_id
uint64 persistent_id
string state          # authorized, scanning, completed, already_enrolled, failed, finished, deleted
string message
uint32 accepted_samples
uint32 required_samples
uint32 total_targets
uint32 completed_targets
uint32 failed_targets
