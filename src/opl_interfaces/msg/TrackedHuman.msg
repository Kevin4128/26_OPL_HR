int32 track_id          # Persistent Identity ID (assigned by IdentityManager)
int32 raw_track_id      # Raw tracking ID (assigned by ByteTrack)
string name             # Recognized name or "Person_X"
float32 confidence      # Detection confidence
float32 match_score     # Cosine similarity score
geometry_msgs/Point position_3d
sensor_msgs/RegionOfInterest bbox
geometry_msgs/Point[] landmarks

# position_3d remains in the registered RGB optical frame; zero is never valid.
bool position_valid
std_msgs/Header position_header          # Corrected depth acquisition time, RGB optical frame
std_msgs/Header depth_header             # Original selected depth stamp and frame convention
uint32 depth_valid_pixels
float32 depth_valid_fraction
float32 depth_mad_m
bool fixed_position_valid
geometry_msgs/PointStamped fixed_position # Configured fixed frame (default odom)
bool observed                            # Actual detector observation, not a predicted-only box
