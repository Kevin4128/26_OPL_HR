int32 track_id          # Persistent Identity ID (assigned by IdentityManager)
int32 raw_track_id      # Raw tracking ID (assigned by the person tracker)
string name             # Recognized name or "Person_X"
float32 confidence      # Detection confidence
float32 match_score     # Cosine similarity score
geometry_msgs/Point position_3d
sensor_msgs/RegionOfInterest bbox
geometry_msgs/Point[] landmarks

# Explicit observation/identity validity; confidence remains detector confidence.
bool observed
bool position_valid
bool association_ambiguous
uint8 UNCONFIRMED=0
uint8 FACE_CONFIRMED=1
uint8 TRACKED=2
uint8 UNCERTAIN=3
uint8 identity_state
