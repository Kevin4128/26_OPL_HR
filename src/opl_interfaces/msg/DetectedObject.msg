# ID identifies this detection only; it is not a tracked object identity.
uint64 detection_id
uint16 class_id
string class_name
float32 confidence
string color
# Fraction of sampled foreground pixels supporting the color, not a probability.
float32 color_confidence
sensor_msgs/RegionOfInterest bbox
# Representative visible surface point, NOT the object's physical center.
geometry_msgs/PoseStamped camera_pose
bool position_valid
# Identity quaternion is a placeholder when orientation_valid is false.
bool orientation_valid
string position_status
std_msgs/Header depth_header
uint32 depth_valid_pixels
float32 depth_mad_m
geometry_msgs/PoseStamped fixed_pose
bool fixed_position_valid
bool fixed_orientation_valid
string fixed_status
