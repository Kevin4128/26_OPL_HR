#include "opl_human_vision/identity_tf.hpp"
#include <cassert>
#include <limits>

int main() {
  using namespace opl_human_vision;
  opl_interfaces::msg::TrackedHuman h;
  h.track_id = 1; h.observed = true; h.position_valid = true;
  h.identity_state = h.TRACKED; h.position_3d.z = 2.0;
  std_msgs::msg::Header header;
  header.frame_id = "camera_color_optical_frame";
  header.stamp.sec = 1;
  auto tf = personTransform(h, header, 1.1);
  assert(tf && tf->child_frame_id == "opl_hr_person_1");
  assert(tf->header.frame_id == header.frame_id && tf->header.stamp.sec == 1);
  assert(tf->transform.translation.z == 2.0 && tf->transform.rotation.w == 1.0);
  h.track_id = 2;
  assert(personTransform(h, header, 1.1)->child_frame_id == "opl_hr_person_2");
  h.identity_state = h.FACE_CONFIRMED;
  assert(personTransform(h, header, 1.1));
  h.identity_state = h.UNCERTAIN;
  assert(!personTransform(h, header, 1.1));
  h.identity_state = h.TRACKED;
  h.association_ambiguous = true;
  assert(!personTransform(h, header, 1.1));
  h.association_ambiguous = false;
  h.observed = false;
  assert(!personTransform(h, header, 1.1));
  h.observed = true;
  h.position_valid = false;
  assert(!personTransform(h, header, 1.1));
  h.position_valid = true;
  assert(!personTransform(h, header, 1.2));
  assert(!personTransform(h, header, 0.9));
  h.track_id = 1000000000;
  assert(!personTransform(h, header, 1.1));
  h.track_id = 1;
  h.position_3d.z = std::numeric_limits<double>::quiet_NaN();
  assert(!personTransform(h, header, 1.1));
  h.position_3d.z = 2.0;
  header.frame_id.clear();
  assert(!personTransform(h, header, 1.1));
}
