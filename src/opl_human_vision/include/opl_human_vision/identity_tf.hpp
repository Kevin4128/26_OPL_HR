#pragma once

#include <cmath>
#include <optional>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <opl_interfaces/msg/tracked_human.hpp>
#include <std_msgs/msg/header.hpp>

namespace opl_human_vision {

inline std::optional<geometry_msgs::msg::TransformStamped> personTransform(
  const opl_interfaces::msg::TrackedHuman& human,
  const std_msgs::msg::Header& header, double now_sec)
{
  const double age = now_sec - (header.stamp.sec + header.stamp.nanosec * 1e-9);
  const auto& p = human.position_3d;
  if (!human.observed || !human.position_valid || human.association_ambiguous ||
      (human.identity_state != human.FACE_CONFIRMED && human.identity_state != human.TRACKED) ||
      human.track_id <= 0 || human.track_id >= 1000000000 || header.frame_id.empty() ||
      !std::isfinite(age) || age < 0 || age > 0.15 ||
      !std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z) || p.z <= 0) {
    return std::nullopt;
  }
  geometry_msgs::msg::TransformStamped transform;
  transform.header = header;
  transform.child_frame_id = "opl_hr_person_" + std::to_string(human.track_id);
  transform.transform.translation.x = p.x;
  transform.transform.translation.y = p.y;
  transform.transform.translation.z = p.z;
  // Frame convention only; body heading is not measured.
  transform.transform.rotation.w = 1.0;
  return transform;
}

}  // namespace opl_human_vision
