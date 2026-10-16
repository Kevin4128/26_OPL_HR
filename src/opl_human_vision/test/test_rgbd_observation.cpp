#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>

#include "opl_human_vision/human_position.hpp"
#include "opl_human_vision/tracker_engine.hpp"
#include <cv_bridge/cv_bridge.hpp>

using namespace opl_human_vision;

static builtin_interfaces::msg::Time stamp(double seconds) {
  return rclcpp::Time(int64_t(std::llround(seconds * 1e9)));
}

static sensor_msgs::msg::Image::SharedPtr
image(double time, const std::string &frame = "rgb") {
  auto result = std::make_shared<sensor_msgs::msg::Image>();
  result->header.stamp = stamp(time);
  result->header.frame_id = frame;
  result->width = 100;
  result->height = 100;
  return result;
}

static opl_interfaces::msg::TrackedHumanArray::SharedPtr
observation(double time, const std::string &frame = "rgb") {
  auto result = std::make_shared<opl_interfaces::msg::TrackedHumanArray>();
  result->header = image(time, frame)->header;
  return result;
}

static sensor_msgs::msg::CameraInfo::SharedPtr calibration() {
  auto info = std::make_shared<sensor_msgs::msg::CameraInfo>();
  info->header.frame_id =
      "rgb"; // Zero timestamp is valid for static calibration.
  info->width = info->height = 100;
  info->k = {100, 0, 50, 0, 100, 50, 0, 0, 1};
  return info;
}

static void timestamps() {
  const int64_t now = 10000000000LL;
  ObservationBuffer b;
  assert(b.settings.valid());
  assert(!b.addImage(image(0), false, now, 0));
  assert(!b.addImage(image(8), false, now, 0));
  assert(!b.addImage(image(11), false, now, 0));
  assert(!b.addImage(image(10, ""), false, now, 0));
  auto bad = image(10);
  bad->header.stamp.nanosec = 1000000000;
  assert(!b.addImage(bad, false, now, 0));
  assert(b.enqueue(observation(10), now, 0));
  assert(!b.enqueue(observation(10), now, 0));
  assert(!b.take(now, 0.05, false));
  assert(b.addImage(image(9.97), false, now, 0.05));
  assert(!b.take(now, 0.05, false)); // Nearby RGB is never a crop source.
  const auto exact = image(10);
  assert(b.addImage(exact, false, now, 0.06));
  assert(!b.addImage(exact, false, now, 0.07));
  const auto selected = b.take(now, 0.07, false);
  assert(selected && selected->rgb == exact && !selected->depth);
  assert(!b.enqueue(observation(9.99), now, 0.08));

  b.clear();
  assert(b.enqueue(observation(10, "wrong"), now, 0));
  assert(b.addImage(image(10), false, now, 0));
  assert(!b.take(now, 0.05, false));
  const auto missing = b.take(now, 0.11, false);
  assert(missing && !missing->rgb);

  // Reordered pending detections are processed in acquisition order.
  b.clear();
  assert(b.enqueue(observation(10), now, 0));
  assert(b.enqueue(observation(9.98), now, 0));
  assert(b.addImage(image(10), false, now, 0));
  assert(b.addImage(image(9.98), false, now, 0));
  assert(stampNs(b.take(now, 0, false)->observation->header.stamp) ==
         9980000000LL);
  assert(stampNs(b.take(now, 0, false)->observation->header.stamp) == now);

  b.clear();
  b.enqueue(observation(10), now, 0);
  b.addImage(image(10), false, now, 0);
  b.enqueue(observation(9.99), now, 0.09);
  const auto expired_predecessor = b.take(now, 0.11, true);
  assert(expired_predecessor && !expired_predecessor->rgb);
  const auto timely_2d = b.take(now, 0.11, true);
  assert(timely_2d && timely_2d->rgb && !timely_2d->depth);

  b.clear();
  b.enqueue(observation(10), now, 0);
  b.addImage(image(10), false, now, 0);
  assert(!b.take(9000000000LL, 0.02,
                 false)); // Backward ROS clock flushes old epoch.
  assert(b.enqueue(observation(9), 9000000000LL, 0.03));
  assert(b.addImage(image(9), false, 9000000000LL, 0.03));
  assert(b.take(9000000000LL, 0.03, false)->rgb);

  b.clear();
  b.settings.capacity = 1;
  b.addImage(image(9.99), false, now, 0);
  b.addImage(image(10), false, now, 0);
  assert(b.enqueue(observation(9.99), now, 0));
  assert(!b.enqueue(observation(10), now, 0));
  assert(!b.take(now, 0.11, false)
              ->rgb); // Bounded memory never substitutes a newer image.
  b.clear();
  b.settings.capacity = 60;
  b.enqueue(observation(10), now, 0);
  b.addImage(image(10), false, now, 0);
  assert(!b.take(now + 2000000000LL, 0.2, false)); // Stale pending observation.
}

static void depthMatching() {
  const int64_t now = 10000000000LL;
  ObservationBuffer b;
  b.setCalibration(calibration(), now);
  b.addImage(image(10), false, now, 0);
  b.enqueue(observation(10), now, 0);
  b.addImage(image(9.8), true, now, 0);
  assert(!b.take(now, 0.05, true));
  auto missing = b.take(now, 0.11, true);
  assert(missing && missing->rgb &&
         !missing->depth); // Valid 2D with absent depth.

  b.clear();
  b.setCalibration(calibration(), now);
  b.addImage(image(10), false, now, 0);
  b.enqueue(observation(10), now, 0);
  b.addImage(image(10, "raw_depth"), true, now, 0);
  assert(!b.take(now, 0.05, true));
  assert(!b.take(now, 0.11, true)->depth);

  b.clear();
  b.settings.registered_frame_aliases = {"registered_depth"};
  b.settings.depth_offset_sec = 0.05;
  b.setCalibration(calibration(), now);
  b.addImage(image(10, "/rgb"), false, now, 0);
  b.enqueue(observation(10), now, 0);
  auto correct_depth = image(9.95, "registered_depth");
  b.addImage(correct_depth, true, now, 0.01);
  b.addImage(image(10.03, "registered_depth"), true, now, 0.02);
  auto selected = b.take(now, 0.02, true);
  assert(selected && selected->depth == correct_depth); // Not the latest depth.

  b.clear();
  b.settings.depth_offset_sec = 0;
  b.setCalibration(calibration(), now);
  b.addImage(image(10), false, now, 0);
  b.enqueue(observation(10), now, 0);
  b.addImage(image(9.99), true, now, 0);
  b.addImage(image(10.01), true, now, 0);
  assert(!b.take(now, 0.11, true)
              ->depth); // Equal nearest neighbours are ambiguous.

  b.clear();
  b.setCalibration(calibration(), now);
  b.addImage(image(10), false, now, 0);
  b.enqueue(observation(10), now, 0);
  auto wrong_size = image(10);
  wrong_size->width = 99;
  b.addImage(wrong_size, true, now, 0);
  assert(!b.take(now, 0.11, true)->depth);

  b.clear();
  b.setCalibration(calibration(), now);
  b.addImage(image(10), false, now, 0);
  b.enqueue(observation(10), now, 0);
  auto unchanged = calibration();
  unchanged->header.stamp = stamp(10.1);
  b.setCalibration(unchanged, now);
  b.addImage(image(10), true, now, 0);
  assert(b.take(now, 0, true)->calibration->header.stamp.sec == 0);
  b.enqueue(observation(10.02), now, 0);
  auto changed = calibration();
  changed->k[0] = 101;
  b.setCalibration(changed, now);
  assert(!b.take(now, 0.2,
                 true)); // Calibration change flushes pending observations.
}

static void depthValidity() {
  ObservationBuffer b;
  auto info = calibration();
  auto rgb = image(10);
  assert(validCalibration(*info, *rgb, b));
  auto invalid = *info;
  invalid.k[0] = 0;
  assert(!validCalibration(invalid, *rgb, b));
  invalid = *info;
  invalid.k[2] = std::numeric_limits<double>::quiet_NaN();
  assert(!validCalibration(invalid, *rgb, b));
  invalid = *info;
  invalid.width = 99;
  assert(!validCalibration(invalid, *rgb, b));
  invalid = *info;
  invalid.header.frame_id = "other";
  assert(!validCalibration(invalid, *rgb, b));
  invalid = *info;
  invalid.header.stamp = stamp(10.1);
  assert(validCalibration(
      invalid, *rgb, b)); // Calibration timestamps are not exposure timestamps.
  invalid = *info;
  invalid.header.stamp.nanosec = 1000000000u;
  assert(!validCalibration(invalid, *rgb, b));
  invalid = *info;
  invalid.distortion_model = "unknown";
  invalid.d = {0.1};
  assert(!validCalibration(invalid, *rgb, b));

  cv::Mat depth(100, 100, CV_32FC1, cv::Scalar(2));
  opl_interfaces::msg::TrackedHuman human;
  DepthSettings settings;
  const cv::Rect bbox(0, 0, 100, 100);
  estimateHumanPosition(depth, bbox, *info, settings, human);
  assert(human.position_valid && human.depth_valid_pixels == 1600);
  assert(human.depth_valid_fraction == 1 && human.depth_mad_m == 0);
  assert(std::abs(human.position_3d.z - 2) < 1e-6);
  assert(std::abs(human.position_3d.y + 0.21) <
         1e-6); // Torso centre, not bbox centre.
  cv::Mat mm(100, 100, CV_16UC1, cv::Scalar(2000));
  estimateHumanPosition(mm, bbox, *info, settings, human);
  assert(human.position_valid && std::abs(human.position_3d.z - 2) < 1e-6);
  depth.setTo(std::numeric_limits<float>::quiet_NaN());
  depth.at<float>(30, 40) = 2;
  estimateHumanPosition(depth, bbox, *info, settings, human);
  assert(!human.position_valid && human.depth_valid_pixels == 1 &&
         human.position_3d.z == 0);
  depth.setTo(0);
  depth(cv::Rect(30, 20, 40, 5))
      .setTo(2); // Enough pixels, insufficient fraction.
  estimateHumanPosition(depth, bbox, *info, settings, human);
  assert(!human.position_valid && human.depth_valid_pixels == 200);
  depth.setTo(2);
  depth(cv::Rect(30, 20, 20, 40)).setTo(5);
  estimateHumanPosition(depth, bbox, *info, settings, human);
  assert(!human.position_valid); // Mixed person/background.
  depth.setTo(2);
  depth(cv::Rect(30, 20, 2, 40)).setTo(9);
  estimateHumanPosition(depth, bbox, *info, settings, human);
  assert(human.position_valid &&
         human.position_3d.z == 2); // Robust to isolated outliers.
  DepthDiagnostics diagnostics;
  depth.setTo(2);
  depth(cv::Rect(30, 20, 12, 40))
      .setTo(5); // 30% background, compact human mode.
  estimateHumanPosition(depth, bbox, *info, settings, human, &diagnostics);
  assert(human.position_valid && human.position_3d.z == 2);
  assert(diagnostics.p90 - diagnostics.p10 > diagnostics.allowed_spread);
  assert(diagnostics.cluster_p90 - diagnostics.cluster_p10 <=
         diagnostics.allowed_spread);
  depth.setTo(2);
  depth(cv::Rect(30, 44, 40, 16)).setTo(1); // Lower foreground furniture.
  estimateHumanPosition(depth, bbox, *info, settings, human, &diagnostics);
  assert(human.position_valid && human.position_3d.z == 2);
  depth.setTo(2);
  depth(cv::Rect(40, 20, 20, 20)).setTo(1); // Upper torso fully blocked.
  estimateHumanPosition(depth, bbox, *info, settings, human, &diagnostics);
  assert(!human.position_valid &&
         diagnostics.reason == "selected_surface_lacks_upper_torso_support");
  depth.setTo(2);
  depth(cv::Rect(45, 35, 10, 10)).setTo(9); // Centre ray belongs to background.
  estimateHumanPosition(depth, bbox, *info, settings, human, &diagnostics);
  assert(human.position_valid);
  int u =
      std::floor(human.position_3d.x / human.position_3d.z * 100 + 50 + 1e-6);
  int v =
      std::floor(human.position_3d.y / human.position_3d.z * 100 + 50 + 1e-6);
  assert(depth.at<float>(v, u) == 2);
  for (int y = 20; y < 60; ++y)
    for (int x = 30; x < 70; ++x)
      depth.at<float>(y, x) = ((x - 30) % 5 < 4 && (y - 20) % 5 < 4) ? 2 : 9;
  estimateHumanPosition(depth, bbox, *info, settings, human, &diagnostics);
  assert(!human.position_valid &&
         diagnostics.reason == "insufficient_connected_torso_support");
  for (int y = 20; y < 60; ++y)
    for (int x = 30; x < 70; ++x)
      depth.at<float>(y, x) = 1.f + (x - 30) * .03f;
  estimateHumanPosition(depth, bbox, *info, settings, human, &diagnostics);
  assert(!human.position_valid &&
         diagnostics.reason == "selected_cluster_depth_spread");
  for (float z :
       {0.f, -1.f, 0.1f, 11.f, std::numeric_limits<float>::infinity()}) {
    depth.setTo(z);
    estimateHumanPosition(depth, bbox, *info, settings, human);
    assert(!human.position_valid && human.position_3d.z == 0);
  }
  estimateHumanPosition(depth, cv::Rect(101, 0, 10, 10), *info, settings,
                        human);
  assert(!human.position_valid);
  cv::Mat unsupported(100, 100, CV_8UC1, cv::Scalar(2));
  estimateHumanPosition(unsupported, bbox, *info, settings, human);
  assert(!human.position_valid);

  // Tracking has no dependency on a depth observation or TF.
  TrackerEngine tracker(30, 1, 0.3, 0);
  human.bbox.width = 50;
  human.bbox.height = 80;
  human.confidence = 0.9;
  const auto tracks = tracker.update({human}, cv::Mat(), 10);
  assert(tracks.size() == 1);
}

static void transforms() {
  auto clock = std::make_shared<rclcpp::Clock>(RCL_ROS_TIME);
  tf2_ros::Buffer buffer(clock);
  opl_interfaces::msg::TrackedHuman human;
  human.position_valid = true;
  human.position_header.frame_id = "rgb";
  human.position_header.stamp = stamp(10);
  human.position_3d.x = 3;
  human.position_3d.z = 2;
  transformHumanPosition(human, buffer, "odom");
  assert(!human.fixed_position_valid && human.position_valid &&
         human.position_3d.x == 3);
  geometry_msgs::msg::TransformStamped tf;
  tf.header.frame_id = "odom";
  tf.child_frame_id = "rgb";
  tf.header.stamp = stamp(10);
  tf.transform.rotation.w = 1;
  assert(buffer.setTransform(tf, "test"));
  tf.header.stamp = stamp(11);
  tf.transform.translation.x = 1;
  tf.transform.rotation.w = std::sqrt(0.5);
  tf.transform.rotation.z = std::sqrt(0.5);
  assert(buffer.setTransform(tf, "test"));
  transformHumanPosition(human, buffer, "odom");
  assert(human.fixed_position_valid &&
         std::abs(human.fixed_position.point.x - 3) < 1e-6);
  assert(human.fixed_position.header.stamp ==
         stamp(10)); // Latest TF would give (1,3,2).
  human.position_header.stamp = stamp(11);
  human.position_3d.x = 0;
  human.position_3d.y = -2;
  transformHumanPosition(human, buffer, "odom");
  assert(human.fixed_position_valid &&
         std::abs(human.fixed_position.point.x - 3) < 1e-6);
  assert(std::abs(human.fixed_position.point.y) <
         1e-6); // Stationary world person, moving camera.
  human.position_header.stamp = stamp(12);
  transformHumanPosition(human, buffer, "odom");
  assert(
      !human.fixed_position_valid); // Extrapolation cannot use latest fallback.
  human.position_header.stamp = stamp(0);
  transformHumanPosition(human, buffer, "odom");
  assert(!human.fixed_position_valid);
  human.position_header.stamp = stamp(10);
  human.position_3d = geometry_msgs::msg::Point();
  transformHumanPosition(human, buffer, "odom");
  assert(!human.fixed_position_valid); // Zero is never a valid camera position.
  human.position_valid = false;
  transformHumanPosition(human, buffer, "odom");
  assert(!human.fixed_position_valid);
}

static void depthPayloads() {
  // This is the same cv_bridge conversion used by the tracker, with padding and
  // byte swapping.
  auto payload = std::make_shared<sensor_msgs::msg::Image>();
  payload->width = 2;
  payload->height = 2;
  payload->step = 6;
  payload->encoding = "16UC1";
  payload->is_bigendian = true;
  payload->data = {0x07, 0xd0, 0x03, 0xe8, 0, 0, 0x0b, 0xb8, 0x0f, 0xa0, 0, 0};
  auto decoded = cv_bridge::toCvShare(payload, payload->encoding);
  assert(decoded->image.at<uint16_t>(0, 0) == 2000);
  assert(decoded->image.at<uint16_t>(1, 1) == 4000);
  payload->data.pop_back();
  bool rejected = false;
  try {
    cv_bridge::toCvShare(payload, payload->encoding);
  } catch (const cv_bridge::Exception &) {
    rejected = true;
  }
  assert(rejected);
}

int main() {
  timestamps();
  depthMatching();
  depthValidity();
  transforms();
  depthPayloads();
  std::cout << "RGB-D timestamp, calibration, depth validity, 2D fallback and "
               "TF checks passed\n";
}
