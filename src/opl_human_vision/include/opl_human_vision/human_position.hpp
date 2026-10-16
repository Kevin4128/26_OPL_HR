#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opl_interfaces/msg/tracked_human.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_ros/buffer.hpp>

#include "opl_human_vision/observation_buffer.hpp"

namespace opl_human_vision {

struct DepthSettings {
  int min_pixels{25};
  double min_fraction{0.25};
  double min_depth{0.2}, max_depth{10.0};
  double max_spread_m{0.3}, max_spread_ratio{0.15};

  bool valid() const {
    return min_pixels > 0 && std::isfinite(min_fraction) && min_fraction > 0 &&
           min_fraction <= 1 && std::isfinite(min_depth) &&
           std::isfinite(max_depth) && min_depth > 0 && max_depth > min_depth &&
           std::isfinite(max_spread_m) && max_spread_m >= 0 &&
           std::isfinite(max_spread_ratio) && max_spread_ratio >= 0;
  }
};

inline bool validCalibration(const sensor_msgs::msg::CameraInfo &info,
                             const sensor_msgs::msg::Image &rgb,
                             const ObservationBuffer &buffer) {
  if (info.header.stamp.sec < 0 || info.header.stamp.nanosec >= 1000000000u ||
      info.width != rgb.width || info.height != rgb.height ||
      !buffer.registeredFrame(info.header.frame_id, rgb.header.frame_id) ||
      info.binning_x > 1 || info.binning_y > 1 || info.roi.x_offset ||
      info.roi.y_offset || (info.roi.width && info.roi.width != info.width) ||
      (info.roi.height && info.roi.height != info.height))
    return false;
  for (const double value : info.k)
    if (!std::isfinite(value))
      return false;
  for (const double value : info.d)
    if (!std::isfinite(value))
      return false;
  if (info.k[0] <= 0 || info.k[4] <= 0 || info.k[2] < 0 ||
      info.k[2] >= info.width || info.k[5] < 0 || info.k[5] >= info.height ||
      info.k[8] != 1.0 || info.k[1] != 0 || info.k[3] != 0 || info.k[6] != 0 ||
      info.k[7] != 0)
    return false;
  const bool distorted = std::any_of(info.d.begin(), info.d.end(),
                                     [](double x) { return x != 0; });
  return !distorted ||
         (info.distortion_model == "plumb_bob" &&
          (info.d.size() == 4 || info.d.size() == 5)) ||
         (info.distortion_model == "rational_polynomial" &&
          info.d.size() == 8) ||
         (info.distortion_model == "equidistant" && info.d.size() == 4);
}

struct DepthDiagnostics {
  std::string reason{"invalid_payload_or_box"};
  double median{0}, p10{0}, p90{0}, allowed_spread{0};
  double cluster_median{0}, cluster_p10{0}, cluster_p90{0};
  uint32_t cluster_pixels{0};
};

inline void estimateHumanPosition(const cv::Mat &depth, const cv::Rect &bbox,
                                  const sensor_msgs::msg::CameraInfo &info,
                                  const DepthSettings &settings,
                                  opl_interfaces::msg::TrackedHuman &human,
                                  DepthDiagnostics *diagnostics = nullptr) {
  DepthDiagnostics unused;
  auto &diagnostic = diagnostics ? *diagnostics : unused;
  diagnostic = DepthDiagnostics{};
  human.position_valid = false;
  human.position_3d = geometry_msgs::msg::Point();
  human.depth_valid_pixels = 0;
  human.depth_valid_fraction = 0;
  human.depth_mad_m = 0;
  if (!settings.valid() || depth.empty() ||
      (depth.type() != CV_16UC1 && depth.type() != CV_32FC1) ||
      bbox.width <= 0 || bbox.height <= 0)
    return;
  // Sample the central torso, not legs or the box's background edges.
  const cv::Rect torso(
      bbox.x + int(bbox.width * 0.3), bbox.y + int(bbox.height * 0.2),
      std::max(1, int(bbox.width * 0.4)), std::max(1, int(bbox.height * 0.4)));
  const auto roi = torso & cv::Rect(0, 0, depth.cols, depth.rows);
  diagnostic.reason = "torso_outside_image";
  if (roi.empty())
    return;
  std::vector<float> samples;
  samples.reserve(roi.area());
  for (int y = roi.y; y < roi.y + roi.height; ++y) {
    for (int x = roi.x; x < roi.x + roi.width; ++x) {
      const float z = depth.type() == CV_16UC1
                          ? depth.at<uint16_t>(y, x) * 0.001f
                          : depth.at<float>(y, x);
      if (std::isfinite(z) && z > settings.min_depth && z < settings.max_depth)
        samples.push_back(z);
    }
  }
  human.depth_valid_pixels = samples.size();
  human.depth_valid_fraction = float(samples.size()) / roi.area();
  diagnostic.reason = "insufficient_valid_pixels_or_fraction";
  if (samples.size() < size_t(settings.min_pixels) ||
      human.depth_valid_fraction < settings.min_fraction)
    return;
  std::sort(samples.begin(), samples.end());
  const size_t n = samples.size();
  const double z = (samples[(n - 1) / 2] + samples[n / 2]) * 0.5;
  diagnostic.median = z;
  diagnostic.p10 = samples[(n - 1) / 10];
  diagnostic.p90 = samples[(n - 1) * 9 / 10];
  diagnostic.allowed_spread =
      std::max(settings.max_spread_m, z * settings.max_spread_ratio);
  diagnostic.cluster_median = z;
  diagnostic.cluster_p10 = diagnostic.p10;
  diagnostic.cluster_p90 = diagnostic.p90;
  diagnostic.cluster_pixels = n;
  // Split separated depth surfaces without trimming a continuous wide surface.
  size_t group_start = 0, best_start = 0, best_end = 0;
  for (size_t i = 1; i <= n; ++i) {
    if (i < n &&
        samples[i] - samples[i - 1] <= std::max(.08f, samples[i - 1] * .04f))
      continue;
    if (i - group_start > best_end - best_start) {
      best_start = group_start;
      best_end = i;
    }
    group_start = i;
  }
  diagnostic.reason = "ambiguous_depth_clusters";
  if (best_end - best_start < .6 * n)
    return;
  cv::Mat mask(roi.size(), CV_8UC1, cv::Scalar(0));
  for (int y = 0; y < roi.height; ++y)
    for (int x = 0; x < roi.width; ++x) {
      const float value = depth.type() == CV_16UC1
                              ? depth.at<uint16_t>(roi.y + y, roi.x + x) * .001f
                              : depth.at<float>(roi.y + y, roi.x + x);
      if (std::isfinite(value) && value >= samples[best_start] &&
          value <= samples[best_end - 1])
        mask.at<uint8_t>(y, x) = 255;
    }
  cv::Mat labels, component_stats, centroids;
  const int components = cv::connectedComponentsWithStats(
      mask, labels, component_stats, centroids, 4, CV_32S);
  int selected = 0, selected_count = 0;
  for (int id = 1; id < components; ++id)
    if (component_stats.at<int>(id, cv::CC_STAT_AREA) > selected_count) {
      selected = id;
      selected_count = component_stats.at<int>(id, cv::CC_STAT_AREA);
    }
  diagnostic.cluster_pixels = selected_count;
  diagnostic.reason = "insufficient_connected_torso_support";
  if (selected_count < settings.min_pixels || selected_count < .6 * n ||
      double(selected_count) / roi.area() < settings.min_fraction)
    return;
  // Upper central torso must support the selected surface, not just background
  // edges or lower furniture.
  const cv::Rect core(roi.width / 4, 0, std::max(1, roi.width / 2),
                      std::max(1, roi.height / 2));
  int core_valid = 0, core_selected = 0;
  for (int y = core.y; y < core.br().y; ++y)
    for (int x = core.x; x < core.br().x; ++x) {
      const float value = depth.type() == CV_16UC1
                              ? depth.at<uint16_t>(roi.y + y, roi.x + x) * .001f
                              : depth.at<float>(roi.y + y, roi.x + x);
      if (std::isfinite(value) && value > settings.min_depth &&
          value < settings.max_depth)
        ++core_valid;
      if (labels.at<int>(y, x) == selected)
        ++core_selected;
    }
  diagnostic.reason = "selected_surface_lacks_upper_torso_support";
  if (core_selected < core.area() * settings.min_fraction || !core_valid ||
      core_selected < .5 * core_valid)
    return;
  std::vector<float> cluster;
  cluster.reserve(selected_count);
  for (int y = 0; y < roi.height; ++y)
    for (int x = 0; x < roi.width; ++x)
      if (labels.at<int>(y, x) == selected)
        cluster.push_back(depth.type() == CV_16UC1
                              ? depth.at<uint16_t>(roi.y + y, roi.x + x) * .001f
                              : depth.at<float>(roi.y + y, roi.x + x));
  std::sort(cluster.begin(), cluster.end());
  const size_t count = cluster.size();
  const double selected_z =
      (cluster[(count - 1) / 2] + cluster[count / 2]) * .5;
  diagnostic.cluster_median = selected_z;
  diagnostic.cluster_p10 = cluster[(count - 1) / 10];
  diagnostic.cluster_p90 = cluster[(count - 1) * 9 / 10];
  diagnostic.allowed_spread =
      std::max(settings.max_spread_m, selected_z * settings.max_spread_ratio);
  std::vector<float> deviations;
  for (float value : cluster)
    deviations.push_back(std::abs(value - selected_z));
  std::nth_element(deviations.begin(), deviations.begin() + count / 2,
                   deviations.end());
  human.depth_mad_m = deviations[count / 2];
  diagnostic.reason = "selected_cluster_depth_spread";
  if (diagnostic.cluster_p90 - diagnostic.cluster_p10 >
      diagnostic.allowed_spread)
    return;
  cv::Point2d pixel(roi.x + (roi.width - 1) * .5,
                    roi.y + (roi.height - 1) * .5);
  cv::Point2d chosen;
  double nearest = std::numeric_limits<double>::infinity();
  for (int y = 0; y < roi.height; ++y)
    for (int x = 0; x < roi.width; ++x)
      if (labels.at<int>(y, x) == selected) {
        const float value =
            depth.type() == CV_16UC1
                ? depth.at<uint16_t>(roi.y + y, roi.x + x) * .001f
                : depth.at<float>(roi.y + y, roi.x + x);
        if (std::abs(value - selected_z) >
            std::max(.02, 3. * human.depth_mad_m))
          continue;
        const double distance =
            std::hypot(roi.x + x - pixel.x, roi.y + y - pixel.y);
        if (distance < nearest) {
          nearest = distance;
          chosen = {double(roi.x + x), double(roi.y + y)};
        }
      }
  diagnostic.reason = "no_representative_cluster_pixel";
  if (!std::isfinite(nearest))
    return;
  // Retain the geometric centre only when its actual pixel belongs to the
  // median surface.
  if (nearest > 1.)
    pixel = chosen;
  diagnostic.reason = "invalid_projection";
  std::vector<cv::Point2d> ray{pixel};
  const cv::Mat k(3, 3, CV_64F, const_cast<double *>(info.k.data()));
  const bool distorted = std::any_of(info.d.begin(), info.d.end(),
                                     [](double x) { return x != 0; });
  if (distorted) {
    if (info.distortion_model == "equidistant")
      cv::fisheye::undistortPoints(ray, ray, k, info.d);
    else
      cv::undistortPoints(ray, ray, k, info.d);
  } else {
    ray[0] = {(pixel.x - info.k[2]) / info.k[0],
              (pixel.y - info.k[5]) / info.k[4]};
  }
  human.position_3d.x = ray[0].x * selected_z;
  human.position_3d.y = ray[0].y * selected_z;
  human.position_3d.z = selected_z;
  human.position_valid = std::isfinite(human.position_3d.x) &&
                         std::isfinite(human.position_3d.y) && selected_z > 0;
  if (!human.position_valid)
    human.position_3d = geometry_msgs::msg::Point();
  else
    diagnostic.reason = "valid";
}

inline void transformHumanPosition(opl_interfaces::msg::TrackedHuman &human,
                                   tf2_ros::Buffer &buffer,
                                   const std::string &target) {
  human.fixed_position_valid = false;
  human.fixed_position = geometry_msgs::msg::PointStamped();
  human.fixed_position.header = human.position_header;
  human.fixed_position.header.frame_id = opticalFrame(target);
  if (!human.position_valid || stampNs(human.position_header.stamp) <= 0 ||
      opticalFrame(human.position_header.frame_id).empty() ||
      opticalFrame(target).empty() || !std::isfinite(human.position_3d.x) ||
      !std::isfinite(human.position_3d.y) ||
      !std::isfinite(human.position_3d.z) || human.position_3d.z <= 0)
    return;
  geometry_msgs::msg::PointStamped source;
  source.header = human.position_header;
  source.header.frame_id = opticalFrame(source.header.frame_id);
  source.point = human.position_3d;
  try {
    // Never use time zero/latest: the robot may have moved since this exposure.
    const auto transform =
        buffer.lookupTransform(opticalFrame(target), source.header.frame_id,
                               rclcpp::Time(source.header.stamp));
    tf2::doTransform(source, human.fixed_position, transform);
    human.fixed_position.header.stamp = source.header.stamp;
    const auto &point = human.fixed_position.point;
    human.fixed_position_valid = std::isfinite(point.x) &&
                                 std::isfinite(point.y) &&
                                 std::isfinite(point.z);
  } catch (const tf2::TransformException &) {
    // Recognition remains usable without a connected or sufficiently old TF
    // tree.
  }
}

} // namespace opl_human_vision
