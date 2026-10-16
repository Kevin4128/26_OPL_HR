#pragma once

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <deque>
#include <optional>
#include <string>
#include <vector>

#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <opl_interfaces/msg/tracked_human_array.hpp>

namespace opl_human_vision {

inline double steadySeconds() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

inline int64_t stampNs(const builtin_interfaces::msg::Time & stamp) {
  if (stamp.sec < 0 || stamp.nanosec >= 1000000000u) return 0;
  return int64_t(stamp.sec) * 1000000000 + stamp.nanosec;
}

inline std::string opticalFrame(const std::string & frame) {
  const auto first = frame.find_first_not_of('/');
  return first == std::string::npos ? "" : frame.substr(first);
}

struct ObservationSettings {
  size_t capacity{60};
  double max_age_sec{1.0};
  double wait_sec{0.1};
  double depth_slop_sec{0.02};
  double depth_offset_sec{0.0};  // Added to depth stamps to express ROS/RGB time.
  std::vector<std::string> registered_frame_aliases;

  bool valid() const {
    return capacity > 0 && capacity <= 1000 && std::isfinite(max_age_sec) &&
      max_age_sec > 0 && std::isfinite(wait_sec) && wait_sec >= 0 &&
      wait_sec < max_age_sec && std::isfinite(depth_slop_sec) &&
      depth_slop_sec >= 0 && depth_slop_sec < max_age_sec &&
      std::isfinite(depth_offset_sec) && std::abs(depth_offset_sec) < 3600;
  }
};

struct MatchedObservation {
  opl_interfaces::msg::TrackedHumanArray::ConstSharedPtr observation;
  sensor_msgs::msg::Image::ConstSharedPtr rgb;
  sensor_msgs::msg::Image::ConstSharedPtr depth;
  sensor_msgs::msg::CameraInfo::ConstSharedPtr calibration;
};

// Call under the owning node's mutex. Steady time bounds waits even if ROS time stops.
class ObservationBuffer {
public:
  ObservationSettings settings;

  uint64_t epoch() const {return epoch_;}
  void clear() {
    ++epoch_;
    rgb_.clear(); depth_.clear(); pending_.clear(); calibration_.reset();
    last_completed_ = 0;
    last_clock_ = 0;
    rgb_frame_.clear(); width_ = height_ = 0;
  }

  bool observeClock(int64_t now) {
    const bool reset = last_clock_ > 0 && now < last_clock_;
    if (reset) clear();
    last_clock_ = now;
    return reset;
  }

  bool addImage(const sensor_msgs::msg::Image::ConstSharedPtr & image,
    bool depth, int64_t now, double steady)
  {
    observeClock(now);
    const auto stamp = stampNs(image->header.stamp);
    const auto corrected = stamp + (depth ? depthOffsetNs() : 0);
    if (stamp <= 0 || !fresh(corrected, now) || opticalFrame(image->header.frame_id).empty() ||
      image->width == 0 || image->height == 0) return false;
    if (!depth && (!rgb_frame_.empty()) &&
      (rgb_frame_ != opticalFrame(image->header.frame_id) ||
      width_ != image->width || height_ != image->height))
    {
      // Reconfiguration cannot reuse observations or calibration from another stream.
      clear(); last_clock_ = now;
    }
    if (!depth) {
      rgb_frame_ = opticalFrame(image->header.frame_id);
      width_ = image->width; height_ = image->height;
    }
    auto & images = depth ? depth_ : rgb_;
    prune(images, steady);
    for (const auto & item : images) {
      if (stampNs(item.image->header.stamp) == stamp &&
        opticalFrame(item.image->header.frame_id) == opticalFrame(image->header.frame_id))
        return false;
    }
    images.push_back({image, steady});
    while (images.size() > settings.capacity) images.pop_front();
    return true;
  }

  void setCalibration(const sensor_msgs::msg::CameraInfo::ConstSharedPtr & info,
    int64_t now)
  {
    observeClock(now);
    if (calibration_) {
      auto previous = *calibration_;
      auto incoming = *info;
      previous.header.stamp = incoming.header.stamp;
      if (previous == incoming) return;  // Static calibration need not share image stamps.
      if (stampNs(info->header.stamp) > 0 && stampNs(calibration_->header.stamp) >
        stampNs(info->header.stamp)) return;  // Delayed old calibration.
      ++epoch_;
      rgb_.clear(); depth_.clear(); pending_.clear();
    }
    calibration_ = info;
  }

  bool enqueue(const opl_interfaces::msg::TrackedHumanArray::ConstSharedPtr & observation,
    int64_t now, double steady)
  {
    observeClock(now);
    const auto stamp = stampNs(observation->header.stamp);
    if (stamp <= last_completed_ || !fresh(stamp, now) ||
      opticalFrame(observation->header.frame_id).empty()) return false;
    for (const auto & item : pending_)
      if (stampNs(item.observation->header.stamp) == stamp) return false;
    // Reject overflow rather than extending the bounded waiting time.
    if (pending_.size() >= settings.capacity) return false;
    auto where = std::upper_bound(pending_.begin(), pending_.end(), stamp,
      [](int64_t value, const Pending & p) {return value < stampNs(p.observation->header.stamp);});
    // Reordering must not postpone a successor's already-running deadline.
    const double received = where == pending_.end() ? steady : std::min(steady, where->received);
    pending_.insert(where, {observation, received});
    return true;
  }

  std::optional<MatchedObservation> take(int64_t now, double steady, bool want_depth) {
    observeClock(now);
    prune(rgb_, steady); prune(depth_, steady);
    while (!pending_.empty()) {
      const auto & pending = pending_.front();
      const auto & header = pending.observation->header;
      const auto stamp = stampNs(header.stamp);
      if (!fresh(stamp, now) || stamp <= last_completed_) {
        pending_.pop_front(); continue;
      }
      MatchedObservation result;
      result.observation = pending.observation;
      for (const auto & item : rgb_) {
        if (stampNs(item.image->header.stamp) == stamp &&
          opticalFrame(item.image->header.frame_id) == opticalFrame(header.frame_id))
          result.rgb = item.image;
      }
      int64_t best = int64_t(settings.depth_slop_sec * 1e9) + 1;
      bool ambiguous = false;
      if (want_depth && result.rgb) {
        for (const auto & item : depth_) {
          if (!registeredFrame(item.image->header.frame_id, header.frame_id) ||
            item.image->width != result.rgb->width || item.image->height != result.rgb->height)
            continue;
          const auto corrected = stampNs(item.image->header.stamp) + depthOffsetNs();
          if (!fresh(corrected, now)) continue;
          const auto diff = std::abs(corrected - stamp);
          if (diff < best) {best = diff; result.depth = item.image; ambiguous = false;}
          else if (diff == best) ambiguous = true;
        }
        if (ambiguous) result.depth.reset();
        result.calibration = calibration_;
      }
      const bool expired = steady - pending.received >= settings.wait_sec;
      if (!expired && (!result.rgb || (want_depth && (!result.depth || !result.calibration))))
        return std::nullopt;
      last_completed_ = stamp;
      pending_.pop_front();
      return result;
    }
    return std::nullopt;
  }

  bool registeredFrame(const std::string & candidate, const std::string & rgb) const {
    if (opticalFrame(candidate).empty() || opticalFrame(rgb).empty()) return false;
    if (opticalFrame(candidate) == opticalFrame(rgb)) return true;
    return std::any_of(settings.registered_frame_aliases.begin(),
      settings.registered_frame_aliases.end(), [&](const auto & alias) {
        return opticalFrame(candidate) == opticalFrame(alias);
      });
  }

  int64_t depthOffsetNs() const {return int64_t(settings.depth_offset_sec * 1e9);}

private:
  struct Image {sensor_msgs::msg::Image::ConstSharedPtr image; double received;};
  struct Pending {opl_interfaces::msg::TrackedHumanArray::ConstSharedPtr observation; double received;};
  std::deque<Image> rgb_, depth_;
  std::deque<Pending> pending_;
  sensor_msgs::msg::CameraInfo::ConstSharedPtr calibration_;
  uint64_t epoch_{0};
  int64_t last_completed_{0}, last_clock_{0};
  std::string rgb_frame_;
  uint32_t width_{0}, height_{0};

  bool fresh(int64_t stamp, int64_t now) const {
    return stamp > 0 && now > 0 && stamp <= now + int64_t(settings.wait_sec * 1e9) &&
      now - stamp <= int64_t(settings.max_age_sec * 1e9);
  }

  void prune(std::deque<Image> & images, double steady) {
    while (!images.empty() && steady - images.front().received > settings.max_age_sec)
      images.pop_front();
  }
};

}  // namespace opl_human_vision
