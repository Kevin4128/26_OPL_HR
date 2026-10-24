#pragma once

#include <cv_bridge/cv_bridge.hpp>
#include <rclcpp/time.hpp>
#include <deque>
#include <cstdlib>

namespace opl_human_vision {

class RgbFrameBuffer {
public:
  void clear() {
    frames_.clear();
    last_clock_ = -1;
  }

  void push(const cv_bridge::CvImageConstPtr& frame, int64_t now) {
    checkClock(now);
    frames_.push_back(frame);
    if (frames_.size() > 30) frames_.pop_front();
  }

  cv_bridge::CvImageConstPtr nearest(const std_msgs::msg::Header& target, int64_t now) {
    checkClock(now);
    cv_bridge::CvImageConstPtr best;
    int64_t distance = 100000001;
    for (const auto& frame : frames_) {
      if (frame->image.empty() || frame->header.frame_id != target.frame_id) continue;
      const auto diff = std::llabs(rclcpp::Time(frame->header.stamp).nanoseconds() -
        rclcpp::Time(target.stamp).nanoseconds());
      if (diff <= 100000000 && diff < distance) {
        best = frame;
        distance = diff;
      }
    }
    return best;
  }

private:
  void checkClock(int64_t now) {
    if (last_clock_ >= 0 && now < last_clock_) frames_.clear();
    last_clock_ = now;
  }

  std::deque<cv_bridge::CvImageConstPtr> frames_;
  int64_t last_clock_{-1};
};

}  // namespace opl_human_vision
