#include "opl_human_vision/rgb_frame_buffer.hpp"
#include <cassert>

int main() {
  using namespace opl_human_vision;
  auto frame = [](int64_t ns, const std::string& name = "optical") {
    auto f = std::make_shared<cv_bridge::CvImage>();
    f->header.frame_id = name;
    f->header.stamp = rclcpp::Time(ns);
    f->image = cv::Mat::zeros(4, 4, CV_8UC3);
    return f;
  };
  RgbFrameBuffer buffer;
  auto a = frame(1000000000), b = frame(1080000000), c = frame(1040000000);
  buffer.push(a, 2000000000);
  buffer.push(b, 2000000000);
  buffer.push(c, 2000000000); // out-of-order delivery is permitted
  auto target = frame(1070000000)->header;
  assert(buffer.nearest(target, 2000000000) == b);
  target.stamp = rclcpp::Time(1180000000);
  assert(buffer.nearest(target, 2000000000) == b); // inclusive 100ms
  target.stamp = rclcpp::Time(1180000001);
  assert(!buffer.nearest(target, 2000000000));
  target = b->header;
  target.frame_id = "another_camera";
  assert(!buffer.nearest(target, 2000000000));
  buffer.clear();
  buffer.push(a, 2000000000);
  for (int i = 0; i < 30; ++i) buffer.push(frame(2000000000 + i), 2000000000);
  assert(!buffer.nearest(a->header, 2000000000)); // evicted beyond 30 entries
  assert(!buffer.nearest(frame(2000000000)->header, 1000000000)); // backward ROS clock clears
  buffer.push(a, 1000000000);
  assert(buffer.nearest(a->header, 1000000000) == a);
  buffer.clear();
  assert(!buffer.nearest(a->header, 1000000000));
}
