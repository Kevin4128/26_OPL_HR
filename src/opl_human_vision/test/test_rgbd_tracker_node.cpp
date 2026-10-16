#include <cassert>
#include <chrono>
#include <future>
#include <iostream>

#include <lifecycle_msgs/msg/state.hpp>
#include "opl_human_vision/human_tracker_node.hpp"

using namespace std::chrono_literals;
using Array = opl_interfaces::msg::TrackedHumanArray;

int main(int argc, char ** argv) {
  rclcpp::init(argc, argv);
  auto options = rclcpp::NodeOptions().arguments({"--ros-args", "-r", "__ns:=/rgbd_m1_test"});
  options.parameter_overrides({
    {"image_topic", "rgb"}, {"depth_topic", "depth"}, {"camera_info_topic", "info"},
    {"min_hits", 1}, {"max_publish_age", 1}, {"appearance_reid_enabled", false},
    {"max_observation_age_sec", 5.0}, {"observation_buffer_size", 10}});
  auto tracker = std::make_shared<opl_human_vision::HumanTrackerNode>(options);
  assert(tracker->configure().id() == lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  assert(tracker->activate().id() == lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);
  auto source = std::make_shared<rclcpp::Node>("rgbd_m1_source", "/rgbd_m1_test");
  const auto qos = rclcpp::SensorDataQoS().keep_last(10);
  auto rgb_pub = source->create_publisher<sensor_msgs::msg::Image>("rgb", qos);
  auto depth_pub = source->create_publisher<sensor_msgs::msg::Image>("depth", qos);
  auto info_pub = source->create_publisher<sensor_msgs::msg::CameraInfo>("info", qos);
  auto detections_pub = source->create_publisher<Array>("raw_detections", qos);
  std::promise<Array> output;
  auto subscription = source->create_subscription<Array>("tracked_humans", qos,
    [&](Array::ConstSharedPtr message) {output.set_value(*message);});
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(tracker->get_node_base_interface()); executor.add_node(source);

  sensor_msgs::msg::Image rgb;
  rgb.header.frame_id = "rgbd_m1_test_optical";
  rgb.width = rgb.height = 100; rgb.encoding = "bgr8"; rgb.step = 300;
  rgb.data.resize(30000, 128);
  Array detections;
  opl_interfaces::msg::TrackedHuman human;
  human.bbox.width = 50; human.bbox.height = 80; human.confidence = 0.9;
  detections.humans.push_back(human);

  auto receive = [&]() {
    auto result = output.get_future();
    assert(executor.spin_until_future_complete(result, 2s) == rclcpp::FutureReturnCode::SUCCESS);
    auto message = result.get();
    output = std::promise<Array>();
    assert(message.humans.size() == 1);
    return message;
  };

  // Detection can arrive before RGB; no depth or calibration is needed for 2D output.
  rgb.header.stamp = source->now(); detections.header = rgb.header;
  detections_pub->publish(detections); rgb_pub->publish(rgb);
  auto result = receive();
  assert(result.header == rgb.header);
  assert(!result.humans[0].position_valid && !result.humans[0].fixed_position_valid);
  assert(result.humans[0].position_3d.z == 0);

  sensor_msgs::msg::CameraInfo info;
  info.header.frame_id = rgb.header.frame_id;
  info.width = info.height = 100; info.k = {100, 0, 50, 0, 100, 50, 0, 0, 1};
  info_pub->publish(info);
  rgb.header.stamp = source->now(); detections.header = rgb.header;
  sensor_msgs::msg::Image depth = rgb;
  depth.encoding = "16UC1"; depth.step = 200; depth.data.resize(20000);
  for (size_t i = 0; i < depth.data.size(); i += 2) {depth.data[i] = 0xd0; depth.data[i + 1] = 0x07;}
  rgb_pub->publish(rgb); depth_pub->publish(depth); detections_pub->publish(detections);
  result = receive();
  const auto & positioned = result.humans[0];
  assert(positioned.position_valid && std::abs(positioned.position_3d.z - 2) < 1e-6);
  assert(positioned.position_header == depth.header && positioned.depth_header == depth.header);
  assert(positioned.depth_valid_pixels >= 25);
  assert(!positioned.fixed_position_valid);  // No TF for this unique test optical frame.

  // An unsupported depth encoding must not prevent the next visual tracker output.
  rgb.header.stamp = source->now(); detections.header = rgb.header;
  depth.header = rgb.header; depth.encoding = "mono8"; depth.step = 100; depth.data.resize(10000);
  rgb_pub->publish(rgb); depth_pub->publish(depth); detections_pub->publish(detections);
  result = receive();
  assert(!result.humans[0].position_valid && result.humans[0].position_3d.z == 0);
  // Predicted boxes are visible briefly but never carry fresh measured 3D.
  rgb.header.stamp=source->now();detections.header=rgb.header;detections.humans.clear();
  rgb_pub->publish(rgb);detections_pub->publish(detections);
  result=receive();
  assert(!result.humans[0].observed && !result.humans[0].position_valid && !result.humans[0].fixed_position_valid);
  assert(result.humans[0].depth_valid_pixels==0 && result.humans[0].position_3d.z==0);
  assert(tracker->deactivate().id() == lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  assert(tracker->cleanup().id() == lifecycle_msgs::msg::State::PRIMARY_STATE_UNCONFIGURED);
  executor.remove_node(tracker->get_node_base_interface()); executor.remove_node(source);
  subscription.reset(); tracker.reset(); source.reset();
  rclcpp::shutdown();
  std::cout << "Synthetic ROS tracker: late RGB, missing depth, calibrated 3D, TF failure and bad depth passed\n";
}
