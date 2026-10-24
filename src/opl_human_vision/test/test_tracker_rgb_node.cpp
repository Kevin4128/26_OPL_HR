#include "opl_human_vision/human_tracker_node.hpp"
#include <cassert>
#include <thread>

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  {
    rclcpp::NodeOptions invalid;
    invalid.parameter_overrides({rclcpp::Parameter("track_high_threshold", 0.05)});
    auto node = std::make_shared<opl_human_vision::HumanTrackerNode>(invalid);
    assert(node->configure().label() == "unconfigured");
  }
  rclcpp::NodeOptions options;
  options.parameter_overrides({rclcpp::Parameter("min_hits", 1),
    rclcpp::Parameter("appearance_reid_enabled", false),
    rclcpp::Parameter("image_topic", "rgb_test/image"),
    rclcpp::Parameter("depth_topic", "rgb_test/depth"),
    rclcpp::Parameter("camera_info_topic", "rgb_test/info"),
    rclcpp::Parameter("detections_topic", "rgb_test/detections"),
    rclcpp::Parameter("tracked_topic", "rgb_test/tracks")});
  auto tracker = std::make_shared<opl_human_vision::HumanTrackerNode>(options);
  assert(tracker->configure().label() == "inactive");
  tracker->activate();
  auto probe = std::make_shared<rclcpp::Node>("tracker_rgb_test");
  auto detections = probe->create_publisher<opl_interfaces::msg::TrackedHumanArray>("rgb_test/detections", rclcpp::SensorDataQoS());
  auto images = probe->create_publisher<sensor_msgs::msg::Image>("rgb_test/image", rclcpp::SensorDataQoS());
  auto depths = probe->create_publisher<sensor_msgs::msg::Image>("rgb_test/depth", rclcpp::SensorDataQoS());
  auto infos = probe->create_publisher<sensor_msgs::msg::CameraInfo>("rgb_test/info", rclcpp::SensorDataQoS());
  int count = 0;
  opl_interfaces::msg::TrackedHumanArray output;
  auto subscription = probe->create_subscription<opl_interfaces::msg::TrackedHumanArray>(
    "rgb_test/tracks", rclcpp::SensorDataQoS(),
    [&](opl_interfaces::msg::TrackedHumanArray::ConstSharedPtr msg) { output = *msg; ++count; });
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(tracker->get_node_base_interface());
  executor.add_node(probe);
  auto spin = [&]() {
    executor.spin_some();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    executor.spin_some();
  };
  for (int i = 0; i < 200 && (images->get_subscription_count() == 0 ||
      detections->get_subscription_count() == 0 || depths->get_subscription_count() == 0 ||
      infos->get_subscription_count() == 0); ++i) spin();
  opl_interfaces::msg::TrackedHumanArray input;
  input.header.frame_id = "camera_color_optical_frame";
  input.humans.resize(1);
  auto& h = input.humans.front();
  h.bbox.x_offset = 10; h.bbox.y_offset = 10;
  h.bbox.width = 20; h.bbox.height = 40; h.confidence = 0.9;
  auto exchange = [&]() {
    const auto before = count;
    for (int i = 0; i < 100 && count == before; ++i) { detections->publish(input); spin(); }
    assert(count > before && output.humans.size() == 1);
    assert(output.humans.front().observed);
  };
  input.header.stamp = probe->now();
  exchange(); // geometry must work before any RGB is received
  const auto id = output.humans.front().raw_track_id;
  assert(!output.humans.front().position_valid);
  for (bool matched : {false, true}) {
    input.header.stamp = probe->now();
    auto rgb_header = input.header;
    if (!matched) rgb_header.stamp = rclcpp::Time(input.header.stamp) - rclcpp::Duration::from_seconds(0.2);
    auto rgb = cv_bridge::CvImage(rgb_header, "bgr8", cv::Mat::zeros(100, 100, CV_8UC3)).toImageMsg();
    auto depth = cv_bridge::CvImage(input.header, "32FC1", cv::Mat(100, 100, CV_32FC1, cv::Scalar(2.0))).toImageMsg();
    sensor_msgs::msg::CameraInfo info;
    info.header = input.header; info.width = 100; info.height = 100;
    info.k = {100, 0, 50, 0, 100, 50, 0, 0, 1};
    images->publish(*rgb); depths->publish(*depth); infos->publish(info);
    for (int i = 0; i < 5; ++i) spin();
    exchange();
    assert(output.humans.front().raw_track_id == id);
    assert(output.humans.front().position_valid == matched);
  }
  input.header.stamp = probe->now();
  h.confidence = 0.3;
  exchange(); // the ROS node must pass weak observations to BoT-SORT
  assert(output.humans.front().raw_track_id == id);
  tracker->deactivate(); tracker->cleanup();
  rclcpp::shutdown();
}
