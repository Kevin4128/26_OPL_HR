#ifndef OPL_HUMAN_VISION__HUMAN_TRACKER_NODE_HPP_
#define OPL_HUMAN_VISION__HUMAN_TRACKER_NODE_HPP_

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>

#include <geometry_msgs/msg/point.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/string.hpp>

#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/opencv.hpp>

#include <mutex>

#include "opl_human_vision/human_position.hpp"
#include "opl_human_vision/tracker_engine.hpp"
#include "opl_interfaces/msg/tracked_human_array.hpp"
#include <tf2_ros/transform_listener.hpp>

namespace opl_human_vision {

class HumanTrackerNode : public rclcpp_lifecycle::LifecycleNode {
public:
  explicit HumanTrackerNode(
      const rclcpp::NodeOptions &options = rclcpp::NodeOptions());

  ~HumanTrackerNode() override = default;

  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
  on_configure(const rclcpp_lifecycle::State &) override;

  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
  on_activate(const rclcpp_lifecycle::State &) override;

  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
  on_deactivate(const rclcpp_lifecycle::State &) override;

  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
  on_cleanup(const rclcpp_lifecycle::State &) override;

private:
  void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr msg);

  void depthCallback(const sensor_msgs::msg::Image::ConstSharedPtr msg);

  void
  cameraInfoCallback(const sensor_msgs::msg::CameraInfo::ConstSharedPtr msg);

  void detectionsCallback(
      const opl_interfaces::msg::TrackedHumanArray::ConstSharedPtr msg);

  void drainObservations();
  void processObservation(const MatchedObservation &observation);

  std::string image_topic_;
  std::string depth_topic_;
  std::string detections_topic_;
  std::string tracked_topic_;

  rclcpp::CallbackGroup::SharedPtr image_cb_group_;

  rclcpp::CallbackGroup::SharedPtr depth_cb_group_;

  rclcpp::CallbackGroup::SharedPtr track_cb_group_;

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_sub_;

  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr info_sub_;

  rclcpp::Subscription<opl_interfaces::msg::TrackedHumanArray>::SharedPtr
      detection_sub_;

  rclcpp_lifecycle::LifecyclePublisher<
      opl_interfaces::msg::TrackedHumanArray>::SharedPtr tracked_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr depth_diagnostics_pub_;

  std::shared_ptr<TrackerEngine> tracker_;

  std::mutex observation_mutex_;
  ObservationBuffer observations_;
  DepthSettings depth_settings_;
  uint64_t tracker_epoch_{0};
  bool depth_registered_{true};
  std::string fixed_frame_{"odom"};
  rclcpp::TimerBase::SharedPtr observation_timer_;
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
};

} // namespace opl_human_vision

#endif // OPL_HUMAN_VISION__HUMAN_TRACKER_NODE_HPP_