#ifndef OPL_HUMAN_VISION__HUMAN_DETECTOR_NODE_HPP_
#define OPL_HUMAN_VISION__HUMAN_DETECTOR_NODE_HPP_

#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/core.hpp>

#include "opl_interfaces/msg/tracked_human_array.hpp"
#include "opl_human_vision/hailo_model.hpp"
#include "opl_human_vision/yolov8_pose_decoder.hpp"

namespace opl_human_vision {

// YOLOv8-pose person detector running on the Hailo-8L.
class HumanDetectorNode : public rclcpp_lifecycle::LifecycleNode {
public:
  explicit HumanDetectorNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());
  ~HumanDetectorNode() override = default;

  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
  on_configure(const rclcpp_lifecycle::State & state) override;

  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
  on_activate(const rclcpp_lifecycle::State & state) override;

  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
  on_deactivate(const rclcpp_lifecycle::State & state) override;

  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
  on_cleanup(const rclcpp_lifecycle::State & state) override;

private:
  void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr msg);

  std::string image_topic_;
  std::string detections_topic_;
  std::string model_path_;

  float conf_thresh_{0.45f};
  float nms_thresh_{0.50f};
  double max_inference_hz_{0.0};
  std::chrono::steady_clock::time_point next_inference_{};

  std::unique_ptr<HailoModel> model_;
  YoloV8PoseDecoder decoder_;
  std::vector<PoseCandidate> candidates_;

  rclcpp::CallbackGroup::SharedPtr image_cb_group_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp_lifecycle::LifecyclePublisher<opl_interfaces::msg::TrackedHumanArray>::SharedPtr detection_pub_;
};

}  // namespace opl_human_vision

#endif  // OPL_HUMAN_VISION__HUMAN_DETECTOR_NODE_HPP_
