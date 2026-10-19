#pragma once

#include <memory>
#include <rclcpp_lifecycle/lifecycle_node.hpp>

namespace opl_human_vision {
class ObjectDetectionNode : public rclcpp_lifecycle::LifecycleNode {
public:
  explicit ObjectDetectionNode(const rclcpp::NodeOptions &options=rclcpp::NodeOptions());
  ~ObjectDetectionNode() override;
  using Return=rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;
  Return on_configure(const rclcpp_lifecycle::State &) override;
  Return on_activate(const rclcpp_lifecycle::State &) override;
  Return on_deactivate(const rclcpp_lifecycle::State &) override;
  Return on_cleanup(const rclcpp_lifecycle::State &) override;
  Return on_shutdown(const rclcpp_lifecycle::State &) override;
private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}
