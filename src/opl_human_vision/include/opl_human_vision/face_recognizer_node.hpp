#pragma once

#include <opl_interfaces/msg/enrollment_status.hpp>
#include <opl_interfaces/msg/identity_status_array.hpp>
#include <opl_interfaces/msg/tracked_human_array.hpp>
#include <opl_interfaces/srv/delete_identity.hpp>
#include <opl_interfaces/srv/enroll_person.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>
#include <sensor_msgs/msg/image.hpp>

#include "opl_human_vision/enrollment_queue.hpp"
#include "opl_human_vision/face_engine.hpp"
#include "opl_human_vision/observation_buffer.hpp"

#include <chrono>
#include <mutex>
#include <set>
#include <unordered_map>

namespace opl_human_vision {

class FaceRecognizerNode : public rclcpp_lifecycle::LifecycleNode {
public:
  explicit FaceRecognizerNode(
      const rclcpp::NodeOptions &options = rclcpp::NodeOptions());

  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
  on_configure(const rclcpp_lifecycle::State &state) override;

  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
  on_activate(const rclcpp_lifecycle::State &state) override;

  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
  on_deactivate(const rclcpp_lifecycle::State &state) override;

  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
  on_cleanup(const rclcpp_lifecycle::State &state) override;

private:
  void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr msg);

  void trackedHumansCallback(
      const opl_interfaces::msg::TrackedHumanArray::ConstSharedPtr msg);

  void drainObservations();
  void enrollmentWatchdog();
  void beginEnrollmentTarget();
  void publishEnrollment(const std::string &state, const std::string &message,
                         uint64_t raw = 0, uint64_t identity = 0,
                         uint32_t samples = 0);
  void processObservation(
      const opl_interfaces::msg::TrackedHumanArray::ConstSharedPtr &msg,
      const cv::Mat &frame_ref);

  void handleEnrollService(
      const std::shared_ptr<opl_interfaces::srv::EnrollPerson::Request> request,

      std::shared_ptr<opl_interfaces::srv::EnrollPerson::Response> response);

  std::string image_topic_;
  std::string tracked_topic_;
  std::string recognized_topic_;

  rclcpp::CallbackGroup::SharedPtr image_cb_group_;

  rclcpp::CallbackGroup::SharedPtr tracked_cb_group_;

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;

  rclcpp::Subscription<opl_interfaces::msg::TrackedHumanArray>::SharedPtr
      tracked_sub_;

  rclcpp_lifecycle::LifecyclePublisher<
      opl_interfaces::msg::TrackedHumanArray>::SharedPtr recognized_pub_;

  rclcpp::Service<opl_interfaces::srv::EnrollPerson>::SharedPtr enroll_srv_;

  std::shared_ptr<FaceEngine> face_engine_;

  EnrollmentQueue enrollment_;
  std::vector<opl_interfaces::msg::TrackedHuman> visible_humans_;
  double last_visible_frame_{0};
  double last_observation_time_{0};
  std::string last_observation_frame_;
  rclcpp::Publisher<opl_interfaces::msg::EnrollmentStatus>::SharedPtr
      enrollment_pub_;
  rclcpp::Service<opl_interfaces::srv::DeleteIdentity>::SharedPtr delete_srv_;

  std::mutex engine_mutex_;

  std::mutex observation_mutex_;
  ObservationBuffer observations_;
  rclcpp::TimerBase::SharedPtr observation_timer_;

  double identity_lock_grace_sec_{1.0};
  rclcpp::Publisher<opl_interfaces::msg::IdentityStatusArray>::SharedPtr
      identity_status_pub_;

  // How often a locked identity must be checked with
  // real AdaFace/OSNet inference.
  double identity_validation_interval_sec_{0.5};
};

} // namespace opl_human_vision
