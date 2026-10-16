#include "opl_human_vision/face_recognizer_node.hpp"
#include "opl_human_vision/scoped_timing.hpp"
#include "rclcpp_components/register_node_macro.hpp"

#include <geometry_msgs/msg/point.hpp>
#include <opl_interfaces/srv/enroll_person.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cv_bridge/cv_bridge.hpp>
#include <filesystem>

namespace opl_human_vision {

FaceRecognizerNode::FaceRecognizerNode(const rclcpp::NodeOptions &options)
    : rclcpp_lifecycle::LifecycleNode("face_recognizer_node", options) {
  cv::setNumThreads(declare_parameter<int>("opencv_threads", 1));
  declare_parameter<int>("observation_buffer_size", 60);
  declare_parameter<double>("max_observation_age_sec", 1.0);
  declare_parameter<double>("observation_wait_sec", 0.1);
  declare_parameter<std::string>("image_topic", "/image_raw");
  declare_parameter<std::string>("tracked_topic", "tracked_humans");
  declare_parameter<std::string>("recognized_topic", "recognized_humans");

  declare_parameter<std::string>("identity_store_path",
                                 OPL_IDENTITY_STORE_DEFAULT);
  declare_parameter<double>("enrollment_missing_sec", 2.0);
  declare_parameter<double>("minimum_learning_confidence", 0.45);
  declare_parameter<double>("enrollment_timeout_sec", 30.0);
  declare_parameter<std::string>("scrfd_model_path", "");
  declare_parameter<std::string>("adaface_model_path", "");
  declare_parameter<std::string>("osnet_model_path", "");

  declare_parameter<float>("match_threshold", 0.55f);
  declare_parameter<float>("body_match_threshold", 0.85f);

  declare_parameter<int>("min_confirm_frames", 20);
  declare_parameter<float>("min_confirm_duration_sec", 2.5f);

  // How long a raw-track -> persistent-ID lock is kept alive
  // after the raw track temporarily disappears from published output.
  declare_parameter<double>("identity_lock_grace_sec", 1.0);
  declare_parameter<double>("identity_validation_interval_sec", 0.5);
  declare_parameter<double>("session_reid_timeout_sec", 2.0);
  declare_parameter<double>("reid_spatial_gate_max_age_sec", 2.0);
  declare_parameter<double>("reid_spatial_gate_base_m", 0.75);
  declare_parameter<double>("reid_spatial_gate_max_speed_mps", 2.5);
  declare_parameter<double>("reid_depth_gate_base_m", 0.75);
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
FaceRecognizerNode::on_configure(const rclcpp_lifecycle::State &) {
  RCLCPP_INFO(get_logger(), "Configuring FaceRecognizerNode...");

  observations_.clear();
  auto &settings = observations_.settings;
  const auto capacity = get_parameter("observation_buffer_size").as_int();
  settings.capacity = capacity > 0 ? capacity : 0;
  settings.max_age_sec = get_parameter("max_observation_age_sec").as_double();
  settings.wait_sec = get_parameter("observation_wait_sec").as_double();
  if (!settings.valid()) {
    RCLCPP_ERROR(get_logger(), "Invalid observation buffer parameters");
    return CallbackReturn::FAILURE;
  }

  image_topic_ = get_parameter("image_topic").as_string();

  tracked_topic_ = get_parameter("tracked_topic").as_string();

  recognized_topic_ = get_parameter("recognized_topic").as_string();

  identity_lock_grace_sec_ =
      get_parameter("identity_lock_grace_sec").as_double();

  identity_validation_interval_sec_ =
      get_parameter("identity_validation_interval_sec").as_double();

  if (identity_validation_interval_sec_ < 0.0) {
    identity_validation_interval_sec_ = 0.0;
  }

  const double session_reid_timeout =
      get_parameter("session_reid_timeout_sec").as_double();

  const std::string scrfd_path = get_parameter("scrfd_model_path").as_string();

  const std::string adaface_path =
      get_parameter("adaface_model_path").as_string();

  const std::string osnet_path = get_parameter("osnet_model_path").as_string();

  const float match_thresh =
      static_cast<float>(get_parameter("match_threshold").as_double());

  const float body_thresh =
      static_cast<float>(get_parameter("body_match_threshold").as_double());

  const int min_confirm_frames = get_parameter("min_confirm_frames").as_int();

  const float min_confirm_duration =
      static_cast<float>(get_parameter("min_confirm_duration_sec").as_double());
  const double spatial_gate_max_age =
      get_parameter("reid_spatial_gate_max_age_sec").as_double();

  const float spatial_gate_base =
      static_cast<float>(get_parameter("reid_spatial_gate_base_m").as_double());

  const float spatial_gate_max_speed = static_cast<float>(
      get_parameter("reid_spatial_gate_max_speed_mps").as_double());

  const float depth_gate_base =
      static_cast<float>(get_parameter("reid_depth_gate_base_m").as_double());

  face_engine_ = std::make_shared<FaceEngine>(
      scrfd_path, adaface_path, osnet_path, match_thresh, body_thresh,
      min_confirm_frames, min_confirm_duration, session_reid_timeout,
      spatial_gate_max_age, spatial_gate_base, spatial_gate_max_speed,
      depth_gate_base,
      get_parameter("minimum_learning_confidence").as_double());

  if (!face_engine_->loadModels()) {
    RCLCPP_ERROR(get_logger(),
                 "Failed to initialize TensorRT models in FaceEngine!");

    return CallbackReturn::FAILURE;
  }

  try {
    const auto private_root = std::filesystem::path(OPL_IDENTITY_STORE_DEFAULT)
                                  .parent_path()
                                  .lexically_normal();
    const auto store_path =
        std::filesystem::absolute(
            get_parameter("identity_store_path").as_string())
            .lexically_normal();
    const auto relative = store_path.lexically_relative(private_root);
    if (relative.empty() || *relative.begin() == "..")
      throw std::invalid_argument(
          "Identity storage must stay under the workspace .private directory");
    face_engine_->configureStore(store_path.string(),
                                 modelFingerprint(adaface_path));
    enrollment_.missing_sec =
        get_parameter("enrollment_missing_sec").as_double();
    enrollment_.timeout_sec =
        get_parameter("enrollment_timeout_sec").as_double();
    if (!std::isfinite(enrollment_.missing_sec) ||
        enrollment_.missing_sec <= 0 ||
        !std::isfinite(enrollment_.timeout_sec) || enrollment_.timeout_sec <= 0)
      throw std::invalid_argument("Invalid enrollment timeouts");
  } catch (const std::exception &error) {
    RCLCPP_ERROR(get_logger(), "Identity storage configuration failed: %s",
                 error.what());
    return CallbackReturn::FAILURE;
  }
  enrollment_pub_ = create_publisher<opl_interfaces::msg::EnrollmentStatus>(
      "~/enrollment_status", rclcpp::QoS(20).reliable().transient_local());
  delete_srv_ = create_service<opl_interfaces::srv::DeleteIdentity>(
      "~/delete_identity",
      [this](const std::shared_ptr<opl_interfaces::srv::DeleteIdentity::Request>
                 request,
             std::shared_ptr<opl_interfaces::srv::DeleteIdentity::Response>
                 response) {
        if (enrollment_.current()) {
          response->message = "Enrollment is active; finish it before deleting";
          return;
        }
        std::lock_guard<std::mutex> engine_lock(engine_mutex_);
        response->success = face_engine_->deleteIdentity(request->persistent_id,
                                                         response->message);
        if (response->success) {
          publishEnrollment("deleted", response->message, 0,
                            request->persistent_id);
        }
      });

  identity_status_pub_ =
      create_publisher<opl_interfaces::msg::IdentityStatusArray>(
          "~/identity_status", 20);

  recognized_pub_ = create_publisher<opl_interfaces::msg::TrackedHumanArray>(
      recognized_topic_,
      rclcpp::SensorDataQoS().keep_last(observations_.settings.capacity));

  enroll_srv_ = create_service<opl_interfaces::srv::EnrollPerson>(
      "~/enroll_person",
      std::bind(&FaceRecognizerNode::handleEnrollService, this,
                std::placeholders::_1, std::placeholders::_2));

  RCLCPP_INFO(get_logger(), "Identity lock grace period: %.2f sec",
              identity_lock_grace_sec_);

  RCLCPP_INFO(get_logger(), "FaceRecognizerNode configured successfully.");

  return CallbackReturn::SUCCESS;
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
FaceRecognizerNode::on_activate(const rclcpp_lifecycle::State &state) {
  LifecycleNode::on_activate(state);

  recognized_pub_->on_activate();

  image_cb_group_ =
      create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

  // Serialize processing with lifecycle transitions and resource cleanup.
  tracked_cb_group_ = get_node_base_interface()->get_default_callback_group();

  rclcpp::SubscriptionOptions img_sub_options;

  img_sub_options.callback_group = image_cb_group_;

  img_sub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;

  image_sub_ = create_subscription<sensor_msgs::msg::Image>(
      image_topic_,
      rclcpp::SensorDataQoS().keep_last(observations_.settings.capacity),
      std::bind(&FaceRecognizerNode::imageCallback, this,
                std::placeholders::_1),
      img_sub_options);

  rclcpp::SubscriptionOptions tracked_sub_options;

  tracked_sub_options.callback_group = tracked_cb_group_;

  tracked_sub_options.use_intra_process_comm =
      rclcpp::IntraProcessSetting::Enable;

  tracked_sub_ = create_subscription<opl_interfaces::msg::TrackedHumanArray>(
      tracked_topic_,
      rclcpp::SensorDataQoS().keep_last(observations_.settings.capacity),
      std::bind(&FaceRecognizerNode::trackedHumansCallback, this,
                std::placeholders::_1),
      tracked_sub_options);

  RCLCPP_INFO(get_logger(), "FaceRecognizerNode activated.");

  {
    std::lock_guard<std::mutex> lock(observation_mutex_);
    observations_.clear();
  }
  observation_timer_ =
      create_wall_timer(std::chrono::milliseconds(5),
                        std::bind(&FaceRecognizerNode::drainObservations, this),
                        tracked_cb_group_);

  return CallbackReturn::SUCCESS;
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
FaceRecognizerNode::on_deactivate(const rclcpp_lifecycle::State &state) {
  LifecycleNode::on_deactivate(state);

  observation_timer_.reset();
  {
    std::lock_guard<std::mutex> lock(observation_mutex_);
    observations_.clear();
  }
  image_sub_.reset();
  tracked_sub_.reset();

  while (const auto *target = enrollment_.current()) {
    const auto raw = target->raw_id;
    face_engine_->cancelEnrollment(raw);
    enrollment_.advance(false, steadySeconds());
    publishEnrollment("failed",
                      "Recognizer deactivated; partial samples discarded", raw);
  }
  recognized_pub_->on_deactivate();

  return CallbackReturn::SUCCESS;
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
FaceRecognizerNode::on_cleanup(const rclcpp_lifecycle::State &) {
  recognized_pub_.reset();
  identity_status_pub_.reset();
  delete_srv_.reset();
  enrollment_pub_.reset();
  enrollment_ = EnrollmentQueue();
  visible_humans_.clear();
  last_visible_frame_ = 0;
  enroll_srv_.reset();
  face_engine_.reset();



  {
    std::lock_guard<std::mutex> lock(observation_mutex_);
    observations_.clear();
  }

  return CallbackReturn::SUCCESS;
}

void FaceRecognizerNode::publishEnrollment(const std::string &state,
                                           const std::string &message,
                                           uint64_t raw, uint64_t identity,
                                           uint32_t samples) {
  if (!enrollment_pub_)
    return;
  opl_interfaces::msg::EnrollmentStatus status;
  status.header.stamp = now();
  status.session_id = enrollment_.session_id;
  status.raw_track_id = raw;
  status.persistent_id = identity;
  status.state = state;
  status.message = message;
  status.accepted_samples = samples;
  status.required_samples = get_parameter("min_confirm_frames").as_int();
  status.total_targets = enrollment_.total();
  status.completed_targets = enrollment_.completed;
  status.failed_targets = enrollment_.failed;
  enrollment_pub_->publish(status);
}

void FaceRecognizerNode::beginEnrollmentTarget() {
  while (const auto *target = enrollment_.current()) {
    try {
      face_engine_->authorizeEnrollment(target->raw_id, target->name);
      publishEnrollment("authorized", "Collecting face samples for this target",
                        target->raw_id);
      return;
    } catch (const std::exception &error) {
      const auto raw = target->raw_id;
      enrollment_.advance(false, steadySeconds());
      publishEnrollment("failed", error.what(), raw);
    }
  }
  publishEnrollment("finished", "Enrollment session finished");
}

void FaceRecognizerNode::enrollmentWatchdog() {
  const auto failure = enrollment_.failure(steadySeconds());
  if (failure.empty())
    return;
  const auto raw = enrollment_.current()->raw_id;
  face_engine_->cancelEnrollment(raw);
  enrollment_.advance(false, steadySeconds());
  publishEnrollment("failed", failure, raw);
  beginEnrollmentTarget();
}

void FaceRecognizerNode::handleEnrollService(
    const std::shared_ptr<opl_interfaces::srv::EnrollPerson::Request> request,
    std::shared_ptr<opl_interfaces::srv::EnrollPerson::Response> response) {
  if (!recognized_pub_ || !recognized_pub_->is_activated()) {
    response->message = "Recognizer is not active";
    return;
  }
  if (steadySeconds() - last_visible_frame_ > 0.5) {
    response->message = "No fresh visible person observations";
    return;
  }
  std::vector<EnrollmentCandidate> candidates;
  for (const auto &human : visible_humans_) {
    if (!human.observed || human.raw_track_id <= 0 ||
        !std::isfinite(human.confidence) ||
        human.confidence <
            get_parameter("minimum_learning_confidence").as_double())
      continue;
    if (request->raw_track_id &&
        request->raw_track_id != uint64_t(human.raw_track_id))
      continue;
    candidates.push_back(
        {uint64_t(human.raw_track_id), human.confidence, request->name});
  }
  if (!request->name.empty() && !validIdentityName(request->name)) {
    response->message = "Invalid name";
    return;
  }
  if (!request->name.empty() && candidates.size() > 1) {
    response->message = "Group enrollment uses generated names; select one "
                        "raw_track_id for a custom name";
    return;
  }
  try {
    enrollment_.start(std::move(candidates), steadySeconds());
    beginEnrollmentTarget();
    response->success = true;
    response->session_id = enrollment_.session_id;
    response->message = "Enrollment authorized; completion is reported on "
                        "~/enrollment_status after durable storage";
  } catch (const std::exception &error) {
    response->message = error.what();
  }
}

void FaceRecognizerNode::imageCallback(
    const sensor_msgs::msg::Image::ConstSharedPtr msg) {
  std::lock_guard<std::mutex> lock(observation_mutex_);
  if (!observations_.addImage(msg, false, now().nanoseconds(), steadySeconds()))
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                         "Rejected invalid, stale or duplicate RGB sample");
}

void FaceRecognizerNode::trackedHumansCallback(
    const opl_interfaces::msg::TrackedHumanArray::ConstSharedPtr msg) {
  if (!recognized_pub_->is_activated())
    return;
  {
    std::lock_guard<std::mutex> lock(observation_mutex_);
    if (!observations_.enqueue(msg, now().nanoseconds(), steadySeconds())) {
      RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 5000,
          "Rejected invalid, stale, out-of-order or duplicate tracks");
      return;
    }
  }
  drainObservations();
}

void FaceRecognizerNode::drainObservations() {
  if (!recognized_pub_ || !recognized_pub_->is_activated())
    return;
  enrollmentWatchdog();
  const auto publish_unrecognized = [this](const auto &input) {
    processObservation(input, cv::Mat{});
  };
  while (true) {
    std::optional<MatchedObservation> observation;
    {
      std::lock_guard<std::mutex> lock(observation_mutex_);
      observation =
          observations_.take(now().nanoseconds(), steadySeconds(), false);
    }
    if (!observation)
      break;
    if (!observation->rgb) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                           "Recognition skipped: original RGB frame "
                           "unavailable; preserving tracked output");
      publish_unrecognized(observation->observation);
      continue;
    }
    try {
      const auto rgb = cv_bridge::toCvShare(observation->rgb,
                                            sensor_msgs::image_encodings::BGR8);
      if (!rgb->image.empty())
        processObservation(observation->observation, rgb->image);
      else
        publish_unrecognized(observation->observation);
    } catch (const cv_bridge::Exception &error) {
      RCLCPP_WARN(get_logger(), "Invalid original RGB payload: %s",
                  error.what());
      publish_unrecognized(observation->observation);
    }
  }
}

void FaceRecognizerNode::processObservation(
    const opl_interfaces::msg::TrackedHumanArray::ConstSharedPtr &msg,
    const cv::Mat &frame) {
  ScopedTiming timing("Recognizer.callback_total");
  const auto processing_time = now();
  const double timestamp = rclcpp::Time(msg->header.stamp).seconds();
  if ((last_observation_time_ > 0 && timestamp < last_observation_time_) ||
      (!last_observation_frame_.empty() &&
       opticalFrame(msg->header.frame_id) != last_observation_frame_)) {
    if (const auto *target = enrollment_.current()) {
      const auto raw = target->raw_id;
      enrollment_.cancel(steadySeconds());
      publishEnrollment(
          "failed", "Observation clock/frame reset; entire session cancelled",
          raw);
      publishEnrollment("finished",
                        "Enrollment cancelled by observation reset");
    }
    face_engine_->resetTemporalState();
  }
  last_observation_time_ = timestamp;
  last_observation_frame_ = opticalFrame(msg->header.frame_id);
  visible_humans_ = msg->humans;
  last_visible_frame_ = steadySeconds();
  auto output = *msg;
  std::set<uint64_t> observed_ids;
  for (auto &human : output.humans) {
    if (human.raw_track_id <= 0)
      human.raw_track_id = human.track_id;
    if (human.observed && human.raw_track_id > 0)
      observed_ids.insert(human.raw_track_id);
  }
  enrollment_.observe(observed_ids, steadySeconds());
  std::vector<FaceResult> features(output.humans.size());
  std::vector<cv::Rect> crops(output.humans.size());
  std::lock_guard<std::mutex> engine_lock(engine_mutex_);
  for (size_t i = 0; i < output.humans.size(); ++i) {
    const auto &human = output.humans[i];
    if (!human.observed || frame.empty() ||
        !face_engine_->needsFeatures(human.raw_track_id, timestamp,
                                     identity_validation_interval_sec_))
      continue;
    const auto &b = human.bbox;
    cv::Rect box(int(b.x_offset), int(b.y_offset), int(b.width), int(b.height));
    crops[i] = box & cv::Rect(0, 0, frame.cols, frame.rows);
    bool isolated = true;
    for (size_t j = 0; j < output.humans.size(); ++j)
      if (i != j && output.humans[j].observed) {
        const auto &o = output.humans[j].bbox;
        cv::Rect other(int(o.x_offset), int(o.y_offset), int(o.width),
                       int(o.height));
        if (float((box & other).area()) /
                std::max(1, box.area() + other.area() - (box & other).area()) >
            .25f)
          isolated = false;
      }
    if (crops[i].area() > 0)
      features[i] = face_engine_->extractFeatures(frame(crops[i]), isolated);
  }
  auto results = face_engine_->processFrameFeatures(
      output.humans, std::move(features), timestamp);
  opl_interfaces::msg::IdentityStatusArray statuses;
  statuses.header = msg->header;
  const auto add_status = [&](uint64_t raw, const FaceResult &result) {
    opl_interfaces::msg::IdentityStatus status;
    status.raw_track_id = raw;
    status.persistent_id = result.persistent_id;
    status.name = result.name;
    status.state = result.identity_state;
    status.reason = result.identity_reason;
    status.last_face_score = result.match_score;
    status.face_age_sec = result.face_age_sec;
    status.body_age_sec = result.body_age_sec;
    statuses.identities.push_back(std::move(status));
  };
  std::set<uint64_t> published_raw;
  for (size_t i = 0; i < output.humans.size(); ++i) {
    auto &human = output.humans[i];
    const auto &result = results[i];
    human.track_id = result.persistent_id;
    human.name = result.name;
    human.match_score = result.match_score;
    human.landmarks.clear();
    for (const auto &lm : result.landmarks) {
      geometry_msgs::msg::Point point;
      point.x = crops[i].x + lm.x;
      point.y = crops[i].y + lm.y;
      human.landmarks.push_back(point);
    }
    add_status(human.raw_track_id, result);
    published_raw.insert(human.raw_track_id);
    if (enrollment_.current() &&
        enrollment_.current()->raw_id == uint64_t(human.raw_track_id) &&
        !result.enrollment_state.empty()) {
      const bool terminal = result.enrollment_state == "completed" ||
                            result.enrollment_state == "already_enrolled" ||
                            result.enrollment_state == "failed";
      if (terminal) {
        face_engine_->cancelEnrollment(human.raw_track_id);
        enrollment_.advance(result.enrollment_state != "failed",
                            steadySeconds());
      }
      publishEnrollment(result.enrollment_state, result.enrollment_message,
                        human.raw_track_id, result.persistent_id,
                        result.accepted_samples);
      if (terminal)
        beginEnrollmentTarget();
    }
  }
  for (const auto &[raw, status] : face_engine_->lostIdentities(timestamp))
    if (!published_raw.count(raw))
      add_status(raw, status);
  if (now() < processing_time ||
      (now() - rclcpp::Time(msg->header.stamp)).seconds() >
          observations_.settings.max_age_sec)
    return;
  recognized_pub_->publish(output);
  identity_status_pub_->publish(statuses);
}

} // namespace opl_human_vision

RCLCPP_COMPONENTS_REGISTER_NODE(opl_human_vision::FaceRecognizerNode)
