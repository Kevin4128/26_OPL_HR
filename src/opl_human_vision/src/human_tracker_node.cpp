#include "opl_human_vision/human_tracker_node.hpp"
#include "opl_human_vision/scoped_timing.hpp"

#include "rclcpp_components/register_node_macro.hpp"

#include <cv_bridge/cv_bridge.hpp>

#include <algorithm>
#include <cmath>
#include <mutex>
#include <sstream>
#include <vector>

namespace opl_human_vision {

HumanTrackerNode::HumanTrackerNode(const rclcpp::NodeOptions &options)
    : rclcpp_lifecycle::LifecycleNode("human_tracker_node", options) {
  cv::setNumThreads(declare_parameter<int>("opencv_threads", 1));
  declare_parameter<int>("observation_buffer_size", 60);
  declare_parameter<double>("track_max_age_sec", 2.0);
  declare_parameter<double>("prediction_hold_sec", 0.5);
  declare_parameter<double>("high_confidence_threshold", 0.45);
  declare_parameter<double>("appearance_recovery_sec", 2.0);
  declare_parameter<double>("max_observation_age_sec", 1.0);
  declare_parameter<double>("observation_wait_sec", 0.1);
  declare_parameter<double>("depth_slop_sec", 0.02);
  declare_parameter<double>("depth_timestamp_offset_sec", 0.0);
  declare_parameter<std::vector<std::string>>("registered_frame_aliases",
                                              std::vector<std::string>{});
  declare_parameter<bool>("depth_registered_to_rgb", true);
  declare_parameter<std::string>("fixed_frame", "odom");
  declare_parameter<int>("depth_min_valid_pixels", 25);
  declare_parameter<double>("depth_min_valid_fraction", 0.25);
  declare_parameter<double>("depth_min_m", 0.2);
  declare_parameter<double>("depth_max_m", 10.0);
  declare_parameter<double>("depth_max_spread_m", 0.3);
  declare_parameter<double>("depth_max_spread_ratio", 0.15);
  declare_parameter<std::string>("image_topic", "camera/image_raw");

  declare_parameter<std::string>(
      "depth_topic", "camera/camera/aligned_depth_to_color/image_raw");

  declare_parameter<std::string>("camera_info_topic", "");

  declare_parameter<std::string>("detections_topic", "raw_detections");

  declare_parameter<std::string>("tracked_topic", "tracked_humans");

  declare_parameter<int>("max_age", 30);

  declare_parameter<int>("min_hits", 3);

  declare_parameter<float>("iou_threshold", 0.3f);

  declare_parameter<int>("max_publish_age", 1);

  // --------------------------------------------------------------------------
  // Appearance-assisted tracker reassociation.
  // --------------------------------------------------------------------------

  declare_parameter<std::string>("osnet_model_path", "");

  declare_parameter<bool>("appearance_reid_enabled", false);

  declare_parameter<float>("appearance_match_threshold", 0.85f);

  declare_parameter<int>("appearance_reid_max_age", 10);

  declare_parameter<int>("appearance_refresh_interval", 15);

  declare_parameter<float>("appearance_spatial_base_ratio", 0.10f);

  declare_parameter<float>("appearance_spatial_speed_ratio_per_sec", 0.75f);
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
HumanTrackerNode::on_configure(const rclcpp_lifecycle::State &) {
  RCLCPP_INFO(get_logger(), "Configuring HumanTrackerNode...");

  observations_.clear();
  auto &settings = observations_.settings;
  const auto capacity = get_parameter("observation_buffer_size").as_int();
  settings.capacity = capacity > 0 ? capacity : 0;
  settings.max_age_sec = get_parameter("max_observation_age_sec").as_double();
  settings.wait_sec = get_parameter("observation_wait_sec").as_double();
  settings.depth_slop_sec = get_parameter("depth_slop_sec").as_double();
  settings.depth_offset_sec =
      get_parameter("depth_timestamp_offset_sec").as_double();
  settings.registered_frame_aliases =
      get_parameter("registered_frame_aliases").as_string_array();
  depth_registered_ = get_parameter("depth_registered_to_rgb").as_bool();
  fixed_frame_ = opticalFrame(get_parameter("fixed_frame").as_string());
  depth_settings_.min_pixels = get_parameter("depth_min_valid_pixels").as_int();
  depth_settings_.min_fraction =
      get_parameter("depth_min_valid_fraction").as_double();
  depth_settings_.min_depth = get_parameter("depth_min_m").as_double();
  depth_settings_.max_depth = get_parameter("depth_max_m").as_double();
  depth_settings_.max_spread_m =
      get_parameter("depth_max_spread_m").as_double();
  depth_settings_.max_spread_ratio =
      get_parameter("depth_max_spread_ratio").as_double();
  if (!settings.valid() || !depth_settings_.valid() || fixed_frame_.empty()) {
    RCLCPP_ERROR(get_logger(), "Invalid observation/depth/frame parameters");
    return CallbackReturn::FAILURE;
  }
  tf_buffer_ = std::make_shared<tf2_ros::Buffer>(get_clock());
  tf_listener_ =
      std::make_shared<tf2_ros::TransformListener>(*tf_buffer_, this, true);

  image_topic_ = get_parameter("image_topic").as_string();

  depth_topic_ = get_parameter("depth_topic").as_string();

  detections_topic_ = get_parameter("detections_topic").as_string();

  tracked_topic_ = get_parameter("tracked_topic").as_string();

  const int max_age = get_parameter("max_age").as_int();

  const int min_hits = get_parameter("min_hits").as_int();

  const float iou_threshold =
      static_cast<float>(get_parameter("iou_threshold").as_double());

  const int max_publish_age = get_parameter("max_publish_age").as_int();

  const std::string osnet_model_path =
      get_parameter("osnet_model_path").as_string();

  const bool appearance_reid_enabled =
      get_parameter("appearance_reid_enabled").as_bool();

  const float appearance_match_threshold = static_cast<float>(
      get_parameter("appearance_match_threshold").as_double());

  const int appearance_reid_max_age =
      get_parameter("appearance_reid_max_age").as_int();

  const int appearance_refresh_interval =
      get_parameter("appearance_refresh_interval").as_int();

  const float appearance_spatial_base_ratio = static_cast<float>(
      get_parameter("appearance_spatial_base_ratio").as_double());

  const float appearance_spatial_speed_ratio_per_sec = static_cast<float>(
      get_parameter("appearance_spatial_speed_ratio_per_sec").as_double());

  tracker_ = std::make_shared<TrackerEngine>(
      max_age, min_hits, iou_threshold, max_publish_age,

      osnet_model_path, appearance_reid_enabled, appearance_match_threshold,
      appearance_reid_max_age, appearance_refresh_interval,
      appearance_spatial_base_ratio, appearance_spatial_speed_ratio_per_sec,
      get_parameter("track_max_age_sec").as_double(),
      get_parameter("appearance_recovery_sec").as_double(),
      get_parameter("high_confidence_threshold").as_double(),
      get_parameter("prediction_hold_sec").as_double());

  RCLCPP_INFO(get_logger(),
              "Tracker configured: "
              "max_age=%d, "
              "max_publish_age=%d, "
              "min_hits=%d, "
              "IoU=%.2f",
              max_age, max_publish_age, min_hits, iou_threshold);

  if (appearance_reid_enabled) {

    if (tracker_->appearanceReidReady()) {
      RCLCPP_INFO(get_logger(),
                  "Tracker appearance ReID enabled: "
                  "threshold=%.3f, "
                  "max_age=%d, "
                  "refresh_interval=%d, "
                  "base_ratio=%.3f, "
                  "speed_ratio=%.3f/sec",
                  appearance_match_threshold, appearance_reid_max_age,
                  appearance_refresh_interval, appearance_spatial_base_ratio,
                  appearance_spatial_speed_ratio_per_sec);

    } else {

      RCLCPP_WARN(get_logger(),
                  "Tracker appearance ReID was requested but "
                  "OSNet could not be initialized. "
                  "Tracker will continue with geometric association only.");
    }
  }

  depth_diagnostics_pub_ =
      create_publisher<std_msgs::msg::String>("~/depth_diagnostics", 10);
  tracked_pub_ = create_publisher<opl_interfaces::msg::TrackedHumanArray>(
      tracked_topic_,
      rclcpp::SensorDataQoS().keep_last(observations_.settings.capacity));

  RCLCPP_INFO(get_logger(), "HumanTrackerNode Configured Successfully.");

  return CallbackReturn::SUCCESS;
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
HumanTrackerNode::on_activate(const rclcpp_lifecycle::State &state) {
  LifecycleNode::on_activate(state);

  RCLCPP_INFO(get_logger(), "Activating HumanTrackerNode...");

  tracked_pub_->on_activate();

  image_cb_group_ =
      create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

  depth_cb_group_ =
      create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

  // Serialize processing with lifecycle transitions and resource cleanup.
  track_cb_group_ = get_node_base_interface()->get_default_callback_group();

  rclcpp::SubscriptionOptions img_sub_options;

  img_sub_options.callback_group = image_cb_group_;

  img_sub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;

  image_sub_ = create_subscription<sensor_msgs::msg::Image>(
      image_topic_,
      rclcpp::SensorDataQoS().keep_last(observations_.settings.capacity),
      std::bind(&HumanTrackerNode::imageCallback, this, std::placeholders::_1),
      img_sub_options);

  std::string info_topic = get_parameter("camera_info_topic").as_string();

  if (info_topic.empty()) {

    info_topic = image_topic_;

    const size_t pos = info_topic.rfind("image_raw");

    if (pos != std::string::npos) {
      info_topic.replace(pos, 9, "camera_info");

    } else {

      info_topic += "/camera_info";
    }
  }

  RCLCPP_INFO(get_logger(), "Listening for Camera Info on: %s",
              info_topic.c_str());

  rclcpp::SubscriptionOptions info_sub_options;

  info_sub_options.callback_group = image_cb_group_;

  info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
      info_topic,
      rclcpp::SensorDataQoS().keep_last(observations_.settings.capacity),
      std::bind(&HumanTrackerNode::cameraInfoCallback, this,
                std::placeholders::_1),
      info_sub_options);

  rclcpp::SubscriptionOptions depth_sub_options;

  depth_sub_options.callback_group = depth_cb_group_;

  depth_sub_options.use_intra_process_comm =
      rclcpp::IntraProcessSetting::Enable;

  depth_sub_ = create_subscription<sensor_msgs::msg::Image>(
      depth_topic_,
      rclcpp::SensorDataQoS().keep_last(observations_.settings.capacity),
      std::bind(&HumanTrackerNode::depthCallback, this, std::placeholders::_1),
      depth_sub_options);

  rclcpp::SubscriptionOptions track_sub_options;

  track_sub_options.callback_group = track_cb_group_;

  track_sub_options.use_intra_process_comm =
      rclcpp::IntraProcessSetting::Enable;

  detection_sub_ = create_subscription<opl_interfaces::msg::TrackedHumanArray>(
      detections_topic_,
      rclcpp::SensorDataQoS().keep_last(observations_.settings.capacity),
      std::bind(&HumanTrackerNode::detectionsCallback, this,
                std::placeholders::_1),
      track_sub_options);

  {
    std::lock_guard<std::mutex> lock(observation_mutex_);
    observations_.clear();
  }
  observation_timer_ = create_wall_timer(
      std::chrono::milliseconds(5),
      std::bind(&HumanTrackerNode::drainObservations, this), track_cb_group_);

  return CallbackReturn::SUCCESS;
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
HumanTrackerNode::on_deactivate(const rclcpp_lifecycle::State &state) {
  LifecycleNode::on_deactivate(state);

  observation_timer_.reset();
  {
    std::lock_guard<std::mutex> lock(observation_mutex_);
    observations_.clear();
  }
  image_sub_.reset();
  depth_sub_.reset();
  info_sub_.reset();
  detection_sub_.reset();

  tracked_pub_->on_deactivate();

  return CallbackReturn::SUCCESS;
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
HumanTrackerNode::on_cleanup(const rclcpp_lifecycle::State &) {
  tracked_pub_.reset();
  depth_diagnostics_pub_.reset();

  tracker_.reset();

  observation_timer_.reset();
  tf_listener_.reset();
  tf_buffer_.reset();
  {
    std::lock_guard<std::mutex> lock(observation_mutex_);
    observations_.clear();
  }
  return CallbackReturn::SUCCESS;
}

void HumanTrackerNode::imageCallback(
    const sensor_msgs::msg::Image::ConstSharedPtr msg) {
  std::lock_guard<std::mutex> lock(observation_mutex_);
  if (!observations_.addImage(msg, false, now().nanoseconds(), steadySeconds()))
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                         "Rejected invalid, stale or duplicate RGB sample");
}

void HumanTrackerNode::depthCallback(
    const sensor_msgs::msg::Image::ConstSharedPtr msg) {
  std::lock_guard<std::mutex> lock(observation_mutex_);
  if (!observations_.addImage(msg, true, now().nanoseconds(), steadySeconds()))
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                         "Rejected invalid, stale or duplicate depth sample");
}

void HumanTrackerNode::cameraInfoCallback(
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr msg) {
  std::lock_guard<std::mutex> lock(observation_mutex_);
  observations_.setCalibration(msg, now().nanoseconds());
}

void HumanTrackerNode::detectionsCallback(
    const opl_interfaces::msg::TrackedHumanArray::ConstSharedPtr msg) {
  if (!tracked_pub_->is_activated())
    return;
  {
    std::lock_guard<std::mutex> lock(observation_mutex_);
    if (!observations_.enqueue(msg, now().nanoseconds(), steadySeconds())) {
      RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 5000,
          "Rejected invalid, stale, out-of-order or duplicate detections");
      return;
    }
  }
  drainObservations();
}

void HumanTrackerNode::drainObservations() {
  if (!tracked_pub_ || !tracked_pub_->is_activated())
    return;
  while (true) {
    std::optional<MatchedObservation> observation;
    {
      std::lock_guard<std::mutex> lock(observation_mutex_);
      if (observations_.epoch() != tracker_epoch_) {
        tracker_->reset();
        tracker_epoch_ = observations_.epoch();
      }
      observation = observations_.take(now().nanoseconds(), steadySeconds(),
                                       depth_registered_);
    }
    if (!observation)
      break;
    if (!observation->rgb) {
      RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 5000,
          "Expired detections without their original RGB frame");
      continue;
    }
    try {
      processObservation(*observation);
    } catch (const cv::Exception &error) {
      RCLCPP_ERROR(get_logger(), "RGB observation conversion failed: %s",
                   error.what());
    } catch (const cv_bridge::Exception &error) {
      RCLCPP_ERROR(get_logger(), "RGB observation conversion failed: %s",
                   error.what());
    }
  }
}

void HumanTrackerNode::processObservation(
    const MatchedObservation &observation) {
  ScopedTiming timing("Tracker.callback_total");
  const auto &msg = observation.observation;
  const auto frame =
      cv_bridge::toCvShare(observation.rgb, sensor_msgs::image_encodings::BGR8);
  const cv::Mat &frame_ref = frame->image;
  if (frame_ref.empty())
    return;
  const auto processing_time = now();
  cv_bridge::CvImageConstPtr depth;
  if (depth_registered_ && observation.depth && observation.calibration &&
      validCalibration(*observation.calibration, *observation.rgb,
                       observations_)) {
    try {
      if (observation.depth->encoding ==
              sensor_msgs::image_encodings::TYPE_16UC1 ||
          observation.depth->encoding ==
              sensor_msgs::image_encodings::TYPE_32FC1)
        depth = cv_bridge::toCvShare(observation.depth,
                                     observation.depth->encoding);
    } catch (const cv_bridge::Exception &error) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                           "Invalid depth payload: %s", error.what());
    }
  }
  if (!depth)
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                         "2D-only observation: depth absent, incompatible, or "
                         "calibration invalid");

  auto spatial_detections = msg->humans;
  size_t detection_index = 0;
  for (auto &detection : spatial_detections) {
    DepthDiagnostics diagnostic;
    diagnostic.reason =
        "no_timestamp_compatible_registered_depth_or_calibration";
    detection.position_valid = false;
    detection.fixed_position_valid = false;
    detection.position_3d = geometry_msgs::msg::Point();
    detection.observed = true;
    if (observation.depth)
      detection.depth_header = observation.depth->header;
    if (depth) {
      detection.position_header = observation.rgb->header;
      detection.position_header.frame_id =
          opticalFrame(detection.position_header.frame_id);
      detection.position_header.stamp =
          rclcpp::Time(stampNs(observation.depth->header.stamp) +
                       observations_.depthOffsetNs());
      try {
        estimateHumanPosition(
            depth->image,
            cv::Rect(detection.bbox.x_offset, detection.bbox.y_offset,
                     detection.bbox.width, detection.bbox.height),
            *observation.calibration, depth_settings_, detection, &diagnostic);
      } catch (const cv::Exception &error) {
        diagnostic.reason = "projection_exception";
        detection.position_valid = false;
        detection.position_3d = geometry_msgs::msg::Point();
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                             "Invalid depth projection: %s", error.what());
      }
    }
    transformHumanPosition(detection, *tf_buffer_, fixed_frame_);
    if (depth_diagnostics_pub_->get_subscription_count() > 0) {
      std::ostringstream data;
      data << "{\"stamp_ns\":" << stampNs(msg->header.stamp)
           << ",\"detection_index\":" << detection_index << ",\"reason\":\""
           << diagnostic.reason << "\",\"median\":" << diagnostic.median
           << ",\"p10\":" << diagnostic.p10 << ",\"p90\":" << diagnostic.p90
           << ",\"allowed_spread\":" << diagnostic.allowed_spread
           << ",\"cluster_median\":" << diagnostic.cluster_median
           << ",\"cluster_p10\":" << diagnostic.cluster_p10
           << ",\"cluster_p90\":" << diagnostic.cluster_p90
           << ",\"cluster_pixels\":" << diagnostic.cluster_pixels
           << ",\"valid_pixels\":" << detection.depth_valid_pixels
           << ",\"mad\":" << detection.depth_mad_m << ",\"position_valid\":"
           << (detection.position_valid ? "true" : "false")
           << ",\"fixed_position_valid\":"
           << (detection.fixed_position_valid ? "true" : "false") << "}";
      std_msgs::msg::String output;
      output.data = data.str();
      depth_diagnostics_pub_->publish(output);
    }
    ++detection_index;
  }

  std::vector<Track> active_tracks;

  {
    ScopedTiming update_timing("Tracker.update");

    const double tracker_timestamp_sec =
        rclcpp::Time(msg->header.stamp).seconds();

    active_tracks =
        tracker_->update(spatial_detections, frame_ref, tracker_timestamp_sec);
  }

  RCLCPP_DEBUG(rclcpp::get_logger("vision_timing"),
               "[timing] Tracker "
               "frame=%d.%09u "
               "detections=%zu "
               "confirmed_tracks=%zu",
               msg->header.stamp.sec, msg->header.stamp.nanosec,
               msg->humans.size(), active_tracks.size());

  opl_interfaces::msg::TrackedHumanArray tracked_array_msg;

  tracked_array_msg.header = msg->header;

  for (const auto &track : active_tracks) {
    opl_interfaces::msg::TrackedHuman human =
        track.time_since_update == 0 ? track.observation
                                     : opl_interfaces::msg::TrackedHuman();
    human.landmarks.clear();
    human.observed = track.time_since_update == 0;

    human.raw_track_id = track.id;

    human.track_id = track.id;

    human.name = track.name;

    human.confidence = track.confidence;

    human.bbox.x_offset = std::max(0, static_cast<int>(track.bbox.x));

    human.bbox.y_offset = std::max(0, static_cast<int>(track.bbox.y));

    human.bbox.width = std::max(0, static_cast<int>(track.bbox.width));

    human.bbox.height = std::max(0, static_cast<int>(track.bbox.height));

    if (human.position_valid && !human.fixed_position_valid)
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                           "Observation-time TF to %s unavailable; preserving "
                           "camera position and visual tracking",
                           fixed_frame_.c_str());

    tracked_array_msg.humans.push_back(human);
  }

  if (now() < processing_time ||
      (now() - rclcpp::Time(msg->header.stamp)).seconds() >
          observations_.settings.max_age_sec)
    return; // A clock reset invalidates this in-flight observation.
  tracked_pub_->publish(tracked_array_msg);
}

} // namespace opl_human_vision

RCLCPP_COMPONENTS_REGISTER_NODE(opl_human_vision::HumanTrackerNode)
