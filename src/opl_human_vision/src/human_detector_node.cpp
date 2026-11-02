#include "opl_human_vision/human_detector_node.hpp"
#include "opl_human_vision/pose_nms.hpp"
#include "opl_human_vision/scoped_timing.hpp"
#include "rclcpp_components/register_node_macro.hpp"
#include <algorithm>

namespace opl_human_vision {

HumanDetectorNode::HumanDetectorNode(const rclcpp::NodeOptions & options)
: rclcpp_lifecycle::LifecycleNode("human_detector_node", options) {
  cv::setNumThreads(declare_parameter<int>("opencv_threads", 1));
  declare_parameter<std::string>("image_topic", "camera/image_raw");
  declare_parameter<std::string>("detections_topic", "raw_detections");
  declare_parameter<std::string>("model_path", "");
  declare_parameter<float>("confidence_threshold", 0.45f);
  declare_parameter<float>("nms_threshold", 0.50f);
  // 0 = run on every frame the accelerator can keep up with.
  declare_parameter<double>("max_inference_hz", 0.0);
  declare_parameter<bool>("hailo_multi_process", false);
  // Pose drives tracking, so it gets the highest scheduler priority by default.
  declare_parameter<int>("hailo_priority", 24);
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
HumanDetectorNode::on_configure(const rclcpp_lifecycle::State &) {
  RCLCPP_INFO(get_logger(), "Configuring HumanDetectorNode...");

  image_topic_ = get_parameter("image_topic").as_string();
  detections_topic_ = get_parameter("detections_topic").as_string();
  model_path_ = get_parameter("model_path").as_string();
  conf_thresh_ = static_cast<float>(get_parameter("confidence_threshold").as_double());
  nms_thresh_ = static_cast<float>(get_parameter("nms_threshold").as_double());
  max_inference_hz_ = get_parameter("max_inference_hz").as_double();

  if (model_path_.empty()) {
    RCLCPP_ERROR(get_logger(), "Parameter 'model_path' is required but not set.");
    return CallbackReturn::FAILURE;
  }

  {
    ScopedTiming timing("YOLO.load (attempt)");
    HailoModelOptions options;
    options.multi_process_service = get_parameter("hailo_multi_process").as_bool();
    options.scheduler_priority = static_cast<int>(get_parameter("hailo_priority").as_int());
    std::string error;
    model_ = HailoModel::create(model_path_, options, error);
    if (!model_ || !decoder_.bind(model_->outputs(), model_->inputWidth(), error)) {
      RCLCPP_ERROR(get_logger(), "Failed to load YOLOv8-pose HEF '%s': %s",
        model_path_.c_str(), error.c_str());
      model_.reset();
      return CallbackReturn::FAILURE;
    }
  }
  RCLCPP_INFO(get_logger(), "Loaded %s on Hailo: input %dx%d, DFL reg_max %d",
    model_->name().c_str(), model_->inputWidth(), model_->inputHeight(), decoder_.regMax());

  detection_pub_ = create_publisher<opl_interfaces::msg::TrackedHumanArray>(
    detections_topic_, rclcpp::SensorDataQoS().keep_last(1)
  );

  return CallbackReturn::SUCCESS;
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
HumanDetectorNode::on_activate(const rclcpp_lifecycle::State & state) {
  LifecycleNode::on_activate(state);

  detection_pub_->on_activate();

  image_cb_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

  rclcpp::SubscriptionOptions sub_options;
  sub_options.callback_group = image_cb_group_;
  sub_options.use_intra_process_comm = rclcpp::IntraProcessSetting::Enable;

  image_sub_ = create_subscription<sensor_msgs::msg::Image>(
    image_topic_,
    rclcpp::SensorDataQoS().keep_last(1),
    std::bind(&HumanDetectorNode::imageCallback, this, std::placeholders::_1),
    sub_options
  );

  return CallbackReturn::SUCCESS;
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
HumanDetectorNode::on_deactivate(const rclcpp_lifecycle::State & state) {
  LifecycleNode::on_deactivate(state);

  image_sub_.reset();
  detection_pub_->on_deactivate();

  return CallbackReturn::SUCCESS;
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
HumanDetectorNode::on_cleanup(const rclcpp_lifecycle::State &) {
  detection_pub_.reset();
  model_.reset();
  return CallbackReturn::SUCCESS;
}

void HumanDetectorNode::imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr msg) {
  if (!detection_pub_->is_activated() || !model_) {
    return;
  }

  if (max_inference_hz_ > 0.0) {
    // Deadline schedule with a small tolerance: a plain "elapsed < period"
    // check aliases against the camera rate (15 Hz cap on 30 fps -> 10 Hz).
    using namespace std::chrono;
    const auto now = steady_clock::now();
    const auto period = duration_cast<steady_clock::duration>(duration<double>(1.0 / max_inference_hz_));
    const auto tolerance = duration_cast<steady_clock::duration>(milliseconds(8));
    if (now + tolerance < next_inference_) {
      return;
    }
    next_inference_ = std::max(next_inference_ + period, now - period / 2);
  }

  try {
    ScopedTiming timing("YOLO.callback_total");
    cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::BGR8);
    if (cv_ptr->image.empty()) {
      return;
    }

    const cv::Mat & img = cv_ptr->image;
    int img_w = img.cols;
    int img_h = img.rows;

    // 1. Letterbox straight into the Hailo input buffer (uint8 RGB; the
    //    normalisation is compiled into the HEF).
    cv::Mat input = model_->inputMat();
    const LetterboxInfo lb = letterboxInto(img, input, cv::Scalar(114, 114, 114), true);

    // 2. Inference on the Hailo-8L (scheduler shares the chip with the other models).
    {
      ScopedTiming inference_timing("YOLO.hailo_inference");
      std::string error;
      if (!model_->run(error)) {
        RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 2000, "%s", error.c_str());
        return;
      }
    }

    // 3. Decode only anchors above the score threshold, then undo the letterbox.
    decoder_.decode(model_->outputs(), conf_thresh_, candidates_);

    std::vector<cv::Rect> bboxes;
    std::vector<float> confidences;
    std::vector<std::vector<cv::Point3f>> keypoints_list;
    bboxes.reserve(candidates_.size());
    confidences.reserve(candidates_.size());
    keypoints_list.reserve(candidates_.size());

    for (const auto & c : candidates_) {
      const cv::Point2f tl = lb.toImage(c.x1, c.y1);
      const cv::Point2f br = lb.toImage(c.x2, c.y2);
      std::vector<cv::Point3f> kps;
      kps.reserve(c.keypoints.size());
      for (const auto & kp : c.keypoints) {
        const cv::Point2f p = lb.toImage(kp.x, kp.y);
        kps.emplace_back(p.x, p.y, kp.z);
      }
      bboxes.emplace_back(
        static_cast<int>(tl.x), static_cast<int>(tl.y),
        static_cast<int>(br.x - tl.x), static_cast<int>(br.y - tl.y));
      confidences.push_back(c.score);
      keypoints_list.push_back(std::move(kps));
    }

    // 5. NMS (Non-Maximum Suppression)
    const auto indices = poseNms(bboxes, confidences, keypoints_list, nms_thresh_);

    opl_interfaces::msg::TrackedHumanArray detections_msg;
    detections_msg.header = msg->header;

    for (int idx : indices) {
      const auto& kps = keypoints_list[idx];

      float min_x = static_cast<float>(img_w), min_y = static_cast<float>(img_h);
      float max_x = 0.0f, max_y = 0.0f;
      int valid_kps = 0;

      for (const auto& kp : kps) {
        if (kp.z > 0.35f) {
          min_x = std::min(min_x, kp.x);
          min_y = std::min(min_y, kp.y);
          max_x = std::max(max_x, kp.x);
          max_y = std::max(max_y, kp.y);
          valid_kps++;
        }
      }

      cv::Rect final_box = bboxes[idx];

      if (valid_kps >= 4) {
        float kp_w = max_x - min_x;
        float kp_h = max_y - min_y;

        int tx1 = static_cast<int>(min_x - kp_w * 0.06f);
        int ty1 = static_cast<int>(min_y - kp_h * 0.08f);
        int tx2 = static_cast<int>(max_x + kp_w * 0.06f);
        int ty2 = static_cast<int>(max_y + kp_h * 0.06f);

        if (tx2 > tx1 && ty2 > ty1) {
          final_box = cv::Rect(tx1, ty1, tx2 - tx1, ty2 - ty1);
        }
      }

      // 6. Clamp to image boundary
      int x1 = std::max(0, std::min(final_box.x, img_w - 1));
      int y1 = std::max(0, std::min(final_box.y, img_h - 1));
      int x2 = std::max(0, std::min(final_box.x + final_box.width, img_w));
      int y2 = std::max(0, std::min(final_box.y + final_box.height, img_h));

      int clamped_w = x2 - x1;
      int clamped_h = y2 - y1;

      if (clamped_w < 10 || clamped_h < 10) continue;

      opl_interfaces::msg::TrackedHuman human_msg;
      human_msg.confidence = confidences[idx];
      human_msg.bbox.x_offset = x1;
      human_msg.bbox.y_offset = y1;
      human_msg.bbox.width = clamped_w;
      human_msg.bbox.height = clamped_h;

      detections_msg.humans.push_back(human_msg);
    }

    detection_pub_->publish(detections_msg);
    RCLCPP_DEBUG(rclcpp::get_logger("vision_timing"),
      "[timing] YOLO frame=%d.%09u detections=%zu",
      msg->header.stamp.sec, msg->header.stamp.nanosec, detections_msg.humans.size());

  } catch (const cv_bridge::Exception& e) {
    RCLCPP_ERROR(get_logger(), "cv_bridge exception: %s", e.what());
  } catch (const std::exception& e) {
    RCLCPP_ERROR(get_logger(), "Standard exception in imageCallback: %s", e.what());
  }
}

}  // namespace opl_human_vision

RCLCPP_COMPONENTS_REGISTER_NODE(opl_human_vision::HumanDetectorNode)
