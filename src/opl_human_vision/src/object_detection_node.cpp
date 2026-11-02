#include "opl_human_vision/object_detection_node.hpp"
#include "opl_human_vision/object_perception.hpp"
#include "opl_human_vision/hailo_model.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <deque>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <cv_bridge/cv_bridge.hpp>
#include <diagnostic_msgs/msg/diagnostic_array.hpp>
#include <opl_interfaces/msg/detected_object_array.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <tf2/exceptions.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

namespace opl_human_vision {
namespace {
using Steady=std::chrono::steady_clock;
double ms(Steady::duration d) {return std::chrono::duration<double,std::milli>(d).count();}
// YOLOv8 COCO HEF with on-chip NMS (yolov8s / yolov8m) on the Hailo-8L.
class ObjectEngine {
  std::unique_ptr<HailoModel> model_;
public:
  void load(const std::string &path, const HailoModelOptions &options, float score_threshold, float iou_threshold) {
    HailoModelOptions o=options;
    o.nms_score_threshold=score_threshold; o.nms_iou_threshold=iou_threshold;
    std::string error;
    model_=HailoModel::create(path,o,error);
    if (!model_) throw std::runtime_error(error);
    if (!model_->hasNms() || model_->nmsNumClasses()!=80)
      throw std::runtime_error("Expected a COCO-80 YOLO HEF with on-chip NMS (e.g. yolov8s.hef): "+path);
  }
  cv::Mat input() {return model_->inputMat();}
  double infer() {
    const auto start=Steady::now();
    std::string error;
    if (!model_->run(error)) throw std::runtime_error(error);
    return ms(Steady::now()-start);
  }
  std::vector<objects::Detection> detections(cv::Size size, objects::Letterbox pad, float confidence) const {
    return objects::decodeHailoNms(model_->nmsData(),model_->nmsFloats(),model_->nmsNumClasses(),
      model_->inputWidth(),model_->inputHeight(),size,pad,confidence);
  }
};
}
struct ObjectDetectionNode::Impl {
  ObjectDetectionNode &node;
  std::unique_ptr<ObjectEngine> engine;
  using Image=sensor_msgs::msg::Image;
  using Info=sensor_msgs::msg::CameraInfo;
  using Array=opl_interfaces::msg::DetectedObjectArray;
  std::mutex mutex,processing_mutex;
  std::atomic<bool> active{false};
  Image::ConstSharedPtr rgb;
  Steady::time_point received_at{},pending_since{},last_start{},activated_at{};
  std::deque<Image::ConstSharedPtr> depths;
  std::deque<Info::ConstSharedPtr> infos;
  rclcpp::Subscription<Image>::SharedPtr rgb_sub,depth_sub;
  rclcpp::Subscription<Info>::SharedPtr info_sub;
  rclcpp::CallbackGroup::SharedPtr reception,work;
  rclcpp::TimerBase::SharedPtr timer;
  rclcpp_lifecycle::LifecyclePublisher<Array>::SharedPtr publisher;
  rclcpp_lifecycle::LifecyclePublisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diagnostics;
  std::unique_ptr<tf2_ros::Buffer> tf;
  std::unique_ptr<tf2_ros::TransformListener> listener;
  double confidence=0.35,nms=0.5,hz=10,slop=0.02,wait=0.1,max_age=0.25,min_depth=0.2,max_depth=10;
  std::string fixed_frame,optical_frame;
  double tf_timeout=0.1,tf_retry_period=0.005;
  struct Pending {
    Array message;
    Steady::time_point received_at,queued_at,deadline,retry_at;
    double inference,processing;
    std::string reason,detail;
  };
  std::deque<Pending> pending_tf;
  uint64_t tf_attempts=0,tf_waited=0,tf_timeouts=0;
  uint64_t received=0,dropped=0,processed=0,errors=0,next_id=0;
  int64_t last_rgb_stamp=-1;
  std::atomic<int64_t> last_processed_stamp{-1};
  std::deque<double> captures,pipelines;
  explicit Impl(ObjectDetectionNode &n):node(n) {}
  static int64_t stamp(const std_msgs::msg::Header &h) {return rclcpp::Time(h.stamp).nanoseconds();}
  void receive(Image::ConstSharedPtr msg) {
    std::lock_guard<std::mutex> lock(mutex);
    ++received;
    const auto s=stamp(msg->header);
    if (s<=last_rgb_stamp) {
      // A backward /clock jump starts a new observation epoch, never reuses old depth/TF.
      if (s<last_rgb_stamp && node.now().nanoseconds()<last_rgb_stamp) {
        depths.clear(); infos.clear(); last_processed_stamp=-1; tf->clear();
      } else {++dropped; return;}
    }
    last_rgb_stamp=s;
    if (rgb) ++dropped;
    else pending_since=Steady::now();
    rgb=msg; received_at=Steady::now();
  }
  void stop() {
    active=false;
    std::lock_guard<std::mutex> processing(processing_mutex);
    timer.reset(); rgb_sub.reset(); depth_sub.reset(); info_sub.reset();
    std::lock_guard<std::mutex> lock(mutex);
    rgb.reset(); depths.clear(); infos.clear(); pending_tf.clear();
    last_rgb_stamp=-1; last_processed_stamp=-1;
  }
  void tick() {
    std::lock_guard<std::mutex> processing(processing_mutex);
    if (!active) return;
    flush_tf();
    if (last_start!=Steady::time_point{} && ms(Steady::now()-last_start)<1000/hz) return;
    Image::ConstSharedPtr image,depth;
    Info::ConstSharedPtr info;
    Steady::time_point reception_time;
    {
      std::lock_guard<std::mutex> lock(mutex);
      if (!rgb) return;
      const int64_t s=stamp(rgb->header);
      int64_t best=static_cast<int64_t>(slop*1e9)+1;
      for (const auto &candidate:depths) {
        const auto delta=std::abs(stamp(candidate->header)-s);
        if (delta<best) {best=delta; depth=candidate;}
      }
      int64_t info_best=static_cast<int64_t>(slop*1e9)+1;
      for (const auto &candidate:infos) {
        const auto delta=std::abs(stamp(candidate->header)-s);
        if (delta<info_best) {info_best=delta; info=candidate;}
      }
      if ((!depth || !info) && ms(Steady::now()-pending_since)<wait*1000) return;
      image=std::exchange(rgb,nullptr); reception_time=received_at;
    }
    last_start=Steady::now();
    const auto s=stamp(image->header);
    const double age=(node.now().nanoseconds()-s)*1e-9;
    if (s<=0 || node.now().nanoseconds()<=0 || age< -slop || age>max_age ||
        s<=last_processed_stamp || image->header.frame_id.empty()) {
      {std::lock_guard<std::mutex> lock(mutex); ++dropped;}
      metrics(image->header,0,0,ms(Steady::now()-reception_time),"rejected_rgb_timestamp_or_frame"); return;
    }
    last_processed_stamp=s;
    try {
      const auto frame=cv_bridge::toCvShare(image,"bgr8");
      cv::Mat input=engine->input();
      const auto pad=objects::letterbox(frame->image,input);
      const double inference=engine->infer();
      const auto detections=engine->detections(frame->image.size(),pad,static_cast<float>(confidence));
      std::string depth_error;
      cv_bridge::CvImageConstPtr depth_cv;
      if (image->header.frame_id!=optical_frame) depth_error="unexpected_optical_frame";
      else if (!depth) depth_error="missing_or_unsynchronized_depth";
      else if (depth->header.frame_id!=image->header.frame_id) depth_error="depth_frame_mismatch";
      else if (depth->width!=image->width || depth->height!=image->height) depth_error="unaligned_depth_dimensions";
      else if (depth->encoding!="16UC1" && depth->encoding!="32FC1") depth_error="unsupported_depth_encoding";
      else {
        try {depth_cv=cv_bridge::toCvShare(depth,depth->encoding);}
        catch (const std::exception &) {depth_error="malformed_depth";}
      }
      if (depth_error.empty()) {
        if (!info) depth_error="missing_or_unsynchronized_camera_info";
        else if (info->header.frame_id!=image->header.frame_id) depth_error="calibration_frame_mismatch";
        else if (info->width!=image->width || info->height!=image->height ||
            info->binning_x>1 || info->binning_y>1 || info->roi.x_offset || info->roi.y_offset)
          depth_error="calibration_dimensions_mismatch";
        else if (std::any_of(info->k.begin(),info->k.end(),[](double v){return !std::isfinite(v);}) ||
            info->k[1]!=0 || info->k[3]!=0 || info->k[6]!=0 || info->k[7]!=0 || info->k[8]!=1 ||
            !std::isfinite(info->k[2]) || !std::isfinite(info->k[5]) ||
            info->k[0]<=0 || info->k[4]<=0 || info->k[2]<0 || info->k[2]>=image->width ||
            info->k[5]<0 || info->k[5]>=image->height ||
            std::any_of(info->d.begin(),info->d.end(),[](double d){return !std::isfinite(d)||std::abs(d)>1e-8;}))
          depth_error="invalid_or_nonrectified_calibration";
      }
      Array output; output.header=image->header;
      for (const auto &det:detections) {
        opl_interfaces::msg::DetectedObject object;
        object.detection_id=++next_id; object.class_id=det.class_id;
        object.class_name=objects::labels.at(det.class_id); object.confidence=det.confidence;
        object.bbox.x_offset=det.box.x; object.bbox.y_offset=det.box.y;
        object.bbox.width=det.box.width; object.bbox.height=det.box.height;
        object.camera_pose.header=image->header; object.camera_pose.pose.orientation.w=1;
        object.fixed_pose.header=image->header; object.fixed_pose.header.frame_id=fixed_frame;
        object.fixed_pose.pose.orientation.w=1;
        if (depth) object.depth_header=depth->header;
        objects::Surface position;
        if (depth_error.empty()) position=objects::surface(depth_cv->image,det.box,
          info->k[0],info->k[4],info->k[2],info->k[5],min_depth,max_depth);
        else position.status=depth_error;
        object.position_valid=position.valid; object.position_status=position.status;
        object.depth_valid_pixels=position.pixels; object.depth_mad_m=position.mad;
        const auto color=objects::color(frame->image,det.box,position.mask);
        object.color=color.name; object.color_confidence=color.confidence;
        object.fixed_status=position.valid?"tf_unavailable":"camera_position_invalid";
        if (position.valid) {
          auto &p=object.camera_pose.pose.position;
          p.x=position.point.x; p.y=position.point.y; p.z=position.point.z;
        }
        output.objects.push_back(std::move(object));
      }
      Pending pending{std::move(output),reception_time,Steady::now(),{}, {},inference,
        ms(Steady::now()-last_start),"",""};
      if (transform_frame(pending) || tf_timeout==0) publish_result(pending);
      else {
        ++tf_waited;
        pending.deadline=pending.queued_at+std::chrono::duration_cast<Steady::duration>(std::chrono::duration<double>(tf_timeout));
        pending.retry_at=Steady::now()+std::chrono::duration_cast<Steady::duration>(std::chrono::duration<double>(tf_retry_period));
        if (pending_tf.size()==3) {
          pending_tf.front().reason="tf_retry_queue_full:"+pending_tf.front().reason;
          publish_result(pending_tf.front()); pending_tf.pop_front();
        }
        pending_tf.push_back(std::move(pending));
      }
    } catch (const std::exception &e) {
      ++errors;
      RCLCPP_ERROR_THROTTLE(node.get_logger(),*node.get_clock(),5000,"Object inference failed: %s",e.what());
      metrics(image->header,0,0,ms(Steady::now()-reception_time),e.what());
    }
  }
  bool transform_frame(Pending &pending) {
    auto &message=pending.message;
    if (std::none_of(message.objects.begin(),message.objects.end(),[](const auto &o){return o.position_valid;})) return true;
    try {
      ++tf_attempts;
      // One timestamped lookup per observation attempt, shared by every object.
      const auto transform=tf->lookupTransform(fixed_frame,message.header.frame_id,
        rclcpp::Time(message.header.stamp),rclcpp::Duration::from_seconds(0));
      for (auto &object:message.objects) if (object.position_valid) {
        geometry_msgs::msg::PointStamped camera,fixed;
        camera.header=message.header; camera.point=object.camera_pose.pose.position;
        tf2::doTransform(camera,fixed,transform);
        const auto &p=fixed.point;
        if (!std::isfinite(p.x)||!std::isfinite(p.y)||!std::isfinite(p.z)) {
          object.fixed_status="tf_nonfinite_position"; continue;
        }
        object.fixed_pose.pose.position=p;
        object.fixed_position_valid=true; object.fixed_status="valid_surface_point";
      }
      return true;
    } catch (const tf2::TransformException &e) {
      pending.detail=e.what();
      if (dynamic_cast<const tf2::LookupException *>(&e)) pending.reason="tf_missing_frame";
      else if (dynamic_cast<const tf2::ConnectivityException *>(&e)) pending.reason="tf_disconnected";
      else if (dynamic_cast<const tf2::ForwardExtrapolationException *>(&e)) pending.reason="tf_future_extrapolation";
      else if (dynamic_cast<const tf2::BackwardExtrapolationException *>(&e)) pending.reason="tf_past_extrapolation";
      else if (dynamic_cast<const tf2::NoDataForExtrapolationException *>(&e)) pending.reason="tf_insufficient_history";
      else if (dynamic_cast<const tf2::ExtrapolationException *>(&e)) pending.reason="tf_extrapolation";
      else pending.reason="tf_error";
      return false;
    }
  }
  void publish_result(Pending &pending) {
    const auto &header=pending.message.header;
    const double age=(node.now().nanoseconds()-stamp(header))*1e-9;
    if (age< -slop || age>max_age) {
      {std::lock_guard<std::mutex> lock(mutex); ++dropped;}
      metrics(header,pending.inference,age*1000,ms(Steady::now()-pending.received_at),"stale_after_inference",pending.processing);
      return;
    }
    for (auto &object:pending.message.objects) if (object.position_valid&&!object.fixed_position_valid&&!pending.reason.empty())
      object.fixed_status=pending.reason;
    if (!pending.reason.empty())
      RCLCPP_WARN_THROTTLE(node.get_logger(),*node.get_clock(),5000,"Object TF (%s): %s",
        pending.reason.c_str(),pending.detail.c_str());
    publisher->publish(pending.message); ++processed;
    metrics(header,pending.inference,age*1000,ms(Steady::now()-pending.received_at),"ok",
      pending.processing,ms(Steady::now()-pending.queued_at));
  }
  void flush_tf() {
    for (auto i=pending_tf.begin();i!=pending_tf.end();) {
      const auto now=Steady::now();
      const double age=(node.now().nanoseconds()-stamp(i->message.header))*1e-9;
      if (age< -slop || age>max_age) {
        publish_result(*i); i=pending_tf.erase(i);
      } else if (now>=i->retry_at || now>=i->deadline) {
        // Inference may span a deadline; use any transform already in the buffer.
        if (transform_frame(*i)) {i->reason.clear();publish_result(*i);i=pending_tf.erase(i);}
        else if (now>=i->deadline) {
          ++tf_timeouts; i->reason="tf_timeout:"+i->reason;
          publish_result(*i); i=pending_tf.erase(i);
        } else {i->retry_at=now+std::chrono::duration_cast<Steady::duration>(std::chrono::duration<double>(tf_retry_period));++i;}
      } else ++i;
    }
  }
  void metrics(const std_msgs::msg::Header &header,double inference,double capture,double pipeline,const std::string &status,double processing=0,double tf_wait=0) {
    if (status=="ok") {
      captures.push_back(capture); pipelines.push_back(pipeline);
      if (captures.size()>300) {captures.pop_front(); pipelines.pop_front();}
    }
    auto p95=[](const std::deque<double> &values) {
      if (values.empty()) return 0.0;
      std::vector<double> sorted(values.begin(),values.end()); std::sort(sorted.begin(),sorted.end());
      return sorted[static_cast<size_t>(std::ceil(0.95*sorted.size()))-1];
    };
    diagnostic_msgs::msg::DiagnosticArray message; message.header=header;
    diagnostic_msgs::msg::DiagnosticStatus state;
    state.name="object_detection"; state.hardware_id="Hailo-8L";
    state.level=status=="ok"?0:1; state.message=status;
    auto add=[&](const std::string &key,auto value) {
      diagnostic_msgs::msg::KeyValue item; item.key=key; item.value=std::to_string(value); state.values.push_back(item);
    };
    {std::lock_guard<std::mutex> lock(mutex); add("received_frames",received); add("dropped_frames",dropped);}
    add("processed_frames",processed); add("inference_errors",errors);
    add("inference_wall_ms",inference); add("capture_to_publication_ros_ms",capture);
    add("reception_to_publication_wall_ms",pipeline);
    add("capture_p95_ros_ms",p95(captures)); add("reception_p95_wall_ms",p95(pipelines));
    add("processed_wall_fps",processed/(ms(Steady::now()-activated_at)/1000));
    add("processing_wall_ms",processing); add("tf_wait_wall_ms",tf_wait);
    add("tf_lookup_attempts",tf_attempts); add("tf_waited_frames",tf_waited); add("tf_timeouts",tf_timeouts);
    message.status.push_back(state); diagnostics->publish(message);
  }
};
ObjectDetectionNode::ObjectDetectionNode(const rclcpp::NodeOptions &options)
:LifecycleNode("object_detection_node",options),impl_(std::make_unique<Impl>(*this)) {
  declare_parameter("model_path",std::string(""));
  declare_parameter("hailo_multi_process",false); declare_parameter("hailo_priority",8);
  declare_parameter("image_topic",std::string("/head_cam/image"));
  declare_parameter("depth_topic",std::string("/head_cam/depth_image"));
  declare_parameter("camera_info_topic",std::string("/head_cam/camera_info"));
  declare_parameter("detections_topic",std::string("/detected_objects"));
  declare_parameter("fixed_frame",std::string("odom"));
  declare_parameter("optical_frame",std::string("head_cam_optical_link"));
  declare_parameter("confidence_threshold",0.35); declare_parameter("nms_threshold",0.5);
  declare_parameter("inference_hz",15.0); declare_parameter("sync_slop_s",0.02);
  declare_parameter("depth_wait_s",0.1); declare_parameter("max_observation_age_s",0.25);
  declare_parameter("tf_timeout_s",0.1); declare_parameter("tf_retry_period_s",0.005);
  declare_parameter("min_depth_m",0.2); declare_parameter("max_depth_m",10.0);
}
ObjectDetectionNode::~ObjectDetectionNode()=default;
ObjectDetectionNode::Return ObjectDetectionNode::on_configure(const rclcpp_lifecycle::State &) {
  try {
    auto &p=*impl_;
    p.confidence=get_parameter("confidence_threshold").as_double(); p.nms=get_parameter("nms_threshold").as_double();
    p.hz=get_parameter("inference_hz").as_double(); p.slop=get_parameter("sync_slop_s").as_double();
    p.wait=get_parameter("depth_wait_s").as_double(); p.max_age=get_parameter("max_observation_age_s").as_double();
    p.min_depth=get_parameter("min_depth_m").as_double(); p.max_depth=get_parameter("max_depth_m").as_double();
    p.tf_timeout=get_parameter("tf_timeout_s").as_double();
    p.tf_retry_period=get_parameter("tf_retry_period_s").as_double();
    p.fixed_frame=get_parameter("fixed_frame").as_string();
    p.optical_frame=get_parameter("optical_frame").as_string();
    for (double v:{p.confidence,p.nms,p.hz,p.slop,p.wait,p.max_age,p.min_depth,p.max_depth,p.tf_timeout,p.tf_retry_period})
      if (!std::isfinite(v)) throw std::runtime_error("Parameters must be finite");
    if (p.confidence<=0 || p.confidence>1 || p.nms<=0 || p.nms>1 || p.hz<=0 || p.hz>60 ||
        p.slop<0 || p.slop>0.1 || p.wait<0 || p.wait>p.max_age || p.max_age<=0 ||
        p.tf_timeout<0 || p.tf_timeout>0.2 || p.tf_retry_period<=0 || p.tf_retry_period>0.05 ||
        p.min_depth<=0 || p.max_depth<=p.min_depth || p.fixed_frame.empty() || p.optical_frame.empty())
      throw std::runtime_error("Invalid thresholds, timing, depth range or fixed frame");
    p.engine=std::make_unique<ObjectEngine>();
    HailoModelOptions hailo;
    hailo.multi_process_service=get_parameter("hailo_multi_process").as_bool();
    hailo.scheduler_priority=static_cast<int>(get_parameter("hailo_priority").as_int());
    p.engine->load(get_parameter("model_path").as_string(),hailo,
      static_cast<float>(p.confidence),static_cast<float>(p.nms));
    p.publisher=create_publisher<Impl::Array>(get_parameter("detections_topic").as_string(),10);
    p.diagnostics=create_publisher<diagnostic_msgs::msg::DiagnosticArray>("/object_detection/diagnostics",10);
    p.tf=std::make_unique<tf2_ros::Buffer>(get_clock());
    p.listener=std::make_unique<tf2_ros::TransformListener>(*p.tf);
    p.reception=create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    p.work=create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    RCLCPP_INFO(get_logger(),"YOLOv8 COCO HEF configured on Hailo-8L (on-chip NMS); fixed_frame=%s",p.fixed_frame.c_str());
    return Return::SUCCESS;
  } catch (const std::exception &e) {
    impl_->engine.reset(); RCLCPP_ERROR(get_logger(),"Object configure failed: %s",e.what()); return Return::FAILURE;
  }
}
ObjectDetectionNode::Return ObjectDetectionNode::on_activate(const rclcpp_lifecycle::State &) {
  auto &p=*impl_;
  p.received=p.dropped=p.processed=p.errors=0; p.tf_attempts=p.tf_waited=p.tf_timeouts=0; p.captures.clear(); p.pipelines.clear();
  p.last_start={}; p.activated_at=Steady::now();
  p.publisher->on_activate(); p.diagnostics->on_activate();
  rclcpp::SubscriptionOptions opts; opts.callback_group=p.reception;
  auto qos=rclcpp::SensorDataQoS().keep_last(5);
  p.rgb_sub=create_subscription<Impl::Image>(get_parameter("image_topic").as_string(),qos,
    [&p](Impl::Image::ConstSharedPtr m){p.receive(m);},opts);
  p.depth_sub=create_subscription<Impl::Image>(get_parameter("depth_topic").as_string(),qos,
    [&p](Impl::Image::ConstSharedPtr m){std::lock_guard<std::mutex> lock(p.mutex);
      p.depths.push_back(m); if (p.depths.size()>20) p.depths.pop_front();},opts);
  p.info_sub=create_subscription<Impl::Info>(get_parameter("camera_info_topic").as_string(),qos,
    [&p](Impl::Info::ConstSharedPtr m){std::lock_guard<std::mutex> lock(p.mutex);
      p.infos.push_back(m); if (p.infos.size()>20) p.infos.pop_front();},opts);
  p.active=true;
  // The timer only polls for a synchronised frame / pending TF; tie its period
  // to the TF retry period instead of a fixed 2 ms (500 wakeups/s on the Pi).
  const auto period=std::chrono::microseconds(static_cast<int64_t>(
    std::clamp(p.tf_retry_period,0.002,0.01)*1e6));
  p.timer=create_wall_timer(period,[&p](){p.tick();},p.work);
  return Return::SUCCESS;
}
ObjectDetectionNode::Return ObjectDetectionNode::on_deactivate(const rclcpp_lifecycle::State &) {
  impl_->stop(); impl_->publisher->on_deactivate(); impl_->diagnostics->on_deactivate(); return Return::SUCCESS;
}
ObjectDetectionNode::Return ObjectDetectionNode::on_cleanup(const rclcpp_lifecycle::State &) {
  impl_->stop(); impl_->listener.reset(); impl_->tf.reset(); impl_->engine.reset();
  impl_->publisher.reset(); impl_->diagnostics.reset(); return Return::SUCCESS;
}
ObjectDetectionNode::Return ObjectDetectionNode::on_shutdown(const rclcpp_lifecycle::State &state) {return on_cleanup(state);}
}
RCLCPP_COMPONENTS_REGISTER_NODE(opl_human_vision::ObjectDetectionNode)
