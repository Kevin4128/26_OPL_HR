#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <functional>
#include <memory>
#include <string>
#include <stdexcept>
#include <utility>
#include <vector>

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>

namespace opl_human_vision {
namespace {

class TrtLogger final : public nvinfer1::ILogger {
public:
  void log(Severity severity, const char * message) noexcept override
  {
    if (severity <= Severity::kWARNING) {
      RCLCPP_WARN(rclcpp::get_logger("yolo11m"), "%s", message);
    }
  }
};

template<typename T>
struct TrtDeleter {
  void operator()(T * value) const { delete value; }
};

struct Candidate {
  int class_id;
  float score;
  cv::Rect box;
};

// COCO class IDs emitted by the pretrained YOLO11m model.
constexpr std::array<int, 9> kDetectionClassIds{39, 41, 42, 43, 44, 56, 57, 59, 60};
constexpr std::array<const char *, 9> kDetectionNames{
  "bottle", "cup", "fork", "knife", "spoon", "chair", "couch", "bed", "dining table"};

int detectionClassIndex(int class_id)
{
  for (size_t i = 0; i < kDetectionClassIds.size(); ++i) {
    if (kDetectionClassIds[i] == class_id) {
      return static_cast<int>(i);
    }
  }
  return -1;
}

bool isLargeFurniture(int class_id)
{
  return class_id == 56 || class_id == 57 || class_id == 59 || class_id == 60;
}

}  // namespace

class ObjectDetectorNode final : public rclcpp::Node {
public:
  ObjectDetectorNode()
  : Node("head_cam_object_detector")
  {
    const auto image_topic = declare_parameter<std::string>("image_topic", "/head_cam/image");
    const auto detections_topic = declare_parameter<std::string>("detections_topic", "/head_cam/objects");
    const auto engine_path = declare_parameter<std::string>("engine_path", "");
    confidence_threshold_ = static_cast<float>(declare_parameter<double>("confidence_threshold", 0.35));
    furniture_confidence_threshold_ = static_cast<float>(
      declare_parameter<double>("furniture_confidence_threshold", 0.55));
    small_object_confidence_threshold_ = static_cast<float>(
      declare_parameter<double>("small_object_confidence_threshold", 0.40));
    nms_threshold_ = static_cast<float>(declare_parameter<double>("nms_threshold", 0.40));

    if (engine_path.empty()) {
      throw std::runtime_error("Set the engine_path parameter to the YOLO11m TensorRT engine.");
    }
    loadEngine(engine_path);
    publisher_ = create_publisher<vision_msgs::msg::Detection2DArray>(
      detections_topic, rclcpp::SensorDataQoS());
    subscription_ = create_subscription<sensor_msgs::msg::Image>(
      image_topic, rclcpp::SensorDataQoS(),
      std::bind(&ObjectDetectorNode::imageCallback, this, std::placeholders::_1));
    RCLCPP_INFO(get_logger(),
      "YOLO11m detector (%zu COCO classes): %s -> %s; confidence furniture=%.2f small=%.2f",
      kDetectionClassIds.size(), image_topic.c_str(), detections_topic.c_str(),
      furniture_confidence_threshold_, small_object_confidence_threshold_);
  }

  ~ObjectDetectorNode() override
  {
    if (device_id_ >= 0) { cudaSetDevice(device_id_); }
    if (stream_) { cudaStreamDestroy(stream_); }
    if (gpu_input_) { cudaFree(gpu_input_); }
    if (gpu_output_) { cudaFree(gpu_output_); }
  }

private:
  void loadEngine(const std::string & path)
  {
    cudaError_t cuda_status = cudaSetDevice(device_id_);
    if (cuda_status != cudaSuccess) {
      throw std::runtime_error(std::string("Could not select CUDA device: ") + cudaGetErrorString(cuda_status));
    }
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) { throw std::runtime_error("Could not open TensorRT engine: " + path); }
    const auto length = file.tellg();
    if (length <= 0) { throw std::runtime_error("TensorRT engine is empty: " + path); }
    std::vector<char> bytes(static_cast<size_t>(length));
    file.seekg(0);
    file.read(bytes.data(), length);

    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    if (!runtime_) { throw std::runtime_error("TensorRT runtime creation failed"); }
    engine_.reset(runtime_->deserializeCudaEngine(bytes.data(), bytes.size()));
    if (!engine_) { throw std::runtime_error("Could not deserialize TensorRT engine: " + path); }
    context_.reset(engine_->createExecutionContext());
    if (!context_) { throw std::runtime_error("Could not create TensorRT execution context"); }

    for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
      const char * name = engine_->getIOTensorName(i);
      const auto dims = engine_->getTensorShape(name);
      if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
        input_name_ = name;
        input_h_ = dims.d[2];
        input_w_ = dims.d[3];
      } else {
        output_name_ = name;
        channels_ = dims.d[1];
        anchors_ = dims.d[2];
      }
    }
    if (input_name_.empty() || output_name_.empty() || channels_ < 64 || anchors_ <= 0) {
      throw std::runtime_error("Unexpected YOLO output; expected [1,84,anchors]");
    }
    input_bytes_ = static_cast<size_t>(3) * input_h_ * input_w_ * sizeof(float);
    output_bytes_ = static_cast<size_t>(channels_) * anchors_ * sizeof(float);
    output_.resize(static_cast<size_t>(channels_) * anchors_);
    if (cudaMalloc(&gpu_input_, input_bytes_) != cudaSuccess ||
      cudaMalloc(&gpu_output_, output_bytes_) != cudaSuccess ||
      cudaStreamCreate(&stream_) != cudaSuccess)
    {
      throw std::runtime_error("Could not allocate YOLO TensorRT buffers");
    }
    context_->setTensorAddress(input_name_.c_str(), gpu_input_);
    context_->setTensorAddress(output_name_.c_str(), gpu_output_);
    RCLCPP_INFO(get_logger(), "Loaded YOLO11m engine (%dx%d, %d channels, %d anchors)",
      input_w_, input_h_, channels_, anchors_);
  }

  std::vector<Candidate> infer(const cv::Mat & frame)
  {
    const float scale = std::min(
      static_cast<float>(input_w_) / frame.cols,
      static_cast<float>(input_h_) / frame.rows);
    const int resized_w = static_cast<int>(std::round(frame.cols * scale));
    const int resized_h = static_cast<int>(std::round(frame.rows * scale));
    const int pad_x = (input_w_ - resized_w) / 2;
    const int pad_y = (input_h_ - resized_h) / 2;
    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(resized_w, resized_h));
    cv::Mat letterboxed(input_h_, input_w_, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(letterboxed(cv::Rect(pad_x, pad_y, resized_w, resized_h)));
    const cv::Mat blob = cv::dnn::blobFromImage(
      letterboxed, 1.0 / 255.0, cv::Size(input_w_, input_h_), cv::Scalar(), true, false);

    cudaMemcpyAsync(gpu_input_, blob.ptr<float>(), input_bytes_, cudaMemcpyHostToDevice, stream_);
    if (!context_->enqueueV3(stream_)) { throw std::runtime_error("TensorRT inference enqueue failed"); }
    cudaMemcpyAsync(output_.data(), gpu_output_, output_bytes_, cudaMemcpyDeviceToHost, stream_);
    if (cudaStreamSynchronize(stream_) != cudaSuccess) {
      throw std::runtime_error("TensorRT inference synchronization failed");
    }

    std::vector<Candidate> results;
    for (int class_id : kDetectionClassIds) {
      const float class_threshold = std::max(
        confidence_threshold_, isLargeFurniture(class_id) ?
        furniture_confidence_threshold_ : small_object_confidence_threshold_);
      std::vector<cv::Rect> boxes;
      std::vector<float> scores;
      for (int anchor = 0; anchor < anchors_; ++anchor) {
        const float score = output_[static_cast<size_t>(4 + class_id) * anchors_ + anchor];
        if (!std::isfinite(score) || score < class_threshold) { continue; }
        const float cx = output_[anchor];
        const float cy = output_[static_cast<size_t>(anchors_) + anchor];
        const float w = output_[static_cast<size_t>(2) * anchors_ + anchor];
        const float h = output_[static_cast<size_t>(3) * anchors_ + anchor];
        int left = static_cast<int>((cx - 0.5f * w - pad_x) / scale);
        int top = static_cast<int>((cy - 0.5f * h - pad_y) / scale);
        int right = static_cast<int>((cx + 0.5f * w - pad_x) / scale);
        int bottom = static_cast<int>((cy + 0.5f * h - pad_y) / scale);
        left = std::clamp(left, 0, frame.cols - 1);
        top = std::clamp(top, 0, frame.rows - 1);
        right = std::clamp(right, 0, frame.cols - 1);
        bottom = std::clamp(bottom, 0, frame.rows - 1);
        if (right > left && bottom > top) {
          boxes.emplace_back(left, top, right - left, bottom - top);
          scores.push_back(score);
        }
      }
      std::vector<int> keep;
      cv::dnn::NMSBoxes(boxes, scores, class_threshold, nms_threshold_, keep);
      for (int idx : keep) { results.push_back({class_id, scores[idx], boxes[idx]}); }
    }
    return results;
  }

  void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr msg)
  {
    try {
      auto frame = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::BGR8)->image;
      vision_msgs::msg::Detection2DArray output;
      output.header = msg->header;
      for (const auto & candidate : infer(frame)) {
        vision_msgs::msg::Detection2D detection;
        detection.header = msg->header;
        detection.id = std::to_string(candidate.class_id);
        detection.bbox.center.position.x = candidate.box.x + candidate.box.width / 2.0;
        detection.bbox.center.position.y = candidate.box.y + candidate.box.height / 2.0;
        detection.bbox.size_x = candidate.box.width;
        detection.bbox.size_y = candidate.box.height;
        vision_msgs::msg::ObjectHypothesisWithPose hypothesis;
        const int index = detectionClassIndex(candidate.class_id);
        if (index < 0) { continue; }
        hypothesis.hypothesis.class_id = kDetectionNames[index];
        hypothesis.hypothesis.score = candidate.score;
        detection.results.push_back(hypothesis);
        output.detections.push_back(std::move(detection));
      }
      publisher_->publish(std::move(output));
    } catch (const std::exception & e) {
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 5000, "Image inference failed: %s", e.what());
    }
  }

  TrtLogger logger_;
  std::unique_ptr<nvinfer1::IRuntime, TrtDeleter<nvinfer1::IRuntime>> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine, TrtDeleter<nvinfer1::ICudaEngine>> engine_;
  std::unique_ptr<nvinfer1::IExecutionContext, TrtDeleter<nvinfer1::IExecutionContext>> context_;
  std::string input_name_;
  std::string output_name_;
  int input_w_{640}, input_h_{640}, channels_{0}, anchors_{0}, device_id_{0};
  float confidence_threshold_{0.35F};
  float furniture_confidence_threshold_{0.55F};
  float small_object_confidence_threshold_{0.40F};
  float nms_threshold_{0.40F};
  size_t input_bytes_{0}, output_bytes_{0};
  void * gpu_input_{nullptr};
  void * gpu_output_{nullptr};
  cudaStream_t stream_{nullptr};
  std::vector<float> output_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
  rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr publisher_;
};

}  // namespace opl_human_vision

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<opl_human_vision::ObjectDetectorNode>());
  } catch (const std::exception & e) {
    RCLCPP_FATAL(rclcpp::get_logger("head_cam_object_detector"), "%s", e.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
