#include "opl_human_vision/trt_engine.hpp"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn/dnn.hpp>

#ifndef HAVE_TENSORRT
#include <opencv2/objdetect.hpp>
#endif

namespace opl_human_vision {

TrtEngine::TrtEngine(const std::string& engine_path, float conf_thresh, float nms_thresh, int device_id)
: engine_path_(engine_path), conf_thresh_(conf_thresh), nms_thresh_(nms_thresh), device_id_(device_id) {
  std::cout << "[TrtEngine] Initializing engine with path: " << engine_path_ << " on GPU device " << device_id_ << std::endl;
  if (!loadEngine()) {
    std::cerr << "[TrtEngine] FAILED to load engine!" << std::endl;
  }
}

TrtEngine::~TrtEngine() {
  cleanup();
  std::cout << "[TrtEngine] Resources released." << std::endl;
}

void TrtEngine::cleanup() {
#ifdef HAVE_TENSORRT
  if (device_id_ >= 0) {
    cudaSetDevice(device_id_);
  }
  if (stream_) { cudaStreamDestroy(stream_); stream_ = nullptr; }
  if (gpu_input_buffer_) { cudaFree(gpu_input_buffer_); gpu_input_buffer_ = nullptr; }
  if (gpu_output_buffer_) { cudaFree(gpu_output_buffer_); gpu_output_buffer_ = nullptr; }

  // Unique pointers safely reset execution context and engine objects
  context_.reset();
  engine_.reset();
  runtime_.reset();
#endif
}

bool TrtEngine::loadEngine() {
#ifdef HAVE_TENSORRT
  cudaError_t err = cudaSetDevice(device_id_);
  if (err != cudaSuccess) {
    std::cerr << "[TrtEngine] Failed to set CUDA device " << device_id_ << std::endl;
    return false;
  }
  std::cout << "[TrtEngine] Loading TensorRT engine into VRAM from " << engine_path_ << std::endl;

  std::ifstream file(engine_path_, std::ios::binary);
  if (!file.good()) {
    std::cerr << "[TrtEngine] Error reading engine file at: " << engine_path_ << std::endl;
    return false;
  }

  file.seekg(0, file.end);
  size_t size = file.tellg();
  file.seekg(0, file.beg);
  std::vector<char> trtModelStream(size);
  file.read(trtModelStream.data(), size);
  file.close();

  runtime_.reset(nvinfer1::createInferRuntime(logger_));
  if (!runtime_) { cleanup(); return false; }

  engine_.reset(runtime_->deserializeCudaEngine(trtModelStream.data(), size));
  if (!engine_) { cleanup(); return false; }

  context_.reset(engine_->createExecutionContext());
  if (!context_) { cleanup(); return false; }

  cudaStreamCreate(&stream_);

  // Dynamically extract tensor names and dimensions
  for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
    const char* t_name = engine_->getIOTensorName(i);
    if (engine_->getTensorIOMode(t_name) == nvinfer1::TensorIOMode::kINPUT) {
      input_name_ = t_name;
      nvinfer1::Dims dims = engine_->getTensorShape(t_name);
      input_h_ = dims.d[2];
      input_w_ = dims.d[3];
      input_size_ = 1 * 3 * input_h_ * input_w_ * sizeof(float);
    } else if (engine_->getTensorIOMode(t_name) == nvinfer1::TensorIOMode::kOUTPUT) {
      output_name_ = t_name;
      nvinfer1::Dims dims = engine_->getTensorShape(t_name);
      num_channels_ = dims.d[1]; // 84 (COCO) or 56 (Pose)
      num_anchors_ = dims.d[2];  // Usually 8400
      output_size_ = 1 * num_channels_ * num_anchors_ * sizeof(float);
    }
  }

  if (input_name_.empty() || output_name_.empty()) { cleanup(); return false; }

  // Detect model architecture type
  is_pose_model_ = (num_channels_ == 56);
  std::cout << "[TrtEngine] Tensor Dimensions: Channels=" << num_channels_
            << ", Anchors=" << num_anchors_
            << " (" << (is_pose_model_ ? "YOLOv8-Pose" : "Standard YOLOv8") << ")" << std::endl;

  cudaMalloc(&gpu_input_buffer_, input_size_);
  cudaMalloc(&gpu_output_buffer_, output_size_);
  cpu_output_buffer_.resize(num_channels_ * num_anchors_);

  context_->setTensorAddress(input_name_.c_str(), gpu_input_buffer_);
  context_->setTensorAddress(output_name_.c_str(), gpu_output_buffer_);

  loaded_ = true;
  return true;
#else
  std::cout << "[TrtEngine] Running CPU Mode: Using OpenCV HOG People Detector." << std::endl;
  return true;
#endif
}

#ifdef HAVE_TENSORRT
void TrtEngine::preprocess(const cv::Mat& frame) {
  cv::Mat blob = cv::dnn::blobFromImage(
      frame, 1.0 / 255.0, cv::Size(input_w_, input_h_),
      cv::Scalar(0, 0, 0), true, false
  );
  cudaMemcpyAsync(gpu_input_buffer_, blob.ptr<float>(), input_size_, cudaMemcpyHostToDevice, stream_);
}

std::vector<Detection> TrtEngine::postprocess(const cv::Size& original_shape) {
  std::vector<Detection> detections;
  std::vector<cv::Rect> boxes;
  std::vector<float> confidences;
  const float scale_x = static_cast<float>(original_shape.width) / input_w_;
  const float scale_y = static_cast<float>(original_shape.height) / input_h_;

  for (int i = 0; i < num_anchors_; ++i) {
    const float confidence = cpu_output_buffer_[4 * num_anchors_ + i];
    if (confidence > conf_thresh_) {
      const float cx = cpu_output_buffer_[i];
      const float cy = cpu_output_buffer_[num_anchors_ + i];
      const float w = cpu_output_buffer_[2 * num_anchors_ + i];
      const float h = cpu_output_buffer_[3 * num_anchors_ + i];
      const int left = static_cast<int>((cx - 0.5f * w) * scale_x);
      const int top = static_cast<int>((cy - 0.5f * h) * scale_y);
      const int width = static_cast<int>(w * scale_x);
      const int height = static_cast<int>(h * scale_y);
      boxes.emplace_back(left, top, width, height);
      confidences.push_back(confidence);
    }
  }

  std::vector<int> indices;
  cv::dnn::NMSBoxes(boxes, confidences, conf_thresh_, nms_thresh_, indices);
  for (int idx : indices) {
    Detection det;
    det.class_id = 0;
    det.confidence = confidences[idx];
    det.box = boxes[idx];
    detections.push_back(det);
  }
  return detections;
}
#endif

std::vector<Detection> TrtEngine::infer(const cv::Mat& frame) {
  if (frame.empty()) return {};

#ifdef HAVE_TENSORRT
  cudaSetDevice(device_id_);

  preprocess(frame);
  context_->enqueueV3(stream_);
  cudaMemcpyAsync(cpu_output_buffer_.data(), gpu_output_buffer_, output_size_, cudaMemcpyDeviceToHost, stream_);
  cudaStreamSynchronize(stream_);

  return postprocess(cv::Size(frame.cols, frame.rows));
#else
  std::vector<Detection> detections;
  static cv::HOGDescriptor hog;
  static bool hog_initialized = false;
  if (!hog_initialized) {
    hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());
    hog_initialized = true;
  }

  cv::Mat small_frame;
  float scale = 1.0f;
  if (frame.cols > 640) {
    scale = static_cast<float>(frame.cols) / 640.0f;
    cv::resize(frame, small_frame, cv::Size(640, static_cast<int>(frame.rows / scale)));
  } else {
    small_frame = frame;
  }

  std::vector<cv::Rect> found_boxes;
  std::vector<double> weights;
  hog.detectMultiScale(small_frame, found_boxes, weights, 0, cv::Size(8, 8), cv::Size(16, 16), 1.05, 2.0);

  for (size_t i = 0; i < found_boxes.size(); ++i) {
    if (weights[i] < 0.2) continue;
    Detection det;
    det.class_id = 0;
    det.confidence = static_cast<float>(weights[i]);
    det.box = cv::Rect(
      static_cast<int>(found_boxes[i].x * scale),
      static_cast<int>(found_boxes[i].y * scale),
      static_cast<int>(found_boxes[i].width * scale),
      static_cast<int>(found_boxes[i].height * scale)
    );
    detections.push_back(det);
  }
  return detections;
#endif
}

}  // namespace opl_human_vision
