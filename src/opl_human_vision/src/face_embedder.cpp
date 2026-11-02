#include "opl_human_vision/face_embedder.hpp"
#include "opl_human_vision/scoped_timing.hpp"

#include <cmath>
#include <iostream>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

namespace opl_human_vision {

FaceEmbedder::FaceEmbedder(const std::string & model_path, HailoModelOptions options)
: model_path_(model_path), options_(options) {}

bool FaceEmbedder::init()
{
  ScopedTiming timing("FaceEmbedder.load (attempt)");
  model_ = HailoModel::shared(model_path_, options_, error_);
  if (!model_) {
    std::cerr << "[FaceEmbedder] " << error_ << std::endl;
    return false;
  }
  if (model_->numOutputs() != 1) {
    error_ = "Face embedding HEF must have exactly one output: " + model_path_;
    std::cerr << "[FaceEmbedder] " << error_ << std::endl;
    model_.reset();
    return false;
  }
  input_w_ = model_->inputWidth();
  input_h_ = model_->inputHeight();
  return true;
}

cv::Mat FaceEmbedder::alignFace(const cv::Mat & src, const std::vector<cv::Point2f> & landmarks) const
{
  if (landmarks.size() != 5 || src.empty()) {return src;}

  // Standard InsightFace 112x112 target landmark positions, scaled to the model input.
  static const std::vector<cv::Point2f> kTemplate112 = {
    {38.2946f, 51.6963f},
    {73.5318f, 51.5014f},
    {56.0252f, 71.7366f},
    {41.5493f, 92.3655f},
    {70.7299f, 92.2041f}
  };
  std::vector<cv::Point2f> dst_pts = kTemplate112;
  const float sx = static_cast<float>(input_w_) / 112.0f;
  const float sy = static_cast<float>(input_h_) / 112.0f;
  for (auto & p : dst_pts) {p.x *= sx; p.y *= sy;}

  cv::Mat M = cv::estimateAffinePartial2D(landmarks, dst_pts);
  cv::Mat aligned;
  if (!M.empty()) {
    cv::warpAffine(src, aligned, M, cv::Size(input_w_, input_h_));
  } else {
    cv::resize(src, aligned, cv::Size(input_w_, input_h_));
  }
  return aligned;
}

std::vector<float> FaceEmbedder::extract(const cv::Mat & aligned_face_bgr)
{
  ScopedTiming timing("FaceEmbedder.extract");
  if (aligned_face_bgr.empty() || !model_) {return {};}

  auto lock = model_->useLock();
  cv::Mat input = model_->inputMat();
  stretchInto(aligned_face_bgr, input);  // no-op resize when already aligned to input size

  if (!model_->run(error_)) {
    std::cerr << "[FaceEmbedder] " << error_ << std::endl;
    return {};
  }
  return normalizedEmbedding(model_->output(0));
}

}  // namespace opl_human_vision
