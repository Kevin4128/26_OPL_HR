#include "opl_human_vision/scrfd_detector.hpp"
#include "opl_human_vision/scoped_timing.hpp"

#include <algorithm>
#include <iostream>

namespace opl_human_vision {

ScrfdDetector::ScrfdDetector(const std::string & model_path, HailoModelOptions options)
: model_path_(model_path), options_(options) {}

bool ScrfdDetector::init()
{
  ScopedTiming timing("SCRFD.load (attempt)");
  model_ = HailoModel::create(model_path_, options_, error_);
  if (!model_ || !decoder_.bind(model_->outputs(), model_->inputWidth(), error_)) {
    std::cerr << "[ScrfdDetector] " << error_ << std::endl;
    model_.reset();
    return false;
  }
  return true;
}

bool ScrfdDetector::detect(
  const cv::Mat & human_crop, std::vector<cv::Point2f> & out_landmarks, float & out_score)
{
  ScopedTiming timing("SCRFD.detect");
  if (human_crop.empty() || !model_) {return false;}

  // Faces live in the upper part of a person box; skip the legs.
  const int search_h = std::min(human_crop.rows, static_cast<int>(human_crop.rows * 0.85));
  if (human_crop.cols < 15 || search_h < 15) {return false;}
  const cv::Mat face_roi = human_crop(cv::Rect(0, 0, human_crop.cols, search_h));

  // Top-left letterbox with zero padding, as the original InsightFace preprocessing.
  cv::Mat input = model_->inputMat();
  const LetterboxInfo lb = letterboxInto(face_roi, input, cv::Scalar(0, 0, 0), false);

  if (!model_->run(error_)) {
    std::cerr << "[ScrfdDetector] " << error_ << std::endl;
    return false;
  }

  decoder_.decode(model_->outputs(), conf_threshold_, faces_);
  if (faces_.empty()) {return false;}
  ScrfdDecoder::nms(faces_, nms_threshold_);

  const FaceCandidate & best = faces_.front();
  out_score = best.score;
  out_landmarks.clear();
  for (const auto & lm : best.landmarks) {
    out_landmarks.push_back(lb.toImage(lm.x, lm.y));
  }
  return true;
}

}  // namespace opl_human_vision
