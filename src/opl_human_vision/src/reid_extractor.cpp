#include "opl_human_vision/reid_extractor.hpp"
#include "opl_human_vision/scoped_timing.hpp"

#include <iostream>

namespace opl_human_vision {

ReidExtractor::ReidExtractor(const std::string & model_path, HailoModelOptions options)
: model_path_(model_path), options_(options) {}

bool ReidExtractor::init()
{
  ScopedTiming timing("ReID.load (attempt)");
  model_ = HailoModel::shared(model_path_, options_, error_);
  if (!model_) {
    std::cerr << "[ReidExtractor] " << error_ << std::endl;
    return false;
  }
  if (model_->numOutputs() != 1) {
    error_ = "Re-ID HEF must have exactly one output: " + model_path_;
    std::cerr << "[ReidExtractor] " << error_ << std::endl;
    model_.reset();
    return false;
  }
  return true;
}

std::vector<float> ReidExtractor::extract(const cv::Mat & human_crop_bgr)
{
  ScopedTiming timing("ReID.extract");
  if (!model_ || human_crop_bgr.empty() || human_crop_bgr.cols < 16 || human_crop_bgr.rows < 16) {
    return {};
  }

  // Stretch resize (no letterbox) to the HEF input, as the original OSNet path.
  // ImageNet mean/std normalisation is compiled into the HEF.
  auto lock = model_->useLock();
  cv::Mat input = model_->inputMat();
  stretchInto(human_crop_bgr, input);

  if (!model_->run(error_)) {
    std::cerr << "[ReidExtractor] " << error_ << std::endl;
    return {};
  }
  return normalizedEmbedding(model_->output(0));
}

}  // namespace opl_human_vision
