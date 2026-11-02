#ifndef OPL_HUMAN_VISION__SCRFD_DETECTOR_HPP_
#define OPL_HUMAN_VISION__SCRFD_DETECTOR_HPP_

#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "opl_human_vision/hailo_model.hpp"
#include "opl_human_vision/scrfd_decoder.hpp"

namespace opl_human_vision {

// SCRFD face + 5-landmark detector on the Hailo-8L. Runs on a person crop and
// returns the single best face.
class ScrfdDetector {
public:
  explicit ScrfdDetector(const std::string & model_path, HailoModelOptions options = {});

  bool init();
  const std::string & lastError() const {return error_;}

  bool detect(const cv::Mat & human_crop, std::vector<cv::Point2f> & out_landmarks, float & out_score);

  void setConfThreshold(float conf_threshold) {conf_threshold_ = conf_threshold;}
  void setNmsThreshold(float nms_threshold) {nms_threshold_ = nms_threshold;}

private:
  std::string model_path_;
  HailoModelOptions options_;
  std::string error_;

  float conf_threshold_{0.50f};
  float nms_threshold_{0.40f};

  std::unique_ptr<HailoModel> model_;
  ScrfdDecoder decoder_;
  std::vector<FaceCandidate> faces_;
};

}  // namespace opl_human_vision

#endif  // OPL_HUMAN_VISION__SCRFD_DETECTOR_HPP_
