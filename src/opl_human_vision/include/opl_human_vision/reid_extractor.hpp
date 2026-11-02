#ifndef OPL_HUMAN_VISION__REID_EXTRACTOR_HPP_
#define OPL_HUMAN_VISION__REID_EXTRACTOR_HPP_

#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "opl_human_vision/hailo_model.hpp"

namespace opl_human_vision {

// Person re-ID body embedder on the Hailo-8L (repvgg_a0_person_reid_512).
// Replaces the Jetson OSNet x1_0 engine; same 128x256 input and 512-d output,
// but a different embedding space, so similarity thresholds need re-tuning.
class ReidExtractor {
public:
  explicit ReidExtractor(const std::string & model_path, HailoModelOptions options = {});

  bool init();
  const std::string & lastError() const {return error_;}

  // Returns an L2-normalised embedding, or empty on failure / tiny crops.
  std::vector<float> extract(const cv::Mat & human_crop_bgr);

private:
  std::string model_path_;
  HailoModelOptions options_;
  std::string error_;
  std::shared_ptr<HailoModel> model_;  // shared per HEF within the process
};

}  // namespace opl_human_vision

#endif  // OPL_HUMAN_VISION__REID_EXTRACTOR_HPP_
