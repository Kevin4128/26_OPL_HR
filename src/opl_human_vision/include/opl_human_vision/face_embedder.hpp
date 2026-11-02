#ifndef OPL_HUMAN_VISION__FACE_EMBEDDER_HPP_
#define OPL_HUMAN_VISION__FACE_EMBEDDER_HPP_

#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "opl_human_vision/hailo_model.hpp"

namespace opl_human_vision {

// ArcFace face embedder on the Hailo-8L (arcface_mobilefacenet by default,
// arcface_r50 optional). Replaces the Jetson AdaFace IR50 TensorRT engine;
// embeddings are NOT compatible with AdaFace, so galleries must be re-enrolled.
class FaceEmbedder {
public:
  explicit FaceEmbedder(const std::string & model_path, HailoModelOptions options = {});

  bool init();
  const std::string & lastError() const {return error_;}

  // 5-point similarity transform to the InsightFace 112x112 template (BGR in/out).
  cv::Mat alignFace(const cv::Mat & src, const std::vector<cv::Point2f> & landmarks) const;

  // Returns an L2-normalised embedding, or empty on failure.
  std::vector<float> extract(const cv::Mat & aligned_face_bgr);

private:
  std::string model_path_;
  HailoModelOptions options_;
  std::string error_;
  std::shared_ptr<HailoModel> model_;  // shared per HEF within the process
  int input_w_{112};
  int input_h_{112};
};

}  // namespace opl_human_vision

#endif  // OPL_HUMAN_VISION__FACE_EMBEDDER_HPP_
