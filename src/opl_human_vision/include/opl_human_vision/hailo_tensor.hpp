#ifndef OPL_HUMAN_VISION__HAILO_TENSOR_HPP_
#define OPL_HUMAN_VISION__HAILO_TENSOR_HPP_

// HailoRT-independent view of one model output, so decoders can be unit-tested
// with synthetic tensors and no accelerator present.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace opl_human_vision {

enum class TensorDType { kUInt8, kUInt16, kFloat32 };

// Output tensor in host NHWC order (or flat NC). Quantised outputs are kept in
// their native integer type and dequantised lazily, which avoids converting
// ~1M values per frame for anchors that never pass the score threshold.
struct TensorView {
  std::string name;
  const void * data{nullptr};
  TensorDType dtype{TensorDType::kFloat32};
  float scale{1.0f};
  float zero_point{0.0f};
  int height{1};
  int width{1};
  int channels{1};

  size_t size() const {return static_cast<size_t>(height) * width * channels;}

  float at(size_t i) const
  {
    switch (dtype) {
      case TensorDType::kUInt8:
        return (static_cast<float>(static_cast<const uint8_t *>(data)[i]) - zero_point) * scale;
      case TensorDType::kUInt16:
        return (static_cast<float>(static_cast<const uint16_t *>(data)[i]) - zero_point) * scale;
      case TensorDType::kFloat32:
      default:
        return static_cast<const float *>(data)[i];
    }
  }
};

inline float sigmoid(float x) {return 1.0f / (1.0f + std::exp(-x));}

// Dequantise an embedding tensor and L2-normalise it.
inline std::vector<float> normalizedEmbedding(const TensorView & tensor)
{
  std::vector<float> embedding(tensor.size());
  float sum_sq = 0.0f;
  for (size_t i = 0; i < embedding.size(); ++i) {
    embedding[i] = tensor.at(i);
    sum_sq += embedding[i] * embedding[i];
  }
  const float norm = std::sqrt(sum_sq) + 1e-6f;
  for (float & v : embedding) {v /= norm;}
  return embedding;
}

// Geometry of a letterbox: model_xy = image_xy * scale + pad.
struct LetterboxInfo {
  float scale{1.0f};
  int pad_x{0};
  int pad_y{0};

  cv::Point2f toImage(float mx, float my) const
  {
    return {(mx - static_cast<float>(pad_x)) / scale, (my - static_cast<float>(pad_y)) / scale};
  }
};

// Resize a BGR image into a preallocated uint8 RGB model input (`dst`, already
// sized HxWx3, typically wrapping the HailoRT input buffer). The BGR->RGB swap
// is done on the already-shrunk image, so it costs almost nothing.
//   centered = true  : YOLO-style centred letterbox
//   centered = false : SCRFD/InsightFace-style top-left letterbox
inline LetterboxInfo letterboxInto(
  const cv::Mat & bgr, cv::Mat & dst, const cv::Scalar & pad_value, bool centered)
{
  LetterboxInfo info;
  if (bgr.empty() || dst.empty()) {return info;}

  info.scale = std::min(
    static_cast<float>(dst.cols) / static_cast<float>(bgr.cols),
    static_cast<float>(dst.rows) / static_cast<float>(bgr.rows));
  const int new_w = std::max(1, std::min(dst.cols, static_cast<int>(std::lround(bgr.cols * info.scale))));
  const int new_h = std::max(1, std::min(dst.rows, static_cast<int>(std::lround(bgr.rows * info.scale))));
  if (centered) {
    info.pad_x = (dst.cols - new_w) / 2;
    info.pad_y = (dst.rows - new_h) / 2;
  }

  // Fill only the padding bands; the image area is overwritten below.
  const cv::Rect roi(info.pad_x, info.pad_y, new_w, new_h);
  if (roi.y > 0) {dst(cv::Rect(0, 0, dst.cols, roi.y)).setTo(pad_value);}
  if (roi.br().y < dst.rows) {
    dst(cv::Rect(0, roi.br().y, dst.cols, dst.rows - roi.br().y)).setTo(pad_value);
  }
  if (roi.x > 0) {dst(cv::Rect(0, roi.y, roi.x, roi.height)).setTo(pad_value);}
  if (roi.br().x < dst.cols) {
    dst(cv::Rect(roi.br().x, roi.y, dst.cols - roi.br().x, roi.height)).setTo(pad_value);
  }

  cv::Mat target = dst(roi);
  if (new_w == bgr.cols && new_h == bgr.rows) {
    cv::cvtColor(bgr, target, cv::COLOR_BGR2RGB);
  } else {
    cv::Mat resized;
    cv::resize(bgr, resized, roi.size(), 0, 0,
      info.scale < 1.0f ? cv::INTER_AREA : cv::INTER_LINEAR);
    cv::cvtColor(resized, target, cv::COLOR_BGR2RGB);
  }
  return info;
}

// Plain stretch resize BGR -> RGB into the model input (re-ID crops).
inline void stretchInto(const cv::Mat & bgr, cv::Mat & dst)
{
  if (bgr.empty() || dst.empty()) {return;}
  cv::Mat resized;
  cv::resize(bgr, resized, dst.size(), 0, 0, cv::INTER_LINEAR);
  cv::cvtColor(resized, dst, cv::COLOR_BGR2RGB);
}

}  // namespace opl_human_vision

#endif  // OPL_HUMAN_VISION__HAILO_TENSOR_HPP_
