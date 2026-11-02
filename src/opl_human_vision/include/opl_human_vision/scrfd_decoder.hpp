#ifndef OPL_HUMAN_VISION__SCRFD_DECODER_HPP_
#define OPL_HUMAN_VISION__SCRFD_DECODER_HPP_

// Host-side decoder for the Hailo-compiled SCRFD HEF (scrfd_2.5g / scrfd_10g).
//
// Per stride (8/16/32), NHWC with 2 anchors per cell interleaved in channels:
//   score : HxWx2   (anchor0, anchor1), sigmoid applied on-chip
//   bbox  : HxWx8   (l,t,r,b) * 2 anchors, in units of stride
//   kps   : HxWx20  (x,y) * 5 landmarks * 2 anchors, in units of stride
// Anchor centre = (gx * stride, gy * stride) (InsightFace convention, no +0.5;
// matches TAPPAS scrfd.cpp). Landmark order: left eye, right eye, nose,
// left mouth, right mouth.

#include <algorithm>
#include <array>
#include <string>
#include <vector>

#include "opl_human_vision/hailo_tensor.hpp"

namespace opl_human_vision {

struct FaceCandidate {
  float x1, y1, x2, y2;
  float score;
  std::array<cv::Point2f, 5> landmarks;
};

class ScrfdDecoder {
public:
  static constexpr int kAnchors = 2;

  bool bind(const std::vector<TensorView> & outputs, int input_w, std::string & error)
  {
    levels_.clear();
    for (int stride : {8, 16, 32}) {
      Level level;
      level.stride = stride;
      const int grid = input_w / stride;
      for (size_t i = 0; i < outputs.size(); ++i) {
        const auto & t = outputs[i];
        if (t.width != grid) {continue;}
        if (t.channels == kAnchors) {level.score = static_cast<int>(i);}
        else if (t.channels == kAnchors * 4) {level.bbox = static_cast<int>(i);}
        else if (t.channels == kAnchors * 10) {level.kps = static_cast<int>(i);}
      }
      if (level.score < 0 || level.bbox < 0 || level.kps < 0) {
        error = "SCRFD HEF is missing score/bbox/kps outputs for stride " + std::to_string(stride);
        return false;
      }
      levels_.push_back(level);
    }
    return true;
  }

  void decode(
    const std::vector<TensorView> & outputs, float score_threshold,
    std::vector<FaceCandidate> & out) const
  {
    out.clear();
    for (const auto & level : levels_) {
      const auto & score_t = outputs[static_cast<size_t>(level.score)];
      const auto & bbox_t = outputs[static_cast<size_t>(level.bbox)];
      const auto & kps_t = outputs[static_cast<size_t>(level.kps)];
      const float stride = static_cast<float>(level.stride);
      for (int gy = 0; gy < score_t.height; ++gy) {
        for (int gx = 0; gx < score_t.width; ++gx) {
          const size_t cell = static_cast<size_t>(gy) * score_t.width + gx;
          for (int a = 0; a < kAnchors; ++a) {
            const size_t idx = cell * kAnchors + static_cast<size_t>(a);
            const float score = score_t.at(idx);
            if (score < score_threshold) {continue;}
            const float cx = static_cast<float>(gx) * stride;
            const float cy = static_cast<float>(gy) * stride;
            FaceCandidate f;
            f.score = score;
            f.x1 = cx - bbox_t.at(idx * 4 + 0) * stride;
            f.y1 = cy - bbox_t.at(idx * 4 + 1) * stride;
            f.x2 = cx + bbox_t.at(idx * 4 + 2) * stride;
            f.y2 = cy + bbox_t.at(idx * 4 + 3) * stride;
            for (int k = 0; k < 5; ++k) {
              f.landmarks[static_cast<size_t>(k)] = cv::Point2f(
                cx + kps_t.at(idx * 10 + static_cast<size_t>(k * 2)) * stride,
                cy + kps_t.at(idx * 10 + static_cast<size_t>(k * 2 + 1)) * stride);
            }
            out.push_back(f);
          }
        }
      }
    }
  }

  static float iou(const FaceCandidate & a, const FaceCandidate & b)
  {
    const float iw = std::max(0.0f, std::min(a.x2, b.x2) - std::max(a.x1, b.x1));
    const float ih = std::max(0.0f, std::min(a.y2, b.y2) - std::max(a.y1, b.y1));
    const float inter = iw * ih;
    const float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    const float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    return inter / (area_a + area_b - inter + 1e-5f);
  }

  static void nms(std::vector<FaceCandidate> & faces, float iou_threshold)
  {
    std::sort(faces.begin(), faces.end(),
      [](const FaceCandidate & a, const FaceCandidate & b) {return a.score > b.score;});
    std::vector<FaceCandidate> kept;
    std::vector<bool> suppressed(faces.size(), false);
    for (size_t i = 0; i < faces.size(); ++i) {
      if (suppressed[i]) {continue;}
      kept.push_back(faces[i]);
      for (size_t j = i + 1; j < faces.size(); ++j) {
        if (!suppressed[j] && iou(faces[i], faces[j]) > iou_threshold) {suppressed[j] = true;}
      }
    }
    faces.swap(kept);
  }

private:
  struct Level {
    int stride{0};
    int score{-1};
    int bbox{-1};
    int kps{-1};
  };
  std::vector<Level> levels_;
};

}  // namespace opl_human_vision

#endif  // OPL_HUMAN_VISION__SCRFD_DECODER_HPP_
