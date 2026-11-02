#ifndef OPL_HUMAN_VISION__YOLOV8_POSE_DECODER_HPP_
#define OPL_HUMAN_VISION__YOLOV8_POSE_DECODER_HPP_

// Host-side decoder for Hailo-compiled YOLOv8-pose HEFs (e.g. yolov8s_pose).
//
// The HEF exposes three tensors per stride (8/16/32), NHWC:
//   box   : HxWx(4*reg_max)  DFL logits for l,t,r,b (reg_max = 16)
//   score : HxWx1            person score, sigmoid already applied on-chip
//   kpts  : HxWx(17*3)       raw x, y, conf-logit per keypoint
// Decoding follows Ultralytics / TAPPAS yolov8pose_postprocess.cpp:
//   centre  = (grid + 0.5) * stride
//   box     = centre -/+ softmax-expectation(DFL) * stride
//   keypt   = (2 * k + grid) * stride,  conf = sigmoid(logit)
// Only anchors that pass the score threshold are dequantised further.

#include <array>
#include <cmath>
#include <string>
#include <vector>

#include "opl_human_vision/hailo_tensor.hpp"

namespace opl_human_vision {

struct PoseCandidate {
  float x1, y1, x2, y2;        // model-input pixel coordinates
  float score;
  std::array<cv::Point3f, 17> keypoints;  // (x, y, conf) in model-input pixels
};

class YoloV8PoseDecoder {
public:
  static constexpr int kKeypoints = 17;

  struct Level {
    int stride{0};
    int box{-1};
    int score{-1};
    int kpts{-1};
  };

  // Match output tensors to strides by channel count and grid size, not by
  // name/order, so recompiled HEFs with different layer names still work.
  bool bind(const std::vector<TensorView> & outputs, int input_w, std::string & error)
  {
    levels_.clear();
    reg_max_ = 0;
    for (int stride : {8, 16, 32}) {
      Level level;
      level.stride = stride;
      const int grid = input_w / stride;
      for (size_t i = 0; i < outputs.size(); ++i) {
        const auto & t = outputs[i];
        if (t.width != grid) {continue;}
        if (t.channels == 1) {
          level.score = static_cast<int>(i);
        } else if (t.channels == kKeypoints * 3) {
          level.kpts = static_cast<int>(i);
        } else if (t.channels % 4 == 0) {
          level.box = static_cast<int>(i);
          reg_max_ = t.channels / 4;
        }
      }
      if (level.box < 0 || level.score < 0 || level.kpts < 0) {
        error = "YOLOv8-pose HEF is missing outputs for stride " + std::to_string(stride) +
          " (expected 1-, 51- and 4*reg_max-channel tensors at grid " + std::to_string(grid) + ")";
        return false;
      }
      levels_.push_back(level);
    }
    return true;
  }

  void decode(
    const std::vector<TensorView> & outputs, float score_threshold,
    std::vector<PoseCandidate> & out) const
  {
    out.clear();
    std::vector<float> dfl(static_cast<size_t>(reg_max_));
    for (const auto & level : levels_) {
      const auto & box_t = outputs[static_cast<size_t>(level.box)];
      const auto & score_t = outputs[static_cast<size_t>(level.score)];
      const auto & kpt_t = outputs[static_cast<size_t>(level.kpts)];
      const int gw = score_t.width;
      const int gh = score_t.height;
      const float stride = static_cast<float>(level.stride);

      for (int gy = 0; gy < gh; ++gy) {
        for (int gx = 0; gx < gw; ++gx) {
          const size_t cell = static_cast<size_t>(gy) * gw + gx;
          const float score = score_t.at(cell);
          if (score < score_threshold) {continue;}

          const float cx = (static_cast<float>(gx) + 0.5f) * stride;
          const float cy = (static_cast<float>(gy) + 0.5f) * stride;

          std::array<float, 4> dist{};
          const size_t box_base = cell * static_cast<size_t>(box_t.channels);
          for (int side = 0; side < 4; ++side) {
            float max_logit = -1e30f;
            for (int b = 0; b < reg_max_; ++b) {
              dfl[static_cast<size_t>(b)] = box_t.at(box_base + static_cast<size_t>(side * reg_max_ + b));
              max_logit = std::max(max_logit, dfl[static_cast<size_t>(b)]);
            }
            float sum = 0.0f;
            float expectation = 0.0f;
            for (int b = 0; b < reg_max_; ++b) {
              const float e = std::exp(dfl[static_cast<size_t>(b)] - max_logit);
              sum += e;
              expectation += e * static_cast<float>(b);
            }
            dist[static_cast<size_t>(side)] = (expectation / sum) * stride;
          }

          PoseCandidate c;
          c.score = score;
          c.x1 = cx - dist[0];
          c.y1 = cy - dist[1];
          c.x2 = cx + dist[2];
          c.y2 = cy + dist[3];

          const size_t kpt_base = cell * static_cast<size_t>(kpt_t.channels);
          for (int k = 0; k < kKeypoints; ++k) {
            const size_t o = kpt_base + static_cast<size_t>(k * 3);
            c.keypoints[static_cast<size_t>(k)] = cv::Point3f(
              (kpt_t.at(o + 0) * 2.0f + static_cast<float>(gx)) * stride,
              (kpt_t.at(o + 1) * 2.0f + static_cast<float>(gy)) * stride,
              sigmoid(kpt_t.at(o + 2)));
          }
          out.push_back(c);
        }
      }
    }
  }

  int regMax() const {return reg_max_;}

private:
  std::vector<Level> levels_;
  int reg_max_{16};
};

}  // namespace opl_human_vision

#endif  // OPL_HUMAN_VISION__YOLOV8_POSE_DECODER_HPP_
