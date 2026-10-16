#pragma once
#include <algorithm>
#include <cmath>
#include <numeric>
#include <opencv2/core.hpp>
#include <vector>

namespace opl_human_vision {
// Partial poses must not shrink the detector's full-person crop.
inline cv::Rect refinePersonBox(const cv::Rect &original,
                                const std::vector<cv::Point3f> &pose) {
  if (original.empty() || pose.size() != 17)
    return original;
  for (int k : {0, 5, 6, 11, 12, 15, 16}) {
    const auto &p = pose[k];
    if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z) ||
        p.z < .65f || p.x < original.x || p.x > original.br().x ||
        p.y < original.y || p.y > original.br().y)
      return original;
  }
  if (std::abs(pose[5].x - pose[6].x) < original.width * .15f ||
      std::abs(pose[11].x - pose[12].x) < original.width * .10f ||
      (pose[5].y + pose[6].y) >= (pose[11].y + pose[12].y) ||
      (pose[11].y + pose[12].y) >= (pose[15].y + pose[16].y))
    return original;
  float x1 = original.br().x, y1 = original.br().y, x2 = original.x,
        y2 = original.y;
  for (const auto &p : pose)
    if (p.z >= .65f && std::isfinite(p.x) && std::isfinite(p.y)) {
      x1 = std::min(x1, p.x);
      y1 = std::min(y1, p.y);
      x2 = std::max(x2, p.x);
      y2 = std::max(y2, p.y);
    }
  const float w = x2 - x1, h = y2 - y1;
  cv::Rect candidate(int(x1 - w * .08f), int(y1 - h * .10f), int(w * 1.16f),
                     int(h * 1.18f));
  const float intersection = (candidate & original).area();
  if (intersection / std::max(1.f, float(candidate.area() + original.area()) -
                                       intersection) <
      .75f)
    return original;
  return cv::Rect(cvRound(.8f * original.x + .2f * candidate.x),
                  cvRound(.8f * original.y + .2f * candidate.y),
                  cvRound(.8f * original.width + .2f * candidate.width),
                  cvRound(.8f * original.height + .2f * candidate.height));
}

// Preserve overlapping boxes when several corresponding, confident joints
// provide evidence of two different people. With weak poses, use ordinary NMS.
inline std::vector<int>
poseNms(const std::vector<cv::Rect> &boxes, const std::vector<float> &scores,
        const std::vector<std::vector<cv::Point3f>> &poses, float threshold) {
  std::vector<int> order(scores.size()), kept;
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(),
                   [&](int a, int b) { return scores[a] > scores[b]; });
  for (int candidate : order) {
    bool duplicate = false;
    for (int previous : kept) {
      const float intersection = (boxes[candidate] & boxes[previous]).area();
      const float area =
          boxes[candidate].area() + boxes[previous].area() - intersection;
      if (area <= 0 || intersection / area <= threshold)
        continue;
      std::vector<float> distances;
      const auto &a = poses[candidate];
      const auto &b = poses[previous];
      const float scale =
          std::max(1.0f, std::min(std::hypot(float(boxes[candidate].width),
                                             float(boxes[candidate].height)),
                                  std::hypot(float(boxes[previous].width),
                                             float(boxes[previous].height))));
      for (size_t k = 0; k < std::min(a.size(), b.size()); ++k) {
        if (a[k].z >= 0.5f && b[k].z >= 0.5f)
          distances.push_back(std::hypot(a[k].x - b[k].x, a[k].y - b[k].y) /
                              scale);
      }
      if (distances.size() >= 4) {
        auto middle = distances.begin() + distances.size() / 2;
        std::nth_element(distances.begin(), middle, distances.end());
        if (*middle > 0.10f)
          continue;
      }
      duplicate = true;
      break;
    }
    if (!duplicate)
      kept.push_back(candidate);
  }
  return kept;
}
} // namespace opl_human_vision
