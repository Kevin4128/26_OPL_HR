#pragma once
#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>
#include <opencv2/core.hpp>

namespace opl_human_vision {
// Preserve overlapping boxes when several corresponding, confident joints
// provide evidence of two different people. With weak poses, use ordinary NMS.
inline std::vector<int> poseNms(
  const std::vector<cv::Rect>& boxes, const std::vector<float>& scores,
  const std::vector<std::vector<cv::Point3f>>& poses, float threshold)
{
  std::vector<int> order(scores.size()), kept;
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return scores[a] > scores[b]; });
  for (int candidate : order) {
    bool duplicate = false;
    for (int previous : kept) {
      const float intersection = (boxes[candidate] & boxes[previous]).area();
      const float area = boxes[candidate].area() + boxes[previous].area() - intersection;
      if (area <= 0 || intersection / area <= threshold) continue;
      std::vector<float> distances;
      const auto& a = poses[candidate];
      const auto& b = poses[previous];
      const float scale = std::max(1.0f, std::min(
        std::hypot(float(boxes[candidate].width), float(boxes[candidate].height)),
        std::hypot(float(boxes[previous].width), float(boxes[previous].height))));
      for (size_t k = 0; k < std::min(a.size(), b.size()); ++k) {
        if (a[k].z >= 0.5f && b[k].z >= 0.5f)
          distances.push_back(std::hypot(a[k].x - b[k].x, a[k].y - b[k].y) / scale);
      }
      if (distances.size() >= 4) {
        auto middle = distances.begin() + distances.size() / 2;
        std::nth_element(distances.begin(), middle, distances.end());
        if (*middle > 0.10f) continue;
      }
      duplicate = true;
      break;
    }
    if (!duplicate) kept.push_back(candidate);
  }
  return kept;
}
}  // namespace opl_human_vision
