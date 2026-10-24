#pragma once
// BoT-SORT XYWH filter, adapted from Nir Aharon's MIT-licensed implementation.
// See third_party/BoT-SORT/LICENSE and README.md for the pinned reference.
#include <algorithm>
#include <cmath>
#include <opencv2/core.hpp>

namespace opl_human_vision {

struct BoTSortKalman {
  cv::Vec<double, 8> mean{};
  cv::Matx<double, 8, 8> covariance = cv::Matx<double, 8, 8>::zeros();

  explicit BoTSortKalman(const cv::Rect2f& box = {}) {
    mean[0] = box.x + box.width / 2;
    mean[1] = box.y + box.height / 2;
    mean[2] = box.width;
    mean[3] = box.height;
    for (int i = 0; i < 8; ++i) {
      const double size = std::max(1.0, mean[2 + i % 2]);
      const double sigma = (i < 4 ? 2.0 / 20 : 10.0 / 160) * size;
      covariance(i, i) = sigma * sigma;
    }
  }

  // Velocities are per nominal 30 Hz update; dt scales them to measurement time.
  void predict(double dt, bool tracked) {
    if (!tracked) mean[6] = mean[7] = 0;
    auto motion = cv::Matx<double, 8, 8>::eye();
    auto noise = cv::Matx<double, 8, 8>::zeros();
    for (int i = 0; i < 8; ++i) {
      if (i < 4) motion(i, i + 4) = dt;
      const double sigma = std::max(1.0, mean[2 + i % 2]) / (i < 4 ? 20 : 160);
      noise(i, i) = sigma * sigma * dt;
    }
    mean = motion * mean;
    covariance = motion * covariance * motion.t() + noise;
  }

  void warp(const cv::Matx23d& affine) {
    auto rotation = cv::Matx<double, 8, 8>::zeros();
    for (int block = 0; block < 4; ++block)
      for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
          rotation(block * 2 + i, block * 2 + j) = affine(i, j);
    mean = rotation * mean;
    mean[0] += affine(0, 2);
    mean[1] += affine(1, 2);
    covariance = rotation * covariance * rotation.t();
  }

  void correct(const cv::Rect2f& box) {
    cv::Matx<double, 4, 8> observation = cv::Matx<double, 4, 8>::zeros();
    cv::Matx44d noise = cv::Matx44d::zeros();
    for (int i = 0; i < 4; ++i) {
      observation(i, i) = 1;
      const double sigma = std::max(1.0, mean[2 + i % 2]) / 20;
      noise(i, i) = sigma * sigma;
    }
    const auto projected = observation * covariance * observation.t() + noise;
    const auto gain = covariance * observation.t() * projected.inv(cv::DECOMP_CHOLESKY);
    const cv::Vec4d measurement(box.x + box.width / 2, box.y + box.height / 2,
                               box.width, box.height);
    mean += gain * (measurement - observation * mean);
    covariance -= gain * projected * gain.t();
    covariance = (covariance + covariance.t()) * 0.5;
  }

  cv::Rect2f box() const {
    const float width = std::max(1.0, mean[2]);
    const float height = std::max(1.0, mean[3]);
    return {float(mean[0]) - width / 2, float(mean[1]) - height / 2, width, height};
  }
};

// BoT-SORT lets a close, sufficiently similar body improve the score-fused IoU cost.
inline float boTSortAssociationCost(float iou, float score, float similarity,
                                    float proximity_threshold, float appearance_threshold) {
  float cost = 1.0f - iou * score;
  if (1.0f - iou <= proximity_threshold && similarity >= appearance_threshold)
    cost = std::min(cost, (1.0f - similarity) / 2.0f);
  return cost;
}

}  // namespace opl_human_vision
