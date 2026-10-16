#include "opl_human_vision/pose_nms.hpp"
#include <cassert>
#include <limits>

int main() {
  using namespace opl_human_vision;
  std::vector<cv::Rect> boxes{{0, 0, 100, 200}, {5, 0, 100, 200}};
  std::vector<float> scores{0.9f, 0.8f};
  std::vector<std::vector<cv::Point3f>> poses(2, std::vector<cv::Point3f>(17));
  for (int k = 0; k < 17; ++k) {
    poses[0][k] = {30.0f, float(k * 10), 0.9f};
    poses[1][k] = {65.0f, float(k * 10), 0.9f};
  }
  assert(poseNms(boxes, scores, poses, 0.65f).size() == 2);
  poses[1] = poses[0];
  assert(poseNms(boxes, scores, poses, 0.65f) == std::vector<int>{0});
  for (auto &p : poses[1])
    p.z = 0.1f;
  assert(poseNms(boxes, scores, poses, 0.65f).size() == 1);
  boxes[1].x = 200;
  assert(poseNms(boxes, scores, poses, 0.65f).size() == 2);
  assert(poseNms({}, {}, {}, 0.65f).empty());
  const cv::Rect original(0, 0, 100, 200);
  std::vector<cv::Point3f> partial(17);
  for (int k = 0; k < 4; ++k)
    partial[k] = {40.f + float(k), 10.f + float(k), .99f};
  assert(refinePersonBox(original, partial) == original);
  std::vector<cv::Point3f> whole(17);
  whole[0] = {50, 10, .9};
  whole[5] = {20, 50, .9};
  whole[6] = {80, 50, .9};
  whole[11] = {30, 110, .9};
  whole[12] = {70, 110, .9};
  whole[15] = {35, 185, .9};
  whole[16] = {65, 185, .9};
  whole[9] = {10, 100, .9};
  whole[10] = {90, 100, .9};
  const auto refined = refinePersonBox(original, whole);
  assert(refined != original && refined.width >= 90 && refined.height >= 190);
  whole[16].y = std::numeric_limits<float>::quiet_NaN();
  assert(refinePersonBox(original, whole) == original);
}
