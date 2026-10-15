#include "opl_human_vision/pose_nms.hpp"
#include <cassert>

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
  for (auto& p : poses[1]) p.z = 0.1f;
  assert(poseNms(boxes, scores, poses, 0.65f).size() == 1);
  boxes[1].x = 200;
  assert(poseNms(boxes, scores, poses, 0.65f).size() == 2);
  assert(poseNms({}, {}, {}, 0.65f).empty());
}
