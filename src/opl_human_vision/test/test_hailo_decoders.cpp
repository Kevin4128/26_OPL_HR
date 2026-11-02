// Synthetic-tensor checks for the Hailo host-side decoders (no accelerator needed).
#include "opl_human_vision/hailo_tensor.hpp"
#include "opl_human_vision/scrfd_decoder.hpp"
#include "opl_human_vision/yolov8_pose_decoder.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

using namespace opl_human_vision;

namespace {

bool near(float a, float b, float tol = 1e-3f) {return std::abs(a - b) <= tol;}

struct FloatTensor {
  std::vector<float> data;
  TensorView view;
  FloatTensor(const std::string & name, int h, int w, int c)
  : data(static_cast<size_t>(h) * w * c, 0.0f)
  {
    view.name = name;
    view.height = h;
    view.width = w;
    view.channels = c;
    view.dtype = TensorDType::kFloat32;
  }
  float & at(int y, int x, int ch) {return data[(static_cast<size_t>(y) * view.width + x) * view.channels + ch];}
  TensorView finish() {view.data = data.data(); return view;}
};

void testDequantisation()
{
  const uint8_t u8[] = {200, 100};
  TensorView v;
  v.data = u8; v.dtype = TensorDType::kUInt8; v.scale = 0.5f; v.zero_point = 100.0f; v.channels = 2;
  assert(near(v.at(0), 50.0f) && near(v.at(1), 0.0f));

  const uint16_t u16[] = {17225 + 1000};
  TensorView w;
  w.data = u16; w.dtype = TensorDType::kUInt16; w.scale = 0.001f; w.zero_point = 17225.0f;
  assert(near(w.at(0), 1.0f));

  const float f[] = {3.0f, 4.0f};
  TensorView e;
  e.data = f; e.channels = 2;
  const auto n = normalizedEmbedding(e);
  assert(near(n[0], 0.6f) && near(n[1], 0.8f));
}

void testPoseDecoder()
{
  // Per stride: box (64 = 4 x 16 DFL bins), score (1), keypoints (51).
  std::vector<FloatTensor> tensors;
  for (int stride : {8, 16, 32}) {
    const int g = 640 / stride;
    tensors.emplace_back("box" + std::to_string(stride), g, g, 64);
    tensors.emplace_back("score" + std::to_string(stride), g, g, 1);
    tensors.emplace_back("kpts" + std::to_string(stride), g, g, 51);
  }
  // One person at stride 8, cell (gx=40, gy=30).
  auto & box = tensors[0];
  auto & score = tensors[1];
  auto & kpts = tensors[2];
  score.at(30, 40, 0) = 0.9f;
  score.at(0, 0, 0) = 0.2f;  // below threshold, must be ignored
  for (int side = 0; side < 4; ++side) {box.at(30, 40, side * 16 + 5) = 30.0f;}  // DFL peak at bin 5
  for (int k = 0; k < 17; ++k) {
    kpts.at(30, 40, k * 3 + 0) = 0.25f;
    kpts.at(30, 40, k * 3 + 1) = 0.5f;
    kpts.at(30, 40, k * 3 + 2) = 2.0f;
  }

  std::vector<TensorView> views;
  for (auto & t : tensors) {views.push_back(t.finish());}
  // HEF output order is not guaranteed; binding must not depend on it.
  std::mt19937 rng(7);
  std::shuffle(views.begin(), views.end(), rng);

  YoloV8PoseDecoder decoder;
  std::string error;
  assert(decoder.bind(views, 640, error));
  assert(decoder.regMax() == 16);

  std::vector<PoseCandidate> out;
  decoder.decode(views, 0.5f, out);
  assert(out.size() == 1);
  const auto & c = out[0];
  // centre = (40.5*8, 30.5*8) = (324, 244); distance = 5 bins * 8 = 40
  assert(near(c.score, 0.9f));
  assert(near(c.x1, 284.0f, 0.05f) && near(c.y1, 204.0f, 0.05f));
  assert(near(c.x2, 364.0f, 0.05f) && near(c.y2, 284.0f, 0.05f));
  // keypoint = (2k + grid) * stride
  assert(near(c.keypoints[0].x, (0.5f + 40.0f) * 8.0f));
  assert(near(c.keypoints[0].y, (1.0f + 30.0f) * 8.0f));
  assert(near(c.keypoints[16].z, sigmoid(2.0f)));

  // A HEF missing a stride must be rejected.
  std::vector<TensorView> broken(views.begin(), views.begin() + 3);
  assert(!decoder.bind(broken, 640, error) && !error.empty());
}

void testScrfdDecoder()
{
  std::vector<FloatTensor> tensors;
  for (int stride : {8, 16, 32}) {
    const int g = 640 / stride;
    tensors.emplace_back("score", g, g, 2);
    tensors.emplace_back("bbox", g, g, 8);
    tensors.emplace_back("kps", g, g, 20);
  }
  auto & score = tensors[0];
  auto & bbox = tensors[1];
  auto & kps = tensors[2];
  // Face at stride 8, cell (gx=10, gy=20), anchor 1 -> centre (80, 160).
  score.at(20, 10, 1) = 0.8f;
  for (int i = 0; i < 4; ++i) {bbox.at(20, 10, 4 + i) = static_cast<float>(i + 1);}
  for (int k = 0; k < 5; ++k) {
    kps.at(20, 10, 10 + k * 2) = 0.5f * static_cast<float>(k);
    kps.at(20, 10, 10 + k * 2 + 1) = 1.0f;
  }
  // Weaker overlapping duplicate on the neighbouring cell, anchor 0.
  score.at(20, 11, 0) = 0.6f;
  for (int i = 0; i < 4; ++i) {bbox.at(20, 11, i) = static_cast<float>(i + 1);}

  std::vector<TensorView> views;
  for (auto & t : tensors) {views.push_back(t.finish());}
  std::reverse(views.begin(), views.end());

  ScrfdDecoder decoder;
  std::string error;
  assert(decoder.bind(views, 640, error));
  std::vector<FaceCandidate> faces;
  decoder.decode(views, 0.5f, faces);
  assert(faces.size() == 2);
  ScrfdDecoder::nms(faces, 0.4f);
  assert(faces.size() == 1);
  const auto & f = faces[0];
  assert(near(f.score, 0.8f));
  assert(near(f.x1, 72.0f) && near(f.y1, 144.0f) && near(f.x2, 104.0f) && near(f.y2, 192.0f));
  assert(near(f.landmarks[2].x, 88.0f) && near(f.landmarks[2].y, 168.0f));
}

void testLetterbox()
{
  cv::Mat bgr(480, 640, CV_8UC3, cv::Scalar(10, 20, 30));
  cv::Mat dst(640, 640, CV_8UC3, cv::Scalar(1, 2, 3));

  auto info = letterboxInto(bgr, dst, cv::Scalar(114, 114, 114), true);
  assert(info.pad_x == 0 && info.pad_y == 80 && near(info.scale, 1.0f));
  assert(dst.at<cv::Vec3b>(0, 0) == cv::Vec3b(114, 114, 114));
  assert(dst.at<cv::Vec3b>(639, 639) == cv::Vec3b(114, 114, 114));
  assert(dst.at<cv::Vec3b>(320, 320) == cv::Vec3b(30, 20, 10));  // BGR -> RGB
  const auto p = info.toImage(100.0f, 180.0f);
  assert(near(p.x, 100.0f) && near(p.y, 100.0f));

  // Top-left variant (SCRFD) on a tall crop: scaled, padding right only.
  cv::Mat crop(320, 160, CV_8UC3, cv::Scalar(0, 0, 255));
  info = letterboxInto(crop, dst, cv::Scalar(0, 0, 0), false);
  assert(info.pad_x == 0 && info.pad_y == 0 && near(info.scale, 2.0f));
  assert(dst.at<cv::Vec3b>(10, 10) == cv::Vec3b(255, 0, 0));
  assert(dst.at<cv::Vec3b>(10, 500) == cv::Vec3b(0, 0, 0));
  const auto q = info.toImage(200.0f, 400.0f);
  assert(near(q.x, 100.0f) && near(q.y, 200.0f));

  // Input buffer must be written in place (it wraps the HailoRT buffer).
  const uchar * before = dst.data;
  stretchInto(crop, dst);
  assert(dst.data == before && dst.at<cv::Vec3b>(0, 0) == cv::Vec3b(255, 0, 0));
}

}  // namespace

int main()
{
  testDequantisation();
  testPoseDecoder();
  testScrfdDecoder();
  testLetterbox();
  std::cout << "Hailo dequantisation, YOLOv8-pose, SCRFD and letterbox decoder checks passed\n";
  return 0;
}
