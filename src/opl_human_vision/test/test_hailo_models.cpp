// On-device smoke test: loads every HEF the pipeline uses on the Hailo-8L and
// runs a real image through pose -> SCRFD -> ArcFace / re-ID.
//
// Usage: test_hailo_models <models_dir> <images_dir>
// Exit codes: 0 pass, 1 fail, 77 skipped (no device / HEFs / device busy).
#include "opl_human_vision/face_embedder.hpp"
#include "opl_human_vision/hailo_model.hpp"
#include "opl_human_vision/reid_extractor.hpp"
#include "opl_human_vision/scrfd_detector.hpp"
#include "opl_human_vision/yolov8_pose_decoder.hpp"

#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <numeric>

#include <opencv2/dnn.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace opl_human_vision;
namespace fs = std::filesystem;

namespace {

constexpr int kSkip = 77;

int fail(const std::string & what)
{
  std::cerr << "FAIL: " << what << std::endl;
  return 1;
}

float dot(const std::vector<float> & a, const std::vector<float> & b)
{
  return std::inner_product(a.begin(), a.end(), b.begin(), 0.0f);
}

bool unitNorm(const std::vector<float> & v)
{
  return v.size() == 512 && std::abs(std::sqrt(dot(v, v)) - 1.0f) < 1e-3f;
}

}  // namespace

int main(int argc, char ** argv)
{
  if (argc < 3) {
    std::cerr << "usage: test_hailo_models <models_dir> <images_dir>" << std::endl;
    return 1;
  }
  const fs::path models(argv[1]);
  const fs::path images(argv[2]);

  if (!fs::exists("/dev/hailo0")) {
    std::cout << "SKIP: /dev/hailo0 not present" << std::endl;
    return kSkip;
  }
  for (const char * name : {"yolov8s_pose.hef", "scrfd_2.5g.hef", "arcface_mobilefacenet.hef",
      "repvgg_a0_person_reid_512.hef"})
  {
    if (!fs::exists(models / name)) {
      std::cout << "SKIP: " << (models / name) << " missing (run scripts/fetch_hailo_models.sh)" << std::endl;
      return kSkip;
    }
  }
  const cv::Mat image = cv::imread((images / "zidane.jpg").string());
  if (image.empty()) {
    std::cout << "SKIP: sample image zidane.jpg not found in " << images << std::endl;
    return kSkip;
  }

  // --- Pose ---------------------------------------------------------------
  std::string error;
  auto pose = HailoModel::create((models / "yolov8s_pose.hef").string(), {}, error);
  if (!pose) {
    if (error.find("OUT_OF_PHYSICAL_DEVICES") != std::string::npos) {
      std::cout << "SKIP: Hailo device busy (" << error << ")" << std::endl;
      return kSkip;
    }
    return fail(error);
  }
  YoloV8PoseDecoder decoder;
  if (!decoder.bind(pose->outputs(), pose->inputWidth(), error)) {return fail(error);}

  cv::Mat input = pose->inputMat();
  const auto lb = letterboxInto(image, input, cv::Scalar(114, 114, 114), true);
  const auto t0 = std::chrono::steady_clock::now();
  if (!pose->run(error)) {return fail(error);}
  const double pose_ms =
    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

  std::vector<PoseCandidate> candidates;
  decoder.decode(pose->outputs(), 0.5f, candidates);
  std::vector<cv::Rect> boxes;
  std::vector<float> scores;
  for (const auto & c : candidates) {
    boxes.emplace_back(cv::Point(lb.toImage(c.x1, c.y1)), cv::Point(lb.toImage(c.x2, c.y2)));
    scores.push_back(c.score);
  }
  std::vector<int> keep;
  cv::dnn::NMSBoxes(boxes, scores, 0.5f, 0.5f, keep);
  std::cout << "pose: " << keep.size() << " people, " << pose_ms << " ms" << std::endl;
  if (keep.size() < 2) {return fail("expected at least 2 people in zidane.jpg");}

  // --- Faces + embeddings ---------------------------------------------------
  ScrfdDetector scrfd((models / "scrfd_2.5g.hef").string());
  FaceEmbedder embedder((models / "arcface_mobilefacenet.hef").string());
  ReidExtractor reid((models / "repvgg_a0_person_reid_512.hef").string());
  if (!scrfd.init()) {return fail("SCRFD: " + scrfd.lastError());}
  if (!embedder.init()) {return fail("ArcFace: " + embedder.lastError());}
  if (!reid.init()) {return fail("re-ID: " + reid.lastError());}

  std::vector<std::vector<float>> faces;
  std::vector<std::vector<float>> bodies;
  for (int idx : keep) {
    const cv::Rect box = boxes[static_cast<size_t>(idx)] & cv::Rect(0, 0, image.cols, image.rows);
    const cv::Mat crop = image(box).clone();
    const auto body = reid.extract(crop);
    if (!unitNorm(body)) {return fail("re-ID embedding is not a unit 512-d vector");}
    bodies.push_back(body);

    std::vector<cv::Point2f> landmarks;
    float face_score = 0.0f;
    if (!scrfd.detect(crop, landmarks, face_score)) {continue;}
    // Eyes above nose above mouth inside the crop.
    if (!(landmarks[0].y < landmarks[3].y && landmarks[1].y < landmarks[4].y)) {
      return fail("SCRFD landmarks are not anatomically ordered");
    }
    const cv::Mat aligned = embedder.alignFace(crop, landmarks);
    const auto face = embedder.extract(aligned);
    if (!unitNorm(face)) {return fail("face embedding is not a unit 512-d vector");}
    cv::Mat flipped;
    cv::flip(aligned, flipped, 1);
    const float self_sim = dot(face, embedder.extract(flipped));
    std::cout << "face: score " << face_score << ", self(flip) similarity " << self_sim << std::endl;
    if (self_sim < 0.5f) {return fail("same face (mirrored) should be similar");}
    faces.push_back(face);
  }
  if (faces.size() < 2) {return fail("expected faces for at least 2 people");}

  const float cross_face = dot(faces[0], faces[1]);
  const float cross_body = dot(bodies[0], bodies[1]);
  std::cout << "cross-person similarity: face " << cross_face << ", body " << cross_body << std::endl;
  if (cross_face > 0.5f) {return fail("different people should not have similar faces");}

  std::cout << "Hailo-8L model smoke test passed" << std::endl;
  return 0;
}
