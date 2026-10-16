#include "opl_human_vision/face_engine.hpp"
#include <cassert>
#include <filesystem>
#include <iostream>
#include <unistd.h>
using namespace opl_human_vision;
std::vector<float> unit(size_t axis) {
  std::vector<float> v(512, 0);
  v[axis] = 1;
  return v;
}
FaceResult sample(size_t axis) {
  FaceResult f;
  f.face_quality_valid = true;
  f.face_embedding = unit(axis);
  f.body_embedding = unit(10);
  return f;
}
int main(int argc, char **argv) {
  assert(argc == 2);
  std::string mode = argv[1];
  auto root = std::filesystem::current_path() /
              ("identity_regression_" + mode + std::to_string(getpid()));
  struct Cleanup {
    std::filesystem::path root;
    ~Cleanup() { std::filesystem::remove_all(root); }
  } cleanup{root};
  FaceEngine engine("", "", "", .55, .85, 2, .1, 2);
  opl_interfaces::msg::TrackedHuman h;
  h.observed = true;
  h.confidence = .9f;
  engine.configureStore(root.string(), "regression-model");
  if (mode == "namespace") {
    engine.authorizeEnrollment(1);
    auto scanning = engine.processFeatures(sample(0), 1, {}, h, 1);
    assert(scanning.persistent_id >= 1000000000ULL);
  } else if (mode == "geometry") {
    h.fixed_position_valid = true;
    h.fixed_position.header.frame_id = "odom";
    h.fixed_position.header.stamp.sec = 1;
    auto first = engine.processFeatures(sample(0), 1, {}, h, 1);
    engine.cleanupStaleRawTracks({2});
    h.fixed_position.point.x = 10;
    h.fixed_position.header.stamp.nanosec = 100000000;
    auto other = engine.processFeatures(sample(0), 2, {}, h, 1.1);
    assert(other.persistent_id != first.persistent_id);
  } else if (mode == "names") {
    engine.authorizeEnrollment(1, "Person_2");
    engine.processFeatures(sample(0), 1, {}, h, 1);
    auto first = engine.processFeatures(sample(0), 1, {}, h, 1.11);
    assert(first.just_scanned);
    engine.authorizeEnrollment(2);
    engine.processFeatures(sample(1), 2, {}, h, 2);
    auto second = engine.processFeatures(sample(1), 2, {}, h, 2.11);
    assert(second.just_scanned && second.name != first.name &&
           engine.registeredCount() == 2);
  } else
    assert(false);
  std::cout << mode << " regression passed\n";
}
