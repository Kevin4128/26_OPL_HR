#include "opl_human_vision/tracker_engine.hpp"
#include <cassert>
#include <cmath>
#include <limits>
#include <stdexcept>

using namespace opl_human_vision;

opl_interfaces::msg::TrackedHuman detection(int x, float confidence = 0.9f) {
  opl_interfaces::msg::TrackedHuman h;
  h.bbox.x_offset = x; h.bbox.y_offset = 20;
  h.bbox.width = 100; h.bbox.height = 200; h.confidence = confidence;
  return h;
}

int main() {
  // Weak detections can maintain an established track, but cannot create one.
  TrackerEngine tracker(60, 1);
  assert(tracker.update({detection(10, 0.3f)}, {}, 1.0).empty());
  auto tracks = tracker.update({detection(10)}, {}, 1.1);
  assert(tracks.size() == 1);
  const auto id = tracks.front().id;
  tracks = tracker.update({detection(12, 0.3f)}, {}, 1.2);
  assert(tracks.size() == 1 && tracks.front().id == id);
  assert(tracks.front().time_since_update == 0);
  assert(std::abs(tracks.front().confidence - 0.3f) < 1e-5);
  tracks = tracker.update({detection(12, 0.05f)}, {}, 1.3);
  assert(tracks.size() == 1 && tracks.front().time_since_update == 1);
  // Once lost, a track must return through the high-confidence association.
  assert(tracker.update({detection(12, 0.3f)}, {}, 1.4).empty());
  tracks = tracker.update({detection(12)}, {}, 1.5);
  assert(tracks.size() == 1 && tracks.front().id == id);

  TrackerEngine tentative(60, 3);
  assert(tentative.update({detection(10)}, {}, 1.0).empty());
  assert(tentative.update({detection(10, 0.3f)}, {}, 1.1).empty());
  assert(tentative.update({detection(10)}, {}, 1.2).empty());
  assert(tentative.update({detection(10)}, {}, 1.3).empty());
  assert(tentative.update({detection(10)}, {}, 1.4).size() == 1);

  // Golden values from upstream BoT-SORT's initiate -> predict -> update.
  BoTSortKalman filter({10, 20, 100, 200});
  filter.predict(1, true);
  filter.correct({20, 20, 100, 200});
  assert(std::abs(filter.mean[0] - 68.67768595041322) < 1e-9);
  assert(std::abs(filter.mean[4] - 2.066115702479339) < 1e-9);
  assert(std::abs(filter.covariance(0, 0) - 21.694214876033072) < 1e-9);
  assert(std::abs(filter.covariance(0, 4) - 5.165289256198349) < 1e-9);
  const double uncertainty = filter.covariance(0, 0);
  filter.predict(1, true);
  assert(filter.covariance(0, 0) > uncertainty);
  const auto original = filter;
  filter.warp(cv::Matx23d(2, 0, 5, 0, 2, -3));
  assert(std::abs(filter.mean[0] - (2 * original.mean[0] + 5)) < 1e-9);
  assert(std::abs(filter.mean[4] - 2 * original.mean[4]) < 1e-9);
  assert(std::abs(filter.covariance(0, 0) - 4 * original.covariance(0, 0)) < 1e-9);
  filter.mean[6] = filter.mean[7] = 10;
  filter.predict(1, false);
  assert(filter.mean[6] == 0 && filter.mean[7] == 0);
  BoTSortKalman half_step({10, 20, 100, 200});
  half_step.mean[4] = 10;
  half_step.predict(0.5, true);
  assert(half_step.mean[0] == 65);

  // Similar clothing cannot authorize an appearance match outside proximity.
  assert(std::abs(boTSortAssociationCost(0.6f, 0.9f, 1.0f, 0.5f, 0.85f)) < 1e-6);
  assert(std::abs(boTSortAssociationCost(0.1f, 0.9f, 1.0f, 0.5f, 0.85f) - 0.91f) < 1e-6);
  assert(std::abs(boTSortAssociationCost(0.6f, 0.9f, 0.8f, 0.5f, 0.85f) - 0.46f) < 1e-6);

  // Motion learned before complete occlusion predicts a moving person's return.
  TrackerEngine moving(60, 1);
  for (int i = 0; i < 20; ++i)
    tracks = moving.update({detection(10 + i * 5)}, {}, 1.0 + i / 30.0);
  const int moving_id = tracks.front().id;
  const auto observed_x = tracks.front().bbox.x;
  tracks = moving.update({}, {}, 1.0 + 20 / 30.0);
  assert(tracks.size() == 1 && tracks.front().time_since_update == 1);
  assert(tracks.front().bbox.x > observed_x);
  for (int i = 21; i < 30; ++i) assert(moving.update({}, {}, 1.0 + i / 30.0).empty());
  tracks = moving.update({detection(160)}, {}, 2.0);
  assert(tracks.size() == 1 && tracks.front().id == moving_id);
  // A backward clock jump starts new trajectories without reusing raw IDs.
  tracks = moving.update({detection(160)}, {}, 0.5);
  assert(tracks.size() == 1 && tracks.front().id != moving_id);

  TrackerEngine invalid_detection(60, 1);
  assert(invalid_detection.update({detection(10, std::numeric_limits<float>::quiet_NaN())}, {}, 1).empty());
  assert(invalid_detection.update({detection(10, 1.1f)}, {}, 2).empty());
  auto zero_box = detection(10); zero_box.bbox.width = 0;
  assert(invalid_detection.update({zero_box}, {}, 3).empty());
  // Threshold ordering and non-finite values fail before loading GPU models.
  for (float value : {-0.1f, 1.1f, std::numeric_limits<float>::quiet_NaN()}) {
    bool rejected = false;
    try { TrackerEngine invalid(60, 1, value); }
    catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);
  }
  bool rejected = false;
  try { TrackerEngine invalid(60, 1, 0.8f, 1, "", false, 0.85f, 60, 0.65f, 0.70f); }
  catch (const std::invalid_argument&) { rejected = true; }
  assert(rejected);
}
