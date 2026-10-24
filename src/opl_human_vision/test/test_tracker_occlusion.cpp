#include "opl_human_vision/tracker_engine.hpp"
#include <cassert>
#include <algorithm>

int main() {
  using namespace opl_human_vision;
  TrackerEngine tracker(60, 3, 0.8f, 1);
  opl_interfaces::msg::TrackedHuman a, b;
  a.bbox.x_offset = 10; a.bbox.y_offset = 10;
  a.bbox.width = 100; a.bbox.height = 200; a.confidence = 0.9f;
  b = a; b.bbox.x_offset = 60;
  double t = 1.0;
  std::vector<Track> tracks;
  for (int i = 0; i < 3; ++i) tracks = tracker.update({a, b}, {}, t += 1.0/30);
  assert(tracks.size() == 2);
  const int first = tracks[0].id, second = tracks[1].id;
  for (int i = 0; i < 20; ++i) tracks = tracker.update({b}, {}, t += 1.0/30);
  assert(tracks.size() == 1 && tracks[0].id == second);
  tracks = tracker.update({a, b}, {}, t += 1.0/30);
  assert(tracks.size() == 2);
  assert(std::any_of(tracks.begin(), tracks.end(), [first](const Track& x) { return x.id == first; }));
  for (int i = 0; i < 61; ++i) tracker.update({}, {}, t += 1.0/30);
  for (int i = 0; i < 3; ++i) tracks = tracker.update({a}, {}, t += 1.0/30);
  assert(tracks.size() == 1 && tracks[0].id != first);
  // Equal geometry cannot distinguish two people, even if assignment is stable.
  TrackerEngine ambiguous(60, 1, 0.8f, 1);
  a.bbox.x_offset = 10; b = a;
  tracks = ambiguous.update({a, b}, {}, 1.0);
  tracks = ambiguous.update({b, a}, {}, 1.0 + 1.0/30);
  assert(tracks.size() == 2);
  assert(tracks[0].association_ambiguous && tracks[1].association_ambiguous);
  tracks = ambiguous.update({a}, {}, 1.0 + 2.0/30);
  assert(std::all_of(tracks.begin(), tracks.end(), [](const Track& x) {
    return x.association_ambiguous;
  }));
  TrackerEngine separate(60, 1, 0.8f, 1);
  b.bbox.x_offset = 250;
  tracks = separate.update({a, b}, {}, 1.0);
  const int left = tracks[0].id, right = tracks[1].id;
  tracks = separate.update({b, a}, {}, 1.0 + 1.0/30);
  assert(tracks.size() == 2);
  for (const auto& x : tracks) {
    assert(!x.association_ambiguous);
    assert((x.bbox.x < 100 && x.id == left) || (x.bbox.x > 200 && x.id == right));
  }
  // Remote people cannot change the certainty of this clearly separated assignment.
  for (int extras : {0, 8}) {
    TrackerEngine crowd(60, 1, 0.8f, 1);
    a.bbox.x_offset = 0; b = a; b.bbox.x_offset = 10;
    std::vector<opl_interfaces::msg::TrackedHuman> detections{a, b};
    for (int i = 0; i < extras; ++i) {
      auto remote = a; remote.bbox.x_offset = 250 + 200 * i;
      detections.push_back(remote);
    }
    crowd.update(detections, {}, 1.0);
    tracks = crowd.update(detections, {}, 1.0 + 1.0/30);
    assert(std::none_of(tracks.begin(), tracks.end(), [](const Track& x) {
      return x.association_ambiguous;
    }));
  }

}
