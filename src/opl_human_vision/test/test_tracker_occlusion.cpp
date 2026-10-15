#include "opl_human_vision/tracker_engine.hpp"
#include <cassert>
#include <algorithm>

int main() {
  using namespace opl_human_vision;
  TrackerEngine tracker(60, 3, 0.35f, 1);
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
}
