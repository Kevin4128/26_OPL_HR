#include "opl_human_vision/tracker_engine.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>

// Link-time stand-in: exercise the real crop/association path without GPU models.
namespace {
int extractions = 0;
bool invalid_feature = false;
}
namespace opl_human_vision {
OsnetExtractor::OsnetExtractor(const std::string&, int) {}
OsnetExtractor::~OsnetExtractor() = default;
bool OsnetExtractor::init() { return true; }
std::vector<float> OsnetExtractor::extract(const cv::Mat& crop) {
  ++extractions;
  if (invalid_feature) return {std::numeric_limits<float>::quiet_NaN(), 0, 0};
  const auto pixel = crop.at<cv::Vec3b>(crop.rows / 2, crop.cols / 2);
  return {float(pixel[0]), float(pixel[1]), float(pixel[2])};
}
}
using namespace opl_human_vision;

opl_interfaces::msg::TrackedHuman detection(int x, float confidence = 0.9f) {
  opl_interfaces::msg::TrackedHuman h;
  h.bbox.x_offset = x; h.bbox.y_offset = 20;
  h.bbox.width = 100; h.bbox.height = 200; h.confidence = confidence;
  return h;
}
cv::Mat image(int first, int second, bool swap = false) {
  cv::Mat frame = cv::Mat::zeros(400, 600, CV_8UC3);
  frame(cv::Rect(first, 20, 100, 200)).setTo(swap ? cv::Scalar(0, 255, 0) : cv::Scalar(255, 0, 0));
  if (second >= 0)
    frame(cv::Rect(second, 20, 100, 200)).setTo(swap ? cv::Scalar(255, 0, 0) : cv::Scalar(0, 255, 0));
  return frame;
}
const Track& byId(const std::vector<Track>& tracks, int id) {
  const auto it = std::find_if(tracks.begin(), tracks.end(), [id](const Track& t) { return t.id == id; });
  assert(it != tracks.end());
  return *it;
}
int main() {
  // Looser proximity deliberately lets appearance challenge the geometric match.
  TrackerEngine tracker(60, 1, 0.8f, 1, "mock", true, 0.85f, 60, 0.65f, 0.1f, 0.7f, 0.8f);
  auto tracks = tracker.update({detection(100), detection(160)}, image(100, 160), 1.0);
  assert(tracks.size() == 2 && extractions == 2);
  const int blue = tracks[0].id, green = tracks[1].id;
  tracks = tracker.update({detection(100), detection(160)}, image(100, 160, true), 1.033);
  assert(extractions == 4 && tracks.size() == 2);
  assert(byId(tracks, blue).bbox.x > 150 && byId(tracks, green).bbox.x < 110);
  assert(!byId(tracks, blue).association_ambiguous && !byId(tracks, green).association_ambiguous);
  const auto blue_feature = byId(tracks, blue).smooth_feature;
  const auto green_feature = byId(tracks, green).smooth_feature;
  tracks = tracker.update({detection(160, 0.3f), detection(100, 0.3f)}, image(100, 160), 1.066);
  assert(extractions == 4 && tracks.size() == 2);
  assert(byId(tracks, blue).smooth_feature == blue_feature);
  assert(byId(tracks, green).smooth_feature == green_feature);
  tracks = tracker.update({detection(100), detection(110)}, image(100, 110), 1.1);
  assert(extractions == 4); // overlapping crops cannot contaminate the EMA
  for (const auto& track : tracks) {
    if (track.id == blue) assert(track.smooth_feature == blue_feature);
    if (track.id == green) assert(track.smooth_feature == green_feature);
  }
  tracker.update({detection(100), detection(110)}, {}, 1.133);
  assert(extractions == 4); // no matched RGB means no appearance extraction

  TrackerEngine smoothing(60, 1, 0.8f, 1, "mock", true);
  tracks = smoothing.update({detection(10)}, image(10, -1), 2.0);
  assert(tracks.size() == 1);
  const int id = tracks.front().id;
  tracks = smoothing.update({detection(10)}, image(10, -1, true), 2.033);
  assert(tracks.size() == 1 && tracks.front().id == id);
  const double norm = std::hypot(0.9, 0.1);
  assert(std::abs(tracks.front().smooth_feature[0] - 0.9 / norm) < 1e-6);
  assert(std::abs(tracks.front().smooth_feature[1] - 0.1 / norm) < 1e-6);
  const auto previous_feature = tracks.front().smooth_feature;
  invalid_feature = true;
  tracks = smoothing.update({detection(10)}, image(10, -1), 2.066);
  assert(tracks.size() == 1 && tracks.front().smooth_feature == previous_feature);
}
