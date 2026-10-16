#include "opl_human_vision/tracker_engine.hpp"
#include <cassert>
#include <iostream>
#include <rclcpp/rclcpp.hpp>

using namespace opl_human_vision;
using Human = opl_interfaces::msg::TrackedHuman;
Human person(int x, double world, double time, bool spatial = true) {
  Human h;
  h.bbox.x_offset = x;
  h.bbox.y_offset = 30;
  h.bbox.width = 80;
  h.bbox.height = 160;
  h.confidence = .9;
  h.fixed_position_valid = spatial;
  h.fixed_position.header.frame_id = "odom";
  h.fixed_position.header.stamp = rclcpp::Time(int64_t(time * 1e9));
  h.fixed_position.point.x = world;
  return h;
}
int main() {
  cv::Mat frame(240, 360, CV_8UC3, cv::Scalar(100));
  // Weak proposals recover established people, never create people by
  // themselves.
  TrackerEngine weak(60, 1, .1, 0);
  auto low = person(20, 0, 1, false);
  low.confidence = .4f;
  assert(weak.update({low}, frame, 1).empty());
  auto strong = person(20, 0, 1.1, false);
  auto established = weak.update({strong}, frame, 1.1);
  assert(established.size() == 1);
  const int weak_id = established[0].id;
  low = person(23, 0, 1.2, false);
  low.confidence = .4f;
  auto recovered_low = weak.update({low}, frame, 1.2);
  assert(recovered_low.size() == 1 && recovered_low[0].id == weak_id);
  auto unrelated_low = person(260, 0, 1.3, false);
  unrelated_low.confidence = .4f;
  assert(weak.update({unrelated_low}, frame, 1.3).empty());
  std::vector<float> a(512, 0), b(512, 0);
  a[0] = 1;
  b[1] = 1;
  TrackerEngine tracker(60, 1, .1, 0, "", true, .85, 60, 1, .2, .75, 2, 2);
  auto tracks =
      tracker.update({person(20, 0, 1), person(200, 3, 1)}, frame, 1, {a, b});
  assert(tracks.size() == 2);
  int first = tracks[0].id, second = tracks[1].id;
  int switches = 0;
  double t = 1;
  for (int step = 1; step <= 8; ++step) {
    t += .1;
    tracks = tracker.update(
        {person(20 + step * 22, 0, t), person(200 - step * 22, 3, t)}, frame, t,
        {a, b});
    assert(tracks.size() == 2);
    for (const auto &track : tracks) {
      int expected =
          track.observation.fixed_position.point.x == 0 ? first : second;
      switches += track.id != expected;
    }
  }
  assert(switches == 0);
  // Partial occlusion followed by both returning, then a full brief occlusion.
  for (int i = 0; i < 5; ++i) {
    t += .1;
    tracker.update({person(196, 0, t)}, frame, t, {a});
  }
  t += .1;
  tracks =
      tracker.update({person(196, 0, t), person(24, 3, t)}, frame, t, {a, b});
  assert(tracks.size() == 2);
  for (const auto &track : tracks)
    assert(track.id ==
           (track.observation.fixed_position.point.x == 0 ? first : second));
  t += .3;
  tracker.update({}, frame, t);
  t += .3;
  tracks =
      tracker.update({person(196, 0, t), person(24, 3, t)}, frame, t, {a, b});
  assert(tracks.size() == 2);
  for (const auto &track : tracks)
    assert(track.id ==
           (track.observation.fixed_position.point.x == 0 ? first : second));
  // Same elapsed expiry at different processed rates; stale tracks expire
  // BEFORE assignment.
  for (int rate : {5, 15, 30}) {
    TrackerEngine timed(60, 1, .1, 0);
    auto initial = timed.update({person(20, 0, 1, false)}, frame, 1);
    int id = initial[0].id;
    for (int i = 1; i < int(1.8 * rate); ++i)
      timed.update({}, frame, 1. + double(i) / rate);
    auto recovered = timed.update({person(20, 0, 2.9, false)}, frame, 2.9);
    assert(recovered[0].id == id);
    auto expired = timed.update({person(20, 0, 5.01, false)}, frame, 5.01);
    assert(expired[0].id != id);
    int newer = expired[0].id;
    assert(timed.update({person(20, 0, 5.01, false)}, frame, 5.01).empty());
    auto reset = timed.update({person(20, 0, 1, false)}, frame, 1);
    assert(reset[0].id != newer);
  }
  // Similar clothing, ambiguous overlap, absent TF: refuse a forced old-ID
  // transfer.
  TrackerEngine similar(60, 1, .1, 0, "", true, .85, 60, 1, .2, .75);
  auto original = similar.update(
      {person(20, 0, 1, false), person(200, 3, 1, false)}, frame, 1, {a, a});
  auto ambiguous =
      similar.update({person(105, 0, 1.2, false), person(115, 3, 1.2, false)},
                     frame, 1.2, {a, a});
  assert(ambiguous
             .empty()); // Ambiguous detections must not create replacement IDs.
  auto separated =
      similar.update({person(20, 0, 1.3, false), person(200, 3, 1.3, false)},
                     frame, 1.3, {a, a});
  assert(separated.size() == 2);
  assert(separated[0].id == original[0].id &&
         separated[1].id == original[1].id);
  // Prediction visibility follows elapsed time at every processed rate.
  for (int rate : {5, 15, 30}) {
    TrackerEngine predicted(60, 1, .1, 1);
    auto visible = predicted.update({person(20, 0, 10, false)}, frame, 10);
    const int id = visible[0].id;
    for (int i = 1; i <= int(.4 * rate); ++i) {
      auto held = predicted.update({}, frame, 10. + double(i) / rate);
      assert(held.size() == 1 && held[0].id == id &&
             held[0].time_since_update > 0);
    }
    assert(predicted.update({}, frame, 10.51).empty());
    auto returned = predicted.update({person(20, 0, 11.8, false)}, frame, 11.8);
    assert(returned.size() == 1 && returned[0].id == id);
  }
  TrackerEngine smoothed(60, 1, .1, 0);
  smoothed.update({person(100, 0, 20, false)}, frame, 20);
  double output_error = 0, raw_error = 0;
  for (int i = 1; i <= 20; ++i) {
    const int x = 100 + (i % 2 ? 6 : -6);
    auto tracks = smoothed.update({person(x, 0, 20. + i / 30., false)}, frame,
                                  20. + i / 30.);
    assert(tracks.size() == 1);
    output_error += std::abs(tracks[0].bbox.x - 100);
    raw_error += std::abs(x - 100);
  }
  assert(output_error < raw_error * .8);
  // A weak recovery must not teach a new appearance that later authorizes
  // drift.
  TrackerEngine guarded_gallery(60, 1, .35, 0, "", true, .85, 60, 1, .2, .75);
  auto gallery_first =
      guarded_gallery.update({person(20, 0, 30, false)}, frame, 30, {a});
  std::vector<float> view(512, 0), intruder(512, 0);
  view[0] = std::cos(CV_PI * .15);
  view[1] = std::sin(CV_PI * .15);
  intruder[0] = std::cos(CV_PI * .30);
  intruder[1] = std::sin(CV_PI * .30);
  auto weak_view = person(20, 0, 30.1, false);
  weak_view.confidence = .4f;
  auto recovered_view =
      guarded_gallery.update({weak_view}, frame, 30.1, {view});
  assert(recovered_view.size() == 1 &&
         recovered_view[0].id == gallery_first[0].id);
  auto foreign = guarded_gallery.update({person(80, 0, 30.2, false)}, frame,
                                        30.2, {intruder});
  assert(foreign.size() == 1 && foreign[0].id != gallery_first[0].id);
  // Synthetic camera rotation, actual image GMC with a textured background.
  cv::Mat texture(240, 360, CV_8UC3);
  cv::RNG rng(123);
  rng.fill(texture, cv::RNG::UNIFORM, 0, 255);
  TrackerEngine rotation(60, 1, .1, 0);
  auto before = rotation.update({person(110, 0, 1)}, texture, 1);
  int id = before[0].id;
  auto affine = cv::getRotationMatrix2D({180, 120}, 10, 1.05);
  cv::Mat moved;
  cv::warpAffine(texture, moved, affine, texture.size());
  std::vector<cv::Point2f> corners{
      {110, 30}, {190, 30}, {190, 190}, {110, 190}},
      warped;
  cv::transform(corners, warped, affine);
  auto box = cv::boundingRect(warped);
  auto observation = person(box.x, 0, 1.1);
  observation.bbox.y_offset = box.y;
  observation.bbox.width = box.width;
  observation.bbox.height = box.height;
  auto after = rotation.update({observation}, moved, 1.1);
  assert(after.size() == 1 && after[0].id == id);
  std::cout << "Synthetic crossing ID switches=" << switches
            << ", partial/full occlusion recovered 2/2, rates 5/15/30 passed\n";
  std::cout << "Similar-clothing ambiguity deferred assignment without new "
               "IDs; camera "
               "rotation retained ID\n";
}
