#include "opl_human_vision/enrollment_queue.hpp"
#include "opl_human_vision/face_engine.hpp"

#include <cassert>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sys/stat.h>
#include <unistd.h>

using namespace opl_human_vision;
namespace fs = std::filesystem;
std::vector<float> embedding(size_t axis) {
  std::vector<float> v(512, 0);
  v[axis] = 1;
  return v;
}
FaceResult features(size_t axis, bool good = true) {
  FaceResult f;
  f.face_quality_valid = good;
  f.face_embedding = embedding(axis);
  f.body_embedding = embedding(10);
  return f;
}
FaceEngine engine(int scans = 3) {
  return FaceEngine("", "", "", 0.55, 0.85, scans, 0.3, 2);
}

int main() {
  auto root =
      fs::current_path() / ("identity_policy_" + std::to_string(getpid()));
  fs::create_directory(root);
  fs::permissions(root, fs::perms::owner_all);
  struct Cleanup {
    fs::path root;
    ~Cleanup() { fs::remove_all(root); }
  } cleanup{root};
  opl_interfaces::msg::TrackedHuman human;
  human.observed = true;
  human.confidence = .9f;
  uint64_t alice = 0;
  {
    auto face = engine();
    face.configureStore((root / "identities").string(), "synthetic-model-v1");
    face.authorizeEnrollment(1, "Alice");
    human.confidence = .4f;
    auto weak_body = face.processFeatures(features(0), 1, {}, human, .9);
    assert(weak_body.accepted_samples == 0 && face.registeredCount() == 0);
    human.confidence = .9f;
    auto poor = face.processFeatures(features(0, false), 1, {}, human, 1);
    assert(poor.accepted_samples == 0 && poor.enrollment_state == "scanning");
    auto first = face.processFeatures(features(0), 1, {}, human, 1.1);
    assert(first.accepted_samples == 1 && face.registeredCount() == 0);
    auto duplicate = face.processFeatures(features(0), 1, {}, human, 1.1);
    assert(duplicate.accepted_samples == 1);
    face.processFeatures(features(0), 1, {}, human, 1.25);
    auto done = face.processFeatures(features(0), 1, {}, human, 1.41);
    assert(done.enrollment_state == "completed" && done.just_scanned &&
           done.persistent_id > 0);
    alice = done.persistent_id;
    assert(face.registeredCount() == 1 &&
           fs::exists(root / "identities/identities.json"));
    struct stat st {};
    stat((root / "identities").c_str(), &st);
    assert((st.st_mode & 0777) == 0700);
    stat((root / "identities/identities.json").c_str(), &st);
    assert((st.st_mode & 0777) == 0600);
    face.authorizeEnrollment(1, "Alice");
    auto repeat = face.processFeatures(features(0), 1, {}, human, 1.6);
    assert(repeat.enrollment_state == "already_enrolled" &&
           repeat.persistent_id == alice && !face.enrollmentAuthorized(1));
    face.authorizeEnrollment(1, "DifferentName");
    assert(
        face.processFeatures(features(0), 1, {}, human, 1.8).enrollment_state ==
        "failed");
    face.cleanupStaleRawTracks({1, 2});
    assert(face.processFeatures(features(0), 2, {}, human, 1.9).name ==
           "unknown");
    assert(face.processFeatures(features(0), 2, {alice}, human, 2).name ==
           "unknown");
    // A verified lineage retains its identity while an observed body remains.
    assert(face.processFeatures(features(0, false), 1, {}, human, 2.1).name ==
           "Alice");
    face.authorizeEnrollment(3, "Alice");
    assert(
        face.processFeatures(features(2), 3, {}, human, 2.2).enrollment_state ==
        "failed");
    face.authorizeEnrollment(4, "NewPerson");
    face.processFeatures(features(3), 4, {}, human, 2.3);
    assert(
        face.processFeatures(features(4), 4, {}, human, 2.5).enrollment_state ==
        "failed");
    assert(face.registeredCount() == 1);
    bool locked = false;
    try {
      IdentityStore other((root / "identities").string(), "synthetic-model-v1");
    } catch (...) {
      locked = true;
    }
    assert(locked);
  }
  {
    auto restarted = engine();
    restarted.configureStore((root / "identities").string(),
                             "synthetic-model-v1");
    assert(restarted.registeredCount() == 1);
    human.raw_track_id = 11;
    human.bbox.x_offset = 100;
    human.bbox.y_offset = 40;
    human.bbox.width = 120;
    human.bbox.height = 300;
    auto body = features(0, false);
    auto run = [&](FaceResult f, double t) {
      return restarted.processFrameFeatures({human}, {f}, t).front();
    };
    assert(run(features(0), 1).identity_state == "VERIFIED");
    for (int i = 1; i <= 100; ++i) {
      auto tracked = run(body, 1 + i * .1);
      assert(tracked.name == "Alice" && tracked.identity_state == "TRACKED" &&
             tracked.match_score > .99f);
    }
    // Face age does not expire an observed verified body lineage.
    assert(run(body, 11.1).face_age_sec > 10);
    human.observed = false;
    auto predicted = run({}, 11.2);
    assert(predicted.identity_state == "LOST" && predicted.body_age_sec > .09);
    human.observed = true;
    human.raw_track_id = 12;
    assert(run(body, 11.4).name ==
           "Alice"); // Unique recent verified body recovery.
    assert(run(features(2), 11.5).identity_state == "AMBIGUOUS");
    assert(run(body, 11.6).name == "unknown");
    assert(run(features(0), 11.7).name == "unknown");
    assert(run({}, 11.75).name == "unknown");
    assert(run(features(0), 11.8).name == "unknown");
    assert(run({}, 11.85).name == "unknown");
    assert(run(features(0), 11.91).name == "Alice");
    restarted.processFrameFeatures({}, {}, 12);
    assert(!restarted.lostIdentities(12).empty());
    human.raw_track_id = 13;
    assert(run(body, 14.1).name == "unknown"); // Recovery window expired.
    restarted.resetTemporalState();
    auto other = human;
    other.raw_track_id = 14;
    other.bbox.x_offset = 400;
    // Identical simultaneous face claims are ambiguous, independent of input
    // order.
    auto conflict = restarted.processFrameFeatures(
        {human, other}, {features(0), features(0)}, 1);
    assert(conflict[0].name == "unknown" && conflict[1].name == "unknown");
    restarted.resetTemporalState();
    // An already visible stranger cannot inherit a disappearing owner's name.
    restarted.processFrameFeatures({human, other}, {features(0), features(2)},
                                   1);
    auto stranger = restarted.processFrameFeatures({other}, {body}, 1.5);
    assert(stranger.front().name == "unknown");
    restarted.resetTemporalState();
    restarted.processFrameFeatures({human, other}, {features(0), features(2)},
                                   1);
    other.bbox.x_offset = human.bbox.x_offset + 10;
    auto overlap =
        restarted.processFrameFeatures({human, other}, {body, body}, 1.1);
    assert(overlap[0].name == "unknown" &&
           overlap[0].identity_state == "AMBIGUOUS");
    other.bbox.x_offset = 400;
    assert(restarted.processFrameFeatures({human, other}, {body, body}, 1.2)[0]
               .name == "unknown");
    assert(restarted
               .processFrameFeatures({human, other}, {features(0), features(2)},
                                     1.3)[0]
               .name == "unknown");
    assert(restarted
               .processFrameFeatures({human, other}, {features(0), features(2)},
                                     1.4)[0]
               .name == "unknown");
    assert(restarted
               .processFrameFeatures({human, other}, {features(0), features(2)},
                                     1.51)[0]
               .name == "Alice");
    restarted.resetTemporalState();
    auto weaker = features(0);
    weaker.face_embedding[0] = .8f;
    weaker.face_embedding[1] = .6f;
    auto distinct = restarted.processFrameFeatures({other, human},
                                                   {weaker, features(0)}, 1);
    assert(distinct[0].name == "unknown" && distinct[1].name == "Alice");
    restarted.resetTemporalState();
    int true_matches = 0, false_matches = 0;
    for (int i = 0; i < 20; ++i) {
      restarted.cleanupStaleRawTracks({1});
      auto known =
          restarted.processFeatures(features(0), 1, {}, human, 10 + i * .2);
      true_matches += known.name == "Alice";
      auto unknown = restarted.processFeatures(features(2), 2, {alice}, human,
                                               10 + i * .2);
      false_matches += unknown.name == "Alice";
    }
    assert(true_matches == 20 && false_matches == 0);
    // Close-but-below-threshold and ambiguous faces remain unknown.
    restarted.cleanupStaleRawTracks({5});
    auto near = features(0);
    near.face_embedding[0] = .5f;
    near.face_embedding[1] = std::sqrt(.75f);
    assert(restarted.processFeatures(near, 5, {}, human, 19).name == "unknown");
    // A stationary world point remains comparable while camera-relative
    // coordinates change.
    human.fixed_position_valid = true;
    human.fixed_position.header.frame_id = "odom";
    human.fixed_position.header.stamp.sec = 20;
    human.fixed_position.point.x = 1;
    human.position_valid = true;
    human.position_3d.z = 1;
    restarted.cleanupStaleRawTracks({1});
    assert(restarted.processFeatures(features(0), 1, {}, human, 20).name ==
           "Alice");
    human.position_3d.z = 8;
    human.fixed_position.header.stamp.nanosec = 100000000;
    assert(restarted.processFeatures(features(0), 1, {}, human, 20.1).name ==
           "Alice");
    human.fixed_position.point.x = 10;
    human.fixed_position.header.stamp.nanosec = 200000000;
    assert(restarted.processFeatures(features(0), 1, {}, human, 20.2).name ==
           "unknown");
    human.fixed_position_valid = false;
    assert(restarted.processFeatures(features(0), 1, {}, human, 20.3).name ==
           "unknown");
    assert(restarted.processFeatures(features(0), 1, {}, human, 20.4).name ==
           "unknown");
    assert(restarted.processFeatures(features(0), 1, {}, human, 20.51).name ==
           "Alice");
    std::string message;
    assert(restarted.deleteIdentity(alice, message));
    assert(!restarted.deleteIdentity(alice, message));
    assert(restarted.processFeatures(features(0), 1, {}, human, 20.6).name ==
           "unknown");
    std::cout << "Synthetic identity metrics: known 20/20, false registered "
                 "matches 0/20\n";
  }
  {
    auto restarted = engine();
    restarted.configureStore((root / "identities").string(),
                             "synthetic-model-v1");
    assert(restarted.registeredCount() == 0);
  }
  // Version-1 migration and bounded, anchored adaptive memory use synthetic
  // files only.
  const auto adaptive_path = root / "adaptive";
  {
    IdentityStore store(adaptive_path.string(), "adaptive-model");
    IdentityRecord record;
    record.persistent_id = 1;
    record.name = "Anchor";
    record.face_gallery = {embedding(0)};
    store.save({record}, 2);
  }
  {
    std::ifstream input(adaptive_path / "identities.json");
    std::string legacy((std::istreambuf_iterator<char>(input)), {});
    const auto version = legacy.find("\"version\": 2");
    assert(version != std::string::npos);
    legacy.replace(version, 12, "\"version\": 1");
    std::ofstream output(adaptive_path / "identities.json");
    output << legacy;
  }
  {
    auto adaptive = engine();
    adaptive.configureStore(adaptive_path.string(), "adaptive-model");
    human.fixed_position_valid = false;
    human.raw_track_id = 1;
    human.confidence = .9f;
    human.observed = true;
    auto angle = [](int axis) {
      auto f = features(0);
      f.face_embedding[0] = .8f;
      f.face_embedding[axis] = .6f;
      return f;
    };
    auto feed = [&](FaceResult f, double t) {
      return adaptive.processFrameFeatures({human}, {f}, t).front();
    };
    auto a = angle(30);
    feed(a, 1);
    feed(a, 1.25);
    auto saved = feed(a, 1.51);
    assert(saved.identity_reason.find("saved durably") != std::string::npos);
    auto gallery_size = [&]() {
      cv::FileStorage file((adaptive_path / "identities.json").string(),
                           cv::FileStorage::READ);
      return file["identities"][0]["faces"].size();
    };
    assert(gallery_size() == 2);
    // Duplicate faces and body-only observations do not grow the gallery.
    feed(a, 2);
    feed(a, 2.3);
    feed(a, 2.6);
    feed(features(0, false), 3);
    human.confidence = .4f;
    feed(angle(31), 7);
    feed(angle(31), 7.25);
    feed(angle(31), 7.51);
    assert(gallery_size() == 2);
    human.confidence = .9f;
    // Many distinct, strongly anchor-compatible views stop at the bounded cap.
    for (int i = 0; i < 15; ++i) {
      const double t = 13 + i * 6;
      auto f = angle(40 + i);
      feed(f, t);
      feed(f, t + .25);
      feed(f, t + .51);
    }
  }
  {
    IdentityStore store(adaptive_path.string(), "adaptive-model");
    uint64_t next = 0;
    auto records = store.load(next);
    assert(records.size() == 1 && records[0].anchor_count == 1 &&
           records[0].face_gallery.size() == 12);
    assert(records[0].face_gallery.front() == embedding(0));
    assert(next == 2);
  }
  bool mismatch = false;
  try {
    auto e = engine();
    e.configureStore((root / "identities").string(), "wrong-model");
  } catch (...) {
    mismatch = true;
  }
  assert(mismatch);
  fs::create_symlink(root / "identities", root / "link");
  bool symlink = false;
  try {
    IdentityStore e((root / "link").string(), "synthetic-model-v1");
  } catch (...) {
    symlink = true;
  }
  assert(symlink);
  chmod((root / "identities/identities.json").c_str(), 0644);
  bool permissions = false;
  try {
    IdentityStore e((root / "identities").string(), "synthetic-model-v1");
    uint64_t n;
    e.load(n);
  } catch (...) {
    permissions = true;
  }
  assert(permissions);
  chmod((root / "identities/identities.json").c_str(), 0600);
  {
    std::ofstream bad(root / "identities/identities.json");
    bad << "invalid data";
  }
  bool corrupt = false;
  try {
    IdentityStore e((root / "identities").string(), "synthetic-model-v1");
    uint64_t n;
    e.load(n);
  } catch (...) {
    corrupt = true;
  }
  assert(corrupt);
  {
    auto no_store = engine();
    no_store.authorizeEnrollment(1);
    no_store.processFeatures(features(0), 1, {}, human, 1);
    no_store.processFeatures(features(0), 1, {}, human, 1.15);
    assert(no_store.processFeatures(features(0), 1, {}, human, 1.31)
               .enrollment_state == "failed");
    assert(no_store.registeredCount() == 0);
  }
  EnrollmentQueue queue;
  queue.start({{1, .8f, ""}, {2, .9f, ""}}, 0);
  assert(queue.current()->raw_id == 2 && queue.total() == 2);
  queue.observe({1, 2, 3}, .5);
  assert(queue.total() == 2 && queue.current()->raw_id == 2);
  queue.observe({1}, 1);
  assert(queue.failure(2.49).empty() && !queue.failure(2.51).empty());
  queue.advance(false, 2.51);
  assert(queue.current()->raw_id == 1);
  queue.observe({1}, 32);
  assert(!queue.failure(33).empty());
  queue.advance(false, 33);
  assert(!queue.current());
  queue.start({{1, .8f, ""}}, 40);
  queue.advance(true, 41);
  assert(queue.completed == 1 && queue.failed == 0);
  queue.missing_sec = 3;
  queue.timeout_sec = 45;
  queue.start({{1, .9f, ""}, {2, .8f, ""}}, 50);
  const auto cancelled_session = queue.session_id;
  queue.cancel(51);
  assert(!queue.current() && queue.failed == 2 &&
         queue.session_id == cancelled_session);
  assert(queue.missing_sec == 3 && queue.timeout_sec == 45);
  queue.start({{1, .9f, ""}}, 52);
  assert(queue.session_id == cancelled_session + 1);
  cv::Mat crop(112, 112, CV_8UC3, cv::Scalar(100));
  std::string reason;
  std::vector<cv::Point2f> points{
      {30, 30}, {70, 30}, {50, 50}, {35, 75}, {65, 75}};
  assert(FaceEngine::acceptableFace(crop, points, .9, reason));
  assert(!FaceEngine::acceptableFace(crop, points, .5, reason));
  assert(!FaceEngine::acceptableAlignedFace(crop, reason)); // Flat/blurred.
  cv::Mat textured(112, 112, CV_8UC3);
  cv::RNG rng(42);
  rng.fill(textured, cv::RNG::UNIFORM, 60, 190);
  assert(FaceEngine::acceptableAlignedFace(textured, reason));
  cv::Mat dark(112, 112, CV_8UC3);
  rng.fill(dark, cv::RNG::UNIFORM, 0, 25);
  assert(!FaceEngine::acceptableAlignedFace(dark, reason));
  cv::Mat bright(112, 112, CV_8UC3);
  rng.fill(bright, cv::RNG::UNIFORM, 235, 255);
  assert(!FaceEngine::acceptableAlignedFace(bright, reason));
  points[2] = {90, 50};
  assert(!FaceEngine::acceptableFace(crop, points, .9, reason));
  points[2] = {50, 50};
  points[0] = {1, 1};
  assert(!FaceEngine::acceptableFace(crop, points, .9, reason));
  std::cout << "Enrollment scans, restart, deletion, ownership, dropout, "
               "storage and queue checks passed\n";
}
