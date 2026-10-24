#include "opl_human_vision/face_engine.hpp"
#include <cassert>

int main() {
  using namespace opl_human_vision;
  auto seen = std::chrono::steady_clock::time_point{};
  const auto start = seen;
  assert(refreshIdentityLock(seen, start + std::chrono::seconds(1), 3.0, true));
  assert(seen == start + std::chrono::seconds(1));
  assert(refreshIdentityLock(seen, start + std::chrono::seconds(2), 3.0, false));
  assert(seen == start + std::chrono::seconds(1));
  assert(!refreshIdentityLock(seen, start + std::chrono::seconds(5), 3.0, true));
  assert(seen == start + std::chrono::seconds(1));
  FaceEngine engine("", "", "", 0.55f, 0.85f, 2, 0.0f);
  engine.configureFaceDetector(0.61f, 0.22f);
  const auto thresholds = engine.detectorThresholds();
  assert(thresholds.first == 0.61f && thresholds.second == 0.22f);
  geometry_msgs::msg::Point p;
  FaceResult face; face.face_embedding = {1, 0}; face.body_embedding = {1, 0};
  FaceResult body; body.body_embedding = {1, 0};
  // Recognition alone never creates an enrolled identity.
  auto r = engine.processFeatures(face, 1, {}, p, 1);
  assert(r.persistent_id >= 1000000000ULL && r.name == "unknown");
  assert(r.identity_state == IdentityState::Unconfirmed);
  engine.authorizeEnrollment(1, "furia");
  engine.processFeatures(face, 1, {}, p, 2);
  r = engine.processFeatures(face, 1, {}, p, 3);
  assert(r.just_scanned && r.name == "furia" && r.persistent_id == 1);
  // Covered/incorrect face embeddings cannot revoke an unambiguous identity.
  FaceResult covered = face; covered.face_embedding = {0, 1};
  r = engine.processFeatures(covered, 1, {}, p, 4);
  assert(r.persistent_id == 1 && r.identity_state == IdentityState::Tracked);
  r = engine.processFeatures(body, 1, {}, p, 5);
  assert(r.persistent_id == 1 && r.name == "furia");
  r = engine.processFeatures({}, 1, {}, p, 6);
  assert(r.persistent_id == 1);
  // New raw tracks cannot acquire by body, even with identical outfits.
  r = engine.processFeatures(body, 2, {}, p, 7);
  assert(r.persistent_id >= 1000000000ULL);
  engine.processFeatures(face, 2, {}, p, 8);
  engine.processFeatures(face, 2, {}, p, 8); // same RGB frame cannot confirm twice
  r = engine.processFeatures(face, 2, {}, p, 9);
  assert(r.persistent_id != 1);
  r = engine.processFeatures(face, 2, {}, p, 10);
  assert(r.persistent_id == 1 && r.identity_state == IdentityState::FaceConfirmed);
  // Ownership guard blocks duplicate use even from the engine's cache.
  r = engine.processFeatures(face, 2, {1}, p, 11);
  assert(r.persistent_id != 1);
  engine.invalidateTrack(1);
  r = engine.processFeatures(body, 1, {}, p, 12);
  assert(r.persistent_id != 1);
  // A second identity requires a second explicit enrollment authorization.
  engine.authorizeEnrollment(3, "other");
  engine.processFeatures(covered, 3, {}, p, 13);
  r = engine.processFeatures(covered, 3, {}, p, 14);
  assert(r.just_scanned && r.persistent_id == 2 && r.name == "other");
  engine.processFeatures(covered, 2, {}, p, 15);
  engine.processFeatures(covered, 2, {}, p, 16);
  r = engine.processFeatures(covered, 2, {}, p, 17);
  assert(r.persistent_id == 2 && r.name == "other");
  engine.authorizeEnrollment(4, "cancelled");
  engine.processFeatures(face, 4, {}, p, 18);
  r = engine.processFeatures(face, 4, {}, p, 18);
  assert(!r.just_scanned);
  engine.invalidateTrack(4);
  r = engine.processFeatures(face, 4, {}, p, 19);
  assert(!r.just_scanned && r.name != "cancelled");
  engine.cleanupStaleRawTracks({});
  r = engine.processFeatures(body, 2, {}, p, 18);
  assert(r.persistent_id != 2);
}
