#include "opl_human_vision/face_engine.hpp"
#include "opl_human_vision/observation_buffer.hpp"
#include "opl_human_vision/scoped_timing.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace opl_human_vision {
namespace {
float cosine(const std::vector<float> &a, const std::vector<float> &b) {
  if (a.empty() || a.size() != b.size())
    return -1;
  double dot = 0, aa = 0, bb = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    if (!std::isfinite(a[i]) || !std::isfinite(b[i]))
      return -1;
    dot += a[i] * b[i];
    aa += a[i] * a[i];
    bb += b[i] * b[i];
  }
  return aa > 1e-12 && bb > 1e-12 ? float(dot / std::sqrt(aa * bb)) : -1;
}
float similarity(const std::vector<float> &query,
                 const std::vector<std::vector<float>> &gallery) {
  float best = -1;
  for (const auto &face : gallery)
    best = std::max(best, cosine(query, face));
  return best;
}
std::vector<float> average(const std::vector<std::vector<float>> &faces) {
  std::vector<float> mean(512, 0);
  for (const auto &face : faces)
    for (size_t i = 0; i < mean.size(); ++i)
      mean[i] += face[i];
  double norm = 0;
  for (float value : mean)
    norm += value * value;
  if (norm < 1e-12)
    return {};
  for (auto &value : mean)
    value /= std::sqrt(norm);
  return mean;
}
bool usableFace(const FaceResult &features) {
  return features.face_quality_valid && features.face_embedding.size() == 512 &&
         cosine(features.face_embedding, features.face_embedding) > 0.99f;
}
bool validFixed(const opl_interfaces::msg::TrackedHuman &human) {
  const auto &p = human.fixed_position.point;
  return human.fixed_position_valid &&
         stampNs(human.fixed_position.header.stamp) > 0 &&
         !opticalFrame(human.fixed_position.header.frame_id).empty() &&
         std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z);
}
} // namespace
FaceEngine::FaceEngine(const std::string &scrfd, const std::string &adaface,
                       const std::string &osnet, float face, float body,
                       int count, float duration, double timeout,
                       double spatial_age, float spatial_base,
                       float spatial_speed, float /*depth_gate*/,
                       float minimum_learning_confidence)
    : scrfd_(std::make_unique<ScrfdDetector>(scrfd)),
      adaface_(std::make_unique<AdaFaceExtractor>(adaface)),
      osnet_(std::make_unique<OsnetExtractor>(osnet)), face_threshold_(face),
      body_threshold_(body),
      minimum_learning_confidence_(minimum_learning_confidence),
      required_samples_(count), required_duration_(duration),
      session_timeout_(timeout), spatial_max_age_(spatial_age),
      spatial_base_(spatial_base), spatial_speed_(spatial_speed) {
  if (!std::isfinite(minimum_learning_confidence) ||
      minimum_learning_confidence <= 0 || minimum_learning_confidence > 1 ||
      !std::isfinite(face) || face <= 0 || face > 1 || !std::isfinite(body) ||
      body <= 0 || body > 1 || count < 1 || count > 200 ||
      !std::isfinite(duration) || duration < 0 || !std::isfinite(timeout) ||
      timeout <= 0 || !std::isfinite(spatial_age) || spatial_age < 0 ||
      !std::isfinite(spatial_base) || spatial_base < 0 ||
      !std::isfinite(spatial_speed) || spatial_speed < 0)
    throw std::invalid_argument("Invalid identity/enrollment parameters");
}
bool FaceEngine::loadModels() {
  return scrfd_->init() && adaface_->init() && osnet_->init();
}
void FaceEngine::configureStore(const std::string &path,
                                const std::string &model) {
  auto store = std::make_unique<IdentityStore>(path, model);
  uint64_t next = 1;
  auto records = store->load(next);
  store_ = std::move(store);
  reid_gallery_ = std::move(records);
  next_persistent_id_ = next;
  resetTemporalState();
}
void FaceEngine::save(const std::vector<IdentityRecord> &records,
                      uint64_t next) {
  if (!store_)
    throw std::runtime_error("Persistent identity storage is not configured");
  store_->save(records, next);
}
bool FaceEngine::deleteIdentity(uint64_t id, std::string &message) {
  auto records = reid_gallery_;
  auto found =
      std::find_if(records.begin(), records.end(),
                   [id](const auto &r) { return r.persistent_id == id; });
  if (found == records.end()) {
    message = "Identity not found";
    return false;
  }
  records.erase(found);
  try {
    save(records, next_persistent_id_);
  } catch (const std::exception &e) {
    message = e.what();
    return false;
  }
  reid_gallery_ = std::move(records);
  for (auto it = verified_.begin(); it != verified_.end();)
    if (it->second.id == id)
      it = verified_.erase(it);
    else
      ++it;
  for (auto it = raw_registered_.begin(); it != raw_registered_.end();)
    if (it->second == id)
      it = raw_registered_.erase(it);
    else
      ++it;
  positions_.erase(id);
  message = "Stored identity deleted";
  return true;
}
void FaceEngine::authorizeEnrollment(uint64_t raw, const std::string &name) {
  if (!raw || (!name.empty() && !validIdentityName(name)))
    throw std::invalid_argument("Invalid target or name");
  PendingEnrollment pending;
  pending.name = name;
  pending.session_id = next_session_id_++;
  enrollments_[raw] = std::move(pending);
}
void FaceEngine::retireSession(uint64_t raw) {
  const auto mapped = raw_session_.find(raw);
  if (mapped == raw_session_.end())
    return;
  const auto id = mapped->second;
  raw_session_.erase(mapped);
  sessions_.erase(std::remove_if(sessions_.begin(), sessions_.end(),
                                 [id](const auto &s) { return s.id == id; }),
                  sessions_.end());
  positions_.erase(id);
}
void FaceEngine::cancelEnrollment(uint64_t raw) { enrollments_.erase(raw); }
bool FaceEngine::enrollmentAuthorized(uint64_t raw) const {
  return enrollments_.count(raw) != 0;
}
void FaceEngine::resetTemporalState() {
  raw_registered_.clear();
  verified_.clear();
  adaptive_candidates_.clear();
  last_gallery_write_ = -5;
  last_extraction_.clear();
  observed_raw_ids_.clear();
  raw_session_.clear();
  sessions_.clear();
  positions_.clear();
  enrollments_.clear();
  last_timestamp_ = 0;
}
void FaceEngine::cleanupStaleRawTracks(const std::set<uint64_t> &active) {
  observed_raw_ids_ = active;
  for (auto it = raw_registered_.begin(); it != raw_registered_.end();)
    if (!active.count(it->first))
      it = raw_registered_.erase(it);
    else
      ++it;
  for (auto it = raw_session_.begin(); it != raw_session_.end();)
    if (!active.count(it->first))
      it = raw_session_.erase(it);
    else
      ++it;
}
bool FaceEngine::spatiallyCompatible(
    uint64_t id, const opl_interfaces::msg::TrackedHuman &human) const {
  if (!validFixed(human))
    return true;
  auto old = positions_.find(id);
  if (old == positions_.end())
    return true;
  const auto &current = human.fixed_position;
  if (opticalFrame(current.header.frame_id) !=
      opticalFrame(old->second.header.frame_id))
    return true;
  double dt =
      (stampNs(current.header.stamp) - stampNs(old->second.header.stamp)) *
      1e-9;
  if (dt < 0)
    return false;
  if (dt > spatial_max_age_)
    return true;
  const auto &a = current.point;
  const auto &b = old->second.point;
  return std::hypot(std::hypot(a.x - b.x, a.y - b.y), a.z - b.z) <=
         spatial_base_ + spatial_speed_ * dt;
}
void FaceEngine::updateIdentityObservation(
    uint64_t id, const opl_interfaces::msg::TrackedHuman &human) {
  if (validFixed(human))
    positions_[id] = human.fixed_position;
}
bool FaceEngine::acceptableFace(const cv::Mat &crop,
                                const std::vector<cv::Point2f> &lm, float score,
                                std::string &reason) {
  reason = "Face confidence " + std::to_string(score) + " below required 0.65";
  if (crop.empty() || lm.size() != 5 || !std::isfinite(score) || score < 0.65f)
    return false;
  double eyes = cv::norm(lm[0] - lm[1]);
  reason = "Face too small or partly outside crop (eye distance " +
           std::to_string(eyes) + " px; minimum 12)";
  if (!std::isfinite(eyes) || eyes < 12)
    return false;
  for (const auto &p : lm)
    if (!std::isfinite(p.x) || !std::isfinite(p.y) || p.x < 4 || p.y < 4 ||
        p.x >= crop.cols - 4 || p.y >= crop.rows - 4)
      return false;
  reason = "Face pose is too oblique";
  auto middle = (lm[0] + lm[1]) * 0.5f;
  if (std::abs(lm[0].y - lm[1].y) > eyes * 0.5 ||
      std::abs(lm[2].x - middle.x) > eyes * 0.65 || lm[2].y <= middle.y ||
      (lm[3].y + lm[4].y) * 0.5 <= lm[2].y)
    return false;
  reason.clear();
  return true;
}
bool FaceEngine::acceptableAlignedFace(const cv::Mat &aligned,
                                       std::string &reason) {
  reason = "Face alignment is invalid";
  if (aligned.empty() || aligned.type() != CV_8UC3 ||
      aligned.size() != cv::Size(112, 112))
    return false;
  cv::Mat gray, lap;
  cv::cvtColor(aligned, gray, cv::COLOR_BGR2GRAY);
  cv::Laplacian(gray, lap, CV_64F);
  cv::Scalar mean, sd;
  cv::meanStdDev(lap, mean, sd);
  const double brightness = cv::mean(gray)[0];
  reason = "Face blurred, too dark or overexposed";
  if (sd[0] * sd[0] < 30 || brightness < 35 || brightness > 225)
    return false;
  reason.clear();
  return true;
}
FaceResult
FaceEngine::processFace(const cv::Mat &crop, uint64_t raw,
                        const std::set<uint64_t> &occupied,
                        const opl_interfaces::msg::TrackedHuman &human,
                        double timestamp, bool /*force*/, bool isolated) {
  ScopedTiming timing("FaceEngine.processFace");
  auto features = extractFeatures(crop, isolated);
  last_extraction_[raw] = timestamp;
  return processFeatures(std::move(features), raw, occupied, human, timestamp,
                         isolated);
}
FaceResult FaceEngine::extractFeatures(const cv::Mat &crop, bool isolated) {
  FaceResult features;
  features.quality_reason = "No usable face detected";
  if (!crop.empty() && crop.cols >= 16 && crop.rows >= 32) {
    float score = 0;
    if (scrfd_->detect(crop, features.landmarks, score) &&
        acceptableFace(crop, features.landmarks, score,
                       features.quality_reason)) {
      auto aligned = adaface_->alignFace(crop, features.landmarks);
      if (!aligned.empty()) {
        if (acceptableAlignedFace(aligned, features.quality_reason)) {
          features.face_embedding = adaface_->extract(aligned);
          features.face_quality_valid = true;
        }
      }
    }
    if (isolated)
      features.body_embedding = osnet_->extract(crop);
  }
  return features;
}
FaceResult
FaceEngine::processFeatures(FaceResult result, uint64_t raw,
                            const std::set<uint64_t> &occupied,
                            const opl_interfaces::msg::TrackedHuman &human,
                            double timestamp, bool isolated) {
  result.persistent_id = 0;
  result.name = "unknown";
  result.match_score = 0;
  result.just_scanned = false;
  result.enrollment_state.clear();
  result.enrollment_message.clear();
  result.accepted_samples = 0;
  if (!raw || !std::isfinite(timestamp) || timestamp <= 0)
    return result;
  if (last_timestamp_ > 0 && timestamp < last_timestamp_)
    resetTemporalState();
  last_timestamp_ = timestamp;
  for (auto it = verified_.begin(); it != verified_.end();) {
    if (timestamp - it->second.last_body > session_timeout_) {
      last_extraction_.erase(it->first);
      raw_registered_.erase(it->first);
      it = verified_.erase(it);
    } else
      ++it;
  }
  result.identity_state = human.observed ? "UNKNOWN" : "LOST";
  result.identity_reason =
      "No unambiguous registered face or verified track lineage";
  const bool learning_allowed =
      isolated && human.observed && std::isfinite(human.confidence) &&
      human.confidence >= minimum_learning_confidence_;
  bool face = usableFace(result) && isolated;
  if (!face)
    result.face_embedding.clear();
  float best = -1, runner = -1;
  const IdentityRecord *winner = nullptr;
  if (face)
    for (const auto &identity : reid_gallery_) {
      float score = similarity(result.face_embedding, identity.face_gallery);
      if (score > best) {
        runner = best;
        best = score;
        winner = &identity;
      } else
        runner = std::max(runner, score);
    }
  bool owned = false;
  if (winner)
    for (const auto &[owner, id] : raw_registered_)
      if (id == winner->persistent_id && owner != raw)
        owned = true;
  bool registered = winner && best >= face_threshold_ &&
                    best - runner >= 0.08f &&
                    !occupied.count(winner->persistent_id) && !owned &&
                    spatiallyCompatible(winner->persistent_id, human);
  auto known = verified_.find(raw);
  if (!enrollments_.count(raw) && known != verified_.end()) {
    auto &track = known->second;
    float self_score = -1;
    for (const auto &record : reid_gallery_)
      if (record.persistent_id == track.id && face)
        self_score = similarity(result.face_embedding, record.face_gallery);
    bool frontal = result.landmarks.empty(); // Synthetic/replayed features.
    if (result.landmarks.size() == 5) {
      const auto eyes = cv::norm(result.landmarks[0] - result.landmarks[1]);
      frontal = eyes > 0 &&
                std::abs(result.landmarks[2].x -
                         (result.landmarks[0].x + result.landmarks[1].x) * .5) <
                    eyes * .25;
    }
    const bool contradiction =
        face && ((winner && best >= face_threshold_ && best - runner >= .08f &&
                  winner->persistent_id != track.id) ||
                 (frontal && self_score < face_threshold_ - .15f));
    if (contradiction || (human.observed && !isolated) ||
        occupied.count(track.id) || !spatiallyCompatible(track.id, human)) {
      track.ambiguous = true;
      raw_registered_.erase(raw);
    }
    if (track.ambiguous) {
      if (human.observed)
        track.last_body = timestamp;
      result.identity_state = "AMBIGUOUS";
      result.identity_reason = "Contradictory face or spatial evidence; "
                               "waiting for repeated face verification";
      if (registered && human.observed) {
        if (track.candidate != winner->persistent_id ||
            timestamp - track.candidate_last_sample > session_timeout_) {
          track.candidate = winner->persistent_id;
          track.confirmations = 0;
          track.candidate_since = timestamp;
        }
        if (timestamp - track.candidate_last_sample >= .05) {
          ++track.confirmations;
          track.candidate_last_sample = timestamp;
        }
        if (track.confirmations >= 3 &&
            timestamp - track.candidate_since >= .2 - 1e-6) {
          verified_.erase(known);
          known = verified_.end();
        } else
          return result;
      } else {
        if (face) {
          track.confirmations = 0;
          track.candidate = 0;
        }
        return result;
      }
    } else if (!occupied.count(track.id)) {
      if (human.observed) {
        track.last_body = timestamp;
        track.observation = human;
        updateIdentityObservation(track.id, human);
        raw_registered_[raw] = track.id;
        if (face && winner && winner->persistent_id == track.id && registered) {
          track.last_face = timestamp;
          track.face_score = best;
        }
        if (learning_allowed && !result.body_embedding.empty() &&
            similarity(result.body_embedding, track.bodies) < .97f) {
          if (track.bodies.size() == 8)
            track.bodies.erase(track.bodies.begin() + 1);
          track.bodies.push_back(result.body_embedding);
        }
      }
      result.persistent_id = track.id;
      result.name = track.name;
      result.match_score = track.face_score;
      result.face_age_sec = timestamp - track.last_face;
      result.body_age_sec = timestamp - track.last_body;
      result.identity_state =
          !human.observed
              ? "LOST"
              : (face && registered && winner->persistent_id == track.id
                     ? "VERIFIED"
                     : "TRACKED");
      result.identity_reason = human.observed
                                   ? "Verified track lineage retained"
                                   : "Predicted track; no fresh measurement";
      return result;
    }
  }
  // Only a recently face-verified lineage can be recovered from body evidence.
  // A new registered name always requires face evidence.
  if (!registered && !face && learning_allowed && !raw_session_.count(raw) &&
      !enrollments_.count(raw) && !result.body_embedding.empty()) {
    float best_body = -1, runner_body = -1;
    uint64_t source = 0;
    for (const auto &[old_raw, track] : verified_) {
      if (track.ambiguous || observed_raw_ids_.count(old_raw) ||
          occupied.count(track.id) ||
          timestamp - track.last_body > session_timeout_ ||
          !spatiallyCompatible(track.id, human))
        continue;
      const auto &a = track.observation.bbox;
      const auto &b = human.bbox;
      const double shift = std::hypot(
          double(a.x_offset) + a.width * .5 - b.x_offset - b.width * .5,
          double(a.y_offset) + a.height * .5 - b.y_offset - b.height * .5);
      const double extent = std::hypot(a.width, a.height);
      if (extent <= 0 || shift > extent * (.5 + timestamp - track.last_body))
        continue;
      const float score = similarity(result.body_embedding, track.bodies);
      if (score > best_body) {
        runner_body = best_body;
        best_body = score;
        source = old_raw;
      } else
        runner_body = std::max(runner_body, score);
    }
    if (source && best_body >= body_threshold_ &&
        best_body - runner_body >= .08f) {
      auto track = verified_.at(source);
      verified_.erase(source);
      raw_registered_.erase(source);
      verified_[raw] = std::move(track);
      return processFeatures(std::move(result), raw, occupied, human, timestamp,
                             isolated);
    }
  }
  if (!human.observed)
    return result;
  auto pending = enrollments_.find(raw);
  if (pending != enrollments_.end()) {
    auto &scan = pending->second;
    result.persistent_id = scan.session_id;
    result.accepted_samples = scan.faces.size();
    result.enrollment_state = "scanning";
    result.name = "Scanning (" + std::to_string(result.accepted_samples) + "/" +
                  std::to_string(required_samples_) + ")";
    if (!learning_allowed) {
      result.enrollment_message =
          "Waiting for a high-confidence observed isolated body";
      return result;
    }
    if (registered) {
      if (!scan.name.empty() && scan.name != winner->name) {
        result.enrollment_state = "failed";
        result.enrollment_message = "Face already enrolled under another name";
        enrollments_.erase(raw);
        return result;
      }
      result.enrollment_state = "already_enrolled";
      result.enrollment_message =
          "Existing identity retained without overwrite";
      enrollments_.erase(raw);
    } else {
      if (winner && best >= face_threshold_) {
        result.enrollment_message =
            "Existing identity ambiguous, occupied or spatially inconsistent";
        return result;
      }
      if (!face) {
        result.enrollment_message =
            isolated ? result.quality_reason
                     : "Overlapping crops; waiting for isolation";
        return result;
      }
      for (const auto &identity : reid_gallery_)
        if (!scan.name.empty() && identity.name == scan.name) {
          result.enrollment_state = "failed";
          result.enrollment_message =
              "Name belongs to another identity; delete explicitly to replace";
          enrollments_.erase(raw);
          return result;
        }
      if (scan.last_sample >= 0 && timestamp - scan.last_sample < 0.1 - 1e-6) {
        result.enrollment_message = "Waiting for distinct observation";
        return result;
      }
      if (!scan.faces.empty() &&
          cosine(result.face_embedding, average(scan.faces)) <
              std::max(0.7f, face_threshold_)) {
        result.enrollment_state = "failed";
        result.enrollment_message =
            "Inconsistent samples; target may have changed";
        enrollments_.erase(raw);
        return result;
      }
      if (scan.first_sample < 0)
        scan.first_sample = timestamp;
      scan.last_sample = timestamp;
      if (scan.faces.size() < size_t(required_samples_))
        scan.faces.push_back(result.face_embedding);
      result.accepted_samples = scan.faces.size();
      result.enrollment_message = "Collecting consistent face samples";
      result.name = "Scanning (" + std::to_string(result.accepted_samples) +
                    "/" + std::to_string(required_samples_) + ")";
      if (scan.faces.size() < size_t(required_samples_) ||
          timestamp - scan.first_sample < required_duration_)
        return result;
      IdentityRecord record;
      record.persistent_id = next_persistent_id_;
      record.name = scan.name;
      if (record.name.empty()) {
        uint64_t suffix = record.persistent_id;
        do {
          record.name = "Person_" + std::to_string(suffix++);
        } while (std::any_of(reid_gallery_.begin(), reid_gallery_.end(),
                             [&](const auto &identity) {
                               return identity.name == record.name;
                             }));
      }
      // Keep actual observations, including a central medoid; never invent an
      // unseen view.
      const auto mean = average(scan.faces);
      auto medoid = std::max_element(scan.faces.begin(), scan.faces.end(),
                                     [&](const auto &a, const auto &b) {
                                       return cosine(a, mean) < cosine(b, mean);
                                     });
      record.face_gallery.push_back(*medoid);
      for (const auto &sample : scan.faces)
        if (record.face_gallery.size() < 4 &&
            similarity(sample, record.face_gallery) < .97f)
          record.face_gallery.push_back(sample);
      record.anchor_count = record.face_gallery.size();
      auto records = reid_gallery_;
      records.push_back(record);
      try {
        save(records, next_persistent_id_ + 1);
      } catch (const std::exception &e) {
        result.enrollment_state = "failed";
        result.enrollment_message = e.what();
        enrollments_.erase(raw);
        return result;
      }
      reid_gallery_ = std::move(records);
      ++next_persistent_id_;
      result.persistent_id = record.persistent_id;
      result.name = record.name;
      result.match_score = 1;
      result.just_scanned = true;
      result.enrollment_state = "completed";
      result.enrollment_message = "Validated identity saved durably";
      enrollments_.erase(raw);
      retireSession(raw);
      raw_registered_[raw] = record.persistent_id;
      updateIdentityObservation(record.persistent_id, human);
      rememberVerification(raw, result, human, timestamp);
      return result;
    }
  }
  if (registered) {
    result.persistent_id = winner->persistent_id;
    result.name = winner->name;
    result.match_score = best;
    retireSession(raw);
    raw_registered_[raw] = winner->persistent_id;
    updateIdentityObservation(winner->persistent_id, human);
    rememberVerification(raw, result, human, timestamp);
    return result;
  }
  // Body appearance supports anonymous continuity; it cannot assign a
  // registered name.
  raw_registered_.erase(raw);
  sessions_.erase(std::remove_if(sessions_.begin(), sessions_.end(),
                                 [&](const auto &s) {
                                   if (timestamp - s.last_seen <=
                                       session_timeout_)
                                     return false;
                                   positions_.erase(s.id);
                                   return true;
                                 }),
                  sessions_.end());
  uint64_t session_id = 0;
  auto mapped = raw_session_.find(raw);
  if (mapped != raw_session_.end() && !occupied.count(mapped->second))
    for (const auto &s : sessions_)
      if (s.id == mapped->second && spatiallyCompatible(s.id, human))
        session_id = s.id;
  if (!session_id && isolated) {
    float score = -1, second = -1;
    uint64_t candidate = 0;
    for (const auto &s : sessions_) {
      bool used = occupied.count(s.id);
      for (const auto &[owner, id] : raw_session_)
        if (owner != raw && id == s.id)
          used = true;
      if (used || !spatiallyCompatible(s.id, human))
        continue;
      float match =
          std::max(face ? similarity(result.face_embedding, s.faces) : -1,
                   similarity(result.body_embedding, s.bodies));
      if (match > score) {
        second = score;
        score = match;
        candidate = s.id;
      } else
        second = std::max(second, match);
    }
    if (score >= body_threshold_ && score - second >= 0.08f)
      session_id = candidate;
  }
  if (!session_id) {
    session_id = next_session_id_++;
    sessions_.push_back({session_id, timestamp, {}, {}});
  }
  raw_session_[raw] = session_id;
  for (auto &s : sessions_)
    if (s.id == session_id) {
      s.last_seen = timestamp;
      if (learning_allowed && s.faces.empty() && face)
        s.faces.push_back(result.face_embedding);
      if (learning_allowed && s.bodies.empty() &&
          !result.body_embedding.empty())
        s.bodies.push_back(result.body_embedding);
    }
  updateIdentityObservation(session_id, human);
  result.persistent_id = session_id;
  result.name = "unknown";
  result.match_score = 0;
  return result;
}
void FaceEngine::learnVerifiedFace(
    FaceResult &result, uint64_t raw,
    const opl_interfaces::msg::TrackedHuman &human, double timestamp,
    bool isolated) {
  if (result.identity_state == "AMBIGUOUS" || result.identity_state == "LOST") {
    adaptive_candidates_.erase(raw);
    return;
  }
  if (result.identity_state != "VERIFIED" || !human.observed || !isolated ||
      human.confidence < minimum_learning_confidence_ || !usableFace(result) ||
      result.match_score < std::max(.65f, face_threshold_ + .1f))
    return;
  auto record = std::find_if(
      reid_gallery_.begin(), reid_gallery_.end(), [&](const auto &item) {
        return item.persistent_id == result.persistent_id;
      });
  if (record == reid_gallery_.end())
    return;
  const size_t anchors =
      record->anchor_count ? record->anchor_count : record->face_gallery.size();
  float anchor_match = -1, rival = -1;
  for (size_t i = 0; i < anchors; ++i)
    anchor_match = std::max(
        anchor_match, cosine(result.face_embedding, record->face_gallery[i]));
  for (const auto &other : reid_gallery_)
    if (other.persistent_id != record->persistent_id)
      rival = std::max(rival,
                       similarity(result.face_embedding, other.face_gallery));
  if (anchor_match < std::max(.65f, face_threshold_ + .1f) ||
      result.match_score - rival < .15f ||
      similarity(result.face_embedding, record->face_gallery) >= .97f) {
    adaptive_candidates_.erase(raw);
    return;
  }
  auto &candidate = adaptive_candidates_[raw];
  if (candidate.embedding.empty() ||
      timestamp - candidate.last > session_timeout_ ||
      cosine(candidate.embedding, result.face_embedding) < .97f) {
    candidate = {result.face_embedding, timestamp, timestamp, 1};
    return;
  }
  if (timestamp - candidate.last < .1)
    return;
  candidate.last = timestamp;
  ++candidate.count;
  if (candidate.count < 3 || timestamp - candidate.first < .5 ||
      timestamp - last_gallery_write_ < 5)
    return;
  auto records = reid_gallery_;
  auto &updated = records[std::distance(reid_gallery_.begin(), record)];
  if (updated.face_gallery.size() < 12)
    updated.face_gallery.push_back(result.face_embedding);
  else {
    // Preserve all anchors and every distinct adaptive view; replace only a
    // redundant adaptive pair.
    float redundancy = .95f;
    size_t replace = updated.face_gallery.size();
    for (size_t i = anchors; i < updated.face_gallery.size(); ++i)
      for (size_t j = 0; j < updated.face_gallery.size(); ++j)
        if (i != j) {
          float score =
              cosine(updated.face_gallery[i], updated.face_gallery[j]);
          if (score > redundancy) {
            redundancy = score;
            replace = i;
          }
        }
    if (replace == updated.face_gallery.size()) {
      adaptive_candidates_.erase(raw);
      return;
    }
    updated.face_gallery[replace] = result.face_embedding;
  }
  updated.anchor_count = anchors;
  try {
    save(records, next_persistent_id_);
    reid_gallery_ = std::move(records);
    result.identity_reason += "; distinct verified face saved durably";
  } catch (const std::exception &error) {
    result.identity_reason +=
        std::string("; adaptive memory save failed: ") + error.what();
  }
  last_gallery_write_ = timestamp;
  adaptive_candidates_.erase(raw);
}
void FaceEngine::rememberVerification(
    uint64_t raw, const FaceResult &result,
    const opl_interfaces::msg::TrackedHuman &human, double timestamp) {
  VerifiedTrack track;
  track.id = result.persistent_id;
  track.name = result.name;
  track.last_face = track.last_body = timestamp;
  track.face_score = result.match_score;
  track.observation = human;
  if (!result.body_embedding.empty())
    track.bodies.push_back(result.body_embedding);
  verified_[raw] = std::move(track);
}
bool FaceEngine::needsFeatures(uint64_t raw, double timestamp,
                               double interval) const {
  auto it = last_extraction_.find(raw);
  return enrollments_.count(raw) || it == last_extraction_.end() ||
         timestamp - it->second >= interval;
}
std::vector<FaceResult> FaceEngine::processFrameFeatures(
    const std::vector<opl_interfaces::msg::TrackedHuman> &humans,
    std::vector<FaceResult> features, double timestamp) {
  if (humans.size() != features.size())
    throw std::invalid_argument("Feature/frame size mismatch");
  std::set<uint64_t> active, blocked, assigned;
  for (const auto &h : humans)
    if (h.observed)
      active.insert(h.raw_track_id);
  cleanupStaleRawTracks(active);
  // Competing usable faces cannot acquire one identity by callback ordering.
  std::unordered_map<uint64_t, std::vector<std::pair<size_t, float>>> claims;
  for (size_t i = 0; i < humans.size(); ++i) {
    if (!humans[i].observed || !usableFace(features[i]))
      continue;
    float best = -1, second = -1;
    uint64_t id = 0;
    for (const auto &record : reid_gallery_) {
      float score = similarity(features[i].face_embedding, record.face_gallery);
      if (score > best) {
        second = best;
        best = score;
        id = record.persistent_id;
      } else
        second = std::max(second, score);
    }
    if (best >= face_threshold_ && best - second >= .08f)
      claims[id].push_back({i, best});
  }
  std::set<size_t> rejected;
  for (auto &[id, candidates] : claims)
    if (candidates.size() > 1) {
      std::sort(
          candidates.begin(), candidates.end(),
          [](const auto &a, const auto &b) { return a.second > b.second; });
      if (candidates[0].second - candidates[1].second < .08f)
        blocked.insert(id);
      else
        for (size_t j = 1; j < candidates.size(); ++j)
          rejected.insert(candidates[j].first);
    }
  std::vector<FaceResult> results(humans.size());
  for (bool observed : {true, false})
    for (size_t i = 0; i < humans.size(); ++i) {
      if (humans[i].observed != observed)
        continue;
      const uint64_t raw = humans[i].raw_track_id;
      bool isolated = true;
      const auto &b = humans[i].bbox;
      cv::Rect2f box(b.x_offset, b.y_offset, b.width, b.height);
      for (size_t j = 0; j < humans.size(); ++j)
        if (i != j && humans[j].observed) {
          const auto &o = humans[j].bbox;
          cv::Rect2f other(o.x_offset, o.y_offset, o.width, o.height);
          if ((box & other).area() / std::max(1.f, box.area() + other.area() -
                                                       (box & other).area()) >
              .25f)
            isolated = false;
        }
      auto occupied = assigned;
      occupied.insert(blocked.begin(), blocked.end());
      auto lineage = verified_.find(raw);
      if (lineage != verified_.end() && blocked.count(lineage->second.id)) {
        lineage->second.ambiguous = true;
        raw_registered_.erase(raw);
      }
      if (rejected.count(i))
        for (const auto &[id, candidates] : claims)
          for (const auto &candidate : candidates)
            if (candidate.first == i)
              occupied.insert(id);
      if (!features[i].quality_reason.empty() ||
          !features[i].face_embedding.empty() ||
          !features[i].body_embedding.empty())
        last_extraction_[raw] = timestamp;
      results[i] = processFeatures(std::move(features[i]), raw, occupied,
                                   humans[i], timestamp, isolated);
      for (const auto &[id, candidates] : claims)
        if (blocked.count(id))
          for (const auto &candidate : candidates)
            if (candidate.first == i) {
              results[i].identity_state = "AMBIGUOUS";
              results[i].identity_reason = "Competing simultaneous face claims";
            }
      if (results[i].persistent_id)
        assigned.insert(results[i].persistent_id);
      if (results[i].persistent_id &&
          results[i].persistent_id < 1000000000ULL &&
          results[i].identity_state == "UNKNOWN") {
        results[i].identity_state = "VERIFIED";
        results[i].face_age_sec = results[i].body_age_sec = 0;
      }
      learnVerifiedFace(results[i], raw, humans[i], timestamp, isolated);
    }
  return results;
}
std::vector<std::pair<uint64_t, FaceResult>>
FaceEngine::lostIdentities(double timestamp) const {
  std::vector<std::pair<uint64_t, FaceResult>> result;
  for (const auto &[raw, track] : verified_)
    if (!observed_raw_ids_.count(raw) &&
        timestamp - track.last_body <= session_timeout_) {
      FaceResult status;
      status.persistent_id = track.id;
      status.name = track.name;
      status.match_score = track.face_score;
      status.identity_state = track.ambiguous ? "AMBIGUOUS" : "LOST";
      status.identity_reason = "No observed body; recovery window active";
      status.face_age_sec = timestamp - track.last_face;
      status.body_age_sec = timestamp - track.last_body;
      result.push_back({raw, std::move(status)});
    }
  return result;
}
} // namespace opl_human_vision
