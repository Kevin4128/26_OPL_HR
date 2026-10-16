#pragma once

#include "opl_human_vision/adaface_extractor.hpp"
#include "opl_human_vision/identity_store.hpp"
#include "opl_human_vision/osnet_extractor.hpp"
#include "opl_human_vision/scrfd_detector.hpp"
#include "opl_interfaces/msg/tracked_human.hpp"

#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace opl_human_vision {

struct FaceResult {
  uint64_t persistent_id{0};
  std::string name{"unknown"};
  float match_score{0};
  bool just_scanned{false};
  std::string identity_state{"UNKNOWN"}, identity_reason;
  double face_age_sec{-1}, body_age_sec{-1};
  bool face_quality_valid{false};
  std::string quality_reason;
  std::string enrollment_state;
  std::string enrollment_message;
  uint32_t accepted_samples{0};
  std::vector<cv::Point2f> landmarks;
  std::vector<float> face_embedding, body_embedding;
};

class FaceEngine {
public:
  FaceEngine(const std::string &scrfd_path, const std::string &adaface_path,
             const std::string &osnet_path, float face_match_threshold = 0.55f,
             float body_match_threshold = 0.85f, int min_confirm_frames = 20,
             float min_confirm_duration_sec = 2.5f,
             double session_reid_timeout_sec = 2.0,
             double spatial_gate_max_age_sec = 2.0,
             float spatial_gate_base_m = 0.75f,
             float spatial_gate_max_speed_mps = 2.5f,
             float depth_gate_base_m = 0.75f,
             float minimum_learning_confidence = 0.45f);
  bool loadModels();
  void configureStore(const std::string &directory, const std::string &model);
  bool deleteIdentity(uint64_t identity, std::string &message);
  void authorizeEnrollment(uint64_t raw_track_id, const std::string &name = "");
  void cancelEnrollment(uint64_t raw_track_id);
  bool enrollmentAuthorized(uint64_t raw_track_id) const;
  void resetTemporalState();
  void cleanupStaleRawTracks(const std::set<uint64_t> &active_raw_ids);
  void
  updateIdentityObservation(uint64_t identity,
                            const opl_interfaces::msg::TrackedHuman &human);
  FaceResult processFace(const cv::Mat &crop, uint64_t raw_track_id,
                         const std::set<uint64_t> &occupied,
                         const opl_interfaces::msg::TrackedHuman &human,
                         double timestamp_sec, bool force_deep_reid = false,
                         bool isolated = true);
  // Shared decision path for live extraction and deterministic/replayed
  // features.
  FaceResult processFeatures(FaceResult features, uint64_t raw_track_id,
                             const std::set<uint64_t> &occupied,
                             const opl_interfaces::msg::TrackedHuman &human,
                             double timestamp_sec, bool isolated = true);
  FaceResult extractFeatures(const cv::Mat &crop, bool isolated = true);
  bool needsFeatures(uint64_t raw, double timestamp, double interval) const;
  std::vector<FaceResult> processFrameFeatures(
      const std::vector<opl_interfaces::msg::TrackedHuman> &humans,
      std::vector<FaceResult> features, double timestamp);
  std::vector<std::pair<uint64_t, FaceResult>>
  lostIdentities(double timestamp) const;
  static bool acceptableFace(const cv::Mat &crop,
                             const std::vector<cv::Point2f> &landmarks,
                             float score, std::string &reason);
  static bool acceptableAlignedFace(const cv::Mat &aligned,
                                    std::string &reason);
  size_t registeredCount() const { return reid_gallery_.size(); }

private:
  struct AdaptiveCandidate {
    std::vector<float> embedding;
    double first{0}, last{0};
    int count{0};
  };
  std::unordered_map<uint64_t, AdaptiveCandidate> adaptive_candidates_;
  double last_gallery_write_{-5};
  void learnVerifiedFace(FaceResult &result, uint64_t raw,
                         const opl_interfaces::msg::TrackedHuman &human,
                         double timestamp, bool isolated);
  struct VerifiedTrack {
    uint64_t id{0}, candidate{0};
    std::string name;
    double last_face{0}, last_body{0}, candidate_since{0},
        candidate_last_sample{0};
    float face_score{0};
    int confirmations{0};
    bool ambiguous{false};
    opl_interfaces::msg::TrackedHuman observation;
    std::vector<std::vector<float>> bodies;
  };
  std::unordered_map<uint64_t, VerifiedTrack> verified_;
  std::unordered_map<uint64_t, double> last_extraction_;
  std::set<uint64_t> observed_raw_ids_;
  void rememberVerification(uint64_t raw, const FaceResult &result,
                            const opl_interfaces::msg::TrackedHuman &human,
                            double timestamp);
  struct SessionIdentity {
    uint64_t id{0};
    double last_seen{0};
    std::vector<std::vector<float>> faces, bodies;
  };
  struct PendingEnrollment {
    std::string name;
    uint64_t session_id{0};
    double first_sample{-1}, last_sample{-1};
    std::vector<std::vector<float>> faces;
  };
  std::unique_ptr<ScrfdDetector> scrfd_;
  std::unique_ptr<AdaFaceExtractor> adaface_;
  std::unique_ptr<OsnetExtractor> osnet_;
  std::unique_ptr<IdentityStore> store_;
  std::vector<IdentityRecord> reid_gallery_;
  std::vector<SessionIdentity> sessions_;
  std::unordered_map<uint64_t, uint64_t> raw_registered_, raw_session_;
  std::unordered_map<uint64_t, PendingEnrollment> enrollments_;
  std::unordered_map<uint64_t, geometry_msgs::msg::PointStamped> positions_;
  uint64_t next_persistent_id_{1}, next_session_id_{1000000000ULL};
  float face_threshold_, body_threshold_, minimum_learning_confidence_;
  int required_samples_;
  double required_duration_, session_timeout_, spatial_max_age_, spatial_base_,
      spatial_speed_;
  double last_timestamp_{0};
  bool
  spatiallyCompatible(uint64_t identity,
                      const opl_interfaces::msg::TrackedHuman &human) const;
  void retireSession(uint64_t raw);
  void save(const std::vector<IdentityRecord> &records, uint64_t next);
};

inline void drawFaceDebugOverlay(cv::Mat &frame, const cv::Rect &person_box,
                                 const FaceResult &result) {
  if (person_box.width <= 0 || person_box.height <= 0) {
    return;
  }

  cv::rectangle(frame, person_box, cv::Scalar(0, 255, 0), 2);

  for (const auto &pt : result.landmarks) {
    cv::Point2f abs_pt(person_box.x + pt.x, person_box.y + pt.y);

    cv::circle(frame, abs_pt, 2, cv::Scalar(0, 0, 255), -1);
  }

  std::string label =
      result.name + " (" +
      std::to_string(static_cast<int>(result.match_score * 100)) + "%)";

  cv::putText(frame, label,
              cv::Point(person_box.x, std::max(0, person_box.y - 10)),
              cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 2);
}

} // namespace opl_human_vision