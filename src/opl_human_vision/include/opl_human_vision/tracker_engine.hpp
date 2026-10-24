#ifndef OPL_HUMAN_VISION__TRACKER_ENGINE_HPP_
#define OPL_HUMAN_VISION__TRACKER_ENGINE_HPP_

#include <memory>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "opl_interfaces/msg/tracked_human.hpp"

#include "opl_human_vision/osnet_extractor.hpp"
#include "opl_human_vision/botsort_kalman.hpp"

namespace opl_human_vision {


enum class TrackState {
  Tentative,
  Confirmed,
  Lost
};


struct Track {
  int id;
  cv::Rect2f bbox;
  float confidence;

  int age{0};

  // Total successful detector associations over this track's lifetime.
  int hits{0};

  // Consecutive successful detector associations.
  int hit_streak{0};

  int time_since_update{0};
  bool association_ambiguous{false};

  TrackState state{TrackState::Tentative};

  std::string name{"unknown"};

  BoTSortKalman kalman;
  std::vector<float> smooth_feature;
  double last_prediction_time_sec{0.0};

};


class TrackerEngine {
public:
  TrackerEngine(
    int max_age = 30,
    int min_hits = 3,
    float match_threshold = 0.8f,
    int max_publish_age = 1,

    const std::string& osnet_model_path = "",
    bool appearance_reid_enabled = false,
    float appearance_match_threshold = 0.85f,
    int appearance_reid_max_age = 10,
    float track_high_threshold = 0.65f,
    float track_low_threshold = 0.10f,
    float new_track_threshold = 0.70f,
    float proximity_threshold = 0.50f
  );

  ~TrackerEngine() = default;

  std::vector<Track> update(
    const std::vector<opl_interfaces::msg::TrackedHuman>& detections,
    const cv::Mat& frame,
    double timestamp_sec
  );

  bool appearanceReidReady() const {
    return
      appearance_reid_enabled_ &&
      appearance_reid_ready_;
  }

private:
  cv::Mat estimateCameraMotion(
    const cv::Mat& current_frame
  );

  float computeIoU(
    const cv::Rect2f& boxA,
    const cv::Rect2f& boxB
  ) const;

  cv::Rect2f detectionBox(
    const opl_interfaces::msg::TrackedHuman& detection
  ) const;

  std::vector<float> extractAppearance(
    const cv::Mat& frame,
    const cv::Rect2f& bbox
  );

  float cosineSimilarity(
    const std::vector<float>& a,
    const std::vector<float>& b
  ) const;

  void updateFeature(Track& track, const std::vector<float>& feature);

  void applyDetectionToTrack(
    Track& track,
    const opl_interfaces::msg::TrackedHuman& detection,
    const std::vector<float>& feature
  );

  int max_age_;
  int min_hits_;
  float match_threshold_;
  int max_publish_age_;


  // --------------------------------------------------------------------------
  // Appearance-assisted Lost-track recovery.
  // --------------------------------------------------------------------------

  bool appearance_reid_enabled_{false};
  bool appearance_reid_ready_{false};

  float appearance_match_threshold_{0.85f};

  int appearance_reid_max_age_{10};
  float track_high_threshold_;
  float track_low_threshold_;
  float new_track_threshold_;
  float proximity_threshold_;
  std::unique_ptr<OsnetExtractor> appearance_osnet_;
  double last_timestamp_sec_{0.0};

  int next_id_{1};

  cv::Mat prev_frame_gray_;

  std::vector<Track> active_tracks_;
};

}  // namespace opl_human_vision

#endif  // OPL_HUMAN_VISION__TRACKER_ENGINE_HPP_