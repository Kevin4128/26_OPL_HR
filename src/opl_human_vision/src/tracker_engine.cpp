#include "opl_human_vision/tracker_engine.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>


namespace {

// -----------------------------------------------------------------------------
// Hungarian minimum-cost assignment solver.
//
// Input:
//   cost[track][detection]
//
// Output:
//   assignment[track] = detection
//
// -1 means unassigned.
// -----------------------------------------------------------------------------

std::vector<int> solveHungarian(
  const std::vector<std::vector<float>>& input_cost)
{
  const std::size_t original_rows =
    input_cost.size();

  if (original_rows == 0) {
    return {};
  }


  const std::size_t original_cols =
    input_cost.front().size();


  std::vector<int> result(
    original_rows,
    -1
  );


  if (original_cols == 0) {
    return result;
  }


  const bool transpose =
    original_rows >
    original_cols;


  std::vector<std::vector<double>> cost;


  if (!transpose) {

    cost.resize(
      original_rows,
      std::vector<double>(
        original_cols,
        0.0
      )
    );


    for (
      std::size_t i = 0;
      i < original_rows;
      ++i)
    {
      for (
        std::size_t j = 0;
        j < original_cols;
        ++j)
      {
        cost[i][j] =
          static_cast<double>(
            input_cost[i][j]
          );
      }
    }

  } else {

    cost.resize(
      original_cols,
      std::vector<double>(
        original_rows,
        0.0
      )
    );


    for (
      std::size_t i = 0;
      i < original_rows;
      ++i)
    {
      for (
        std::size_t j = 0;
        j < original_cols;
        ++j)
      {
        cost[j][i] =
          static_cast<double>(
            input_cost[i][j]
          );
      }
    }
  }


  const int n =
    static_cast<int>(
      cost.size()
    );

  const int m =
    static_cast<int>(
      cost.front().size()
    );


  std::vector<double> u(
    n + 1,
    0.0
  );

  std::vector<double> v(
    m + 1,
    0.0
  );

  std::vector<int> p(
    m + 1,
    0
  );

  std::vector<int> way(
    m + 1,
    0
  );


  for (int i = 1; i <= n; ++i) {

    p[0] =
      i;


    int j0 =
      0;


    std::vector<double> minv(
      m + 1,
      std::numeric_limits<double>::infinity()
    );

    std::vector<bool> used(
      m + 1,
      false
    );


    do {

      used[j0] =
        true;


      const int i0 =
        p[j0];


      double delta =
        std::numeric_limits<double>::infinity();


      int j1 =
        0;


      for (int j = 1; j <= m; ++j) {

        if (used[j]) {
          continue;
        }


        const double current_cost =
          cost[i0 - 1][j - 1] -
          u[i0] -
          v[j];


        if (
          current_cost <
          minv[j])
        {
          minv[j] =
            current_cost;

          way[j] =
            j0;
        }


        if (
          minv[j] <
          delta)
        {
          delta =
            minv[j];

          j1 =
            j;
        }
      }


      for (int j = 0; j <= m; ++j) {

        if (used[j]) {

          u[p[j]] +=
            delta;

          v[j] -=
            delta;

        } else {

          minv[j] -=
            delta;
        }
      }


      j0 =
        j1;

    } while (
      p[j0] != 0
    );


    do {

      const int j1 =
        way[j0];


      p[j0] =
        p[j1];


      j0 =
        j1;

    } while (
      j0 != 0
    );
  }


  if (!transpose) {

    for (int j = 1; j <= m; ++j) {

      if (p[j] == 0) {
        continue;
      }


      const int track_index =
        p[j] - 1;

      const int detection_index =
        j - 1;


      if (
        track_index >= 0 &&
        track_index <
          static_cast<int>(
            original_rows
          ))
      {
        result[track_index] =
          detection_index;
      }
    }

  } else {

    for (int j = 1; j <= m; ++j) {

      if (p[j] == 0) {
        continue;
      }


      const int original_detection =
        p[j] - 1;

      const int original_track =
        j - 1;


      if (
        original_track >= 0 &&
        original_track <
          static_cast<int>(
            original_rows
          ) &&
        original_detection >= 0 &&
        original_detection <
          static_cast<int>(
            original_cols
          ))
      {
        result[original_track] =
          original_detection;
      }
    }
  }


  return result;
}

// Dummy columns make leaving a track unmatched part of the optimization.
std::vector<int> solveAssignment(const std::vector<std::vector<float>>& costs, float limit) {
  if (costs.empty()) return {};
  const size_t columns = costs.front().size();
  auto padded = costs;
  for (auto& row : padded) {
    for (auto& cost : row) if (cost > limit) cost = 1000000.0f;
    row.resize(columns + costs.size(), limit + 1e-6f);
  }
  auto result = solveHungarian(padded);
  for (size_t i = 0; i < result.size(); ++i)
    if (result[i] < 0 || size_t(result[i]) >= columns || costs[i][result[i]] > limit)
      result[i] = -1;
  return result;
}

// Preserve the identity layer's ambiguity signal across all association stages.
std::vector<bool> ambiguousAssignments(const std::vector<std::vector<float>>& costs,
                                      const std::vector<int>& assignment, float limit) {
  std::vector<bool> ambiguous(costs.size(), false);
  auto score = [&](const std::vector<int>& match) {
    double sum = 0;
    for (size_t row = 0; row < match.size(); ++row)
      sum += match[row] < 0 ? limit : costs[row][match[row]];
    return sum;
  };
  const double best = score(assignment);
  for (size_t row = 0; row < assignment.size(); ++row) {
    const int col = assignment[row];
    if (col < 0) continue;
    auto alternative_costs = costs;
    alternative_costs[row][col] = 1000000.0f;
    const auto alternative = solveAssignment(alternative_costs, limit);
    size_t changed = 0;
    for (size_t i = 0; i < assignment.size(); ++i)
      if (assignment[i] != alternative[i]) ++changed;
    if (score(alternative) > best + 0.05 * changed + 1e-6) continue;
    for (size_t i = 0; i < assignment.size(); ++i)
      if (assignment[i] != alternative[i]) ambiguous[i] = true;
  }
  return ambiguous;
}

}  // namespace


namespace opl_human_vision {


TrackerEngine::TrackerEngine(
  int max_age, int min_hits, float match_threshold, int max_publish_age,
  const std::string& osnet_model_path, bool appearance_reid_enabled,
  float appearance_match_threshold, int appearance_reid_max_age,
  float track_high_threshold, float track_low_threshold,
  float new_track_threshold, float proximity_threshold)
: max_age_(max_age), min_hits_(min_hits), match_threshold_(match_threshold),
  max_publish_age_(max_publish_age), appearance_reid_enabled_(appearance_reid_enabled),
  appearance_match_threshold_(appearance_match_threshold),
  appearance_reid_max_age_(std::min(appearance_reid_max_age, max_age)),
  track_high_threshold_(track_high_threshold), track_low_threshold_(track_low_threshold),
  new_track_threshold_(new_track_threshold), proximity_threshold_(proximity_threshold)
{
  for (float value : {match_threshold, appearance_match_threshold, track_high_threshold,
                      track_low_threshold, new_track_threshold, proximity_threshold})
    if (!std::isfinite(value) || value < 0 || value > 1)
      throw std::invalid_argument("BoT-SORT thresholds must be finite and within [0, 1]");
  if (track_low_threshold >= track_high_threshold || new_track_threshold < track_high_threshold ||
      max_age < 1 || min_hits < 1 || max_publish_age < 0 || max_publish_age > max_age ||
      appearance_reid_max_age < 0)
    throw std::invalid_argument("Invalid BoT-SORT threshold ordering or track lifetime");
  if (appearance_reid_enabled_ && !osnet_model_path.empty()) {
    appearance_osnet_ = std::make_unique<OsnetExtractor>(osnet_model_path);
    appearance_reid_ready_ = appearance_osnet_->init();
  }
}

// -----------------------------------------------------------------------------
// Convert detector message bbox to OpenCV rectangle.
// -----------------------------------------------------------------------------

cv::Rect2f TrackerEngine::detectionBox(
  const opl_interfaces::msg::TrackedHuman& detection) const
{
  return cv::Rect2f(
    static_cast<float>(
      detection.bbox.x_offset
    ),
    static_cast<float>(
      detection.bbox.y_offset
    ),
    static_cast<float>(
      detection.bbox.width
    ),
    static_cast<float>(
      detection.bbox.height
    )
  );
}


// -----------------------------------------------------------------------------
// Global camera-motion estimation.
// -----------------------------------------------------------------------------

cv::Mat TrackerEngine::estimateCameraMotion(
  const cv::Mat& current_frame)
{
  cv::Mat transform =
    cv::Mat::eye(
      2,
      3,
      CV_64F
    );


  if (current_frame.empty()) {
    prev_frame_gray_.release();
    return transform;
  }


  cv::Mat gray;
  cv::Mat small_gray;


  if (
    current_frame.channels() ==
    3)
  {
    cv::cvtColor(
      current_frame,
      gray,
      cv::COLOR_BGR2GRAY
    );

  } else {

    gray =
      current_frame;
  }


  float scale =
    1.0f;


  if (gray.cols > 320) {

    scale =
      320.0f /
      static_cast<float>(
        gray.cols
      );


    cv::resize(
      gray,
      small_gray,
      cv::Size(),
      scale,
      scale,
      cv::INTER_NEAREST
    );

  } else {

    small_gray =
      gray;
  }


  if (
    prev_frame_gray_.empty() || prev_frame_gray_.size() != small_gray.size())
  {
    prev_frame_gray_ =
      small_gray.clone();

    return transform;
  }


  cv::Mat mask =
    cv::Mat::ones(
      small_gray.size(),
      CV_8UC1
    ) * 255;


  for (
    const auto& track :
    active_tracks_)
  {
    const cv::Rect2f scaled_box(
      track.bbox.x * scale,
      track.bbox.y * scale,
      track.bbox.width * scale,
      track.bbox.height * scale
    );


    const int x1 =
      std::max(
        0,
        static_cast<int>(
          std::floor(
            scaled_box.x
          )
        )
      );


    const int y1 =
      std::max(
        0,
        static_cast<int>(
          std::floor(
            scaled_box.y
          )
        )
      );


    const int x2 =
      std::min(
        small_gray.cols,
        static_cast<int>(
          std::ceil(
            scaled_box.x +
            scaled_box.width
          )
        )
      );


    const int y2 =
      std::min(
        small_gray.rows,
        static_cast<int>(
          std::ceil(
            scaled_box.y +
            scaled_box.height
          )
        )
      );


    if (
      x2 > x1 &&
      y2 > y1)
    {
      mask(
        cv::Rect(
          x1,
          y1,
          x2 - x1,
          y2 - y1
        )
      ).setTo(0);
    }
  }


  std::vector<cv::Point2f>
    prev_pts;

  std::vector<cv::Point2f>
    curr_pts;


  cv::goodFeaturesToTrack(
    prev_frame_gray_,
    prev_pts,
    200,
    0.01,
    10,
    mask
  );


  if (
    prev_pts.size() >= 10)
  {
    std::vector<uchar>
      status;

    std::vector<float>
      err;


    cv::calcOpticalFlowPyrLK(
      prev_frame_gray_,
      small_gray,
      prev_pts,
      curr_pts,
      status,
      err
    );


    std::vector<cv::Point2f>
      valid_prev;

    std::vector<cv::Point2f>
      valid_curr;


    for (
      std::size_t i = 0;
      i < status.size();
      ++i)
    {
      if (status[i]) {

        valid_prev.push_back(
          prev_pts[i]
        );

        valid_curr.push_back(
          curr_pts[i]
        );
      }
    }


    if (
      valid_prev.size() >= 6)
    {
      cv::Mat small_transform =
        cv::estimateAffinePartial2D(
          valid_prev,
          valid_curr
        );


      if (
        !small_transform.empty() &&
        cv::checkRange(
          small_transform
        ))
      {
        small_transform.at<double>(
          0,
          2
        ) /= scale;


        small_transform.at<double>(
          1,
          2
        ) /= scale;


        transform =
          small_transform;
      }
    }
  }


  prev_frame_gray_ =
    small_gray.clone();


  return transform;
}


// -----------------------------------------------------------------------------
// IoU.
// -----------------------------------------------------------------------------

float TrackerEngine::computeIoU(
  const cv::Rect2f& boxA,
  const cv::Rect2f& boxB) const
{
  const float x1 =
    std::max(
      boxA.x,
      boxB.x
    );

  const float y1 =
    std::max(
      boxA.y,
      boxB.y
    );

  const float x2 =
    std::min(
      boxA.x +
        boxA.width,
      boxB.x +
        boxB.width
    );

  const float y2 =
    std::min(
      boxA.y +
        boxA.height,
      boxB.y +
        boxB.height
    );


  const float inter_area =
    std::max(
      0.0f,
      x2 - x1
    ) *
    std::max(
      0.0f,
      y2 - y1
    );


  const float box_a_area =
    boxA.width *
    boxA.height;

  const float box_b_area =
    boxB.width *
    boxB.height;


  return
    inter_area /
    (
      box_a_area +
      box_b_area -
      inter_area +
      1e-6f
    );
}


// -----------------------------------------------------------------------------
// Extract OSNet body embedding from a bbox.
// -----------------------------------------------------------------------------

std::vector<float> TrackerEngine::extractAppearance(
  const cv::Mat& frame,
  const cv::Rect2f& bbox)
{
  if (
    !appearanceReidReady() ||
    frame.empty())
  {
    return {};
  }


  const int x1 =
    std::max(
      0,
      std::min(
        static_cast<int>(
          std::floor(
            bbox.x
          )
        ),
        frame.cols
      )
    );


  const int y1 =
    std::max(
      0,
      std::min(
        static_cast<int>(
          std::floor(
            bbox.y
          )
        ),
        frame.rows
      )
    );


  const int x2 =
    std::max(
      0,
      std::min(
        static_cast<int>(
          std::ceil(
            bbox.x +
            bbox.width
          )
        ),
        frame.cols
      )
    );


  const int y2 =
    std::max(
      0,
      std::min(
        static_cast<int>(
          std::ceil(
            bbox.y +
            bbox.height
          )
        ),
        frame.rows
      )
    );


  const int width =
    x2 - x1;

  const int height =
    y2 - y1;


  if (
    width < 16 ||
    height < 32)
  {
    return {};
  }


  const cv::Mat person_crop =
    frame(
      cv::Rect(
        x1,
        y1,
        width,
        height
      )
    ).clone();


  return
    appearance_osnet_->extract(
      person_crop
    );
}


// -----------------------------------------------------------------------------
// Safe cosine similarity.
// -----------------------------------------------------------------------------

float TrackerEngine::cosineSimilarity(
  const std::vector<float>& a,
  const std::vector<float>& b) const
{
  if (
    a.empty() ||
    b.empty() ||
    a.size() != b.size())
  {
    return -1.0f;
  }


  double dot =
    0.0;

  double norm_a =
    0.0;

  double norm_b =
    0.0;


  for (
    std::size_t i = 0;
    i < a.size();
    ++i)
  {
    dot +=
      static_cast<double>(
        a[i]
      ) *
      static_cast<double>(
        b[i]
      );


    norm_a +=
      static_cast<double>(
        a[i]
      ) *
      static_cast<double>(
        a[i]
      );


    norm_b +=
      static_cast<double>(
        b[i]
      ) *
      static_cast<double>(
        b[i]
      );
  }


  if (
    !std::isfinite(dot) || !std::isfinite(norm_a) || !std::isfinite(norm_b) ||
    norm_a < 1e-12 ||
    norm_b < 1e-12)
  {
    return -1.0f;
  }


  const double similarity =
    dot /
    (
      std::sqrt(norm_a) *
      std::sqrt(norm_b)
    );


  return
    static_cast<float>(
      std::max(
        -1.0,
        std::min(
          1.0,
          similarity
        )
      )
    );
}


void TrackerEngine::updateFeature(Track& track, const std::vector<float>& feature) {
  if (feature.empty()) return;
  double norm = 0;
  for (float value : feature) {
    if (!std::isfinite(value)) return;
    norm += double(value) * value;
  }
  if (norm < 1e-12) return;
  std::vector<float> normalized = feature;
  for (auto& value : normalized) value /= std::sqrt(norm);
  if (track.smooth_feature.size() != normalized.size()) {
    track.smooth_feature = normalized;
    return;
  }
  norm = 0;
  for (size_t i = 0; i < normalized.size(); ++i) {
    track.smooth_feature[i] = 0.9f * track.smooth_feature[i] + 0.1f * normalized[i];
    norm += double(track.smooth_feature[i]) * track.smooth_feature[i];
  }
  if (norm > 1e-12)
    for (auto& value : track.smooth_feature) value /= std::sqrt(norm);
}

void TrackerEngine::applyDetectionToTrack(
  Track& track, const opl_interfaces::msg::TrackedHuman& detection,
  const std::vector<float>& feature)
{
  track.kalman.correct(detectionBox(detection));
  track.bbox = track.kalman.box();
  track.confidence = detection.confidence;
  ++track.hits;
  ++track.hit_streak;
  track.time_since_update = 0;
  if (track.state == TrackState::Lost || track.hit_streak >= min_hits_)
    track.state = TrackState::Confirmed;
  if (!track.association_ambiguous) updateFeature(track, feature);
}

std::vector<Track> TrackerEngine::update(
  const std::vector<opl_interfaces::msg::TrackedHuman>& detections,
  const cv::Mat& frame, double timestamp_sec)
{
  const bool timestamp_valid = std::isfinite(timestamp_sec) && timestamp_sec > 0;
  if (timestamp_valid && last_timestamp_sec_ > timestamp_sec) {
    active_tracks_.clear();
    prev_frame_gray_.release();
  }
  if (timestamp_valid) last_timestamp_sec_ = timestamp_sec;

  const cv::Mat motion = estimateCameraMotion(frame);
  cv::Matx23d warp(1, 0, 0, 0, 1, 0);
  if (motion.rows == 2 && motion.cols == 3 && cv::checkRange(motion))
    for (int row = 0; row < 2; ++row)
      for (int col = 0; col < 3; ++col) warp(row, col) = motion.at<double>(row, col);

  std::vector<int> pool, tentative;
  for (size_t i = 0; i < active_tracks_.size(); ++i) {
    auto& track = active_tracks_[i];
    ++track.age;
    ++track.time_since_update;
    track.association_ambiguous = false;
    double dt = 1;
    if (timestamp_valid && track.last_prediction_time_sec > 0)
      dt = std::clamp((timestamp_sec - track.last_prediction_time_sec) * 30, 0.0, 6.0);
    if (timestamp_valid) track.last_prediction_time_sec = timestamp_sec;
    // Upstream predicts tracked/lost tracks, then applies GMC to both pools.
    if (track.state != TrackState::Tentative)
      track.kalman.predict(dt, track.state == TrackState::Confirmed);
    track.kalman.warp(warp);
    track.bbox = track.kalman.box();
    if (track.time_since_update > max_age_) continue;
    (track.state == TrackState::Tentative ? tentative : pool).push_back(int(i));
  }

  std::vector<int> high, low;
  std::vector<bool> overlapping(detections.size(), false);
  std::vector<std::vector<float>> features(detections.size());
  for (size_t i = 0; i < detections.size(); ++i) {
    const auto& detection = detections[i];
    if (!std::isfinite(detection.confidence) || detection.confidence > 1 ||
        !detection.bbox.width || !detection.bbox.height ||
        detection.confidence <= track_low_threshold_) continue;
    (detection.confidence >= track_high_threshold_ ? high : low).push_back(int(i));
  }
  // Keep occluding crops out of both the query and the persistent appearance EMA.
  for (size_t i = 0; i < detections.size(); ++i) {
    if (!detections[i].bbox.width || !detections[i].bbox.height ||
        !std::isfinite(detections[i].confidence) ||
        detections[i].confidence <= track_low_threshold_) continue;
    for (size_t j = i + 1; j < detections.size(); ++j) {
      if (!std::isfinite(detections[j].confidence) ||
          detections[j].confidence <= track_low_threshold_) continue;
      if (computeIoU(detectionBox(detections[i]), detectionBox(detections[j])) > 0.30f)
        overlapping[i] = overlapping[j] = true;
    }
  }
  for (int index : high)
    if (!overlapping[index]) features[index] = extractAppearance(frame, detectionBox(detections[index]));

  std::vector<bool> matched_track(active_tracks_.size(), false);
  std::vector<bool> matched_detection(detections.size(), false);
  auto associate = [&](const std::vector<int>& tracks, const std::vector<int>& observations,
                       float limit, bool use_appearance) {
    std::vector<std::vector<float>> costs(tracks.size(), std::vector<float>(observations.size()));
    for (size_t row = 0; row < tracks.size(); ++row) {
      const auto& track = active_tracks_[tracks[row]];
      for (size_t col = 0; col < observations.size(); ++col) {
        const int index = observations[col];
        const float iou = computeIoU(track.bbox, detectionBox(detections[index]));
        float similarity = -1;
        if (use_appearance && track.time_since_update <= appearance_reid_max_age_)
          similarity = cosineSimilarity(track.smooth_feature, features[index]);
        costs[row][col] = use_appearance ?
          boTSortAssociationCost(iou, detections[index].confidence, similarity,
                                proximity_threshold_, appearance_match_threshold_) : 1.0f - iou;
      }
    }
    const auto matches = solveAssignment(costs, limit);
    const auto ambiguous = ambiguousAssignments(costs, matches, limit);
    for (size_t row = 0; row < tracks.size(); ++row) {
      auto& track = active_tracks_[tracks[row]];
      track.association_ambiguous = track.association_ambiguous || ambiguous[row];
      const int col = matches[row];
      if (col < 0) continue;
      const int index = observations[col];
      matched_track[tracks[row]] = true;
      matched_detection[index] = true;
      applyDetectionToTrack(track, detections[index], use_appearance ? features[index] : std::vector<float>{});
    }
  };

  // BoT-SORT: tracked+lost / strong, remaining tracked / weak, tentative / strong.
  associate(pool, high, match_threshold_, true);
  std::vector<int> remaining_tracked;
  for (int index : pool)
    if (!matched_track[index] && active_tracks_[index].state == TrackState::Confirmed)
      remaining_tracked.push_back(index);
  associate(remaining_tracked, low, 0.5f, false);
  std::vector<int> remaining_high;
  for (int index : high) if (!matched_detection[index]) remaining_high.push_back(index);
  associate(tentative, remaining_high, 0.7f, true);

  for (size_t i = 0; i < active_tracks_.size(); ++i) {
    auto& track = active_tracks_[i];
    if (matched_track[i]) continue;
    track.hit_streak = 0;
    if (track.state == TrackState::Tentative)
      track.time_since_update = max_age_ + 1;
    else
      track.state = TrackState::Lost;
  }
  for (int index : remaining_high) {
    if (matched_detection[index] || detections[index].confidence < new_track_threshold_) continue;
    Track track;
    track.id = next_id_++;
    track.confidence = detections[index].confidence;
    track.bbox = detectionBox(detections[index]);
    track.kalman = BoTSortKalman(track.bbox);
    track.age = track.hits = track.hit_streak = 1;
    track.last_prediction_time_sec = timestamp_valid ? timestamp_sec : 0;
    if (min_hits_ == 1) track.state = TrackState::Confirmed;
    updateFeature(track, features[index]);
    active_tracks_.push_back(std::move(track));
  }

  // Remove tracked/lost duplicates using the longer observed trajectory.
  std::vector<bool> duplicate(active_tracks_.size(), false);
  for (size_t i = 0; i < active_tracks_.size(); ++i) {
    const auto& tracked = active_tracks_[i];
    if (tracked.state != TrackState::Confirmed) continue;
    for (size_t j = 0; j < active_tracks_.size(); ++j) {
      const auto& lost = active_tracks_[j];
      if (lost.state != TrackState::Lost || lost.time_since_update > max_age_) continue;
      if (computeIoU(tracked.bbox, lost.bbox) > 0.85f) {
        // Indistinguishable trajectories must not silently retain an identity lock.
        active_tracks_[i].association_ambiguous = true;
        active_tracks_[j].association_ambiguous = true;
        duplicate[tracked.age > lost.age - lost.time_since_update ? j : i] = true;
      }
    }
  }
  std::vector<Track> output;
  size_t index = 0;
  active_tracks_.erase(std::remove_if(active_tracks_.begin(), active_tracks_.end(),
    [&](const Track& track) { return duplicate[index++] || track.time_since_update > max_age_; }),
    active_tracks_.end());
  for (const auto& track : active_tracks_)
    if (track.state != TrackState::Tentative && track.time_since_update <= max_publish_age_)
      output.push_back(track);
  return output;
}

}  // namespace opl_human_vision
