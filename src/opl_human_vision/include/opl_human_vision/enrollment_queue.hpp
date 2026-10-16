#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace opl_human_vision {
struct EnrollmentCandidate {
  uint64_t raw_id;
  float confidence;
  std::string name;
};

class EnrollmentQueue {
public:
  void start(std::vector<EnrollmentCandidate> candidates, double now) {
    if (current() || candidates.empty())
      throw std::invalid_argument("Enrollment busy or no fresh candidates");
    std::set<uint64_t> ids;
    for (const auto &candidate : candidates)
      if (!candidate.raw_id || !std::isfinite(candidate.confidence) ||
          !ids.insert(candidate.raw_id).second)
        throw std::invalid_argument(
            "Invalid or duplicate enrollment candidate");
    std::sort(
        candidates.begin(), candidates.end(), [](const auto &a, const auto &b) {
          return a.confidence == b.confidence ? a.raw_id < b.raw_id
                                              : a.confidence > b.confidence;
        });
    candidates_ = std::move(candidates);
    index_ = 0;
    ++session_id;
    completed = failed = 0;
    started_ = last_seen_ = now;
  }
  const EnrollmentCandidate *current() const {
    return index_ < candidates_.size() ? &candidates_[index_] : nullptr;
  }
  void observe(const std::set<uint64_t> &visible, double now) {
    if (current() && visible.count(current()->raw_id))
      last_seen_ = now;
  }
  std::string failure(double now) const {
    if (!current())
      return "";
    if (now - last_seen_ >= missing_sec)
      return "Target disappeared; partial samples discarded";
    if (now - started_ >= timeout_sec)
      return "Insufficient acceptable scans before enrollment deadline";
    return "";
  }
  void advance(bool success, double now) {
    if (!current())
      return;
    if (success)
      ++completed;
    else
      ++failed;
    ++index_;
    started_ = last_seen_ = now;
  }
  void cancel(double now) {
    while (current())
      advance(false, now);
  }
  size_t total() const { return candidates_.size(); }
  uint64_t session_id{0};
  uint32_t completed{0}, failed{0};
  double missing_sec{2.0}, timeout_sec{30.0};

private:
  std::vector<EnrollmentCandidate> candidates_;
  size_t index_{0};
  double started_{0}, last_seen_{0};
};
} // namespace opl_human_vision
