#pragma once

#include <cstdint>
#include <chrono>

namespace opl_human_vision {

enum class IdentityState : uint8_t {
  Unconfirmed = 0, FaceConfirmed = 1, Tracked = 2, Uncertain = 3
};

struct IdentityDecision {
  uint64_t id;
  IdentityState state;
};

inline bool refreshIdentityLock(
  std::chrono::steady_clock::time_point& last_seen,
  std::chrono::steady_clock::time_point now, double grace_sec, bool observed)
{
  if (std::chrono::duration<double>(now - last_seen).count() > grace_sec) return false;
  if (observed) last_seen = now;
  return true;
}

struct FaceConfirmation {
  uint64_t candidate{0};
  int hits{0};
  int64_t last_stamp{-1};
};

// Only distinct RGB observations count; missing face evidence cannot revoke continuity.
inline IdentityDecision decideIdentityAction(
  FaceConfirmation& confirmation, uint64_t cached_id,
  uint64_t face_candidate, int64_t image_stamp)
{
  if (image_stamp > confirmation.last_stamp) {
    confirmation.hits = face_candidate == confirmation.candidate ? confirmation.hits + 1 : 1;
    confirmation.candidate = face_candidate;
    confirmation.last_stamp = image_stamp;
  }
  if (face_candidate && confirmation.candidate == face_candidate && confirmation.hits >= 3) {
    return {face_candidate, IdentityState::FaceConfirmed};
  }
  if (cached_id && (!face_candidate || face_candidate == cached_id)) {
    return {cached_id, face_candidate ? IdentityState::FaceConfirmed : IdentityState::Tracked};
  }
  return {0, cached_id ? IdentityState::Uncertain : IdentityState::Unconfirmed};
}

}  // namespace opl_human_vision
