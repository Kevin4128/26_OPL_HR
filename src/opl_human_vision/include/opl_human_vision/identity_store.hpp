#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace opl_human_vision {

struct IdentityRecord {
  uint64_t persistent_id{0};
  std::string name;
  std::vector<std::vector<float>> face_gallery;
  // Zero in caller-created/legacy records means every existing template is an
  // anchor.
  uint32_t anchor_count{0};
};

bool validIdentityName(const std::string &name);
std::string modelFingerprint(const std::string &path);

// Owns an exclusively locked directory FD; all data operations are relative to
// it.
class IdentityStore {
public:
  IdentityStore(const std::string &directory, const std::string &model);
  ~IdentityStore();
  IdentityStore(const IdentityStore &) = delete;
  IdentityStore &operator=(const IdentityStore &) = delete;
  std::vector<IdentityRecord> load(uint64_t &next_id) const;
  void save(const std::vector<IdentityRecord> &identities,
            uint64_t next_id) const;

private:
  int directory_fd_{-1}, lock_fd_{-1};
  std::string model_;
  mutable bool commit_uncertain_{false};
};

} // namespace opl_human_vision
