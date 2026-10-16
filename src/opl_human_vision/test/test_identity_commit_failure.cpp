#include "opl_human_vision/identity_store.hpp"
#include <cassert>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <unistd.h>
using namespace opl_human_vision;
int main() {
  auto root = std::filesystem::current_path() /
              ("commit_failure_" + std::to_string(getpid()));
  struct Cleanup {
    std::filesystem::path root;
    ~Cleanup() { std::filesystem::remove_all(root); }
  } cleanup{root};
  {
    IdentityStore store(root.string(), "test");
    IdentityRecord record;
    record.persistent_id = 1;
    record.name = "Alice";
    record.face_gallery = {std::vector<float>(512, 0)};
    record.face_gallery[0][0] = 1;
    bool failed = false;
    try {
      store.save({record}, 2);
    } catch (...) {
      failed = true;
    }
    assert(failed);
    unsetenv("OPL_TEST_FAIL_DIRECTORY_FSYNC");
    bool blocked = false;
    try {
      store.save({}, 1);
    } catch (...) {
      blocked = true;
    }
    assert(blocked);
  }
  IdentityStore restarted(root.string(), "test");
  uint64_t next = 0;
  assert(restarted.load(next).size() == 1 && next == 2);
  std::cout << "Uncertain commit blocks further mutations until "
               "restart/reconciliation\n";
}
