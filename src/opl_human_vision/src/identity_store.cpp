#include "opl_human_vision/identity_store.hpp"

#include <fcntl.h>
#include <opencv2/core.hpp>
#include <openssl/evp.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>

namespace opl_human_vision {
namespace {
constexpr size_t MAX_BYTES = 32 * 1024 * 1024;

void checkFile(int fd) {
  struct stat st {};
  if (fstat(fd, &st) || !S_ISREG(st.st_mode) || st.st_uid != getuid() ||
      (st.st_mode & 0777) != 0600 || st.st_nlink != 1 ||
      st.st_size > int64_t(MAX_BYTES))
    throw std::runtime_error("Identity file must be an owner-only regular file "
                             "(0600), without links");
}

uint64_t parseId(const cv::FileNode &value) {
  if (!value.isString())
    throw std::runtime_error("Identity ID must be a string");
  const std::string text = value;
  if (text.empty() || text.size() > 10 ||
      text.find_first_not_of("0123456789") != std::string::npos)
    throw std::runtime_error("Invalid identity ID");
  const auto id = std::stoull(text);
  if (id == 0 || id >= 1000000000ULL)
    throw std::runtime_error("Identity ID out of range");
  return id;
}

void validate(const std::vector<IdentityRecord> &identities, uint64_t next) {
  if (identities.size() > 1000 || next == 0 || next >= 1000000000ULL)
    throw std::runtime_error("Identity store capacity exceeded");
  std::set<uint64_t> ids;
  std::set<std::string> names;
  for (const auto &identity : identities) {
    if (!identity.persistent_id || identity.persistent_id >= next ||
        !ids.insert(identity.persistent_id).second ||
        !validIdentityName(identity.name) ||
        !names.insert(identity.name).second || identity.face_gallery.empty() ||
        identity.face_gallery.size() > 20 ||
        identity.anchor_count > identity.face_gallery.size())
      throw std::runtime_error("Invalid or duplicate identity metadata");
    for (const auto &embedding : identity.face_gallery) {
      double norm = 0;
      if (embedding.size() != 512)
        throw std::runtime_error("Invalid face embedding dimension");
      for (const auto value : embedding) {
        if (!std::isfinite(value))
          throw std::runtime_error("Nonfinite face embedding");
        norm += value * value;
      }
      if (std::abs(norm - 1) > 0.02)
        throw std::runtime_error("Face embedding is not normalized");
    }
  }
}
} // namespace

bool validIdentityName(const std::string &name) {
  if (name.empty() || name.size() > 80 || name.front() == ' ' ||
      name.back() == ' ')
    return false;
  for (const unsigned char c : name)
    if (c < 32 || c == 127 || c == '/' || c == '\\')
      return false;
  return true;
}

std::string modelFingerprint(const std::string &path) {
  std::ifstream input(path, std::ios::binary);
  if (!input)
    throw std::runtime_error(
        "Cannot read face model for identity compatibility check");
  auto context = std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)>(
      EVP_MD_CTX_new(), EVP_MD_CTX_free);
  if (!context || EVP_DigestInit_ex(context.get(), EVP_sha256(), nullptr) != 1)
    throw std::runtime_error("Cannot initialize model fingerprint");
  std::array<char, 65536> bytes{};
  while (input.read(bytes.data(), bytes.size()) || input.gcount()) {
    if (EVP_DigestUpdate(context.get(), bytes.data(), input.gcount()) != 1)
      throw std::runtime_error("Cannot fingerprint face model");
  }
  if (!input.eof())
    throw std::runtime_error("Face model read failed");
  std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
  unsigned int length = 0;
  if (EVP_DigestFinal_ex(context.get(), digest.data(), &length) != 1)
    throw std::runtime_error("Cannot finish model fingerprint");
  std::ostringstream text;
  for (unsigned int i = 0; i < length; ++i)
    text << std::hex << std::setw(2) << std::setfill('0') << int(digest[i]);
  return text.str();
}

IdentityStore::IdentityStore(const std::string &directory,
                             const std::string &model)
    : model_(model) {
  if (model.empty() || directory.empty())
    throw std::runtime_error("Identity store path/model is empty");
  int current = open("/", O_DIRECTORY | O_RDONLY | O_CLOEXEC);
  if (current < 0)
    throw std::runtime_error("Cannot open identity directory root");
  try {
    const auto absolute =
        std::filesystem::absolute(directory).lexically_normal();
    for (const auto &component : absolute.relative_path()) {
      int next = openat(current, component.c_str(),
                        O_DIRECTORY | O_NOFOLLOW | O_RDONLY | O_CLOEXEC);
      if (next < 0 && errno == ENOENT) {
        if (mkdirat(current, component.c_str(), 0700) && errno != EEXIST)
          throw std::runtime_error("Cannot create private identity directory");
        next = openat(current, component.c_str(),
                      O_DIRECTORY | O_NOFOLLOW | O_RDONLY | O_CLOEXEC);
      }
      if (next < 0)
        throw std::runtime_error(
            "Identity directory is inaccessible or contains a symlink");
      close(current);
      current = next;
    }
    struct stat st {};
    if (fstat(current, &st) || st.st_uid != getuid() ||
        (st.st_mode & 0777) != 0700)
      throw std::runtime_error("Identity directory must be owned by this user "
                               "with permissions 0700");
    lock_fd_ = openat(current, ".writer.lock",
                      O_CREAT | O_RDWR | O_NOFOLLOW | O_CLOEXEC, 0600);
    if (lock_fd_ < 0)
      throw std::runtime_error("Cannot open identity writer lock");
    checkFile(lock_fd_);
    if (flock(lock_fd_, LOCK_EX | LOCK_NB))
      throw std::runtime_error("Identity store already has a writer");
    directory_fd_ = current;
  } catch (...) {
    close(current);
    if (lock_fd_ >= 0)
      close(lock_fd_);
    lock_fd_ = -1;
    throw;
  }
}

IdentityStore::~IdentityStore() {
  if (lock_fd_ >= 0)
    close(lock_fd_);
  if (directory_fd_ >= 0)
    close(directory_fd_);
}

std::vector<IdentityRecord> IdentityStore::load(uint64_t &next_id) const {
  int fd = openat(directory_fd_, "identities.json",
                  O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
  if (fd < 0 && errno == ENOENT) {
    next_id = 1;
    return {};
  }
  if (fd < 0)
    throw std::runtime_error(
        "Cannot open identity file (symlinks are refused)");
  std::string data;
  try {
    checkFile(fd);
    std::array<char, 8192> bytes{};
    while (true) {
      const auto count = read(fd, bytes.data(), bytes.size());
      if (count < 0 && errno == EINTR)
        continue;
      if (count < 0)
        throw std::runtime_error("Cannot read identity file");
      if (!count)
        break;
      data.append(bytes.data(), count);
      if (data.size() > MAX_BYTES)
        throw std::runtime_error("Identity file exceeds size limit");
    }
    close(fd);
    fd = -1;
    cv::FileStorage file(data, cv::FileStorage::READ | cv::FileStorage::MEMORY);
    const int version = int(file["version"]);
    if (!file.isOpened() || (version != 1 && version != 2) ||
        std::string(file["model_sha256"]) != model_)
      throw std::runtime_error(
          "Unsupported identity file version or different face model");
    const auto next = parseId(file["next_id"]);
    auto records = file["identities"];
    if (!records.isSeq() || records.size() > 1000)
      throw std::runtime_error("Invalid identities list");
    std::vector<IdentityRecord> identities;
    for (const auto &item : records) {
      IdentityRecord identity;
      identity.persistent_id = parseId(item["id"]);
      if (!item["name"].isString())
        throw std::runtime_error("Invalid identity name");
      identity.name = std::string(item["name"]);
      auto gallery = item["faces"];
      if (!gallery.isSeq() || gallery.empty() || gallery.size() > 20)
        throw std::runtime_error("Invalid face gallery");
      for (const auto &face : gallery) {
        if (!face.isSeq() || face.size() != 512)
          throw std::runtime_error("Invalid face vector");
        std::vector<float> embedding;
        face >> embedding;
        identity.face_gallery.push_back(std::move(embedding));
      }
      if (version == 1)
        identity.anchor_count = identity.face_gallery.size();
      else {
        const auto anchors = item["anchor_count"];
        if (!anchors.isInt() || int(anchors) < 1 ||
            int(anchors) > int(identity.face_gallery.size()))
          throw std::runtime_error("Invalid immutable anchor count");
        identity.anchor_count = int(anchors);
      }
      identities.push_back(std::move(identity));
    }
    validate(identities, next);
    next_id = next;
    return identities;
  } catch (...) {
    if (fd >= 0)
      close(fd);
    throw;
  }
}

void IdentityStore::save(const std::vector<IdentityRecord> &identities,
                         uint64_t next_id) const {
  if (commit_uncertain_)
    throw std::runtime_error("Previous commit uncertain; restart to reload "
                             "identity storage before further changes");
  validate(identities, next_id);
  cv::FileStorage file("", cv::FileStorage::WRITE | cv::FileStorage::MEMORY |
                               cv::FileStorage::FORMAT_JSON);
  file << "version" << 2 << "model_sha256" << model_ << "next_id"
       << std::to_string(next_id);
  file << "identities" << "[";
  for (const auto &identity : identities) {
    file << "{" << "id" << std::to_string(identity.persistent_id) << "name"
         << identity.name << "anchor_count"
         << int(identity.anchor_count ? identity.anchor_count
                                      : identity.face_gallery.size())
         << "faces" << "[";
    for (const auto &face : identity.face_gallery)
      file << face;
    file << "]" << "}";
  }
  file << "]";
  const auto data = file.releaseAndGetString();
  if (data.size() > MAX_BYTES)
    throw std::runtime_error("Identity store exceeds size limit");
  const std::string temporary =
      ".pending." + std::to_string(getpid()) + "." +
      std::to_string(
          std::chrono::steady_clock::now().time_since_epoch().count());
  int fd = openat(directory_fd_, temporary.c_str(),
                  O_CREAT | O_EXCL | O_WRONLY | O_NOFOLLOW | O_CLOEXEC, 0600);
  if (fd < 0)
    throw std::runtime_error("Cannot create private identity transaction");
  try {
    size_t offset = 0;
    while (offset < data.size()) {
      const auto written =
          write(fd, data.data() + offset, data.size() - offset);
      if (written < 0 && errno == EINTR)
        continue;
      if (written <= 0)
        throw std::runtime_error("Cannot write identity transaction");
      offset += written;
    }
    if (fsync(fd))
      throw std::runtime_error("Cannot flush identity transaction");
    close(fd);
    fd = -1;
    if (renameat(directory_fd_, temporary.c_str(), directory_fd_,
                 "identities.json"))
      throw std::runtime_error("Cannot commit identity transaction");
    if (fsync(directory_fd_)) {
      commit_uncertain_ = true;
      throw std::runtime_error("Identity commit uncertain after directory "
                               "flush failure; restart to reload storage");
    }
  } catch (...) {
    if (fd >= 0)
      close(fd);
    unlinkat(directory_fd_, temporary.c_str(), 0);
    throw;
  }
}

} // namespace opl_human_vision
