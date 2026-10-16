#include <cerrno>
#include <cstdlib>
#include <dlfcn.h>
#include <sys/stat.h>
#include <unistd.h>
extern "C" int fsync(int fd) {
  struct stat st {};
  if (std::getenv("OPL_TEST_FAIL_DIRECTORY_FSYNC") && !fstat(fd, &st) &&
      S_ISDIR(st.st_mode)) {
    errno = EIO;
    return -1;
  }
  using Sync = int (*)(int);
  static auto original = reinterpret_cast<Sync>(dlsym(RTLD_NEXT, "fsync"));
  return original(fd);
}
