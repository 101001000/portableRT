#include <string>
#include <unistd.h>

inline std::string get_executable_path() {
  char result[1024];
  ssize_t count = readlink("/proc/self/exe", result, 1024);
  return std::string(result, (count > 0) ? count : 0);
}

inline std::string get_executable_dir() {
  std::string full_path = get_executable_path();
  size_t found = full_path.find_last_of("/\\");
  return full_path.substr(0, found);
}