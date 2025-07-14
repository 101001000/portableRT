#include <array>
#include <iostream>
#include <portableRT/portableRT.hpp>
#include <portableRT/version.hpp>

int main() {

  std::cout << "portableRT version: " << PORTABLERT_VERSION << std::endl;

  std::cout << "Printing all compiled backends: " << std::endl;
  for (auto backend : portableRT::all_backends()) {
    std::cout << "\t" << backend->name() << std::endl;
  }

  std::cout << std::endl;

  std::cout << "Printing all available backends: " << std::endl;
  for (auto backend : portableRT::available_backends()) {
    portableRT::select_backend(
        backend); // It's necessary to initialize the backend to know the device
    std::cout << "\t" << backend->name() << " (" << backend->device_name()
              << ")" << std::endl;
  }

  std::cout << std::endl;
  return 0;
}