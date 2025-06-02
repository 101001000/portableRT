#include <array>
#include <iostream>
#include <portableRT/portableRT.hpp>

int main() {

  std::cout << "Printing all compiled backends: ";
  for (auto backend : portableRT::all_backends()) {
    std::cout << backend->name() << ", ";
  }

  std::cout << std::endl;

  std::cout << "Printing all available backends: ";
  for (auto backend : portableRT::available_backends()) {
    std::cout << backend->name() << ", ";
  }

  std::cout << std::endl;
  return 0;
}