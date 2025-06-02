#pragma once

#include <array>
namespace portableRT {

struct Ray {
  std::array<float, 3> origin;
  std::array<float, 3> direction;
};

} // namespace portableRT