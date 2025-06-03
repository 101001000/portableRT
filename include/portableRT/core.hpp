#pragma once

#include <array>
namespace portableRT {

struct Ray {
  std::array<float, 3> origin;
  std::array<float, 3> direction;
};

using Tris = std::vector<std::array<float, 9>>;

} // namespace portableRT