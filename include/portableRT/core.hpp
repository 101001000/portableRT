#pragma once

#include <array>
namespace portableRT {

struct Ray {
  std::array<float, 3> origin;
  std::array<float, 3> direction;
};

enum class BackendType { CPU, OPTIX, HIP, EMBREE_SYCL, EMBREE_CPU, SYCL };
} // namespace portableRT