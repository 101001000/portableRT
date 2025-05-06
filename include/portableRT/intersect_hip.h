#pragma once
#include "core.h"

namespace portableRT{
template<>
bool intersect_tri<Backend::HIP>(const std::array<float, 9> &v, const Ray &ray);
}