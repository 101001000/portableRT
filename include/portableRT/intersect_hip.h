#pragma once
#include "core.h"

namespace portableRT{
template<>
bool intersect_tri<BackendType::HIP>(const std::array<float, 9> &v, const Ray &ray);
namespace {
    inline RegisterBackend<BackendType::HIP, intersect_tri<BackendType::HIP>> register_hip("HIP");
}
}