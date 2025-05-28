#pragma once
#include "core.h"

namespace portableRT{
template<>
bool intersect_tri<BackendType::SYCL>(const std::array<float, 9> &v, const Ray &ray);
namespace {
    inline RegisterBackend<BackendType::SYCL, intersect_tri<BackendType::SYCL>> register_sycl("SYCL");
}
}