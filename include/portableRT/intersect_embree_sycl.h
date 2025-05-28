#pragma once
#include "core.h"

namespace portableRT{
template<>
bool intersect_tri<BackendType::EMBREE_SYCL>(const std::array<float, 9> &v, const Ray &ray);
namespace {
    inline RegisterBackend<BackendType::EMBREE_SYCL, intersect_tri<BackendType::EMBREE_SYCL>> register_embree_sycl("Embree SYCL");
}
}