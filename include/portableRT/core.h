#pragma once

#include <array>
namespace portableRT {

    enum class Backend{
        CPU,
        OPTIX,
        HIP,
        EMBREE_SYCL,
        EMBREE_CPU
    };

    struct Ray {
        std::array<float, 3> origin;
        std::array<float, 3> direction;
    };

    template<Backend B>
    bool intersect_tri(const std::array<float, 9> &vertices, const Ray &ray);
}