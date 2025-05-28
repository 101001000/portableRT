#pragma once

#include <array>
namespace portableRT {


    struct Ray {
        std::array<float, 3> origin;
        std::array<float, 3> direction;
    };
    

    enum class BackendType{
        CPU,
        OPTIX,
        HIP,
        EMBREE_SYCL,
        EMBREE_CPU,
        SYCL
    };
    template<BackendType B>
    bool intersect_tri(const std::array<float, 9> &vertices, const Ray &ray);
}