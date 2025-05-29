#pragma once
#include "core.h"
#include "backend.h"

namespace portableRT{
template<>
bool intersect_tri<BackendType::EMBREE_SYCL>(const std::array<float, 9> &v, const Ray &ray);


class EmbreeSyclBackend : public InvokableBackend<EmbreeSyclBackend> {
public:
    EmbreeSyclBackend() : InvokableBackend(BackendType::EMBREE_SYCL, "Embree SYCL") {
        static RegisterBackend reg(*this);
    }

    bool intersect_tri(const std::array<float,9>& vertices, const Ray& ray) const;
    bool is_available() const;
};


static EmbreeSyclBackend embreesycl_backend;

}