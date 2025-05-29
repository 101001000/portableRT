#pragma once
#include "core.h"
#include "backend.h"

namespace portableRT{
template<>
bool intersect_tri<BackendType::HIP>(const std::array<float, 9> &v, const Ray &ray);


class HIPBackend : public InvokableBackend<HIPBackend> {
public:
    HIPBackend() : InvokableBackend(BackendType::HIP, "HIP") {
        static RegisterBackend reg(*this);
    }

    bool intersect_tri(const std::array<float,9>& vertices, const Ray& ray) const;
    bool is_available() const;
};


static HIPBackend hip_backend;

}