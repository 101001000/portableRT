#pragma once
#include "core.h"
#include "backend.h"

namespace portableRT{

class SYCLBackend : public InvokableBackend<SYCLBackend> {
public:
    SYCLBackend() : InvokableBackend(BackendType::SYCL, "SYCL") {
        static RegisterBackend reg(*this);
    }

    bool intersect_tri(const std::array<float,9>& vertices, const Ray& ray) const;
    bool is_available() const;
};


static SYCLBackend sycl_backend;

}