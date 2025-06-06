#pragma once
#include "backend.hpp"
#include "core.hpp"

namespace portableRT {

class HIPBackend : public InvokableBackend<HIPBackend> {
public:
  HIPBackend() : InvokableBackend("HIP") {
    static RegisterBackend reg(*this);
  }

  bool intersect_tri(const std::array<float, 9> &vertices, const Ray &ray);
  bool is_available() const override;
  void init() override;
  void shutdown() override;
};

static HIPBackend hip_backend;

} // namespace portableRT