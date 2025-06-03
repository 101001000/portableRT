#pragma once
#include "backend.hpp"
#include "core.hpp"

namespace portableRT {

class HIPBackend : public InvokableBackend<HIPBackend> {
public:
  HIPBackend() : InvokableBackend("HIP") { static RegisterBackend reg(*this); }

  bool intersect_tris(const Tris &tris, const Ray &ray);
  bool is_available() const override;
  void init() override;
  void shutdown() override;
};

static HIPBackend hip_backend;

} // namespace portableRT