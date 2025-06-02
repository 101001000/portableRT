#pragma once
#include "backend.hpp"
#include "core.hpp"

namespace portableRT {

class CPUBackend : public InvokableBackend<CPUBackend> {
public:
  CPUBackend() : InvokableBackend("CPU") {
    static RegisterBackend reg(*this);
  }

  bool intersect_tri(const std::array<float, 9> &vertices, const Ray &ray);
  bool is_available() const override;
  void init() override;
  void shutdown() override;
};

static CPUBackend cpu_backend;

} // namespace portableRT