#pragma once
#include "backend.hpp"
#include "core.hpp"

namespace portableRT {

class CPUBackend : public InvokableBackend<CPUBackend> {
public:
  CPUBackend() : InvokableBackend("CPU") { static RegisterBackend reg(*this); }

  bool intersect_tris(const Ray &ray);
  bool is_available() const override;
  void init() override;
  void shutdown() override;
  void set_tris(const Tris &tris) override;

private:
  Tris m_tris;
};

static CPUBackend cpu_backend;

} // namespace portableRT