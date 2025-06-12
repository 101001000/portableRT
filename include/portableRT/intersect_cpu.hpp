#pragma once
#include "backend.hpp"
#include "bvh.hpp"
#include "core.hpp"

namespace portableRT {

class CPUBackend : public InvokableBackend<CPUBackend> {
public:
  CPUBackend() : InvokableBackend("CPU") { static RegisterBackend reg(*this); }

  std::vector<float> nearest_hits(const std::vector<Ray> &rays);
  bool is_available() const override;
  void init() override;
  void shutdown() override;
  void set_tris(const Tris &tris) override;
  std::string device_name() const override;

private:
  Tris m_tris;
  BVH *m_bvh;
};

static CPUBackend cpu_backend;

} // namespace portableRT