#pragma once
#include "backend.hpp"
#include "core.hpp"
#include "bvh.hpp"

namespace portableRT {

class CPUBackend : public InvokableBackend<CPUBackend> {
public:
  CPUBackend() : InvokableBackend("CPU") { static RegisterBackend reg(*this); }

  std::vector<HitReg> nearest_hits(const std::vector<Ray> &rays);
  bool is_available() const override;
  void init() override;
  void shutdown() override;
  void set_tris(const Tris &tris) override;
  std::string device_name() const override;

private:
  BVH2 m_bvh;
};

static CPUBackend cpu_backend;

} // namespace portableRT