#pragma once
#include "backend.hpp"
#include "bvh.hpp"
#include "core.hpp"

namespace portableRT {

class HIPBackend : public InvokableBackend<HIPBackend> {
public:
  HIPBackend() : InvokableBackend("HIP") { static RegisterBackend reg(*this); }

  bool intersect_tris(const Ray &ray);
  bool is_available() const override;
  void init() override;
  void shutdown() override;
  void set_tris(const Tris &tris) override;

private:
  Tris m_tris;
  uint64_t m_rootidx;
  void *m_dbvh;
};

static HIPBackend hip_backend;

} // namespace portableRT