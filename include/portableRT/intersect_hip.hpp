#pragma once
#include "backend.hpp"
#include "core.hpp"
#include "bvh.hpp"

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
  void* m_dbvh;
};

static HIPBackend hip_backend;

} // namespace portableRT