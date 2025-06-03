#pragma once
#include "backend.hpp"
#include "core.hpp"
#include <embree4/rtcore.h>

namespace portableRT {

class EmbreeCPUBackend : public InvokableBackend<EmbreeCPUBackend> {
public:
  EmbreeCPUBackend() : InvokableBackend("Embree CPU") {
    static RegisterBackend reg(*this);
  }

  void initializeScene();

  bool intersect_tris(const Tris &tris, const Ray &ray);
  bool is_available() const override;
  void init() override;
  void shutdown() override;

private:
  RTCDevice m_device;
  RTCScene m_scene;
  RTCGeometry m_tri;
};

static EmbreeCPUBackend embreecpu_backend;

} // namespace portableRT