#pragma once
#include "backend.hpp"
#include "core.hpp"
#include <embree4/rtcore.h>

namespace portableRT {

class EmbreeCPUBackend : public InvokableBackend<EmbreeCPUBackend> {
  public:
	EmbreeCPUBackend() : InvokableBackend("Embree CPU") { static RegisterBackend reg(*this); }

	void initializeScene();

	std::vector<HitReg> nearest_hits(const std::vector<Ray> &rays);
	bool is_available() const override;
	void init() override;
	void shutdown() override;
	void set_tris(const Tris &tris) override;
	std::string device_name() const override;

  private:
	RTCDevice m_device;
	RTCScene m_scene;
	RTCGeometry m_tri;
	unsigned int m_geom_id = RTC_INVALID_GEOMETRY_ID;
};

static EmbreeCPUBackend embreecpu_backend;

} // namespace portableRT