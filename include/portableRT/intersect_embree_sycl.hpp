#pragma once
#include <embree4/rtcore.h>
#include <memory>

#include "backend.hpp"
#include "core.hpp"

namespace portableRT {

struct EmbreeSYCLBackendImpl;

class EmbreeSYCLBackend : public InvokableBackend<EmbreeSYCLBackend> {
  public:
	EmbreeSYCLBackend();
	~EmbreeSYCLBackend();

	template <class... Tags>
	std::vector<HitReg<Tags...>> nearest_hits(const std::vector<Ray> &rays);
	bool is_available() const override;
	void init() override;
	void shutdown() override;
	void set_tris(const Tris &tris) override;
	std::string device_name() const override;

  private:
	// Ugly solution to overcome the issues with forward declarations of the sycl
	// type alias in the intel implementation I need to move the sycl import to
	// the .cpp instead the .h so the hip compiler ignores the sycl headers. Other
	// option is to pack the sycl headers, but that would break sycl dependency.
	std::unique_ptr<EmbreeSYCLBackendImpl> m_impl;

	RTCDevice m_rtcdevice;
	RTCScene m_rtcscene;
	RTCTraversable m_rtctraversable;
	RTCGeometry m_tri;
	unsigned int m_geom_id = RTC_INVALID_GEOMETRY_ID;
};

static EmbreeSYCLBackend embreesycl_backend;

} // namespace portableRT