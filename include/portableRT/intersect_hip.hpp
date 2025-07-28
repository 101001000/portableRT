#pragma once
#include "backend.hpp"
#include "bvh.hpp"
#include "core.hpp"

namespace portableRT {

class HIPBackend : public InvokableBackend<HIPBackend> {
  public:
	HIPBackend() : InvokableBackend("HIP") { static RegisterBackend reg(*this); }

	template <class... Tags>
	std::vector<HitReg<Tags...>> nearest_hits(const std::vector<Ray> &rays);
	bool is_available() const override;
	void init() override;
	void shutdown() override;
	void set_tris(const Tris &tris) override;
	std::string device_name() const override;

  private:
	void *m_dbvh;
};

static HIPBackend hip_backend;

} // namespace portableRT