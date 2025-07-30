#pragma once
#include "backend.hpp"
#include "core.hpp"
#include <embree4/rtcore.h>

namespace portableRT {

template <class... Tags>
inline portableRT::HitReg<Tags...> castRay(RTCScene scene, float ox, float oy, float oz, float dx,
                                           float dy, float dz) {

	struct RTCRayHit rayhit;
	rayhit.ray.org_x = ox;
	rayhit.ray.org_y = oy;
	rayhit.ray.org_z = oz;
	rayhit.ray.dir_x = dx;
	rayhit.ray.dir_y = dy;
	rayhit.ray.dir_z = dz;
	rayhit.ray.tnear = 0;
	rayhit.ray.tfar = std::numeric_limits<float>::infinity();
	rayhit.ray.mask = -1;
	rayhit.ray.flags = 0;
	rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
	rayhit.hit.primID = RTC_INVALID_GEOMETRY_ID;
	rayhit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;

	rtcIntersect1(scene, &rayhit);

	portableRT::HitReg<Tags...> hit_reg;
	hit_reg.px = ox + dx * rayhit.ray.tfar;
	hit_reg.py = oy + dy * rayhit.ray.tfar;
	hit_reg.pz = oz + dz * rayhit.ray.tfar;
	hit_reg.t = rayhit.ray.tfar;
	hit_reg.primitive_id = rayhit.hit.primID;
	hit_reg.u = rayhit.hit.u;
	hit_reg.v = rayhit.hit.v;
	hit_reg.valid = rayhit.hit.primID != RTC_INVALID_GEOMETRY_ID;

	return hit_reg;
}

class EmbreeCPUBackend : public InvokableBackend<EmbreeCPUBackend> {
  public:
	EmbreeCPUBackend() : InvokableBackend("Embree CPU") { static RegisterBackend reg(*this); }

	void initializeScene();

	template <class... Tags>
	void check_hit(int i, const std::vector<Ray> &rays, std::vector<HitReg<Tags...>> &hits, int N) {
		for (int r = 0; r < N; r++) {
			auto hit_reg = castRay<Tags...>(m_scene, rays[i + r].origin[0], rays[i + r].origin[1],
			                                rays[i + r].origin[2], rays[i + r].direction[0],
			                                rays[i + r].direction[1], rays[i + r].direction[2]);
			hits[i + r] = hit_reg;
		}
	}

	template <class... Tags>
	std::vector<HitReg<Tags...>> nearest_hits(const std::vector<Ray> &rays) {
		std::vector<HitReg<Tags...>> hits;
		hits.resize(rays.size());

		clear_affinity();

		unsigned n = std::thread::hardware_concurrency();
		std::vector<std::thread> threads;
		threads.reserve(n);

		int rays_per_thread = rays.size() / n;

		for (unsigned i = 0; i < n; ++i) {
			threads.emplace_back(&EmbreeCPUBackend::check_hit<Tags...>, this, i * rays_per_thread,
			                     std::cref(rays), std::ref(hits), rays_per_thread);
		}

		for (auto &thread : threads)
			thread.join();

		check_hit<Tags...>(n * rays_per_thread, rays, hits, rays.size() % n);

		return hits;
	}

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