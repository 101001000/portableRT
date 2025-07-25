#pragma once
#include "backend.hpp"
#include "bvh.hpp"
#include "core.hpp"

namespace portableRT {

class CPUBackend : public InvokableBackend<CPUBackend> {
  public:
	CPUBackend() : InvokableBackend("CPU") { static RegisterBackend reg(*this); }

	template <class... Tags>
	static void check_hit(int i, const std::vector<Ray> &rays, std::vector<HitRegA<Tags...>> &hits,
	                      int N, BVH2 &m_bvh) {
		for (int r = 0; r < N; r++) {
			auto hit_reg = m_bvh.nearest_tri(rays[i + r]);
			if constexpr (has_tag<filter::t, Tags...>) {
				hits[i + r].t = hit_reg.t;
			}
			if constexpr (has_tag<filter::uv, Tags...>) {
				hits[i + r].u = hit_reg.u;
				hits[i + r].v = hit_reg.v;
			}
			if constexpr (has_tag<filter::primitive_id, Tags...>) {
				hits[i + r].primitive_id = hit_reg.primitive_id;
			}
		}
	}

	template <class... Tags>
	std::vector<HitRegA<Tags...>> nearest_hits(const std::vector<Ray> &rays) {
		std::vector<HitRegA<Tags...>> hits(rays.size());

		clear_affinity();

		unsigned n = std::thread::hardware_concurrency();
		std::vector<std::thread> threads;
		threads.reserve(n);

		int rays_per_thread = rays.size() / n;

		for (unsigned i = 0; i < n; ++i) {
			threads.emplace_back(check_hit<Tags...>, i * rays_per_thread, std::cref(rays),
			                     std::ref(hits), rays_per_thread, std::ref(m_bvh));
		}

		for (auto &thread : threads)
			thread.join();

		check_hit<Tags...>(n * rays_per_thread, rays, hits, rays.size() % n, m_bvh);

		return hits;
	}

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