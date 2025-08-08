#pragma once
#include "backend.hpp"
#include "bvh.hpp"
#include "core.hpp"

namespace portableRT {

class CPUBackend : public InvokableBackend<CPUBackend> {
  public:
	CPUBackend() : InvokableBackend("CPU") { static RegisterBackend reg(*this); }

	template <class... Tags>
	static void check_hit(int i, const std::vector<Ray> &rays, std::vector<HitReg<Tags...>> &hits,
	                      int N, BVH2 &m_bvh) {
		for (int r = 0; r < N; r++) {
			hits[i + r] = m_bvh.template nearest_tri<Tags...>(rays[i + r]);
		}
	}

	template <class... Tags>
	std::vector<HitReg<Tags...>> nearest_hits(const std::vector<Ray> &rays) {
		std::vector<HitReg<Tags...>> hits(rays.size());

		int n = std::thread::hardware_concurrency();
		int rays_per_thread = rays.size() / n;

		if (rays_per_thread != 0) {

			clear_affinity();
			threads.clear();
			threads.reserve(n);

			for (unsigned i = 0; i < n; ++i) {
				threads.emplace_back(check_hit<Tags...>, i * rays_per_thread, std::cref(rays),
				                     std::ref(hits), rays_per_thread, std::ref(m_bvh));
			}
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
	std::vector<std::thread> threads;
};

static CPUBackend cpu_backend;

} // namespace portableRT