#include "../include/portableRT/intersect_cpu.hpp"
#include <array>
#include <fstream>
#include <functional>
#include <thread>

namespace portableRT {

void check_hit(int i, const std::vector<Ray> &rays, std::vector<HitReg> &hits, int N, BVH2 &m_bvh) {
	for (int r = 0; r < N; r++) {
		auto hit_reg = m_bvh.nearest_tri(rays[i + r]);
		hits[i + r] = hit_reg;
	}
}

std::vector<HitReg> CPUBackend::nearest_hits(const std::vector<Ray> &rays) {
	std::vector<HitReg> hits(rays.size());

	clear_affinity();

	unsigned n = std::thread::hardware_concurrency();
	std::vector<std::thread> threads;
	threads.reserve(n);

	int rays_per_thread = rays.size() / n;

	for (unsigned i = 0; i < n; ++i) {
		threads.emplace_back(check_hit, i * rays_per_thread, std::cref(rays), std::ref(hits),
		                     rays_per_thread, std::ref(m_bvh));
	}

	for (auto &thread : threads)
		thread.join();

	check_hit(n * rays_per_thread, rays, hits, rays.size() % n, m_bvh);

	return hits;
}

bool CPUBackend::is_available() const { return true; }
void CPUBackend::init() {}
void CPUBackend::shutdown() {}
void CPUBackend::set_tris(const Tris &tris) { m_bvh.build(tris); }

std::string CPUBackend::device_name() const {
	std::ifstream cpuinfo("/proc/cpuinfo");
	std::string line;
	while (std::getline(cpuinfo, line)) {
		if (line.rfind("model name", 0) == 0) {
			auto pos = line.find(':');
			if (pos != std::string::npos) {
				size_t start = line.find_first_not_of(" \t", pos + 1);
				return line.substr(start);
			}
		}
	}
	return "unsupported";
}

} // namespace portableRT
