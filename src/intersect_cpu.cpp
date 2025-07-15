#include "../include/portableRT/intersect_cpu.hpp"
#include <array>
#include <fstream>
#include <functional>
#include <thread>

namespace portableRT {

void check_hit(int i, BVH *m_bvh, const std::vector<Ray> &rays,
               std::vector<HitReg> &hits, int N) {
  for (int r = 0; r < N; r++) {
    Hit hit;
    hit.valid = false;
    m_bvh->transverse(rays[i + r], hit);

    HitReg hitReg;
    hitReg.t = hit.valid ? hit.t : std::numeric_limits<float>::infinity();
    hitReg.primitive_id = hit.valid ? hit.triIdx : -1;

    hits[i + r] = hitReg;
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
    threads.emplace_back(check_hit, i * rays_per_thread, m_bvh, std::cref(rays),
                         std::ref(hits), rays_per_thread);
  }

  for (auto &thread : threads)
    thread.join();

  check_hit(n * rays_per_thread, m_bvh, rays, hits, rays.size() % n);

  return hits;
}

bool CPUBackend::is_available() const { return true; }
void CPUBackend::init() {}
void CPUBackend::shutdown() {}
void CPUBackend::set_tris(const Tris &tris) {

  m_tris = tris;

  m_bvh = new BVH();
  m_bvh->triIndices = new int[tris.size()];
  m_bvh->tris = m_tris.data();
  m_bvh->build(&tris);
}

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
