#include "../include/portableRT/intersect_cpu.hpp"
#include <array>

namespace portableRT {

std::vector<float> CPUBackend::nearest_hits(const std::vector<Ray> &rays) {
  std::vector<float> hits(rays.size());
  for (size_t i = 0; i < rays.size(); i++) {
    Hit hit;
    hit.valid = false;
    m_bvh->transverse(rays[i], hit);
    hits[i] = hit.valid ? hit.t : std::numeric_limits<float>::infinity();
  }
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

} // namespace portableRT
