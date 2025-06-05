#include "../include/portableRT/intersect_cpu.hpp"
#include <array>

namespace portableRT {

bool CPUBackend::intersect_tris(const Ray &ray) {
  Hit hit;
  hit.valid = false;
  m_bvh->transverse(ray, hit);
  return hit.valid;
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
