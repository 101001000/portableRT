#include <sycl/sycl.hpp>

#include "../include/portableRT/intersect_sycl.hpp"

namespace portableRT {

struct SYCLBackendImpl {
  sycl::queue m_q;
  sycl::device m_dev;
};

SYCLBackend::SYCLBackend() : InvokableBackend("SYCL") {
  static RegisterBackend reg(*this);
}

SYCLBackend::~SYCLBackend() = default;

void SYCLBackend::init() {
  m_impl = std::make_unique<SYCLBackendImpl>();
  m_impl->m_dev = sycl::device(sycl::default_selector_v);
  m_impl->m_q = sycl::queue(m_impl->m_dev);
}

// TODO: free this
void SYCLBackend::set_tris(const Tris &tris) {
  m_tris = tris;

  m_bvh = new BVH();
  m_bvh->triIndices = new int[tris.size()];
  m_bvh->tris = m_tris.data();
  m_bvh->build(&tris);

  m_dbvh = sycl::malloc_device<BVH>(1, m_impl->m_q);

  int *dev_triIndices = sycl::malloc_device<int>(tris.size(), m_impl->m_q);
  m_impl->m_q
      .memcpy(dev_triIndices, m_bvh->triIndices, sizeof(int) * tris.size())
      .wait();

  Tri *dev_tris = sycl::malloc_device<Tri>(tris.size(), m_impl->m_q);
  m_impl->m_q.memcpy(dev_tris, m_tris.data(), sizeof(Tri) * tris.size()).wait();

  m_impl->m_q.memcpy(m_dbvh, m_bvh, sizeof(BVH)).wait();
  m_impl->m_q.memcpy(&(m_dbvh->tris), &(dev_tris), sizeof(Tri *)).wait();
  m_impl->m_q.memcpy(&(m_dbvh->triIndices), &(dev_triIndices), sizeof(int *))
      .wait();
}

void SYCLBackend::shutdown() { m_impl.reset(); }

bool SYCLBackend::intersect_tris(const Ray &ray) {
  try {
    bool *res = sycl::malloc_shared<bool>(1, m_impl->m_q);

    BVH *bvh = m_dbvh;

    m_impl->m_q
        .submit([&](sycl::handler &cgh) {
          cgh.single_task([=]() {
            Hit hit;
            hit.valid = false;
            bvh->transverse(ray, hit);
            *res = hit.valid;
          });
        })
        .wait();

    bool hit = *res;
    sycl::free(res, m_impl->m_q);
    if (hit)
      return true;
  } catch (sycl::_V1::exception &e) {
    std::cout << e.what() << std::endl;
  }
  return false;
}

bool SYCLBackend::is_available() const {
  try {
    auto devices = sycl::device::get_devices();
    return !devices.empty();
  } catch (const sycl::exception &e) {
    return false;
  }
}
} // namespace portableRT