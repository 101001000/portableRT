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

std::vector<HitReg> SYCLBackend::nearest_hits(const std::vector<Ray> &rays) {
  try {
    HitReg *res = sycl::malloc_shared<HitReg>(rays.size(), m_impl->m_q);
    Ray *rays_dev = sycl::malloc_device<Ray>(rays.size(), m_impl->m_q);
    m_impl->m_q.memcpy(rays_dev, rays.data(), sizeof(Ray) * rays.size()).wait();

    BVH *bvh = m_dbvh;

    m_impl->m_q
        .submit([&](sycl::handler &cgh) {
          cgh.parallel_for(sycl::range<1>(rays.size()), [=](sycl::id<1> id) {
            Hit hit;
            hit.valid = false;
            bvh->transverse(rays_dev[id], hit);
            res[id].t = hit.valid ? hit.t : std::numeric_limits<float>::infinity();
            res[id].primitive_id = hit.valid ? hit.triIdx : static_cast<uint32_t>(-1);
          });
        })
        .wait();

    std::vector<HitReg> hits(rays.size());
    m_impl->m_q.memcpy(hits.data(), res, sizeof(HitReg) * rays.size()).wait();
    sycl::free(res, m_impl->m_q);
    sycl::free(rays_dev, m_impl->m_q);
    return hits;
  } catch (sycl::_V1::exception &e) {
    std::cout << e.what() << std::endl;
    return std::vector<HitReg>(rays.size(),
                              HitReg{std::numeric_limits<float>::infinity(), static_cast<uint32_t>(-1)});
  }
}

bool SYCLBackend::is_available() const {
  try {
    auto devices = sycl::device::get_devices();
    return !devices.empty();
  } catch (const sycl::exception &e) {
    return false;
  }
}

std::string SYCLBackend::device_name() const {
  return m_impl->m_dev.get_info<sycl::info::device::name>();
}

} // namespace portableRT