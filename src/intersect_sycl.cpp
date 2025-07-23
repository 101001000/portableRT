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
  m_bvh.build(tris);

  m_dbvh = sycl::malloc_device<temp::BVH2>(1, m_impl->m_q);
  Tri *dev_tris = sycl::malloc_device<Tri>(tris.size(), m_impl->m_q);
  m_impl->m_q.memcpy(dev_tris, tris.data(), sizeof(Tri) * tris.size()).wait();

  temp::BVH2::Node *dev_nodes = sycl::malloc_device<temp::BVH2::Node>(m_bvh.m_node_count, m_impl->m_q);
  m_impl->m_q.memcpy(dev_nodes, m_bvh.m_nodes, sizeof(temp::BVH2::Node) * m_bvh.m_node_count).wait();

  m_impl->m_q.memcpy(m_dbvh, &m_bvh, sizeof(temp::BVH2)).wait();
  m_impl->m_q.memcpy(&(m_dbvh->m_tris), &(dev_tris), sizeof(Tri *)).wait();
  m_impl->m_q.memcpy(&(m_dbvh->m_nodes), &(dev_nodes), sizeof(temp::BVH2::Node *))
      .wait();
}

void SYCLBackend::shutdown() { m_impl.reset(); }

std::vector<HitReg> SYCLBackend::nearest_hits(const std::vector<Ray> &rays) {
  try {
    HitReg *res = sycl::malloc_shared<HitReg>(rays.size(), m_impl->m_q);
    Ray *rays_dev = sycl::malloc_device<Ray>(rays.size(), m_impl->m_q);
    m_impl->m_q.memcpy(rays_dev, rays.data(), sizeof(Ray) * rays.size()).wait();

    temp::BVH2 *bvh = m_dbvh;

    m_impl->m_q
        .submit([&](sycl::handler &cgh) {
          cgh.parallel_for(sycl::range<1>(rays.size()), [=](sycl::id<1> id) {
            auto [nearest_tri_idx, t] = bvh->nearest_tri(rays_dev[id]);
            HitReg hitReg;
            hitReg.t = t;
            hitReg.primitive_id = nearest_tri_idx;
            res[id] = hitReg;
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
                               HitReg{std::numeric_limits<float>::infinity(),
                                      static_cast<uint32_t>(-1)});
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