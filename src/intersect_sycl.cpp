#include <sycl/sycl.hpp>

#include "../include/portableRT/intersect_sycl.hpp"
#include "../include/portableRT/nearesthit_inst.hpp"

namespace portableRT {

struct SYCLBackendImpl {
	sycl::queue m_q;
	sycl::device m_dev;
};

SYCLBackend::SYCLBackend() : InvokableBackend("SYCL") { static RegisterBackend reg(*this); }

SYCLBackend::~SYCLBackend() = default;

void SYCLBackend::init() {
	m_impl = std::make_unique<SYCLBackendImpl>();
	m_impl->m_dev = sycl::device(sycl::default_selector_v);
	m_impl->m_q = sycl::queue(m_impl->m_dev);
}

// TODO: free this
void SYCLBackend::set_tris(const Tris &tris) {
	m_bvh.build(tris);

	m_dbvh = sycl::malloc_device<BVH2>(1, m_impl->m_q);
	Tri *dev_tris = sycl::malloc_device<Tri>(tris.size(), m_impl->m_q);
	m_impl->m_q.memcpy(dev_tris, tris.data(), sizeof(Tri) * tris.size()).wait();

	BVH2::Node *dev_nodes = sycl::malloc_device<BVH2::Node>(m_bvh.m_node_count, m_impl->m_q);
	m_impl->m_q.memcpy(dev_nodes, m_bvh.m_nodes, sizeof(BVH2::Node) * m_bvh.m_node_count).wait();

	m_impl->m_q.memcpy(m_dbvh, &m_bvh, sizeof(BVH2)).wait();
	m_impl->m_q.memcpy(&(m_dbvh->m_tris), &(dev_tris), sizeof(Tri *)).wait();
	m_impl->m_q.memcpy(&(m_dbvh->m_nodes), &(dev_nodes), sizeof(BVH2::Node *)).wait();
}

void SYCLBackend::shutdown() { m_impl.reset(); }

template <class... Tags>
std::vector<HitReg<Tags...>> SYCLBackend::nearest_hits(const std::vector<Ray> &rays) {
	try {
		HitReg<Tags...> *res = sycl::malloc_shared<HitReg<Tags...>>(rays.size(), m_impl->m_q);
		Ray *rays_dev = sycl::malloc_device<Ray>(rays.size(), m_impl->m_q);
		m_impl->m_q.memcpy(rays_dev, rays.data(), sizeof(Ray) * rays.size()).wait();

		BVH2 *bvh = m_dbvh;

		m_impl->m_q
		    .submit([&](sycl::handler &cgh) {
			    cgh.parallel_for(sycl::range<1>(rays.size()), [=](sycl::id<1> id) {
				    res[id] = bvh->nearest_tri<Tags...>(rays_dev[id]);
			    });
		    })
		    .wait();

		std::vector<HitReg<Tags...>> hits(res, res + rays.size());
		sycl::free(res, m_impl->m_q);
		sycl::free(rays_dev, m_impl->m_q);
		return hits;
	} catch (sycl::_V1::exception &e) {
		std::cout << e.what() << std::endl;
		HitReg<Tags...> hit;
		hit.valid = false;
		return std::vector<HitReg<Tags...>>(rays.size(), hit);
	}
}

// Manual instantiation
#define X(...) INSTANTIATE_HITREG(portableRT::SYCLBackend, __VA_ARGS__)
TAG_COMBOS
#undef X

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