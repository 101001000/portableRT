#include <sycl/sycl.hpp>

#include "../include/portableRT/core.hpp"
#include "../include/portableRT/intersect_embree_sycl.hpp"

void enablePersistentJITCache() {
#if defined(_WIN32)
	_putenv_s("SYCL_CACHE_PERSISTENT", "1");
	_putenv_s("SYCL_CACHE_DIR", "cache");
#else
	setenv("SYCL_CACHE_PERSISTENT", "1", 1);
	setenv("SYCL_CACHE_DIR", "cache", 1);
#endif
}

const sycl::specialization_id<RTCFeatureFlags> feature_mask;
const RTCFeatureFlags required_features = RTC_FEATURE_FLAG_TRIANGLE;

struct Result {
	unsigned geomID;
	unsigned primID;
	float tfar;
};

template <typename T>
T *alignedSYCLMallocDeviceReadWrite(const sycl::queue &queue, size_t count, size_t align) {
	if (count == 0)
		return nullptr;

	assert((align & (align - 1)) == 0);
	T *ptr = (T *)sycl::aligned_alloc(align, count * sizeof(T), queue, sycl::usm::alloc::shared);
	if (count != 0 && ptr == nullptr)
		throw std::bad_alloc();

	return ptr;
}

template <typename T>
T *alignedSYCLMallocDeviceReadOnly(const sycl::queue &queue, size_t count, size_t align) {
	if (count == 0)
		return nullptr;

	assert((align & (align - 1)) == 0);
	T *ptr = (T *)sycl::aligned_alloc_shared(align, count * sizeof(T), queue,
	                                         sycl::ext::oneapi::property::usm::device_read_only());
	if (count != 0 && ptr == nullptr)
		throw std::bad_alloc();

	return ptr;
}

void alignedSYCLFree(const sycl::queue &queue, void *ptr) {
	if (ptr)
		sycl::free(ptr, queue);
}

inline void errorFunctionSYCL(void *userPtr, enum RTCError error, const char *str) {
	printf("error %s: %s\n", rtcGetErrorString(error), str);
}

RTCDevice initializeDevice(sycl::context &sycl_context, sycl::device &sycl_device) {
	RTCDevice device = rtcNewSYCLDevice(sycl_context, "");
	rtcSetDeviceSYCLDevice(device, sycl_device);

	if (!device) {
		printf("error %s: cannot create device. (reason: %s)\n",
		       rtcGetErrorString(rtcGetDeviceError(NULL)), rtcGetDeviceLastErrorMessage(NULL));
		exit(1);
	}

	rtcSetDeviceErrorFunction(device, errorFunctionSYCL, NULL);
	return device;
}

void castRay(sycl::queue &queue, const RTCTraversable traversable,
             const std::vector<portableRT::Ray> &rays, Result *results) {

	portableRT::Ray *d_rays = sycl::malloc_device<portableRT::Ray>(rays.size(), queue);
	queue.memcpy(d_rays, rays.data(), sizeof(portableRT::Ray) * rays.size());

	queue.submit([=](sycl::handler &cgh) {
		cgh.set_specialization_constant<feature_mask>(required_features);

		cgh.parallel_for(
		    sycl::range<1>(rays.size()), [=](sycl::item<1> item, sycl::kernel_handler kh) {
			    /*
			     * The intersect arguments can be used to pass a feature
			     * mask, which improves performance and JIT compile times
			     * on the GPU
			     */
			    RTCIntersectArguments args;
			    rtcInitIntersectArguments(&args);

			    const RTCFeatureFlags features = kh.get_specialization_constant<feature_mask>();
			    args.feature_mask = features;

			    /*
			     * The ray hit structure holds both the ray and the hit.
			     * The user must initialize it properly -- see API
			     * documentation for rtcIntersect1() for details.
			     */
			    struct RTCRayHit rayhit;
			    rayhit.ray.org_x = d_rays[item].origin[0];
			    rayhit.ray.org_y = d_rays[item].origin[1];
			    rayhit.ray.org_z = d_rays[item].origin[2];
			    rayhit.ray.dir_x = d_rays[item].direction[0];
			    rayhit.ray.dir_y = d_rays[item].direction[1];
			    rayhit.ray.dir_z = d_rays[item].direction[2];
			    rayhit.ray.tnear = 0;
			    rayhit.ray.tfar = std::numeric_limits<float>::infinity();
			    rayhit.ray.mask = -1;
			    rayhit.ray.flags = 0;
			    rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
			    rayhit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;

			    /*
			     * There are multiple variants of rtcIntersect. This one
			     * intersects a single ray with the scene.
			     */
			    rtcTraversableIntersect1(traversable, &rayhit, &args);

			    /*
			     * write hit result to output buffer
			     */
			    results[item].geomID = rayhit.hit.geomID;
			    results[item].primID = rayhit.hit.primID;
			    results[item].tfar = rayhit.ray.tfar;
		    });
	});
	queue.wait_and_throw();
	sycl::free(d_rays, queue);
}

namespace portableRT {

struct EmbreeSYCLBackendImpl {
	sycl::queue m_q;
	sycl::device m_dev;
	sycl::context m_context;
};

void EmbreeSYCLBackend::set_tris(const Tris &tris) {
	if (m_geom_id != RTC_INVALID_GEOMETRY_ID) {
		rtcDetachGeometry(m_rtcscene, m_geom_id);
		rtcReleaseGeometry(m_tri);
		// TODO: Fix leaking
		// alignedSYCLFree(m_impl->m_q, m_vertices);
		// alignedSYCLFree(m_impl->m_q, m_indices);
		m_geom_id = RTC_INVALID_GEOMETRY_ID;
	}

	const std::size_t nTris = tris.size();
	const std::size_t nVerts = nTris * 3;

	m_tri = rtcNewGeometry(m_rtcdevice, RTC_GEOMETRY_TYPE_TRIANGLE);
	float *vertices = alignedSYCLMallocDeviceReadWrite<float>(m_impl->m_q, nVerts * 3, 64);
	unsigned int *indices = alignedSYCLMallocDeviceReadOnly<unsigned>(m_impl->m_q, nTris * 3, 64);

	for (std::size_t i = 0; i < nTris; ++i) {
		std::memcpy(vertices + i * 9, tris[i].data(), 9 * sizeof(float));
		indices[i * 3 + 0] = static_cast<unsigned>(i * 3 + 0);
		indices[i * 3 + 1] = static_cast<unsigned>(i * 3 + 1);
		indices[i * 3 + 2] = static_cast<unsigned>(i * 3 + 2);
	}

	rtcSetSharedGeometryBuffer(m_tri, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, vertices, 0,
	                           sizeof(float) * 3, nVerts);

	rtcSetSharedGeometryBuffer(m_tri, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, indices, 0,
	                           sizeof(unsigned) * 3, nTris);

	rtcCommitGeometry(m_tri);
	m_geom_id = rtcAttachGeometry(m_rtcscene, m_tri);
	rtcCommitScene(m_rtcscene);

	m_rtctraversable = rtcGetSceneTraversable(m_rtcscene);
}

std::vector<HitReg> EmbreeSYCLBackend::nearest_hits(const std::vector<Ray> &rays) {
	try {

		std::vector<HitReg> hits;
		Result *result = alignedSYCLMallocDeviceReadWrite<Result>(m_impl->m_q, rays.size(), 16);
		// result->geomID = RTC_INVALID_GEOMETRY_ID;

		/* This will hit the triangle at t=1. */
		castRay(m_impl->m_q, m_rtctraversable, rays, result);

		for (int i = 0; i < rays.size(); ++i) {

			HitReg hit;

			if (result[i].geomID == RTC_INVALID_GEOMETRY_ID) {
				hit.t = std::numeric_limits<float>::infinity();
				hit.primitive_id = static_cast<uint32_t>(-1);
			} else {
				hit.t = result[i].tfar;
				hit.primitive_id = result[i].primID;
			}

			hits.push_back(hit);
		}
		alignedSYCLFree(m_impl->m_q, result);
		return hits;

	} catch (sycl::_V1::exception &e) {
		std::cout << e.what() << std::endl;
		return std::vector<HitReg>(
		    rays.size(), HitReg{std::numeric_limits<float>::infinity(), static_cast<uint32_t>(-1)});
	}
}

bool EmbreeSYCLBackend::is_available() const {
	try {
		sycl::device gpu{rtcSYCLDeviceSelector};

		if (!rtcIsSYCLDeviceSupported(gpu))
			return false;

		return true;
	} catch (const sycl::exception &) {
		return false;
	}
}

EmbreeSYCLBackend::EmbreeSYCLBackend() : InvokableBackend("Embree SYCL") {
	static RegisterBackend reg(*this);
}

EmbreeSYCLBackend::~EmbreeSYCLBackend() = default;

void EmbreeSYCLBackend::init() {

	m_impl = std::make_unique<EmbreeSYCLBackendImpl>();
	enablePersistentJITCache();

	m_impl->m_dev = sycl::device(rtcSYCLDeviceSelector);

	m_impl->m_q = sycl::queue(m_impl->m_dev);
	m_impl->m_context = sycl::context(m_impl->m_dev);

	m_rtcdevice = initializeDevice(m_impl->m_context, m_impl->m_dev);
	m_rtcscene = rtcNewScene(m_rtcdevice);
}

void EmbreeSYCLBackend::shutdown() {
	rtcReleaseGeometry(m_tri);
	rtcReleaseScene(m_rtcscene);
	rtcReleaseDevice(m_rtcdevice);
	m_impl.reset();
}

std::string EmbreeSYCLBackend::device_name() const {
	return m_impl->m_dev.get_info<sycl::info::device::name>();
}

} // namespace portableRT