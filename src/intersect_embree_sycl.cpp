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
T *alignedSYCLMallocDeviceReadWrite(const sycl::queue &queue, size_t count,
                                    size_t align) {
  if (count == 0)
    return nullptr;

  assert((align & (align - 1)) == 0);
  T *ptr = (T *)sycl::aligned_alloc(align, count * sizeof(T), queue,
                                    sycl::usm::alloc::shared);
  if (count != 0 && ptr == nullptr)
    throw std::bad_alloc();

  return ptr;
}

template <typename T>
T *alignedSYCLMallocDeviceReadOnly(const sycl::queue &queue, size_t count,
                                   size_t align) {
  if (count == 0)
    return nullptr;

  assert((align & (align - 1)) == 0);
  T *ptr = (T *)sycl::aligned_alloc_shared(
      align, count * sizeof(T), queue,
      sycl::ext::oneapi::property::usm::device_read_only());
  if (count != 0 && ptr == nullptr)
    throw std::bad_alloc();

  return ptr;
}

void alignedSYCLFree(const sycl::queue &queue, void *ptr) {
  if (ptr)
    sycl::free(ptr, queue);
}

inline void errorFunctionSYCL(void *userPtr, enum RTCError error,
                              const char *str) {
  printf("error %s: %s\n", rtcGetErrorString(error), str);
}

RTCDevice initializeDevice(sycl::context &sycl_context,
                           sycl::device &sycl_device) {
  RTCDevice device = rtcNewSYCLDevice(sycl_context, "");
  rtcSetDeviceSYCLDevice(device, sycl_device);

  if (!device) {
    printf("error %s: cannot create device. (reason: %s)\n",
           rtcGetErrorString(rtcGetDeviceError(NULL)),
           rtcGetDeviceLastErrorMessage(NULL));
    exit(1);
  }

  rtcSetDeviceErrorFunction(device, errorFunctionSYCL, NULL);
  return device;
}

void castRay(sycl::queue &queue, const RTCTraversable traversable, float ox,
             float oy, float oz, float dx, float dy, float dz, Result *result) {
  queue.submit([=](sycl::handler &cgh) {
    cgh.set_specialization_constant<feature_mask>(required_features);

    cgh.parallel_for(sycl::range<1>(1),
                     [=](sycl::item<1> item, sycl::kernel_handler kh) {
                       /*
                        * The intersect arguments can be used to pass a feature
                        * mask, which improves performance and JIT compile times
                        * on the GPU
                        */
                       RTCIntersectArguments args;
                       rtcInitIntersectArguments(&args);

                       const RTCFeatureFlags features =
                           kh.get_specialization_constant<feature_mask>();
                       args.feature_mask = features;

                       /*
                        * The ray hit structure holds both the ray and the hit.
                        * The user must initialize it properly -- see API
                        * documentation for rtcIntersect1() for details.
                        */
                       struct RTCRayHit rayhit;
                       rayhit.ray.org_x = ox;
                       rayhit.ray.org_y = oy;
                       rayhit.ray.org_z = oz;
                       rayhit.ray.dir_x = dx;
                       rayhit.ray.dir_y = dy;
                       rayhit.ray.dir_z = dz;
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
                       result->geomID = rayhit.hit.geomID;
                       result->primID = rayhit.hit.primID;
                       result->tfar = rayhit.ray.tfar;
                     });
  });
  queue.wait_and_throw();
}

namespace portableRT {

struct EmbreeSYCLBackendImpl {
  sycl::queue m_q;
  sycl::device m_dev;
  sycl::context m_context;
};

void EmbreeSYCLBackend::initializeScene() {
  m_tri = rtcNewGeometry(m_rtcdevice, RTC_GEOMETRY_TYPE_TRIANGLE);

  float *vertices =
      alignedSYCLMallocDeviceReadWrite<float>(m_impl->m_q, 3 * 3, 16);

  rtcSetSharedGeometryBuffer(m_tri, RTC_BUFFER_TYPE_VERTEX, 0,
                             RTC_FORMAT_FLOAT3, vertices, 0, 3 * sizeof(float),
                             3);

  unsigned *indices =
      alignedSYCLMallocDeviceReadOnly<unsigned>(m_impl->m_q, 3, 16);

  rtcSetSharedGeometryBuffer(m_tri, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3,
                             indices, 0, 3 * sizeof(unsigned), 1);

  if (vertices && indices) {
    vertices[0] = 0;
    vertices[1] = 0;
    vertices[2] = 0;
    vertices[3] = 0;
    vertices[4] = 0;
    vertices[5] = 0;
    vertices[6] = 0;
    vertices[7] = 0;
    vertices[8] = 0;

    indices[0] = 0;
    indices[1] = 1;
    indices[2] = 2;
  }

  rtcCommitGeometry(m_tri);

  rtcAttachGeometry(m_rtcscene, m_tri);
  rtcCommitScene(m_rtcscene);
}

bool EmbreeSYCLBackend::intersect_tri(const std::array<float, 9> &tri,
                                      const Ray &ray) {
  try {

    float *vertices =
        (float *)rtcGetGeometryBufferData(m_tri, RTC_BUFFER_TYPE_VERTEX, 0);
    for (int i = 0; i < 9; i++) {
      vertices[i] = tri[i];
    }
    rtcCommitGeometry(m_tri);
    rtcCommitScene(m_rtcscene);
    m_rtctraversable = rtcGetSceneTraversable(m_rtcscene);

    Result *result =
        alignedSYCLMallocDeviceReadWrite<Result>(m_impl->m_q, 1, 16);
    result->geomID = RTC_INVALID_GEOMETRY_ID;

    /* This will hit the triangle at t=1. */
    castRay(m_impl->m_q, m_rtctraversable, ray.origin[0], ray.origin[1],
            ray.origin[2], ray.direction[0], ray.direction[1], ray.direction[2],
            result);

    bool res = result->geomID != RTC_INVALID_GEOMETRY_ID;

    alignedSYCLFree(m_impl->m_q, result);

    return res;

  } catch (sycl::_V1::exception &e) {
    std::cout << e.what() << std::endl;
    return false;
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

EmbreeSYCLBackend::EmbreeSYCLBackend()
    : InvokableBackend("Embree SYCL") {
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
  initializeScene();
  m_rtctraversable = rtcGetSceneTraversable(m_rtcscene);
}

void EmbreeSYCLBackend::shutdown() {
  rtcReleaseGeometry(m_tri);
  rtcReleaseScene(m_rtcscene);
  rtcReleaseDevice(m_rtcdevice);
  m_impl.reset();
}

} // namespace portableRT