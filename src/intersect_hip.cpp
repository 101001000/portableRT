#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <hip/hip_runtime.h>
#include <iostream>

#include "../include/portableRT/intersect_hip.hpp"

// https://github.com/GPUOpen-LibrariesAndSDKs/HIPRT/blob/main/hiprt/impl/hiprt_device_impl.h
// https://github.com/GPUOpen-LibrariesAndSDKs/HIPRT/blob/main/hiprt/impl/BvhNode.h

#define CHK(x)                                                                 \
  if ((x) != hipSuccess) {                                                     \
    std::cerr << "HIP " << hipGetErrorString(x) << '\n';                       \
    return 0;                                                                  \
  }

constexpr uint32_t InvalidValue = ~0u;

struct Aabb {
  Aabb() {}
  Aabb(float3 min, float3 max) : m_min(min), m_max(max) {}
  float3 m_min;
  float3 m_max;
};

struct alignas(64) BoxNode {
  uint32_t m_childIndex0 = InvalidValue;
  uint32_t m_childIndex1 = InvalidValue;
  uint32_t m_childIndex2 = InvalidValue;
  uint32_t m_childIndex3 = InvalidValue;
  Aabb m_box0;
  Aabb m_box1;
  Aabb m_box2;
  Aabb m_box3;
  uint32_t m_parentAddr = InvalidValue;
  uint32_t m_updateCounter = 0;
  uint32_t m_childCount = 2;
};
static_assert(sizeof(BoxNode) == 128);

struct alignas(alignof(float3)) TrianglePair {
  float3 m_v0;
  float3 m_v1;
  float3 m_v2;
  float3 m_v3;
};
struct alignas(64) TriangleNode {
  TrianglePair m_triPair;
  uint32_t padding;
  uint32_t m_primIndex0 = InvalidValue;
  uint32_t m_primIndex1 = InvalidValue;
  uint32_t m_flags;
};
static_assert(sizeof(TriangleNode) == 64);

__host__ __device__ uint4 make_descriptor(const void *nodes) {
  uint32_t boxSortHeuristic = 0u;
  uint32_t boxGrowUlp = 6u;
  uint32_t boxSortEnabled = 1u;
  uint64_t size = -1ull;
  uint32_t bigPage = 0u;
  uint32_t llcNoalloc = 0u;

  uint4 descriptor;
  uint32_t triangleReturnMode = 0u;
  uint64_t baseAddress = reinterpret_cast<const uint64_t>(nodes);
  baseAddress = (baseAddress >> 8ull) & 0xffffffffffull;
  boxSortHeuristic &= 0x3;
  boxGrowUlp &= 0xff;
  boxSortEnabled &= 0x1;
  size &= 0x3ffffffffffull;
  bigPage &= 0x1;
  uint32_t type = 0x8;
  llcNoalloc &= 0x3;
  descriptor.x = baseAddress & 0xffffffff;
  descriptor.y = (baseAddress >> 32ull) | (boxSortHeuristic << 21u) |
                 (boxGrowUlp << 23u) | (boxSortEnabled << 31u);
  descriptor.z = size & 0xffffffff;
  descriptor.w = (size >> 32ull) | (triangleReturnMode << 24u) |
                 (llcNoalloc << 25u) | (bigPage << 27u) | (type << 28u);
  return descriptor;
}

static uint64_t encodeBaseAddr(const void *baseAddr, uint32_t nodeIndex) {
  uint64_t baseIndex = reinterpret_cast<uint64_t>(baseAddr) >> 3ull;
  return baseIndex + nodeIndex;
}

__global__ void kernel(void *bvh, uint4 *out, float4 ray_o, float4 ray_d) {

  uint4 desc = make_descriptor(bvh);

  float ray_extent = 100;

  float4 ray_id;
  ray_id.x = 1 / (ray_d.x + 0.001);
  ray_id.y = 1 / (ray_d.y + 0.001);
  ray_id.z = 1 / (ray_d.z + 0.001);
  // V4Ui Ui f V4f V4f V4f V4Ui
  auto res = __builtin_amdgcn_image_bvh_intersect_ray_l(
      0, ray_extent, ray_o.data, ray_d.data, ray_id.data, desc.data);
  out->x = res[0];
  out->y = res[1];
  out->z = res[2];
  out->w = res[3];
}

namespace portableRT {

void HIPBackend::set_tris(const Tris &tris) { m_tris = tris; }

bool HIPBackend::intersect_tris(const Ray &ray) {

  for (int i = 0; i < m_tris.size(); ++i) {
    std::array<float, 9> v = m_tris[i];

    float3 min_b;
    min_b.x = std::min(v[0], std::min(v[3], v[6]));
    min_b.y = std::min(v[1], std::min(v[4], v[7]));
    min_b.z = std::min(v[2], std::min(v[5], v[8]));

    float3 max_b;
    max_b.x = std::max(v[0], std::max(v[3], v[6]));
    max_b.y = std::max(v[1], std::max(v[4], v[7]));
    max_b.z = std::max(v[2], std::max(v[5], v[8]));

    TriangleNode leaf{};

    leaf.m_triPair.m_v0 = {v[0], v[1], v[2]};
    leaf.m_triPair.m_v1 = {v[3], v[4], v[5]};
    leaf.m_triPair.m_v2 = {v[6], v[7], v[8]};

    leaf.m_flags = (2 << 2) | (1 << 0);

    void *d_bvh;
    CHK(hipMalloc(&d_bvh, sizeof(TriangleNode)));
    CHK(hipMemcpy(d_bvh, &leaf, sizeof(TriangleNode), hipMemcpyHostToDevice));

    uint4 *dHit;
    CHK(hipMalloc(&dHit, sizeof(uint4)));

    kernel<<<1, 1>>>(d_bvh, dHit,
                     {ray.origin[0], ray.origin[1], ray.origin[2], 0},
                     {ray.direction[0], ray.direction[1], ray.direction[2], 0});
    CHK(hipDeviceSynchronize());
    uint4 hv;
    CHK(hipMemcpy(&hv, dHit, sizeof(uint4), hipMemcpyDeviceToHost));

    CHK(hipFree(d_bvh));
    CHK(hipFree(dHit));

    if (hv.w)
      return true;
  }

  return false;
}

void HIPBackend::init() {}

void HIPBackend::shutdown() {}

bool HIPBackend::is_available() const {

  if (hipInit(0) != hipSuccess)
    return false;
  int devCount = 0;
  if (hipGetDeviceCount(&devCount) != hipSuccess || devCount == 0)
    return false;

  for (int id = 0; id < devCount; ++id) {
    hipDeviceProp_t prop{};
    if (hipGetDeviceProperties(&prop, id) != hipSuccess)
      continue;

    const std::string arch = prop.gcnArchName;

    if ((arch.rfind("gfx103", 0) == 0 || arch.rfind("gfx110", 0) == 0))
      return true;
  }
  return false;
}

} // namespace portableRT