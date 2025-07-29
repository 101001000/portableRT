#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <hip/hip_runtime.h>
#include <iostream>
#include <stack>

#include "../include/portableRT/intersect_hip.hpp"
#include "../include/portableRT/nearesthit_inst.hpp"

// https://github.com/GPUOpen-LibrariesAndSDKs/HIPRT/blob/main/hiprt/impl/hiprt_device_impl.h
// https://github.com/GPUOpen-LibrariesAndSDKs/HIPRT/blob/main/hiprt/impl/BvhNode.h

#define CHK(x)                                                                                     \
	if ((x) != hipSuccess) {                                                                       \
		std::cerr << "HIP " << hipGetErrorString(x) << '\n';                                       \
	}

constexpr uint32_t InvalidValue = ~0u;

struct Aabb {

	Aabb() { reset(); }
	Aabb(const portableRT::Tri &tri) {
		m_min.x = std::min(tri[0], std::min(tri[3], tri[6]));
		m_min.y = std::min(tri[1], std::min(tri[4], tri[7]));
		m_min.z = std::min(tri[2], std::min(tri[5], tri[8]));
		m_max.x = std::max(tri[0], std::max(tri[3], tri[6]));
		m_max.y = std::max(tri[1], std::max(tri[4], tri[7]));
		m_max.z = std::max(tri[2], std::max(tri[5], tri[8]));
	}
	Aabb(const Aabb &a, const Aabb &b) {
		m_min.x = std::min(a.m_min.x, b.m_min.x);
		m_min.y = std::min(a.m_min.y, b.m_min.y);
		m_min.z = std::min(a.m_min.z, b.m_min.z);
		m_max.x = std::max(a.m_max.x, b.m_max.x);
		m_max.y = std::max(a.m_max.y, b.m_max.y);
		m_max.z = std::max(a.m_max.z, b.m_max.z);
	}
	Aabb(float3 min, float3 max) : m_min(min), m_max(max) {}
	float3 m_min;
	float3 m_max;
	void reset() {
		m_min = float3(3.402823466e+38f);
		m_max = float3(-3.402823466e+38f);
	}
	void print() {
		std::cout << m_min.x << " " << m_min.y << " " << m_min.z << " " << m_max.x << " " << m_max.y
		          << " " << m_max.z << std::endl;
	}
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

__host__ __device__ uint4 make_descriptor(const void *nodes, uint64_t size) {
	uint32_t boxSortHeuristic = 0u;
	uint32_t boxGrowUlp = 6u;
	uint32_t boxSortEnabled = 1u;
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
	descriptor.y = (baseAddress >> 32ull) | (boxSortHeuristic << 21u) | (boxGrowUlp << 23u) |
	               (boxSortEnabled << 31u);
	descriptor.z = size & 0xffffffff;
	descriptor.w = (size >> 32ull) | (triangleReturnMode << 24u) | (llcNoalloc << 25u) |
	               (bigPage << 27u) | (type << 28u);
	return descriptor;
}

static uint32_t encodeNodeIndex(uint32_t nodeAddr, uint32_t nodeType) {
	if (nodeType == 5)
		nodeAddr *= 2;
	return (nodeAddr << 3) | nodeType;
}

__device__ __host__ static uint64_t encodeBaseAddr(const void *baseAddr, uint32_t nodeIndex) {
	uint64_t baseIndex = reinterpret_cast<uint64_t>(baseAddr) >> 3ull;
	return baseIndex + nodeIndex;
}

__device__ float2 barycentrics(float3 v0, float3 v1, float3 v2, float3 p) {
	float3 e0 = v1 - v0;
	float3 e1 = v2 - v0;
	float3 d = p - v0;

	float d00 = e0.x * e0.x + e0.y * e0.y + e0.z * e0.z;
	float d01 = e0.x * e1.x + e0.y * e1.y + e0.z * e1.z;
	float d11 = e1.x * e1.x + e1.y * e1.y + e1.z * e1.z;
	float d20 = d.x * e0.x + d.y * e0.y + d.z * e0.z;
	float d21 = d.x * e1.x + d.y * e1.y + d.z * e1.z;

	float denom = d00 * d11 - d01 * d01;

	const float tol = 1e-8f;
	if (fabsf(denom) < tol) {
		return {1.0f, 0.0f};
	}

	float v = (d11 * d20 - d01 * d21) / denom;
	float w = (d00 * d21 - d01 * d20) / denom;

	return {v, w};
}

__global__ void nearest_hit(void *bvh, portableRT::FullHitReg *out, portableRT::Ray *rays,
                            uint64_t size, uint64_t pos, uint64_t num_rays) {

	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= num_rays)
		return;

	float4 ray_o;
	ray_o.x = rays[idx].origin[0];
	ray_o.y = rays[idx].origin[1];
	ray_o.z = rays[idx].origin[2];

	float4 ray_d;
	ray_d.x = rays[idx].direction[0];
	ray_d.y = rays[idx].direction[1];
	ray_d.z = rays[idx].direction[2];

	float4 ray_id;
	ray_id.x = 1 / (rays[idx].direction[0]);
	ray_id.y = 1 / (rays[idx].direction[1]);
	ray_id.z = 1 / (rays[idx].direction[2]);

	uint4 desc = make_descriptor(bvh, size);
	uint64_t stack[64];
	stack[0] = pos;
	stack[1] = InvalidValue;
	int stack_ptr = 0;

	float t_near = std::numeric_limits<float>::infinity();
	uint32_t id = InvalidValue;
	float u, v;
	float3 p;

	while (stack_ptr >= 0) {

		uint32_t type = stack[stack_ptr] & 0x7;
		uint64_t node_dir = (static_cast<uint64_t>(stack[stack_ptr]) * 8);

		auto res = __builtin_amdgcn_image_bvh_intersect_ray_l(stack[stack_ptr], 100, ray_o.data,
		                                                      ray_d.data, ray_id.data, desc.data);

		// printf("Pop %lu, type: %u, res: %d, %d, %d, %d\n", stack[stack_ptr],
		//        type, res[0], res[1], res[2], res[3]);

		stack_ptr--;

		if (type != 0) {
			for (int i = 0; i < 4; i++) {
				if (res[i] == InvalidValue)
					break;
				// printf("Push %d\n", res[i]);
				stack[++stack_ptr] = res[i];
			}
		} else {
			float t = __int_as_float(res[0]) / __int_as_float(res[1]);
			if (t < t_near) {
				t_near = t;
				id = res[2]; // TODO: wrong result (?)

				const TriangleNode *node =
				    reinterpret_cast<const TriangleNode *>(static_cast<uint8_t *>(bvh) + node_dir);

				float3 v0 = node->m_triPair.m_v0;
				float3 v1 = node->m_triPair.m_v1;
				float3 v2 = node->m_triPair.m_v2;
				p = {ray_o.x + ray_d.x * t, ray_o.y + ray_d.y * t, ray_o.z + ray_d.z * t};
				float2 bary = barycentrics(v0, v1, v2, p);
				u = bary.x;
				v = bary.y;
			}
		}
	}

	out[idx] = {u, v, t_near, id, t_near < std::numeric_limits<float>::infinity(), {p.x, p.y, p.z}};
}

namespace portableRT {

void push_bytes(std::vector<uint8_t> &v, const void *p, size_t n) {
	const uint8_t *s = static_cast<const uint8_t *>(p);
	v.insert(v.end(), s, s + n);
}

void parse_bvh3(BVH2 *bvh, std::vector<uint8_t> &data) {

	auto make_f3 = [](const std::array<float, 3> &v) { return float3(v[0], v[1], v[2]); };

	std::vector<uint32_t> offsets;
	std::vector<uint32_t> parents(bvh->m_node_count, InvalidValue);
	offsets.reserve(bvh->m_node_count);
	uint32_t offset = 0;

	for (int i = 0; i < bvh->m_node_count; i++) {
		const auto node = bvh->m_nodes[i];
		offsets.push_back(offset);
		if (!node.is_leaf) {
			parents[node.left] = i;
			parents[node.right] = i;
		}
		offset += node.is_leaf ? sizeof(TriangleNode) : sizeof(BoxNode);
	}

	for (int i = 0; i < bvh->m_node_count; i++) {
		const auto node = bvh->m_nodes[i];
		if (node.is_leaf) {
			TriangleNode leaf{};
			leaf.m_flags = (1 << 2) | (1 << 0);
			if (node.tri != -1) {
				Tri tri = bvh->m_tris[node.tri];
				leaf.m_triPair.m_v0 = {tri[0], tri[1], tri[2]};
				leaf.m_triPair.m_v1 = {tri[3], tri[4], tri[5]};
				leaf.m_triPair.m_v2 = {tri[6], tri[7], tri[8]};
			}
			push_bytes(data, &leaf, sizeof(leaf));
		} else {

			BVH2::Node left = bvh->m_nodes[node.left];
			BVH2::Node right = bvh->m_nodes[node.right];

			BoxNode box{};
			box.m_childCount = 2;

			box.m_box0 = Aabb(make_f3(left.bounds.first), make_f3(left.bounds.second));
			box.m_box1 = Aabb(make_f3(right.bounds.first), make_f3(right.bounds.second));

			box.m_childIndex0 =
			    left.is_leaf ? (offsets[node.left] >> 3) : (offsets[node.left] >> 3) + 5;
			box.m_childIndex1 =
			    right.is_leaf ? (offsets[node.right] >> 3) : (offsets[node.right] >> 3) + 5;

			if (parents[i] != InvalidValue) {
				box.m_parentAddr = (offsets[parents[i]] >> 3) + 5;
			}

			push_bytes(data, &box, sizeof(box));
		}
	}
}

void *parse_bvh(BVH2 *bvh) {

	std::vector<uint8_t> buff{};

	parse_bvh3(bvh, buff);

	void *d_bvh;
	CHK(hipMalloc(&d_bvh, buff.size()));
	CHK(hipMemcpy(d_bvh, buff.data(), buff.size(), hipMemcpyHostToDevice));

	return d_bvh;
}

void HIPBackend::set_tris(const Tris &tris) {
	BVH2 bvh;
	bvh.build(tris);
	m_dbvh = parse_bvh(&bvh);
}

template <class... Tags>
std::vector<HitReg<Tags...>> HIPBackend::nearest_hits(const std::vector<Ray> &rays) {

	FullHitReg *dHit;
	Ray *dRays;
	CHK(hipMalloc(&dHit, sizeof(FullHitReg) * rays.size()));
	CHK(hipMalloc(&dRays, sizeof(Ray) * rays.size()));
	CHK(hipMemcpy(dRays, rays.data(), sizeof(Ray) * rays.size(), hipMemcpyHostToDevice));

	int block_size = 64;
	int blocks = (rays.size() + block_size - 1) / block_size;

	nearest_hit<<<blocks, block_size>>>(m_dbvh, dHit, dRays, 1000000000, 5, rays.size());
	CHK(hipDeviceSynchronize());
	FullHitReg *hHit = new FullHitReg[rays.size()];
	CHK(hipMemcpy(hHit, dHit, sizeof(FullHitReg) * rays.size(), hipMemcpyDeviceToHost));
	CHK(hipFree(dHit));
	CHK(hipFree(dRays));

	std::vector<HitReg<Tags...>> hits;
	hits.reserve(rays.size());
	std::transform(hHit, hHit + rays.size(), std::back_inserter(hits),
	               [](const FullHitReg &h) { return slice<Tags...>(h); });
	delete[] hHit;

	return hits;
}

// Manual instantiation
NEAREST_HITS_INSTANTIATE(HIPBackend)

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

std::string HIPBackend::device_name() const {
	int device = 0;
	hipDeviceProp_t props{};
	if (hipGetDevice(&device) != hipSuccess)
		return "unsupported";

	if (hipGetDeviceProperties(&props, device) != hipSuccess)
		return "unsupported";

	return std::string(props.name);
}

} // namespace portableRT