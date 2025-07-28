#pragma once

#include <array>
#include <limits>
#include <sched.h>
#include <thread>
namespace portableRT {

// SYCL can mess up with affinity for pinning cores.
inline void clear_affinity() {
	cpu_set_t mask;
	CPU_ZERO(&mask);
	for (int c = 0; c < std::thread::hardware_concurrency(); ++c)
		CPU_SET(c, &mask);
	sched_setaffinity(0, sizeof(mask), &mask);
}

struct Ray {
	std::array<float, 3> origin;
	std::array<float, 3> direction;
};

struct Empty {};

#define ALL_TAGS filter::uv, filter::t, filter::primitive_id, filter::p, filter::valid

template <bool HasUV, bool HasT, bool HasPrimitiveId, bool HasP, bool HasValid> struct HitRegImpl {
	using has_uv = std::bool_constant<HasUV>;
	using has_t = std::bool_constant<HasT>;
	using has_primitive_id = std::bool_constant<HasPrimitiveId>;
	using has_p = std::bool_constant<HasP>;
	using has_valid = std::bool_constant<HasValid>;

	std::conditional_t<HasUV, float, Empty> u;
	std::conditional_t<HasUV, float, Empty> v;
	std::conditional_t<HasT, float, Empty> t;
	std::conditional_t<HasPrimitiveId, uint32_t, Empty> primitive_id;
	std::conditional_t<HasValid, bool, Empty> valid;
	std::conditional_t<HasP, std::array<float, 3>, Empty> p;
};

template <typename T, typename... List> constexpr bool has_tag = (std::is_same_v<T, List> || ...);

namespace filter {
struct uv {};
struct t {};
struct primitive_id {};
struct p {};
struct valid {};
} // namespace filter

template <class... Tags>
using HitReg = HitRegImpl<has_tag<filter::uv, Tags...>, has_tag<filter::t, Tags...>,
                          has_tag<filter::primitive_id, Tags...>, has_tag<filter::p, Tags...>,
                          has_tag<filter::valid, Tags...>>;

using FullHitReg = HitReg<filter::uv, filter::t, filter::primitive_id, filter::p, filter::valid>;

template <class... Tags> HitReg<Tags...> slice(const FullHitReg &hit) {
	HitReg<Tags...> res;
	if constexpr (has_tag<filter::uv, Tags...>) {
		res.u = hit.u;
		res.v = hit.v;
	}
	if constexpr (has_tag<filter::t, Tags...>) {
		res.t = hit.t;
	}
	if constexpr (has_tag<filter::primitive_id, Tags...>) {
		res.primitive_id = hit.primitive_id;
	}
	if constexpr (has_tag<filter::p, Tags...>) {
		res.p = hit.p;
	}
	if constexpr (has_tag<filter::valid, Tags...>) {
		res.valid = hit.valid;
	}
	return res;
}

using Tri = std::array<float, 9>;
using Tris = std::vector<Tri>;

inline bool intersect_tri(const std::array<float, 9> &vertices, const Ray &ray, float &t, float &u,
                          float &v) {
	std::array<float, 3> v0 = {vertices[0], vertices[1], vertices[2]};
	std::array<float, 3> v1 = {vertices[3], vertices[4], vertices[5]};
	std::array<float, 3> v2 = {vertices[6], vertices[7], vertices[8]};

	auto edge1 = std::array<float, 3>{v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]};

	auto edge2 = std::array<float, 3>{v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]};

	auto pvec = std::array<float, 3>{ray.direction[1] * edge2[2] - ray.direction[2] * edge2[1],
	                                 ray.direction[2] * edge2[0] - ray.direction[0] * edge2[2],
	                                 ray.direction[0] * edge2[1] - ray.direction[1] * edge2[0]};

	float det = edge1[0] * pvec[0] + edge1[1] * pvec[1] + edge1[2] * pvec[2];
	if (det == 0.0f)
		return false;

	float invDet = 1.0f / det;

	auto tvec =
	    std::array<float, 3>{ray.origin[0] - v0[0], ray.origin[1] - v0[1], ray.origin[2] - v0[2]};

	u = (tvec[0] * pvec[0] + tvec[1] * pvec[1] + tvec[2] * pvec[2]) * invDet;
	if (u < 0.0f || u > 1.0f)
		return false;

	auto qvec = std::array<float, 3>{tvec[1] * edge1[2] - tvec[2] * edge1[1],
	                                 tvec[2] * edge1[0] - tvec[0] * edge1[2],
	                                 tvec[0] * edge1[1] - tvec[1] * edge1[0]};

	v = (ray.direction[0] * qvec[0] + ray.direction[1] * qvec[1] + ray.direction[2] * qvec[2]) *
	    invDet;
	if (v < 0.0f || u + v > 1.0f)
		return false;

	t = (edge2[0] * qvec[0] + edge2[1] * qvec[1] + edge2[2] * qvec[2]) * invDet;
	return true;
}

} // namespace portableRT