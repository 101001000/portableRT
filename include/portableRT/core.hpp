#pragma once

#include "hitreg.hpp"
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
	float tmin = 0.0f;
	float tmax = std::numeric_limits<float>::max();
	uint32_t self_id = -1;
};

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

template <class Tag> inline constexpr std::string_view tag_name_v = "unknown";

template <> inline constexpr std::string_view tag_name_v<filter::uv> = "uv";
template <> inline constexpr std::string_view tag_name_v<filter::t> = "t";
template <> inline constexpr std::string_view tag_name_v<filter::primitive_id> = "primitive_id";
template <> inline constexpr std::string_view tag_name_v<filter::p> = "p";
template <> inline constexpr std::string_view tag_name_v<filter::valid> = "valid";

template <class... Tags> inline std::string hitreg_name() {
	std::string out;
	((out += tag_name_v<Tags>, out += '_'), ...);
	if (!out.empty())
		out.pop_back();
	return out;
}

inline std::vector<std::string> all_tags_combinations() {
	std::vector<std::string> res;

#define X(...) res.emplace_back(hitreg_name<ADD_FILTER(__VA_ARGS__)>());

	TAG_COMBOS
#undef X
	return res;
}

} // namespace portableRT