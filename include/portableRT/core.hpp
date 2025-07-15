#pragma once

#include <array>
#include <limits>
#include <sched.h>
#include <thread>
namespace portableRT {

// SYCL can mess up with affinity for pinning cores.
inline void clear_affinity()
{
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

struct HitReg{
  float t;
  uint32_t primitive_id;
};

using Tri = std::array<float, 9>;
using Tris = std::vector<Tri>;

inline float intersect_tri(const std::array<float, 9> &vertices,
                           const Ray &ray) {
  std::array<float, 3> v0 = {vertices[0], vertices[1], vertices[2]};
  std::array<float, 3> v1 = {vertices[3], vertices[4], vertices[5]};
  std::array<float, 3> v2 = {vertices[6], vertices[7], vertices[8]};

  auto edge1 =
      std::array<float, 3>{v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]};

  auto edge2 =
      std::array<float, 3>{v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]};

  auto pvec = std::array<float, 3>{
      ray.direction[1] * edge2[2] - ray.direction[2] * edge2[1],
      ray.direction[2] * edge2[0] - ray.direction[0] * edge2[2],
      ray.direction[0] * edge2[1] - ray.direction[1] * edge2[0]};

  float det = edge1[0] * pvec[0] + edge1[1] * pvec[1] + edge1[2] * pvec[2];
  if (det == 0.0f)
    return std::numeric_limits<float>::infinity();

  float invDet = 1.0f / det;

  auto tvec = std::array<float, 3>{ray.origin[0] - v0[0], ray.origin[1] - v0[1],
                                   ray.origin[2] - v0[2]};

  float u =
      (tvec[0] * pvec[0] + tvec[1] * pvec[1] + tvec[2] * pvec[2]) * invDet;
  if (u < 0.0f || u > 1.0f)
    return std::numeric_limits<float>::infinity();

  auto qvec = std::array<float, 3>{tvec[1] * edge1[2] - tvec[2] * edge1[1],
                                   tvec[2] * edge1[0] - tvec[0] * edge1[2],
                                   tvec[0] * edge1[1] - tvec[1] * edge1[0]};

  float v = (ray.direction[0] * qvec[0] + ray.direction[1] * qvec[1] +
             ray.direction[2] * qvec[2]) *
            invDet;
  if (v < 0.0f || u + v > 1.0f)
    return std::numeric_limits<float>::infinity();

  float t =
      (edge2[0] * qvec[0] + edge2[1] * qvec[1] + edge2[2] * qvec[2]) * invDet;
  return t;
}

} // namespace portableRT