#pragma once

// I'm using this monstrosity temporaly.

#include <algorithm>
#include <array>
#include <chrono>
#include <limits>
#include <vector>

#include "core.hpp"

#define BVH_DEPTH 18
#define BVH_SAHBINS 14

// Workaround for compiling with non sycl compilers
#ifndef SYCL_EXTERNAL
#define SYCL_EXTERNAL
#endif

namespace portableRT {

using Vector3 = std::array<float, 3>;
inline std::array<float, 3> operator+(const std::array<float, 3> &a,
                                      const std::array<float, 3> &b) {
  return {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
}
inline std::array<float, 3> operator-(const std::array<float, 3> &a,
                                      const std::array<float, 3> &b) {
  return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}
inline std::array<float, 3> operator*(float s, const std::array<float, 3> &v) {
  return {s * v[0], s * v[1], s * v[2]};
}

struct Hit {
  bool valid;
  Vector3 position;
  unsigned int triIdx;
  float t;
};

struct BVHTri {

  Tri tri;
  int index;

  BVHTri(Tri _tri, int _index);

  BVHTri();
};

struct Node {

  Vector3 b1, b2;
  int from, to, idx, depth;
  bool valid;

  Node();

  Node(int _idx, Vector3 _b1, Vector3 _b2, int _from, int _to, int _depth);
};

// Data structure which holds all the geometry data organized so it can be
// intersected fast with light rays.

class BVH {

public:
  int nodeIdx = 0;
  int allocatedTris = 0;
  int totalTris = 0;

  Tri *tris;

  Node nodes[2 << BVH_DEPTH];

  int triIdx = 0;
  int *triIndices;

  BVH() {}

  bool intersect(Ray ray, Vector3 b1, Vector3 b2);

  SYCL_EXTERNAL void transverse(Ray ray, Hit &nearestHit);

  void intersectNode(Ray ray, Node node, Hit &nearestHit);

  Node leftChild(int idx, int depth);

  Node rightChild(int idx, int depth);

  void build(const std::vector<Tri> *_fullTris);

  void buildAux(int depth, std::vector<BVHTri> *_tris);

  static void dividePlane(std::vector<BVHTri> *tris,
                          std::vector<BVHTri> *trisLeft,
                          std::vector<BVHTri> *trisRight);

  static void divideSAH(std::vector<BVHTri> *tris,
                        std::vector<BVHTri> *trisLeft,
                        std::vector<BVHTri> *trisRight);

  static void boundsUnion(Vector3 b1, Vector3 b2, Vector3 b3, Vector3 b4,
                          Vector3 &b5, Vector3 &b6);

  static float boundsArea(Vector3 b1, Vector3 b2);

  static void bounds(Tri tri, Vector3 &b1, Vector3 &b2);

  static void bounds(std::vector<BVHTri> *tris, Vector3 &b1, Vector3 &b2);
};

} // namespace portableRT