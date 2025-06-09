#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <hip/hip_runtime.h>
#include <iostream>
#include <stack>

#include "../include/portableRT/intersect_hip.hpp"

// https://github.com/GPUOpen-LibrariesAndSDKs/HIPRT/blob/main/hiprt/impl/hiprt_device_impl.h
// https://github.com/GPUOpen-LibrariesAndSDKs/HIPRT/blob/main/hiprt/impl/BvhNode.h

#define CHK(x)                                                                 \
  if ((x) != hipSuccess) {                                                     \
    std::cerr << "HIP " << hipGetErrorString(x) << '\n';                       \
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
    std::cout << m_min.x << " " << m_min.y << " " << m_min.z << " " << m_max.x
              << " " << m_max.y << " " << m_max.z << std::endl;
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
  descriptor.y = (baseAddress >> 32ull) | (boxSortHeuristic << 21u) |
                 (boxGrowUlp << 23u) | (boxSortEnabled << 31u);
  descriptor.z = size & 0xffffffff;
  descriptor.w = (size >> 32ull) | (triangleReturnMode << 24u) |
                 (llcNoalloc << 25u) | (bigPage << 27u) | (type << 28u);
  return descriptor;
}

static uint32_t encodeNodeIndex(uint32_t nodeAddr, uint32_t nodeType) {
  if (nodeType == 5)
    nodeAddr *= 2;
  return (nodeAddr << 3) | nodeType;
}

__device__ __host__ static uint64_t encodeBaseAddr(const void *baseAddr,
                                                   uint32_t nodeIndex) {
  uint64_t baseIndex = reinterpret_cast<uint64_t>(baseAddr) >> 3ull;
  return baseIndex + nodeIndex;
}

__global__ void nearest_hit(void *bvh, float *out, portableRT::Ray *rays,
                            uint64_t size, uint64_t pos, uint64_t num_rays) {

  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if(idx >= num_rays)
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

  float hit = std::numeric_limits<float>::infinity();

  while (stack_ptr >= 0) {

    uint32_t type = stack[stack_ptr] & 0x7;
    auto res = __builtin_amdgcn_image_bvh_intersect_ray_l(
        stack[stack_ptr], 100, ray_o.data, ray_d.data, ray_id.data, desc.data);

    // printf("Pop %lu, type: %u, res: %d, %d, %d, %d\n", stack[stack_ptr],
    // type, res[0], res[1], res[2], res[3]);

    stack_ptr--;

    if (type != 0) {
      for (int i = 0; i < 4; i++) {
        if (res[i] == InvalidValue)
          break;
        // printf("Push %d\n", res[i]);
        stack[++stack_ptr] = res[i];
      }
    } else {
      hit = std::min(hit, float(res[3]));
    }
  }

  out[idx] = hit;
}

namespace portableRT {

void push_bytes(std::vector<uint8_t> &v, const void *p, size_t n) {
  const uint8_t *s = static_cast<const uint8_t *>(p);
  v.insert(v.end(), s, s + n);
}

void parse_bvh(BVH *bvh, Node node, std::vector<uint8_t> &data, int &offset,
               uint32_t parentAddr) {

  auto is_leaf = [](const Node &node) { return node.depth == BVH_DEPTH; };

  auto make_f3 = [](const Vector3 &v) { return float3(v[0], v[1], v[2]); };

  if (is_leaf(node)) {
    if (abs(node.to - node.from) > 1) {
      std::cout << "More tris per node than allowed" << abs(node.to - node.from)
                << std::endl;
    }
    if (node.to - node.from == 1) {

      const auto tri = bvh->tris[bvh->triIndices[node.from]];

      TriangleNode leaf{};
      leaf.m_triPair.m_v0 = {tri[0], tri[1], tri[2]};
      leaf.m_triPair.m_v1 = {tri[3], tri[4], tri[5]};
      leaf.m_triPair.m_v2 = {tri[6], tri[7], tri[8]};
      leaf.m_flags = (1 << 2) | (1 << 0);

      // std::cout << "T()" << std::flush;

      push_bytes(data, &leaf, sizeof(leaf));
      offset += sizeof(leaf);
    } else {

      TriangleNode leaf{};
      leaf.m_flags = (1 << 2) | (1 << 0);

      // std::cout << "E()" << std::flush;

      push_bytes(data, &leaf, sizeof(leaf));
      offset += sizeof(leaf);
    }
  } else {
    Node left = bvh->leftChild(node.idx, node.depth);
    Node right = bvh->rightChild(node.idx, node.depth);

    int k = BVH_DEPTH - node.depth;
    int N = pow(2, k) - 1; // Número de nodos del arbol izquierdo
    int H = pow(2, k - 1); // Número de hojas del árbol izquierdo.
    int I = N - H;         // Número de nodos interiores del árbol izquierdo.

    int left_off = (offset + sizeof(BoxNode)) >> 3;
    int right_off = (offset + sizeof(BoxNode) + sizeof(BoxNode) * I +
                     sizeof(TriangleNode) * H) >>
                    3;

    BoxNode box{};
    box.m_childCount = 2;
    box.m_box0 = Aabb(make_f3(left.b1), make_f3(left.b2));
    box.m_childIndex0 = is_leaf(left) ? left_off : left_off + 5;
    box.m_box1 = Aabb(make_f3(right.b1), make_f3(right.b2));
    box.m_childIndex1 = is_leaf(right) ? right_off : right_off + 5;
    box.m_parentAddr = parentAddr;

    push_bytes(data, &box, sizeof(box));
    offset += sizeof(box);
    // std::cout << left.b1[0] << " " << left.b1[1] << " " << left.b1[2] << " "
    // << left.b2[0] << " " << left.b2[1] << " " << left.b2[2] << " | ";
    // std::cout << right.b1[0] << " " << right.b1[1] << " " << right.b1[2] << "
    // " << right.b2[0] << " " << right.b2[1] << " " << right.b2[2] << " | ";
    // std::cout << "B(" << left_off << ", " << right_off << ")" << std::flush;

    parse_bvh(bvh, left, data, offset, (offset >> 3) + 5);
    parse_bvh(bvh, right, data, offset, (offset >> 3) + 5);
  }
}

struct ExNode {

  bool leaf;
  Tri tri;
  int idx;

  std::unique_ptr<ExNode> left_child;
  std::unique_ptr<ExNode> right_child;
  ExNode *parent;

  Aabb aabb;

  void flatten(std::vector<uint8_t> &data) {
    if (leaf) {
      TriangleNode leaf{};
      leaf.m_triPair.m_v0 = {tri[0], tri[1], tri[2]};
      leaf.m_triPair.m_v1 = {tri[3], tri[4], tri[5]};
      leaf.m_triPair.m_v2 = {tri[6], tri[7], tri[8]};
      leaf.m_flags = (1 << 2) | (1 << 0);
      push_bytes(data, &leaf, sizeof(leaf));
      // std::cout << "L(" << idx << ")" << std::flush;
    } else {
      BoxNode box{};
      box.m_childCount = 2;
      box.m_box0 = left_child->aabb;
      box.m_childIndex0 = left_child->idx;
      box.m_box1 = right_child->aabb;
      box.m_childIndex1 = right_child->idx;
      // box.m_parentAddr  = parent->idx;
      push_bytes(data, &box, sizeof(box));
      // std::cout << "B(" << idx << ")->[" << left_child->idx << ", " <<
      // right_child->idx << "]" << std::flush;
    }
  }
};

void process(ExNode *node, std::vector<uint8_t> &data) {
  if (node == nullptr)
    return;
  process(node->left_child.get(), data);
  process(node->right_child.get(), data);
  if (node->leaf) {
    node->aabb = Aabb(node->tri);
    node->idx = data.size() >> 3;
    node->flatten(data);
  } else {
    node->aabb = Aabb(node->left_child->aabb, node->right_child->aabb);
    node->idx = (data.size() >> 3) + 5;
    node->flatten(data);
  }
}

void make_extended_tree_second(BVH *bvh, ExNode *node, int from, int to) {
  if (to - from <= 1) {
    node->leaf = true;
    const auto tri = bvh->tris[bvh->triIndices[from]];
    node->tri = {tri[0], tri[1], tri[2], tri[3], tri[4],
                 tri[5], tri[6], tri[7], tri[8]};
  } else {
    node->leaf = false;
    int h = (to - from) / 2;
    node->left_child = std::make_unique<ExNode>();
    node->right_child = std::make_unique<ExNode>();
    node->left_child->parent = node;
    node->right_child->parent = node;
    make_extended_tree_second(bvh, node->left_child.get(), from, from + h);
    make_extended_tree_second(bvh, node->right_child.get(), from + h, to);
  }
}

std::unique_ptr<ExNode> make_extended_tree_r(BVH *bvh, ExNode *parent,
                                             Node i_node) {

  std::unique_ptr<ExNode> node = std::make_unique<ExNode>();
  node->parent = parent;

  if (i_node.depth == BVH_DEPTH) {
    if (i_node.to - i_node.from == 1) {
      node->leaf = true;
      const auto tri = bvh->tris[bvh->triIndices[i_node.from]];
      node->tri = {tri[0], tri[1], tri[2], tri[3], tri[4],
                   tri[5], tri[6], tri[7], tri[8]};
    } else {
      make_extended_tree_second(bvh, node.get(), i_node.from, i_node.to);
    }
  } else {
    node->leaf = false;
    node->left_child = make_extended_tree_r(
        bvh, node.get(), bvh->leftChild(i_node.idx, i_node.depth));
    node->right_child = make_extended_tree_r(
        bvh, node.get(), bvh->rightChild(i_node.idx, i_node.depth));
  }

  return std::move(node);
}

std::unique_ptr<ExNode> make_extended_tree(BVH *bvh) {
  std::unique_ptr<ExNode> root = std::make_unique<ExNode>();
  root->leaf = false;
  root->parent = nullptr;
  root->left_child = make_extended_tree_r(
      bvh, root.get(), bvh->leftChild(bvh->nodes[0].idx, bvh->nodes[0].depth));
  root->right_child = make_extended_tree_r(
      bvh, root.get(), bvh->rightChild(bvh->nodes[0].idx, bvh->nodes[0].depth));
  return std::move(root);
}

void *parse_bvh(BVH *bvh, uint64_t &rootidx) {

  std::vector<uint8_t> buff{};

  std::unique_ptr<ExNode> root = make_extended_tree(bvh);
  process(root.get(), buff);

  rootidx = root->idx;
  // std::cout << "root idx " << rootidx << std::endl;

  void *d_bvh;
  CHK(hipMalloc(&d_bvh, buff.size()));
  CHK(hipMemcpy(d_bvh, buff.data(), buff.size(), hipMemcpyHostToDevice));

  return d_bvh;
}

void *parse_bvh2(BVH *bvh) {

  std::vector<uint8_t> buff{};
  int offset = 0;
  parse_bvh(bvh, bvh->nodes[0], buff, offset, InvalidValue);

  for (int i = 0; i < (int)buff.size(); i++) {
    // if (i % 64 == 0) std::cout << '\n';

    int valor = static_cast<int>(buff[i]);
    std::string s = std::to_string(valor);
    int len = s.length();
    int ancho = 3;
    int padL = (ancho - len) / 2;
    int padR = ancho - len - padL;

    /// std::cout << std::string(padL, ' ') << s << std::string(padR, ' ');
    // std::cout << ' ';
  }
  // std::cout << '\n';

  void *d_bvh;
  CHK(hipMalloc(&d_bvh, buff.size()));
  CHK(hipMemcpy(d_bvh, buff.data(), buff.size(), hipMemcpyHostToDevice));

  return d_bvh;
}

void HIPBackend::set_tris(const Tris &tris) {

  m_tris = tris;

  BVH *bvh = new BVH();
  bvh->triIndices = new int[tris.size()];
  bvh->tris = m_tris.data();
  bvh->build(&m_tris);

  m_dbvh = parse_bvh(bvh, m_rootidx);
  delete[] bvh->triIndices;
  delete (bvh);
}

std::vector<float> HIPBackend::nearest_hits(const std::vector<Ray> &rays) {

  float *dHit;
  Ray *dRays;
  CHK(hipMalloc(&dHit, sizeof(float) * rays.size()));
  CHK(hipMalloc(&dRays, sizeof(Ray) * rays.size()));
  CHK(hipMemcpy(dRays, rays.data(), sizeof(Ray) * rays.size(),
                hipMemcpyHostToDevice));

  int block_size = 64;
  int blocks = (rays.size() + block_size - 1) / block_size;

  nearest_hit<<<blocks, block_size>>>(
      m_dbvh, dHit, dRays, 1000000000,
      m_rootidx, rays.size());
  CHK(hipDeviceSynchronize());
  float *hHit = new float[rays.size()];
  CHK(hipMemcpy(hHit, dHit, sizeof(float) * rays.size(),
                hipMemcpyDeviceToHost));
  CHK(hipFree(dHit));
  CHK(hipFree(dRays));

  return std::vector<float>(hHit, hHit + rays.size());
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