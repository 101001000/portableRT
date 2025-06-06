#include "../include/portableRT/bvh.hpp"
#include <iostream>

namespace portableRT {

float map(float a, float b, float c, float d, float e) {
  return d + ((a - b) / (c - b)) * (e - d);
}

Vector3 map(Vector3 a, Vector3 b, Vector3 c, Vector3 d, Vector3 e) {
  return {map(a[0], b[0], c[0], d[0], e[0]), map(a[1], b[1], c[1], d[1], e[1]),
          map(a[2], b[2], c[2], d[2], e[2])};
}

BVHTri::BVHTri(Tri _tri, int _index) {
  tri = _tri;
  index = _index;
}

Node::Node() { valid = false; }

Node::Node(int _idx, Vector3 _b1, Vector3 _b2, int _from, int _to, int _depth) {
  idx = _idx;
  b1 = _b1;
  b2 = _b2;
  from = _from;
  to = _to;
  valid = true;
  depth = _depth;
}

bool BVH::intersect(Ray ray, Vector3 b1, Vector3 b2) {

  // Direct implementation of
  // https://gamedev.stackexchange.com/questions/18436/most-efficient-aabb-vs-ray-collision-algorithms
  Vector3 dirfrac;

  // r.dir is unit direction vector of ray
  dirfrac[0] = 1.0f / ray.direction[0];
  dirfrac[1] = 1.0f / ray.direction[1];
  dirfrac[2] = 1.0f / ray.direction[2];

  // lb is the corner of AABB with minimal coordinates - left bottom, rt is
  // maximal corner r.org is origin of ray
  float t1 = (b1[0] - ray.origin[0]) * dirfrac[0];
  float t2 = (b2[0] - ray.origin[0]) * dirfrac[0];
  float t3 = (b1[1] - ray.origin[1]) * dirfrac[1];
  float t4 = (b2[1] - ray.origin[1]) * dirfrac[1];
  float t5 = (b1[2] - ray.origin[2]) * dirfrac[2];
  float t6 = (b2[2] - ray.origin[2]) * dirfrac[2];

  float tmin =
      std::max(std::max(std::min(t1, t2), std::min(t3, t4)), std::min(t5, t6));
  float tmax =
      std::min(std::min(std::max(t1, t2), std::max(t3, t4)), std::max(t5, t6));

  // if tmax < 0, ray (line) is intersecting AABB, but the whole AABB is behind
  // us
  if (tmax < 0) {
    return false;
  }

  // if tmin > tmax, ray doesn't intersect AABB
  if (tmin > tmax) {
    return false;
  }

  return true;
}

void BVH::transverse(Ray ray, Hit &nearestHit) {

  Node stack[64];

  Node *stackPtr = stack;

  (*stackPtr++).valid = false;

  Node node = nodes[0];

  do {

    Node lChild = leftChild(node.idx, node.depth);
    Node rChild = rightChild(node.idx, node.depth);

    bool lOverlap = intersect(ray, lChild.b1, lChild.b2);
    bool rOverlap = intersect(ray, rChild.b1, rChild.b2);

    if (node.depth == (BVH_DEPTH - 1) && rOverlap) {
      intersectNode(ray, rChild, nearestHit);
    }

    if (node.depth == (BVH_DEPTH - 1) && lOverlap) {
      intersectNode(ray, lChild, nearestHit);
    }

    bool traverseL = (lOverlap && node.depth != (BVH_DEPTH - 1));
    bool traverseR = (rOverlap && node.depth != (BVH_DEPTH - 1));

    if (!traverseL && !traverseR) {
      node = *--stackPtr;

    } else {
      node = (traverseL) ? lChild : rChild;
      if (traverseL && traverseR) {
        *stackPtr++ = rChild;
      }
    }

  } while (node.valid);
}

void BVH::intersectNode(Ray ray, Node node, Hit &nearestHit) {

  auto length2 = [](const Vector3 &v) {
    return v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
  };

  for (int i = node.from; i < node.to; i++) {

    float t = intersect_tri(tris[triIndices[i]], ray);

    if (t > 0) {

      Hit hit;
      hit.valid = true;
      hit.position = ray.origin + t * ray.direction;

      // If previous hit was not valid, then the new is going to be nearer in
      // any case.
      if (!nearestHit.valid || length2(hit.position - ray.origin) <
                                   length2(nearestHit.position - ray.origin)) {
        nearestHit = hit;
        nearestHit.triIdx = i;
      }
    }
  }
}

Node BVH::leftChild(int idx, int depth) {
  if (depth == BVH_DEPTH) {
    return Node();
  }
  return nodes[idx + 1];
}

Node BVH::rightChild(int idx, int depth) {
  if (depth == BVH_DEPTH) {
    return Node();
  }
  return nodes[idx + (2 << (BVH_DEPTH - depth - 1))];
}

void BVH::build(const std::vector<Tri> *_fullTris) {

  std::vector<BVHTri> *_tris = new std::vector<BVHTri>();

  for (int i = 0; i < _fullTris->size(); i++) {
    BVHTri bvhTri(_fullTris->at(i), i);
    _tris->push_back(bvhTri);
  }

  buildAux(0, _tris);

  std::vector<Tri> *sortedTris = new std::vector<Tri>();

  for (int i = 0; i < _tris->size(); i++) {
    sortedTris->push_back(_fullTris->at(triIndices[i]));
  }

  for (int i = 0; i < _tris->size(); i++) {
    Tri t = sortedTris->at(i);
    //_fullTris->data()[i] = t;
  }

  delete (_tris);
  delete (sortedTris);
}

void BVH::buildAux(int depth, std::vector<BVHTri> *_tris) {

  Vector3 b1, b2;

  if (depth == 0) {
    totalTris = _tris->size();
  }

  if (depth == 7) {
    // printf("\rAllocated tris: %d / %d, %d%%", allocatedTris, totalTris,
    //        (100 * allocatedTris) / totalTris);
  }

  bounds(_tris, b1, b2);

  if (depth == BVH_DEPTH) {

    nodes[nodeIdx++] =
        Node(nodeIdx, b1, b2, triIdx, triIdx + _tris->size(), depth);

    for (int i = 0; i < _tris->size(); i++) {
      triIndices[triIdx++] = _tris->at(i).index;
    }

    allocatedTris += _tris->size();
  } else {

    nodes[nodeIdx++] = Node(nodeIdx, b1, b2, 0, 0, depth);

    std::vector<BVHTri> *trisLeft = new std::vector<BVHTri>();
    std::vector<BVHTri> *trisRight = new std::vector<BVHTri>();

    divideSAH(_tris, trisLeft, trisRight);

    buildAux(depth + 1, trisLeft);
    buildAux(depth + 1, trisRight);

    trisLeft->clear();
    trisRight->clear();

    delete (trisLeft);
    delete (trisRight);
  }
}

void BVH::divideSAH(std::vector<BVHTri> *tris, std::vector<BVHTri> *trisLeft,
                    std::vector<BVHTri> *trisRight) {

  if (tris->size() <= 0) {
    return;
  }

  Vector3 totalB1, totalB2;

  int bestBin = 0;
  int bestAxis = 0;
  float bestHeuristic = std::numeric_limits<float>::max();

  auto centroid = [](const Tri &tri) -> std::array<float, 3> {
    return {(tri[0] + tri[3] + tri[6]) / 3.0f,
            (tri[1] + tri[4] + tri[7]) / 3.0f,
            (tri[2] + tri[5] + tri[8]) / 3.0f};
  };

  bounds(tris, totalB1, totalB2);

  for (int axis = 0; axis < 3; axis++) {

    Vector3 b1s[BVH_SAHBINS];
    Vector3 b2s[BVH_SAHBINS];

    int count[BVH_SAHBINS];

    // Bin initialization
    for (int i = 0; i < BVH_SAHBINS; i++) {
      count[i] = 0;
      b1s[i] = Vector3();
      b2s[i] = Vector3();
    }

    // Bin filling
    for (int i = 0; i < tris->size(); i++) {

      int bin = 0;
      Vector3 b1, b2;

      // The bin which corresponds to certain triangle is calculated
      if (totalB1[axis] != totalB2[axis]) {
        float c = centroid(tris->at(i).tri)[axis];
        bin = map(c, totalB1[axis], totalB2[axis], 0, BVH_SAHBINS - 1);
        bin = std::min(std::max(int(bin), 0), BVH_SAHBINS - 1);
      }

      count[bin]++;
      bounds(tris->at(i).tri, b1, b2);
      boundsUnion(b1s[bin], b2s[bin], b1, b2, b1s[bin], b2s[bin]);
    }

    for (int i = 0; i < BVH_SAHBINS; i++) {

      // Tris in the first and second child
      int count1 = 0;
      int count2 = 0;

      // b1, b2 are the boundings for first child, b3, b4 for the second one
      Vector3 b1, b2, b3, b4;

      // First swept for first child from 0 to i
      for (int j = 0; j < i; j++) {
        count1 += count[j];
        boundsUnion(b1, b2, b1s[j], b2s[j], b1, b2);
      }

      // Second swept for second child from i to BVH_SAHBINS - 1
      for (int k = i; k < BVH_SAHBINS; k++) {
        count2 += count[k];
        boundsUnion(b3, b4, b1s[k], b2s[k], b3, b4);
      }

      float heuristic = boundsArea(b1, b2) * static_cast<float>(count1) +
                        boundsArea(b3, b4) * static_cast<float>(count2);

      if (heuristic < bestHeuristic) {
        bestHeuristic = heuristic;
        bestBin = i;
        bestAxis = axis;
      }
    }
  }

  // Tri division depending on the bin
  for (int i = 0; i < tris->size(); i++) {

    float c = centroid(tris->at(i).tri)[bestAxis];
    int bin = map(c, totalB1[bestAxis], totalB2[bestAxis], 0, BVH_SAHBINS - 1);
    bin = std::min(std::max(int(bin), 0), BVH_SAHBINS - 1);

    if (bin < bestBin) {
      trisLeft->push_back(tris->at(i));
    } else {
      trisRight->push_back(tris->at(i));
    }
  }

  //std::cout << "Left: " << trisLeft->size() << " Right: " << trisRight->size() << std::endl;
}

void BVH::boundsUnion(Vector3 b1, Vector3 b2, Vector3 b3, Vector3 b4,
                      Vector3 &b5, Vector3 &b6) {

  if (boundsArea(b1, b2) <= 0 || boundsArea(b3, b4) <= 0) {

    if (boundsArea(b1, b2) <= 0) {
      b5 = b3;
      b6 = b4;
    }

    if (boundsArea(b3, b4) <= 0) {
      b5 = b1;
      b6 = b2;
    }
  } else {

    b5[0] = std::min(b1[0], std::min(b2[0], std::min(b3[0], b4[0])));
    b5[1] = std::min(b1[1], std::min(b2[1], std::min(b3[1], b4[1])));
    b5[2] = std::min(b1[2], std::min(b2[2], std::min(b3[2], b4[2])));

    b6[0] = std::max(b1[0], std::max(b2[0], std::max(b3[0], b4[0])));
    b6[1] = std::max(b1[1], std::max(b2[1], std::max(b3[1], b4[1])));
    b6[2] = std::max(b1[2], std::max(b2[2], std::max(b3[2], b4[2])));
  }
}

float BVH::boundsArea(Vector3 b1, Vector3 b2) {

  float x = b2[0] - b1[0];
  float y = b2[1] - b1[1];
  float z = b2[2] - b1[2];

  return 2 * (x * y + x * z + y * z);
}

void BVH::bounds(Tri tri, Vector3 &b1, Vector3 &b2) {

  b1[0] = std::min(tri[0], std::min(tri[3], tri[6]));
  b1[1] = std::min(tri[1], std::min(tri[4], tri[7]));
  b1[2] = std::min(tri[2], std::min(tri[5], tri[8]));

  b2[0] = std::max(tri[0], std::max(tri[3], tri[6]));
  b2[1] = std::max(tri[1], std::max(tri[4], tri[7]));
  b2[2] = std::max(tri[2], std::max(tri[5], tri[8]));
}

void BVH::bounds(std::vector<BVHTri> *tris, Vector3 &b1, Vector3 &b2) {

  if (tris->size() <= 0) {
    return;
  }

  b1 = {tris->at(0).tri[0], tris->at(0).tri[1], tris->at(0).tri[2]};
  b2 = b1;

  for (int i = 0; i < tris->size(); i++) {

    b1[0] = std::min(tris->at(i).tri[0], b1[0]);
    b1[1] = std::min(tris->at(i).tri[1], b1[1]);
    b1[2] = std::min(tris->at(i).tri[2], b1[2]);

    b1[0] = std::min(tris->at(i).tri[3], b1[0]);
    b1[1] = std::min(tris->at(i).tri[4], b1[1]);
    b1[2] = std::min(tris->at(i).tri[5], b1[2]);

    b1[0] = std::min(tris->at(i).tri[6], b1[0]);
    b1[1] = std::min(tris->at(i).tri[7], b1[1]);
    b1[2] = std::min(tris->at(i).tri[8], b1[2]);

    b2[0] = std::max(tris->at(i).tri[0], b2[0]);
    b2[1] = std::max(tris->at(i).tri[1], b2[1]);
    b2[2] = std::max(tris->at(i).tri[2], b2[2]);

    b2[0] = std::max(tris->at(i).tri[3], b2[0]);
    b2[1] = std::max(tris->at(i).tri[4], b2[1]);
    b2[2] = std::max(tris->at(i).tri[5], b2[2]);

    b2[0] = std::max(tris->at(i).tri[6], b2[0]);
    b2[1] = std::max(tris->at(i).tri[7], b2[1]);
    b2[2] = std::max(tris->at(i).tri[8], b2[2]);
  }
}

} // namespace portableRT