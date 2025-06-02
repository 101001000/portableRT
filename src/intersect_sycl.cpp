#include "../include/portableRT/intersect_sycl.h"
#include "../include/portableRT/core.h"
#include <sycl/sycl.hpp>

namespace portableRT {

struct SYCLBackendImpl {
  sycl::queue m_q;
  sycl::device m_dev;
};

SYCLBackend::SYCLBackend() : InvokableBackend(BackendType::SYCL, "SYCL") {
  static RegisterBackend reg(*this);
}

SYCLBackend::~SYCLBackend() = default;

void SYCLBackend::init() {
  m_impl = std::make_unique<SYCLBackendImpl>();
  m_impl->m_dev = sycl::device(sycl::default_selector_v);
  m_impl->m_q = sycl::queue(m_impl->m_dev);
}

void SYCLBackend::shutdown() { m_impl.reset(); }

bool SYCLBackend::intersect_tri(const std::array<float, 9> &v, const Ray &ray) {
  try {

    bool *res = sycl::malloc_shared<bool>(1, m_impl->m_q);

    m_impl->m_q
        .submit([&](sycl::handler &cgh) {
          cgh.single_task([=]() {
            std::array<float, 3> v0 = {v[0], v[1], v[2]};
            std::array<float, 3> v1 = {v[3], v[4], v[5]};
            std::array<float, 3> v2 = {v[6], v[7], v[8]};

            auto edge1 = std::array<float, 3>{v1[0] - v0[0], v1[1] - v0[1],
                                              v1[2] - v0[2]};

            auto edge2 = std::array<float, 3>{v2[0] - v0[0], v2[1] - v0[1],
                                              v2[2] - v0[2]};

            auto pvec = std::array<float, 3>{
                ray.direction[1] * edge2[2] - ray.direction[2] * edge2[1],
                ray.direction[2] * edge2[0] - ray.direction[0] * edge2[2],
                ray.direction[0] * edge2[1] - ray.direction[1] * edge2[0]};

            float det =
                edge1[0] * pvec[0] + edge1[1] * pvec[1] + edge1[2] * pvec[2];
            if (det == 0.0f) {
              *res = false;
              return;
            };

            float invDet = 1.0f / det;

            auto tvec = std::array<float, 3>{ray.origin[0] - v0[0],
                                             ray.origin[1] - v0[1],
                                             ray.origin[2] - v0[2]};

            float u =
                (tvec[0] * pvec[0] + tvec[1] * pvec[1] + tvec[2] * pvec[2]) *
                invDet;
            if (u < 0.0f || u > 1.0f) {
              *res = false;
              return;
            };

            auto qvec =
                std::array<float, 3>{tvec[1] * edge1[2] - tvec[2] * edge1[1],
                                     tvec[2] * edge1[0] - tvec[0] * edge1[2],
                                     tvec[0] * edge1[1] - tvec[1] * edge1[0]};

            float v = (ray.direction[0] * qvec[0] + ray.direction[1] * qvec[1] +
                       ray.direction[2] * qvec[2]) *
                      invDet;
            if (v < 0.0f || u + v > 1.0f) {
              *res = false;
              return;
            };

            float t =
                (edge2[0] * qvec[0] + edge2[1] * qvec[1] + edge2[2] * qvec[2]) *
                invDet;
            *res = t > 0.0f;
          });
        })
        .wait();

    bool hit = *res;
    sycl::free(res, m_impl->m_q);
    return hit;

  } catch (sycl::_V1::exception &e) {
    std::cout << e.what() << std::endl;
    return false;
  }
}

bool SYCLBackend::is_available() const {
  try {
    auto devices = sycl::device::get_devices();
    return !devices.empty();
  } catch (const sycl::exception &e) {
    return false;
  }
}
} // namespace portableRT