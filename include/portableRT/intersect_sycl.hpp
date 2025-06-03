#pragma once
#include <memory>

#include "backend.hpp"
#include "core.hpp"

namespace portableRT {

struct SYCLBackendImpl;

class SYCLBackend : public InvokableBackend<SYCLBackend> {
public:
  SYCLBackend();
  ~SYCLBackend();

  bool intersect_tris(const Ray &ray);
  bool is_available() const override;
  void init() override;
  void shutdown() override;
  void set_tris(const Tris &tris) override;

private:
  // Ugly solution to overcome the issues with forward declarations of the sycl
  // type alias in the intel implementation I need to move the sycl import to
  // the .cpp instead the .h so the hip compiler ignores the sycl headers. Other
  // option is to pack the sycl headers, but that would break sycl dependency.
  std::unique_ptr<SYCLBackendImpl> m_impl;
  Tris m_tris;
};

static SYCLBackend sycl_backend;

} // namespace portableRT