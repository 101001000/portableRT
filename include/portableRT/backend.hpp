#pragma once

#include <iostream>
#include <string>
#include <vector>

#include "core.hpp"

namespace portableRT {

using IntersectDispatchFn = bool (*)(void *, const std::array<float, 9> &,
                                     const Ray &);

class Backend {
public:
  Backend(std::string name, void *self_ptr,
          IntersectDispatchFn fn)
      : name_{std::move(name)}, self_{self_ptr}, intersect_{fn} {}

  const std::string &name() const { return name_; }

  bool intersect_tri(const std::array<float, 9> &tri, const Ray &r) {
    return intersect_(self_, tri, r);
  }

  virtual bool is_available() const = 0;
  virtual void init() = 0;
  virtual void shutdown() = 0;

  IntersectDispatchFn intersect_;
  void *self_;

private:
  std::string name_;
};

template <class Derived> class InvokableBackend : public Backend {
public:
  InvokableBackend(std::string name)
      : Backend{std::move(name), static_cast<void *>(this), &dispatch} {}

protected:
  ~InvokableBackend() = default;

private:
  static bool dispatch(void *self, const std::array<float, 9> &tri,
                       const Ray &r) {
    return static_cast<Derived *>(self)->intersect_tri(tri, r);
  }
};

inline Backend *selected_backend = nullptr;

// TODO: See if I need lazy initialization for this, as it should be UB
inline std::vector<Backend *> all_backends_;
inline const std::vector<Backend *> &all_backends() { return all_backends_; }

inline std::vector<Backend *> available_backends_;
inline const std::vector<Backend *> &available_backends() {
  return available_backends_;
}

inline IntersectDispatchFn intersect_tri_call = nullptr;

inline bool intersect_tri(const std::array<float, 9> &v, const Ray &r) {
  return intersect_tri_call(selected_backend->self_, v, r);
}

inline void select_backend(Backend *backend) {
  if (selected_backend)
    selected_backend->shutdown();
  selected_backend = backend;
  backend->init();
  intersect_tri_call = backend->intersect_;
}

struct RegisterBackend {
  RegisterBackend(Backend &b) {
    if (b.is_available()) {
      if (selected_backend == nullptr) {
        select_backend(&b);
      }
      available_backends_.push_back(&b);
    }
    all_backends_.push_back(&b);
  }
};

} // namespace portableRT