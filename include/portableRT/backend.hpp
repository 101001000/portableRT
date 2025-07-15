#pragma once

#include <iostream>
#include <string>
#include <vector>

#include "core.hpp"

namespace portableRT {

using NearestHitsDispatchFn = std::vector<HitReg> (*)(void *,
                                                     const std::vector<Ray> &);

class Backend {
public:
  Backend(std::string name, void *self_ptr, NearestHitsDispatchFn fn)
      : name_{std::move(name)}, self_{self_ptr}, nearest_hits_{fn} {}

  const std::string &name() const { return name_; }

  std::vector<HitReg> nearest_hits(const std::vector<Ray> &rays) {
    return nearest_hits_(self_, rays);
  }

  // TODO make move version
  virtual void set_tris(const Tris &tris) = 0;
  virtual bool is_available() const = 0;
  virtual void init() = 0;
  virtual void shutdown() = 0;
  virtual std::string device_name() const = 0;

  NearestHitsDispatchFn nearest_hits_;
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
  static std::vector<HitReg> dispatch(void *self, const std::vector<Ray> &rays) {
    return static_cast<Derived *>(self)->nearest_hits(rays);
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

inline NearestHitsDispatchFn nearest_hits_call = nullptr;

inline std::vector<HitReg> nearest_hits(const std::vector<Ray> &rays) {
  return nearest_hits_call(selected_backend->self_, rays);
}

inline void select_backend(Backend *backend) {
  if (selected_backend)
    selected_backend->shutdown();
  selected_backend = backend;
  backend->init();
  nearest_hits_call = backend->nearest_hits_;
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