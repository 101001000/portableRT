#pragma once

#include <iostream>
#include <string>
#include <variant>
#include <vector>

#include "core.hpp"

namespace portableRT {

class Backend {
  public:
	Backend(std::string name, void *self_ptr) : name_{std::move(name)}, self_{self_ptr} {}

	const std::string &name() const { return name_; }

	// TODO make move version
	virtual void set_tris(const Tris &tris) = 0;
	virtual bool is_available() const = 0;
	virtual void init() = 0;
	virtual void shutdown() = 0;
	virtual std::string device_name() const = 0;

	void *self_;

  private:
	std::string name_;
};

template <class Derived> class InvokableBackend : public Backend {
  public:
	InvokableBackend(std::string name) : Backend{std::move(name), static_cast<void *>(this)} {}

  protected:
	~InvokableBackend() = default;
};

inline Backend *selected_backend = nullptr;

// TODO: See if I need lazy initialization for this, as it should be UB
inline std::vector<Backend *> all_backends_;
inline const std::vector<Backend *> &all_backends() { return all_backends_; }

inline std::vector<Backend *> available_backends_;
inline const std::vector<Backend *> &available_backends() { return available_backends_; }

class OptiXBackend;
class HIPBackend;
class EmbreeSYCLBackend;
class EmbreeCPUBackend;
class SYCLBackend;
class CPUBackend;

using BackendVar = std::variant<
#if defined(USE_OPTIX)
    OptiXBackend *,
#endif
#if defined(USE_HIP)
    HIPBackend *,
#endif
#if defined(USE_EMBREE_SYCL)
    EmbreeSYCLBackend *,
#endif
#if defined(USE_EMBREE_CPU)
    EmbreeCPUBackend *,
#endif
#if defined(USE_SYCL)
    SYCLBackend *,
#endif
    CPUBackend *>;

inline BackendVar var_selected{static_cast<CPUBackend *>(nullptr)};

template <class... Tags>
inline std::vector<HitReg<Tags...>> nearest_hits(const std::vector<Ray> &rays) {
	return std::visit(
	    [&](auto *ptr) -> std::vector<HitReg<Tags...>> {
		    if (!ptr)
			    throw std::runtime_error("Unknown backend");
		    return ptr->template nearest_hits<Tags...>(rays);
	    },
	    var_selected);
}

inline std::vector<HitReg<filter::uv, filter::t, filter::primitive_id>>
nearest_hits(const std::vector<Ray> &rays) {
	return nearest_hits<filter::uv, filter::t, filter::primitive_id>(rays);
}

BackendVar to_variant(Backend *backend);

void select_backend(Backend *backend);

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