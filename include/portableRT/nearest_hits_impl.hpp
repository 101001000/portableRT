#pragma once
#include "intersect_cpu.hpp"

#ifdef USE_OPTIX
#include "intersect_optix.hpp"
#endif

#ifdef USE_HIP
#include "intersect_hip.hpp"
#endif

#ifdef USE_EMBREE_SYCL
#include "intersect_embree_sycl.hpp"
#endif

#ifdef USE_EMBREE_CPU
#include "intersect_embree_cpu.hpp"
#endif

#ifdef USE_SYCL
#include "intersect_sycl.hpp"
#endif

#include "backend.hpp"

namespace portableRT {

template <class... Tags> std::vector<HitReg<Tags...>> nearest_hits(const std::vector<Ray> &rays) {
	return std::visit(
	    [&](auto *ptr) -> std::vector<HitReg<Tags...>> {
		    if (!ptr)
			    throw std::runtime_error("Unknown backend");
		    return ptr->template nearest_hits<Tags...>(rays);
	    },
	    var_selected);
}

} // namespace portableRT