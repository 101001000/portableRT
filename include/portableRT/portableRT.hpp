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

#include "nearest_hits_impl.hpp"