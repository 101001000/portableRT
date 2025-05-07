#pragma once

#include "intersect_cpu.h"

#ifdef USE_OPTIX
#include "intersect_optix.h"
#endif

#ifdef USE_HIP
#include "intersect_hip.h"
#endif

#ifdef USE_EMBREE_SYCL
#include "intersect_embree_sycl.h"
#endif