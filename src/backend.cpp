#include "../include/portableRT/intersect_cpu.hpp"

#ifdef USE_OPTIX
#include "../include/portableRT/intersect_optix.hpp"
#endif

#ifdef USE_HIP
#include "../include/portableRT/intersect_hip.hpp"
#endif

#ifdef USE_EMBREE_SYCL
#include "../include/portableRT/intersect_embree_sycl.hpp"
#endif

#ifdef USE_EMBREE_CPU
#include "../include/portableRT/intersect_embree_cpu.hpp"
#endif

#ifdef USE_SYCL
#include "../include/portableRT/intersect_sycl.hpp"
#endif

namespace portableRT {
BackendVar to_variant(Backend *backend) {
#ifdef USE_OPTIX
	if (auto *q = dynamic_cast<OptiXBackend *>(backend))
		return q;
#endif
#ifdef USE_HIP
	if (auto *q = dynamic_cast<HIPBackend *>(backend))
		return q;
#endif
#ifdef USE_EMBREE_SYCL
	if (auto *q = dynamic_cast<EmbreeSYCLBackend *>(backend))
		return q;
#endif
#ifdef USE_EMBREE_CPU
	if (auto *q = dynamic_cast<EmbreeCPUBackend *>(backend))
		return q;
#endif
#ifdef USE_SYCL
	if (auto *q = dynamic_cast<SYCLBackend *>(backend))
		return q;
#endif
	if (auto *q = dynamic_cast<CPUBackend *>(backend))
		return q;

	throw std::runtime_error("Unknown backend");
}
void select_backend(Backend *backend) {
	if (selected_backend)
		selected_backend->shutdown();
	selected_backend = backend;
	var_selected = to_variant(backend);
	backend->init();
}

} // namespace portableRT
