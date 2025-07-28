#pragma once

#include "backend.hpp"
#include "core.hpp"
#include <array>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nvrtc.h>
#include <optix.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>
#include <sstream>
#include <stdexcept>

struct __align__(16) Params {
	OptixTraversableHandle handle;
	float4 *origins;
	float4 *directions;
	portableRT::FullTags *results;
};

static inline float3 toFloat3(const std::array<float, 3> &a) {
	return make_float3(a[0], a[1], a[2]);
}

static inline float4 toFloat4(const std::array<float, 3> &a) {
	return make_float4(a[0], a[1], a[2], 0);
}

#define CUDA_CHECK(call)                                                                           \
	do {                                                                                           \
		cudaError_t _e = (call);                                                                   \
		if (_e != cudaSuccess) {                                                                   \
			std::fprintf(stderr, "CUDA error \"%s\" en %s:%d — %s\n", cudaGetErrorString(_e),      \
			             __FILE__, __LINE__, #call);                                               \
		}                                                                                          \
	} while (0)

#define OPTIX_CHECK(call)                                                                          \
	do {                                                                                           \
		OptixResult _r = (call);                                                                   \
		if (_r != OPTIX_SUCCESS) {                                                                 \
			std::fprintf(stderr, "OptiX error (%d) en %s:%d — %s\n", static_cast<int>(_r),         \
			             __FILE__, __LINE__, #call);                                               \
		}                                                                                          \
	} while (0)

namespace portableRT {

class OptiXBackend : public InvokableBackend<OptiXBackend> {
  public:
	OptiXBackend() : InvokableBackend("OptiX") { static RegisterBackend reg(*this); }

	bool is_available() const override;
	void init() override;
	void shutdown() override;
	void set_tris(const Tris &tris) override;
	std::string device_name() const override;

	template <class... Tags>
	std::vector<HitReg<Tags...>> nearest_hits(const std::vector<Ray> &rays) {

		std::vector<HitReg<Tags...>> hits;
		hits.reserve(rays.size());

		CUdeviceptr d_res;
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_res), sizeof(FullTags) * rays.size()));
		CUdeviceptr d_origins;
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_origins), sizeof(float4) * rays.size()));
		CUdeviceptr d_directions;
		CUDA_CHECK(
		    cudaMalloc(reinterpret_cast<void **>(&d_directions), sizeof(float4) * rays.size()));

		std::vector<float4> origins(rays.size());
		std::vector<float4> directions(rays.size());

		for (int i = 0; i < rays.size(); i++) {
			origins[i] = toFloat4(rays[i].origin);
			directions[i] = toFloat4(rays[i].direction);
		}

		CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d_origins), origins.data(),
		                      sizeof(float4) * rays.size(), cudaMemcpyHostToDevice));
		CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d_directions), directions.data(),
		                      sizeof(float4) * rays.size(), cudaMemcpyHostToDevice));

		Params params;
		params.handle = m_gas_handle;
		params.origins = reinterpret_cast<float4 *>(d_origins);
		params.directions = reinterpret_cast<float4 *>(d_directions);
		params.results = reinterpret_cast<FullTags *>(d_res);

		CUdeviceptr d_params;
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_params), sizeof(params)));
		CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d_params), &params, sizeof(params),
		                      cudaMemcpyHostToDevice));
		OPTIX_CHECK(
		    optixLaunch(m_pipeline, m_stream, d_params, sizeof(params), &m_sbt, rays.size(), 1, 1));

		CUDA_CHECK(cudaStreamSynchronize(m_stream));

		FullTags *h_res = new FullTags[rays.size()];
		CUDA_CHECK(cudaMemcpy(h_res, (void *)d_res, sizeof(FullTags) * rays.size(),
		                      cudaMemcpyDeviceToHost));

		std::transform(h_res, h_res + rays.size(), std::back_inserter(hits),
		               [](const FullTags &h) { return slice<Tags...>(h); });

		delete[] h_res;

		CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_params)));
		CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_res)));
		CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_origins)));
		CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_directions)));

		return hits;
	}

  private:
	OptixDeviceContext m_context;

	OptixModule m_module{};
	OptixModuleCompileOptions m_mco{};
	OptixPipelineCompileOptions m_pco{};
	OptixPipeline m_pipeline = nullptr;
	OptixShaderBindingTable m_sbt = {};
	OptixProgramGroup m_raygen_prog_group = nullptr;
	OptixProgramGroup m_miss_prog_group = nullptr;
	OptixProgramGroup m_hitgroup_prog_group = nullptr;
	CUstream m_stream;
	OptixTraversableHandle m_gas_handle;
	CUdeviceptr m_d_gas_output_buffer;
};

static OptiXBackend optix_backend;

} // namespace portableRT