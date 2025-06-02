#pragma once

#include "backend.hpp"
#include "core.hpp"
#include <optix.h>

namespace portableRT {

class OptiXBackend : public InvokableBackend<OptiXBackend> {
public:
  OptiXBackend() : InvokableBackend(BackendType::OPTIX, "OptiX") {
    static RegisterBackend reg(*this);
  }

  bool intersect_tri(const std::array<float, 9> &vertices, const Ray &ray);
  bool is_available() const override;
  void init() override;
  void shutdown() override;

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
  CUdeviceptr m_d_res;
};

static OptiXBackend optix_backend;

} // namespace portableRT