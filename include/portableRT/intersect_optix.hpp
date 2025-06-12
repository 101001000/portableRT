#pragma once

#include "backend.hpp"
#include "core.hpp"
#include <optix.h>

namespace portableRT {

class OptiXBackend : public InvokableBackend<OptiXBackend> {
public:
  OptiXBackend() : InvokableBackend("OptiX") {
    static RegisterBackend reg(*this);
  }

  std::vector<float> nearest_hits(const std::vector<Ray> &rays);
  bool is_available() const override;
  void init() override;
  void shutdown() override;
  void set_tris(const Tris &tris) override;
  std::string device_name() const override;

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