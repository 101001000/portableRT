#include <array>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nvrtc.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>
#include <sstream>
#include <stdexcept>

#include "../include/portableRT/intersect_optix.hpp"

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t _e = (call);                                                   \
    if (_e != cudaSuccess) {                                                   \
      std::fprintf(stderr, "CUDA error \"%s\" en %s:%d — %s\n",                \
                   cudaGetErrorString(_e), __FILE__, __LINE__, #call);         \
    }                                                                          \
  } while (0)

#define OPTIX_CHECK(call)                                                      \
  do {                                                                         \
    OptixResult _r = (call);                                                   \
    if (_r != OPTIX_SUCCESS) {                                                 \
      std::fprintf(stderr, "OptiX error (%d) en %s:%d — %s\n",                 \
                   static_cast<int>(_r), __FILE__, __LINE__, #call);           \
    }                                                                          \
  } while (0)

template <typename T> struct SbtRecord {
  __align__(
      OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
  T data;
};

struct RayGenData {};

struct MissData {
  float3 bg_color;
};

struct HitGroupData {};

typedef SbtRecord<RayGenData> RayGenSbtRecord;
typedef SbtRecord<MissData> MissSbtRecord;
typedef SbtRecord<HitGroupData> HitGroupSbtRecord;

std::string loadFile(const std::string &filename) {
  namespace fs = std::filesystem;
  fs::path exePath = fs::canonical("/proc/self/exe").parent_path();
  fs::path fullPath = exePath / "shaders" / filename;

  std::ifstream file(fullPath);
  if (!file)
    throw std::runtime_error("Cannot open file: " + fullPath.string());

  return std::string(std::istreambuf_iterator<char>(file), {});
}

struct Params {
  OptixTraversableHandle handle;
  float3 origin;
  float3 direction;
  int *result;
};

static inline float3 toFloat3(const std::array<float, 3> &a) {
  return make_float3(a[0], a[1], a[2]);
}

namespace portableRT {

void OptiXBackend::init() {
  m_context = nullptr;
  {
    // Initialize CUDA
    CUDA_CHECK(cudaFree(0));

    // Initialize the OptiX API, loading all API entry points
    OPTIX_CHECK(optixInit());

    // Specify context options
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = nullptr;
    options.logCallbackLevel = 4;

    // Associate a CUDA context (and therefore a specific GPU) with this
    // device context
    CUcontext cuCtx = 0; // zero means take the current context
    OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &m_context));
  }

  m_pco.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
  m_pco.usesMotionBlur = false;
  m_pco.exceptionFlags = OPTIX_EXCEPTION_FLAG_DEBUG |
                         OPTIX_EXCEPTION_FLAG_TRACE_DEPTH |
                         OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
  m_pco.numPayloadValues = 1; // ← usas optixSetPayload_0
  m_pco.numAttributeValues = 3;
  m_pco.pipelineLaunchParamsVariableName =
      "params"; // ← tu PTX declara .const params
  m_pco.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

  std::string PTX = loadFile("tri.ptx");
  OPTIX_CHECK(optixModuleCreateFromPTX(m_context, &m_mco, &m_pco, PTX.c_str(),
                                       PTX.size(), nullptr, nullptr,
                                       &m_module));

  {
    OptixProgramGroupOptions program_group_options = {}; // Initialize to zeros

    OptixProgramGroupDesc raygen_prog_group_desc = {}; //
    raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygen_prog_group_desc.raygen.module = m_module;
    raygen_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";
    OPTIX_CHECK(optixProgramGroupCreate(m_context, &raygen_prog_group_desc,
                                        1, // num program groups
                                        &program_group_options, nullptr, 0,
                                        &m_raygen_prog_group));

    OptixProgramGroupDesc miss_prog_group_desc = {};
    miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_prog_group_desc.miss.module = m_module;
    miss_prog_group_desc.miss.entryFunctionName = "__miss__ms";
    OPTIX_CHECK(optixProgramGroupCreate(m_context, &miss_prog_group_desc,
                                        1, // num program groups
                                        &program_group_options, nullptr, 0,
                                        &m_miss_prog_group));

    OptixProgramGroupDesc hitgroup_prog_group_desc = {};
    hitgroup_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitgroup_prog_group_desc.hitgroup.moduleCH = m_module;
    hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
    OPTIX_CHECK(optixProgramGroupCreate(m_context, &hitgroup_prog_group_desc,
                                        1, // num program groups
                                        &program_group_options, nullptr, 0,
                                        &m_hitgroup_prog_group));
  }

  //
  // Link pipeline
  //

  {
    const uint32_t max_trace_depth = 1;
    OptixProgramGroup program_groups[] = {
        m_raygen_prog_group, m_miss_prog_group, m_hitgroup_prog_group};

    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth = max_trace_depth;
    pipeline_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
    OPTIX_CHECK(optixPipelineCreate(
        m_context, &m_pco, &pipeline_link_options, program_groups,
        sizeof(program_groups) / sizeof(program_groups[0]), nullptr, 0,
        &m_pipeline));

    OptixStackSizes stack_sizes = {};
    for (auto &prog_group : program_groups) {
      OPTIX_CHECK(optixUtilAccumulateStackSizes(prog_group, &stack_sizes));
    }

    uint32_t direct_callable_stack_size_from_traversal;
    uint32_t direct_callable_stack_size_from_state;
    uint32_t continuation_stack_size;
    OPTIX_CHECK(optixUtilComputeStackSizes(
        &stack_sizes, max_trace_depth,
        0, // maxCCDepth
        0, // maxDCDEpth
        &direct_callable_stack_size_from_traversal,
        &direct_callable_stack_size_from_state, &continuation_stack_size));
    OPTIX_CHECK(optixPipelineSetStackSize(
        m_pipeline, direct_callable_stack_size_from_traversal,
        direct_callable_stack_size_from_state, continuation_stack_size,
        1 // maxTraversableDepth
        ));
  }
  // std::cout << "Link pipeline ok" << std::endl;
  //
  //  Set up shader binding table
  //

  {
    CUdeviceptr raygen_record;
    const size_t raygen_record_size = sizeof(RayGenSbtRecord);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&raygen_record),
                          raygen_record_size));
    RayGenSbtRecord rg_sbt;
    OPTIX_CHECK(optixSbtRecordPackHeader(m_raygen_prog_group, &rg_sbt));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(raygen_record), &rg_sbt,
                          raygen_record_size, cudaMemcpyHostToDevice));

    CUdeviceptr miss_record;
    size_t miss_record_size = sizeof(MissSbtRecord);
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&miss_record), miss_record_size));
    MissSbtRecord ms_sbt;
    ms_sbt.data = {0.3f, 0.1f, 0.2f};
    OPTIX_CHECK(optixSbtRecordPackHeader(m_miss_prog_group, &ms_sbt));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(miss_record), &ms_sbt,
                          miss_record_size, cudaMemcpyHostToDevice));

    CUdeviceptr hitgroup_record;
    size_t hitgroup_record_size = sizeof(HitGroupSbtRecord);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&hitgroup_record),
                          hitgroup_record_size));
    HitGroupSbtRecord hg_sbt;
    OPTIX_CHECK(optixSbtRecordPackHeader(m_hitgroup_prog_group, &hg_sbt));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(hitgroup_record), &hg_sbt,
                          hitgroup_record_size, cudaMemcpyHostToDevice));

    m_sbt.raygenRecord = raygen_record;
    m_sbt.missRecordBase = miss_record;
    m_sbt.missRecordStrideInBytes = sizeof(MissSbtRecord);
    m_sbt.missRecordCount = 1;
    m_sbt.hitgroupRecordBase = hitgroup_record;
    m_sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
    m_sbt.hitgroupRecordCount = 1;
  }

  CUDA_CHECK(cudaStreamCreate(&m_stream));

  CUDA_CHECK(cudaMalloc((void **)&m_d_res, sizeof(int)));
}

void OptiXBackend::shutdown() {
  CUDA_CHECK(cudaFree(reinterpret_cast<void *>(m_sbt.raygenRecord)));
  CUDA_CHECK(cudaFree(reinterpret_cast<void *>(m_sbt.missRecordBase)));
  CUDA_CHECK(cudaFree(reinterpret_cast<void *>(m_sbt.hitgroupRecordBase)));
  OPTIX_CHECK(optixProgramGroupDestroy(m_hitgroup_prog_group));
  OPTIX_CHECK(optixProgramGroupDestroy(m_miss_prog_group));
  OPTIX_CHECK(optixProgramGroupDestroy(m_raygen_prog_group));
  CUDA_CHECK(cudaFree(reinterpret_cast<void *>(m_d_res)));
  CUDA_CHECK(cudaStreamDestroy(m_stream));
  OPTIX_CHECK(optixPipelineDestroy(m_pipeline));
  OPTIX_CHECK(optixModuleDestroy(m_module));
  OPTIX_CHECK(optixDeviceContextDestroy(m_context));
}

void OptiXBackend::set_tris(const Tris &tris) {
  CUdeviceptr d_gas_output_buffer;
  {
    // Use default options for simplicity.  In a real use case we would want
    // to enable compaction, etc
    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_NONE;
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    // Triangle build input: simple list of three vertices

    const size_t vertices_size = sizeof(float) * tris.size() * 9;
    CUdeviceptr d_vertices = 0;
    CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void **>(&d_vertices), vertices_size));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d_vertices), tris.data(),
                          vertices_size, cudaMemcpyHostToDevice));

    // Our build input is a simple list of non-indexed triangle vertices
    const uint32_t triangle_input_flags[1] = {OPTIX_GEOMETRY_FLAG_NONE};
    OptixBuildInput triangle_input = {};
    triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    triangle_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    triangle_input.triangleArray.vertexStrideInBytes = sizeof(float3);
    triangle_input.triangleArray.numVertices =
        static_cast<uint32_t>(tris.size() * 3);
    triangle_input.triangleArray.vertexBuffers = &d_vertices;
    triangle_input.triangleArray.flags = triangle_input_flags;
    triangle_input.triangleArray.numSbtRecords = 1;

    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(m_context, &accel_options,
                                             &triangle_input,
                                             1, // Number of build inputs
                                             &gas_buffer_sizes));
    CUdeviceptr d_temp_buffer_gas;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_temp_buffer_gas),
                          gas_buffer_sizes.tempSizeInBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_gas_output_buffer),
                          gas_buffer_sizes.outputSizeInBytes));

    OPTIX_CHECK(optixAccelBuild(
        m_context,
        0, // CUDA stream
        &accel_options, &triangle_input,
        1, // num build inputs
        d_temp_buffer_gas, gas_buffer_sizes.tempSizeInBytes,
        d_gas_output_buffer, gas_buffer_sizes.outputSizeInBytes, &m_gas_handle,
        nullptr, // emitted property list
        0        // num emitted properties
        ));

    // We can now free the scratch space buffer used during build and the
    // vertex inputs, since they are not needed by our trivial shading method
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_temp_buffer_gas)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_vertices)));
  }

  { CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_gas_output_buffer))); }
}

bool OptiXBackend::intersect_tris(const Ray &ray) {

  int h_res = 0;
  {

    Params params;
    params.handle = m_gas_handle;
    params.origin = toFloat3(ray.origin);
    params.direction = toFloat3(ray.direction);
    params.result = reinterpret_cast<int *>(m_d_res);

    CUdeviceptr d_param;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_param), sizeof(Params)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d_param), &params,
                          sizeof(params), cudaMemcpyHostToDevice));

    OPTIX_CHECK(optixLaunch(m_pipeline, m_stream, d_param, sizeof(Params),
                            &m_sbt, 1, 1, 1));

    CUDA_CHECK(cudaStreamSynchronize(m_stream));
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_param)));

    CUDA_CHECK(cudaMemcpy(&h_res, (void *)m_d_res, sizeof(int),
                          cudaMemcpyDeviceToHost));
  }

  return h_res;
}

bool OptiXBackend::is_available() const {
  if (cudaFree(0) != cudaSuccess)
    return false;
  if (optixInit() != OPTIX_SUCCESS)
    return false;
  return true;
}

} // namespace portableRT
