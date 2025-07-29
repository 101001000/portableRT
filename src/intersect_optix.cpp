#include "../include/portableRT/intersect_optix.hpp"

#include <optix_function_table_definition.h>

template <typename T> struct SbtRecord {
	__align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
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

	std::string PTX = loadFile("tri.ptx");

	std::vector<std::string> tags = all_tags_combinations();

	m_modules.reserve(tags.size());
	m_pcos.reserve(tags.size());
	m_pipelines.reserve(tags.size());
	m_raygen_prog_groups.reserve(tags.size());
	m_miss_prog_groups.reserve(tags.size());
	m_hitgroup_prog_groups.reserve(tags.size());
	m_hitreg_names.reserve(tags.size());
	m_sbts.reserve(tags.size());

	for (int i = 0; i < tags.size(); i++) {

		m_modules.push_back(OptixModule());
		m_pcos.push_back(OptixPipelineCompileOptions());
		m_pipelines.push_back(OptixPipeline());
		m_raygen_prog_groups.push_back(OptixProgramGroup());
		m_miss_prog_groups.push_back(OptixProgramGroup());
		m_hitgroup_prog_groups.push_back(OptixProgramGroup());
		m_sbts.push_back(OptixShaderBindingTable());
		m_hitreg_names.push_back(tags[i]);

		OptixPipelineCompileOptions &pco = m_pcos.back();
		OptixModule &mod = m_modules.back();
		OptixPipeline &pipeline = m_pipelines.back();
		OptixProgramGroup &raygen_prog_group = m_raygen_prog_groups.back();
		OptixProgramGroup &miss_prog_group = m_miss_prog_groups.back();
		OptixProgramGroup &hitgroup_prog_group = m_hitgroup_prog_groups.back();
		OptixShaderBindingTable &sbt = m_sbts.back();

		pco.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
		pco.usesMotionBlur = false;
		pco.exceptionFlags = OPTIX_EXCEPTION_FLAG_DEBUG | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH |
		                     OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
		pco.numPayloadValues = 4;
		pco.numAttributeValues = 3;
		std::string param_name = "g_params_" + tags[i];
		pco.pipelineLaunchParamsVariableName = param_name.c_str();
		pco.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

		OPTIX_CHECK(optixModuleCreateFromPTX(m_context, &m_mco, &pco, PTX.c_str(), PTX.size(),
		                                     nullptr, nullptr, &mod));

		{
			OptixProgramGroupOptions program_group_options = {}; // Initialize to zeros

			OptixProgramGroupDesc raygen_prog_group_desc = {}; //
			raygen_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
			raygen_prog_group_desc.raygen.module = mod;
			std::string entry_name = "__raygen__rg__" + tags[i];
			raygen_prog_group_desc.raygen.entryFunctionName = entry_name.c_str();
			OPTIX_CHECK(optixProgramGroupCreate(m_context, &raygen_prog_group_desc,
			                                    1, // num program groups
			                                    &program_group_options, nullptr, 0,
			                                    &raygen_prog_group));

			OptixProgramGroupDesc miss_prog_group_desc = {};
			miss_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
			miss_prog_group_desc.miss.module = mod;
			miss_prog_group_desc.miss.entryFunctionName = "__miss__ms";
			OPTIX_CHECK(optixProgramGroupCreate(m_context, &miss_prog_group_desc,
			                                    1, // num program groups
			                                    &program_group_options, nullptr, 0,
			                                    &miss_prog_group));

			OptixProgramGroupDesc hitgroup_prog_group_desc = {};
			hitgroup_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
			hitgroup_prog_group_desc.hitgroup.moduleCH = mod;
			hitgroup_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
			OPTIX_CHECK(optixProgramGroupCreate(m_context, &hitgroup_prog_group_desc,
			                                    1, // num program groups
			                                    &program_group_options, nullptr, 0,
			                                    &hitgroup_prog_group));
		}

		//
		// Link pipeline
		//

		{
			const uint32_t max_trace_depth = 1;
			OptixProgramGroup program_groups[] = {raygen_prog_group, miss_prog_group,
			                                      hitgroup_prog_group};

			OptixPipelineLinkOptions pipeline_link_options = {};
			pipeline_link_options.maxTraceDepth = max_trace_depth;
			pipeline_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
			OPTIX_CHECK(optixPipelineCreate(m_context, &pco, &pipeline_link_options, program_groups,
			                                sizeof(program_groups) / sizeof(program_groups[0]),
			                                nullptr, 0, &pipeline));

			OptixStackSizes stack_sizes = {};
			for (auto &prog_group : program_groups) {
				OPTIX_CHECK(optixUtilAccumulateStackSizes(prog_group, &stack_sizes));
			}

			uint32_t direct_callable_stack_size_from_traversal;
			uint32_t direct_callable_stack_size_from_state;
			uint32_t continuation_stack_size;
			OPTIX_CHECK(optixUtilComputeStackSizes(&stack_sizes, max_trace_depth,
			                                       0, // maxCCDepth
			                                       0, // maxDCDEpth
			                                       &direct_callable_stack_size_from_traversal,
			                                       &direct_callable_stack_size_from_state,
			                                       &continuation_stack_size));
			OPTIX_CHECK(optixPipelineSetStackSize(
			    pipeline, direct_callable_stack_size_from_traversal,
			    direct_callable_stack_size_from_state, continuation_stack_size,
			    1 // maxTraversableDepth
			    ));
		}
		// std::cout << "Link pipeline ok" << std::endl;
		//
		//  Set up shader binding table
		//

		CUdeviceptr raygen_record;
		const size_t raygen_record_size = sizeof(RayGenSbtRecord);
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&raygen_record), raygen_record_size));
		RayGenSbtRecord rg_sbt;
		OPTIX_CHECK(optixSbtRecordPackHeader(raygen_prog_group, &rg_sbt));
		CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(raygen_record), &rg_sbt, raygen_record_size,
		                      cudaMemcpyHostToDevice));

		CUdeviceptr miss_record;
		size_t miss_record_size = sizeof(MissSbtRecord);
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&miss_record), miss_record_size));
		MissSbtRecord ms_sbt;
		ms_sbt.data = {0.3f, 0.1f, 0.2f};
		OPTIX_CHECK(optixSbtRecordPackHeader(miss_prog_group, &ms_sbt));
		CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(miss_record), &ms_sbt, miss_record_size,
		                      cudaMemcpyHostToDevice));

		CUdeviceptr hitgroup_record;
		size_t hitgroup_record_size = sizeof(HitGroupSbtRecord);
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&hitgroup_record), hitgroup_record_size));
		HitGroupSbtRecord hg_sbt;
		OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_prog_group, &hg_sbt));
		CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(hitgroup_record), &hg_sbt,
		                      hitgroup_record_size, cudaMemcpyHostToDevice));

		sbt.raygenRecord = raygen_record;
		sbt.missRecordBase = miss_record;
		sbt.missRecordStrideInBytes = sizeof(MissSbtRecord);
		sbt.missRecordCount = 1;
		sbt.hitgroupRecordBase = hitgroup_record;
		sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
		sbt.hitgroupRecordCount = 1;
	}

	CUDA_CHECK(cudaStreamCreate(&m_stream));
}

void OptiXBackend::shutdown() {
	for (auto &sbt : m_sbts) {
		CUDA_CHECK(cudaFree(reinterpret_cast<void *>(sbt.raygenRecord)));
		CUDA_CHECK(cudaFree(reinterpret_cast<void *>(sbt.missRecordBase)));
		CUDA_CHECK(cudaFree(reinterpret_cast<void *>(sbt.hitgroupRecordBase)));
	}
	CUDA_CHECK(cudaFree(reinterpret_cast<void *>(m_d_gas_output_buffer)));
	for (auto &hitgroup_prog_group : m_hitgroup_prog_groups) {
		OPTIX_CHECK(optixProgramGroupDestroy(hitgroup_prog_group));
	}
	for (auto &miss_prog_group : m_miss_prog_groups) {
		OPTIX_CHECK(optixProgramGroupDestroy(miss_prog_group));
	}
	for (auto &raygen_prog_group : m_raygen_prog_groups) {
		OPTIX_CHECK(optixProgramGroupDestroy(raygen_prog_group));
	}
	CUDA_CHECK(cudaStreamDestroy(m_stream));
	for (auto &pipeline : m_pipelines) {
		OPTIX_CHECK(optixPipelineDestroy(pipeline));
	}
	for (auto &module : m_modules) {
		OPTIX_CHECK(optixModuleDestroy(module));
	}
	OPTIX_CHECK(optixDeviceContextDestroy(m_context));
}

void OptiXBackend::set_tris(const Tris &tris) {

	{
		// Use default options for simplicity.  In a real use case we would want
		// to enable compaction, etc
		OptixAccelBuildOptions accel_options = {};
		accel_options.buildFlags = OPTIX_BUILD_FLAG_NONE;
		accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

		// Triangle build input: simple list of three vertices

		const size_t vertices_size = sizeof(float) * tris.size() * 9;
		CUdeviceptr d_vertices = 0;
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_vertices), vertices_size));
		CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(d_vertices), tris.data(), vertices_size,
		                      cudaMemcpyHostToDevice));

		// Our build input is a simple list of non-indexed triangle vertices
		const uint32_t triangle_input_flags[1] = {OPTIX_GEOMETRY_FLAG_NONE};
		OptixBuildInput triangle_input = {};
		triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
		triangle_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
		triangle_input.triangleArray.vertexStrideInBytes = sizeof(float3);
		triangle_input.triangleArray.numVertices = static_cast<uint32_t>(tris.size() * 3);
		triangle_input.triangleArray.vertexBuffers = &d_vertices;
		triangle_input.triangleArray.flags = triangle_input_flags;
		triangle_input.triangleArray.numSbtRecords = 1;

		OptixAccelBufferSizes gas_buffer_sizes;
		OPTIX_CHECK(optixAccelComputeMemoryUsage(m_context, &accel_options, &triangle_input,
		                                         1, // Number of build inputs
		                                         &gas_buffer_sizes));
		CUdeviceptr d_temp_buffer_gas;
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_temp_buffer_gas),
		                      gas_buffer_sizes.tempSizeInBytes));
		CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&m_d_gas_output_buffer),
		                      gas_buffer_sizes.outputSizeInBytes));

		OPTIX_CHECK(optixAccelBuild(m_context,
		                            0, // CUDA stream
		                            &accel_options, &triangle_input,
		                            1, // num build inputs
		                            d_temp_buffer_gas, gas_buffer_sizes.tempSizeInBytes,
		                            m_d_gas_output_buffer, gas_buffer_sizes.outputSizeInBytes,
		                            &m_gas_handle,
		                            nullptr, // emitted property list
		                            0        // num emitted properties
		                            ));

		// We can now free the scratch space buffer used during build and the
		// vertex inputs, since they are not needed by our trivial shading method
		CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_temp_buffer_gas)));
		CUDA_CHECK(cudaFree(reinterpret_cast<void *>(d_vertices)));
	}
}

bool OptiXBackend::is_available() const {
	if (cudaFree(0) != cudaSuccess)
		return false;
	if (optixInit() != OPTIX_SUCCESS)
		return false;
	return true;
}

std::string OptiXBackend::device_name() const {
	int device = 0;
	cudaDeviceProp props{};
	if (cudaGetDevice(&device) != cudaSuccess)
		return "unsupported";

	if (cudaGetDeviceProperties(&props, device) != cudaSuccess)
		return "unsupported";

	return props.name;
}

} // namespace portableRT
