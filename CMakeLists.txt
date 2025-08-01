cmake_minimum_required(VERSION 3.22)
project(portableRT VERSION 0.3.3)

configure_file(
  version.hpp.in
  ${CMAKE_BINARY_DIR}/include/portableRT/version.hpp
  @ONLY
)

option(USE_OPTIX "Enable OptiX backend" OFF)
option(USE_HIP "Enable HIP backend" OFF)
option(USE_EMBREE_SYCL "Enable Embree SYCL backend" OFF)
option(USE_EMBREE_CPU "Enable Embree CPU backend" OFF)
option(USE_SYCL "Enable SYCL backend" OFF)

# All the files that don't require an specific compiler
add_library(portableRT src/portableRT.cpp src/intersect_cpu.cpp src/backend.cpp)

target_include_directories(portableRT PUBLIC ${CMAKE_BINARY_DIR}/include)

# OPTIX
if(USE_OPTIX)
    message(STATUS "Building with OptiX support")
    target_compile_definitions(portableRT PUBLIC USE_OPTIX)

    set(OptiX_ROOT "/homelocal/encalle_local/Downloads/NVIDIA-OptiX-SDK-7.5.0-linux64-x86_64" CACHE PATH "OptiX SDK")

    find_package(CUDAToolkit 11.5 REQUIRED)

    target_include_directories(portableRT PUBLIC
        "${OptiX_ROOT}/include"
        "${CUDAToolkit_INCLUDE_DIRS}"
    )

    target_link_libraries(portableRT PUBLIC
        CUDA::cudart
    )

    file(GLOB CU_FILES "${CMAKE_SOURCE_DIR}/shaders/*.cu")

    set(PTX_OUTPUT_DIR "${CMAKE_BINARY_DIR}/bin/shaders")
    file(MAKE_DIRECTORY ${PTX_OUTPUT_DIR})

    foreach(CU_FILE ${CU_FILES})
        get_filename_component(FILENAME_WE ${CU_FILE} NAME_WE)
        set(PTX_FILE "${PTX_OUTPUT_DIR}/${FILENAME_WE}.ptx")

        add_custom_command(
            OUTPUT ${PTX_FILE}
            COMMAND ${CUDAToolkit_NVCC_EXECUTABLE}
                    -ptx ${CU_FILE}
                    -I "${OptiX_ROOT}/include"
                    -o ${PTX_FILE}
            DEPENDS ${CU_FILE}
            COMMENT "Compiling ${CU_FILE} to PTX"
        )

        list(APPEND PTX_FILES ${PTX_FILE})
    endforeach()

    add_custom_target(compile_shaders ALL DEPENDS ${PTX_FILES})
    add_dependencies(portableRT compile_shaders)

    target_sources(portableRT PUBLIC src/intersect_optix.cpp)
else()
    message(STATUS "Building without OptiX")
endif()

if(USE_HIP)
    message(STATUS "Building with HIP support")
    target_compile_definitions(portableRT PUBLIC USE_HIP)

    set(CMAKE_HIP_ARCHITECTURES
    gfx1030;gfx1031;gfx1032;gfx1033;gfx1034;gfx1035;gfx1036;
    gfx1100;gfx1101;gfx1102;gfx1103)

    set(HIP_C_COMPILER /opt/rocm/llvm/bin/clang)
    set(HIP_CXX_COMPILER /opt/rocm/llvm/bin/clang++)

    set(HIP_DIR "${HIP_ROOT}/lib/cmake/hip")
    find_package(HIP REQUIRED)

    enable_language(HIP)

    set_property(TARGET portableRT PROPERTY HIP_ARCHITECTURES gfx1031)
    set(HIP_SRC src/intersect_hip.cpp)
    set_source_files_properties(${HIP_SRC} PROPERTIES LANGUAGE HIP)

    add_library(portableRT_hip OBJECT ${HIP_SRC})
    target_compile_features(portableRT_hip PUBLIC cxx_std_17)

    target_link_libraries(portableRT PUBLIC portableRT_hip)
    
    find_library(HIP_LIB amdhip64 HINTS ${HIP_ROOT}/lib REQUIRED)
    find_library(HSA_LIB hsa-runtime64 HINTS ${HIP_ROOT}/lib REQUIRED)

    target_compile_options(portableRT_hip PRIVATE -fPIC)
    target_link_libraries(portableRT PUBLIC ${HIP_LIB} ${HSA_LIB})
else()
    message(STATUS "Building without HIP")
endif()

if(USE_EMBREE_CPU AND NOT USE_EMBREE_SYCL)
    message(STATUS "Building with Embree-cpu support")
    target_compile_definitions(portableRT PUBLIC USE_EMBREE_CPU)
   
    set(embree_DIR "${embree_ROOT}/lib/cmake/embree-4.4.0/" CACHE PATH "Embree")

    FIND_PACKAGE(embree 4 REQUIRED)
    
    target_include_directories(portableRT PUBLIC
    "${embree_ROOT}/include"    
    )
    target_sources(portableRT PUBLIC src/intersect_embree_cpu.cpp)
    target_link_libraries(portableRT PUBLIC embree) 
else()
    message(STATUS "Building without Embree-cpu")
endif()

if(USE_EMBREE_SYCL)
    message(STATUS "Building with Embree-sycl support")
    target_compile_definitions(portableRT PUBLIC USE_EMBREE_SYCL)
    target_compile_definitions(portableRT PUBLIC USE_EMBREE_CPU)

    add_library(portableRT_embree_sycl OBJECT src/intersect_embree_sycl.cpp)
    set_source_files_properties(src/intersect_embree_sycl.cpp PROPERTIES LANGUAGE CXX)

    set(embree_DIR "${embree_ROOT}/lib/cmake/embree-4.4.0/" CACHE PATH "Embree")

    FIND_PACKAGE(embree 4 REQUIRED)
    TARGET_LINK_LIBRARIES(portableRT_embree_sycl PUBLIC embree)
    target_include_directories(portableRT PUBLIC
    "${embree_ROOT}/include"    
    )
    TARGET_LINK_LIBRARIES(portableRT_embree_sycl PUBLIC embree)
    target_link_options(portableRT_embree_sycl  PUBLIC -fsycl)
    target_compile_options(portableRT_embree_sycl PRIVATE -fsycl -Xclang -fsycl-allow-func-ptr -Wno-deprecated-declarations) 
    target_link_libraries(portableRT PUBLIC portableRT_embree_sycl)
    target_sources(portableRT PUBLIC src/intersect_embree_cpu.cpp)
else()
    message(STATUS "Building without Embree-sycl")
endif()

if(USE_SYCL)
    message(STATUS "Building with SYCL support")
    target_compile_definitions(portableRT PUBLIC USE_SYCL)
    add_library(portableRT_sycl OBJECT src/intersect_sycl.cpp)

    target_compile_options(portableRT_sycl PRIVATE -fsycl -Xclang -fsycl-allow-func-ptr -Wno-deprecated-declarations) 
    target_link_options(portableRT_sycl  PUBLIC -fsycl)
    target_link_libraries(portableRT PUBLIC portableRT_sycl)
else()
    message(STATUS "Building without SYCL")
endif()

set_target_properties(portableRT PROPERTIES 
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

install(TARGETS portableRT
    RUNTIME DESTINATION bin
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)


set_target_properties(portableRT PROPERTIES LINKER_LANGUAGE CXX)
install(DIRECTORY include/ DESTINATION include)

set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/install)


#target_compile_options(portableRT PUBLIC -fsanitize=address,undefined -g -O0)
#target_link_options(portableRT PUBLIC -fsanitize=address,undefined)

add_subdirectory(examples)
