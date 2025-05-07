cmake_minimum_required(VERSION 3.22)
project(portableRT)

option(USE_OPTIX "Enable OptiX backend" OFF)
option(USE_HIP "Enable HIP backend" OFF)
option(USE_EMBREE_SYCL "Enable Embree SYCL backend" OFF)
option(USE_EMBREE_CPU "Enable Embree CPU backend" ON)

add_library(portableRT src/portableRT.cpp)

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
else()
    message(STATUS "Building without OptiX")
endif()

if(USE_HIP)
    message(STATUS "Building with HIP support")
    target_compile_definitions(portableRT PUBLIC USE_HIP)

    set(HIP_ROOT "/opt/rocm-5.4.3/" CACHE PATH "HIP SDK")
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

    target_link_libraries(portableRT PUBLIC ${HIP_LIB} ${HSA_LIB})
else()
    message(STATUS "Building without HIP")
endif()

if(USE_EMBREE_SYCL)
    message(STATUS "Building with Embree-sycl support")
    target_compile_definitions(portableRT PUBLIC USE_EMBREE_SYCL)

    set(embree_DIR "/scratch/homelocal/encalle/embree/lib/cmake/embree-4.4.0/" CACHE PATH "Embree")

    FIND_PACKAGE(embree 4 REQUIRED)
    TARGET_LINK_LIBRARIES(portableRT PUBLIC embree)
    target_compile_options(portableRT PUBLIC -fsycl -Xclang -fsycl-allow-func-ptr -Wno-deprecated-declarations) 
    target_link_options(portableRT  PUBLIC -fsycl)
else()
    message(STATUS "Building without Embree-sycl")
endif()

if(USE_EMBREE_CPU)
    message(STATUS "Building with Embree-cpu support")
    target_compile_definitions(portableRT PUBLIC USE_EMBREE_CPU)

    set(embree_DIR "/scratch/homelocal/encalle/embree/lib/cmake/embree-4.4.0/" CACHE PATH "Embree")

    FIND_PACKAGE(embree 4 REQUIRED)
    TARGET_LINK_LIBRARIES(portableRT PUBLIC embree)
else()
    message(STATUS "Building without Embree-cpu")
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

install(DIRECTORY include/ DESTINATION include)

set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/install)


add_subdirectory(examples)
