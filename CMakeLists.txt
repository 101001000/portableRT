cmake_minimum_required(VERSION 3.16)
project(portableRT)

option(USE_OPTIX "Enable OptiX backend" ON)

add_library(portableRT src/portableRT.cpp)

# OPTIX
if(USE_OPTIX)
    message(STATUS "Building with OptiX support")

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