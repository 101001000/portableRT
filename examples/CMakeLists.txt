include(FetchContent)
find_package(SDL2 REQUIRED)

function(add_example name)
    add_executable(${name} ${name}/main.cpp)
    target_include_directories(${name} PRIVATE ${CMAKE_SOURCE_DIR}/include)
    target_link_libraries(${name} PRIVATE portableRT)
    set_target_properties(${name} PROPERTIES
        LINKER_LANGUAGE CXX
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endfunction()


FetchContent_Declare(
  stb
  GIT_REPOSITORY https://github.com/nothings/stb.git
  GIT_TAG        f58f558c120e9b32c217290b80bad1a0729fbb2c 
)
FetchContent_MakeAvailable(stb)
add_library(stb_image INTERFACE)
target_include_directories(stb_image INTERFACE ${stb_SOURCE_DIR})

FetchContent_Declare(
  cgltf
  GIT_REPOSITORY https://github.com/jkuhlmann/cgltf.git
  GIT_TAG        v1.13
)
FetchContent_MakeAvailable(cgltf)
add_library(cgltf INTERFACE)
target_include_directories(cgltf INTERFACE ${cgltf_SOURCE_DIR})

FetchContent_Declare(
  tinyobj
  GIT_REPOSITORY https://github.com/tinyobjloader/tinyobjloader.git
  GIT_TAG        v2.0.0rc13
)
FetchContent_MakeAvailable(tinyobj)
add_library(tinyobj INTERFACE)
target_include_directories(tinyobj INTERFACE ${tinyobj_SOURCE_DIR})

FetchContent_Declare(
  sponza_scene
  GIT_REPOSITORY https://github.com/KhronosGroup/glTF-Sample-Models.git
  GIT_TAG        d7a3cc8e51d7c573771ae77a57f16b0662a905c6
  GIT_SHALLOW    TRUE                
  SOURCE_SUBDIR  2.0/Sponza/glTF
)
FetchContent_MakeAvailable(sponza_scene)

add_example(triangle)
add_example(backends)
add_example(bunny)
add_example(validation)
add_example(sponza)


add_custom_command(TARGET sponza PRE_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          ${sponza_scene_SOURCE_DIR}/2.0/Sponza/glTF
          $<TARGET_FILE_DIR:sponza>/assets/Sponza)

target_link_libraries(sponza PRIVATE SDL2::SDL2 cgltf)
target_link_libraries(bunny PRIVATE stb_image tinyobj)
target_link_libraries(validation PRIVATE stb_image tinyobj)

add_custom_target(copy_common ALL
  COMMAND ${CMAKE_COMMAND} -E copy_directory
  ${CMAKE_SOURCE_DIR}/examples/common
  ${CMAKE_BINARY_DIR}/bin/common
)