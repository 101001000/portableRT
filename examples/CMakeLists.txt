function(add_example name)
    add_executable(${name} ${name}/main.cpp)
    target_include_directories(${name} PRIVATE ${CMAKE_SOURCE_DIR}/include)
    target_link_libraries(${name} PRIVATE portableRT)
    set_target_properties(${name} PROPERTIES
        LINKER_LANGUAGE CXX
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endfunction()

add_example(triangle)
add_example(backends)
add_example(bunny)
add_example(validation)

add_custom_target(copy_common ALL
  COMMAND ${CMAKE_COMMAND} -E copy_directory
  ${CMAKE_SOURCE_DIR}/examples/common
  ${CMAKE_BINARY_DIR}/bin/common
)