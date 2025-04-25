add_executable(triangle triangle/main.cpp)
target_include_directories(triangle PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(triangle PRIVATE portableRT)
set_target_properties(triangle PROPERTIES 
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
