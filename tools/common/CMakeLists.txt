add_library(ngx_tools_common STATIC
  ngx_sign.cpp)
target_include_directories(ngx_tools_common PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(ngx_tools_common PRIVATE dl)
