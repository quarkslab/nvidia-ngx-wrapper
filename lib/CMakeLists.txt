add_subdirectory(core)

add_library(nvidia-ngx-dlisr SHARED
  ngx-wrapper.cpp
  isr/cuda_wrappers.cpp
)
target_include_directories(nvidia-ngx-dlisr PRIVATE isr core)
set_target_properties(nvidia-ngx-dlisr PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  C_VISIBILITY_PRESET hidden
)
target_link_libraries(nvidia-ngx-dlisr PRIVATE QBDL ngx_core)

if (NGX_USE_NVCUDA_DLL)
  add_subdirectory(nvcuda)
endif()
