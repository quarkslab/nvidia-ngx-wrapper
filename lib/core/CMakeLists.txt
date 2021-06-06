add_library(ngx_core STATIC
  cudnn_wrappers.cpp
  unicode.cpp
  winapi.cpp)
set_target_properties(ngx_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(ngx_core PUBLIC dl /usr/local/cuda-10.0/lib64/libcudnn.so CUDA::cudart)

if (NGX_USE_NVCUDA_DLL)
  target_compile_definitions(ngx_core PUBLIC NGX_USE_NVCUDA_DLL)
  add_dependencies(ngx_core nvcuda.dll)
endif()
