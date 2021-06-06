add_library(nvcuda.dll SHARED
  internal.c
  nvcuda.c)
target_link_libraries(nvcuda.dll PRIVATE CUDA::cudart)
target_link_options(nvcuda.dll PRIVATE -rdynamic)
