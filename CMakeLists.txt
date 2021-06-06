cmake_minimum_required(VERSION 3.18)
project(ngx_linux)

option(NGX_USE_NVCUDA_DLL "Uses a wrapper around libnvcuda instead of hooking cudart (WARNING: does not work yet)" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED 17)

find_package(CUDAToolkit 10.0 REQUIRED EXACT)

find_package(LIEF REQUIRED)
find_package(QBDL REQUIRED)

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")

add_subdirectory(lib)
add_subdirectory(tools)
