find_package(OpenCV REQUIRED)
add_executable(isr isr.cpp)
target_link_libraries(isr PRIVATE CUDA::cudart ${OpenCV_LIBS} ngx_tools_common /lib/x86_64-linux-gnu/libnvidia-ngx.so.1)

# We need a symlink of libnvidia-ngx-dlisr.so next to the isr binary
add_dependencies(isr nvidia-ngx-dlisr)
add_custom_command(TARGET isr
  POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E create_symlink $<TARGET_FILE:nvidia-ngx-dlisr> "$<TARGET_FILE_DIR:isr>/$<TARGET_FILE_NAME:nvidia-ngx-dlisr>")
