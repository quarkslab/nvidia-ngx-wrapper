FROM ubuntu:20.04
ENV NGX_WIN_DLL_DIR=/opt/ngx_dlls

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y git p7zip-full build-essential ninja-build wget libopencv-dev clang

RUN wget https://github.com/Kitware/CMake/releases/download/v3.21.2/cmake-3.21.2-linux-x86_64.sh && \
    bash ./cmake-3.21.2-linux-x86_64.sh --prefix=/usr/local --skip-license && \
    rm ./cmake-3.21.2-linux-x86_64.sh

COPY cudnn-10.0-linux-x64-*.tgz /tmp
RUN tar xf /tmp/cudnn-10.0-linux-x64-*.tgz -C /usr/local && mv /usr/local/cuda /usr/local/cuda-10.0 && rm /tmp/cudnn*

COPY cuda_10.0.*.run /tmp
RUN bash /tmp/cuda_10.0.*.run --no-drm --no-opengl-libs --toolkit --silent --override && rm /tmp/cuda*

COPY NGX_SDK_EA1.1.exe /tmp

COPY NVIDIA-Linux-x86_64-465.24.02.run /tmp
RUN bash /tmp/NVIDIA-Linux-x86_64-465.24.02.run -x && mv NVIDIA-Linux-x86_64-465.24.02/libnvidia-ngx.so.465.24.02 /lib/x86_64-linux-gnu/ && \
    ln -s /lib/x86_64-linux-gnu/libnvidia-ngx.so.465.24.02 /lib/x86_64-linux-gnu/libnvidia-ngx.so.1 && \
    ln -s /lib/x86_64-linux-gnu/libnvidia-ngx.so.1 /lib/x86_64-linux-gnu/libnvidia-ngx.so && \
    ln -s /lib/x86_64-linux-gnu/libnvidia-ml.so.465.24.02 /lib/x86_64-linux-gnu/libnvidia-ml.so && \
    rm -rf /tmp/NVIDIA-*

COPY ngx_dlls /opt/ngx_dlls

RUN cd /tmp && git clone https://github.com/lief-project/LIEF && \
    cd LIEF && git checkout 255d09c92d09ebacb9473744550e1b90a8f4fd4e && mkdir build && \
    cd build && cmake -DLIEF_PYTHON_API=OFF -DCMAKE_BUILD_TYPE=Release -G Ninja .. && \
    ninja install && \
    rm -rf /tmp/LIEF

RUN cd /tmp && git clone https://github.com/quarkslab/QBDL && \
    cd QBDL && git checkout 129dd67ac7b488ec1656dc1faadc8f68bb37f601 && mkdir build &&\
    cd build && cmake -DCMAKE_BUILD_TYPE=Release -G Ninja .. && ninja install && \
    rm -rf /tmp/QBDL

RUN ldconfig
