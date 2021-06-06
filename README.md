# NVIDIA NGX SDK on Linux

## Installation

Here are the prerequisites:

* NVIDIA Linux driver 465.24.02. This only has been tested with this exact
  version.  Latest version should work, but might need adaptation of the
  wrapper. Beware than some distribution drivers do not ship the NGX library.
* CUDA SDK 10.0 . The exact 10.0 version is important, as this is what the NGX
  components are using.
* cuDNN SDK 10.0
* NGX Windows SDK 1.1 archive
* [7-zip](https://www.7-zip.org/) (to extract the SDK files)
* LIEF commit 255d09c92d09ebacb9473744550e1b90a8f4fd4e
* QBDL commit 129dd67ac7b488ec1656dc1faadc8f68bb37f601
* and, finally, [cmake](https://cmake.org/)

Once you have downloaded the NGX SDK, use the ``extract_ngx_sdk.sh`` scriopt to
extract the necessary files from the NGX SDK, and copy them in the expected
places in the project's source tree. Indeed, we don't have the rights to
distribute these files ourselves.

### Build using Docker

You also need to place these files at the root of the cloned repository, after downloading them for NVIDIA's website:

* ``cudnn-10.0-linux-x64-*.tgz`` (https://developer.nvidia.com/cudnn-download-survey, need an NVIDIA developper account)
* ``cuda_10.0.*.run`` (https://developer.nvidia.com/compute/cuda/10.0/Prod/local_installers/cuda_10.0.130_410.48_linux)
* ``NVIDIA-Linux-x86_64-465.24.02.run`` (https://www.nvidia.com/download/driverResults.aspx/172836/en-us)

They are used by the Docker image build process. Indeed, an Ubuntu 20-based
Dockerfile is provided to easily build the tools:

```
$ cd /path/to/repo
$ docker build -t ngx .
```

Then, you can build the project directly from the newly created docker image,
from the repository root directory:

```
$ docker run --gpus all --rm -it -v $PWD:/ngx ngx /bin/bash
# cd /ngx && mkdir build && cd build
# cmake -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Release .. && make -j$(nproc)
# ldconfig
# ./tools/isr/isr --input /ngx/myimage.jpg --output /ngx/myimage_x2.jpg --factor 2
```

The docker option ``--gpus`` support comes with
https://nvidia.github.io/nvidia-container-runtime/. Some Debian systems might
also be affected by
https://github.com/NVIDIA/nvidia-docker/issues/1447#issuecomment-760059332 .

Note that running this with a NVIDIA host driver different from 465.24.02 is
undefined behavior.
