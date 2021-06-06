#include "cuda_wrappers.h"
#include "log.h"

// Generate wrappers on CUDA APIs
#define CUDA_DEF_WRAPPER(Name, Offset, ...)                                    \
  CDECL_MSABI cudaError_t CAT(ms_,                                             \
                              Name)(FOR_EACH(VFUNC_param_decl, __VA_ARGS__)) { \
    const cudaError_t ret = Name(FOR_EACH(VFUNC_param_use, __VA_ARGS__));      \
    LOG_FUNC(" = %d", ret);                                                    \
    return ret;                                                                \
  }

#define CUDA_DEF_WRAPPER_VOID(Name, Offset)                                    \
  CDECL_MSABI cudaError_t CAT(ms_, Name)() {                                   \
    const cudaError_t ret = Name();                                            \
    LOG_FUNC(" = %d", ret);                                                    \
    return ret;                                                                \
  }

CUDA_HOOK_FUNCS_(CUDA_DEF_WRAPPER, CUDA_DEF_WRAPPER_VOID)
#undef CUDA_DEF_WRAPPER
#undef CUDA_DEF_WRAPPER_VOID


// Internal CUDA APIs
extern "C" void CUDARTAPI __cudaRegisterFunction(
    void **fatCubinHandle, const char *hostFun, char *deviceFun,
    const char *deviceName, int thread_limit, uint3 *tid, uint3 *bid,
    dim3 *bDim, dim3 *gDim, int *wSize);
extern "C" void **CUDARTAPI __cudaRegisterFatBinary(void *fatCubin);
extern "C" cudaError_t CUDARTAPI __cudaPopCallConfiguration(dim3 *gridDim,
                                                            dim3 *blockDim,
                                                            size_t *sharedMem,
                                                            void *stream);

extern "C" cudaError_t CUDARTAPI __cudaPushCallConfiguration(dim3 gridDim,
                                                             dim3 blockDim,
                                                             size_t sharedMem,
                                                             void *stream);

CDECL_MSABI void **ms___cudaRegisterFatBinary(void *fatCubin) {
  void **const ret = __cudaRegisterFatBinary(fatCubin);
  LOG_FUNC(" = %p", ret);
  return ret;
}

CDECL_MSABI void ms___cudaRegisterFunction(void **fatCubinHandle,
                                           const char *hostFun, char *deviceFun,
                                           const char *deviceName,
                                           int thread_limit, uint3 *tid,
                                           uint3 *bid, dim3 *bDim, dim3 *gDim,
                                           int *wSize) {
  LOG_FUNC();
  __cudaRegisterFunction(fatCubinHandle, hostFun, deviceFun, deviceName,
                         thread_limit, tid, bid, bDim, gDim, wSize);
}

CDECL_MSABI cudaError_t ms___cudaPopCallConfiguration(dim3 *gridDim,
                                                      dim3 *blockDim,
                                                      size_t *sharedMem,
                                                      void *stream) {
  const auto ret =
      __cudaPopCallConfiguration(gridDim, blockDim, sharedMem, stream);
  LOG_FUNC("ret = %d", ret);
  return ret;
}

CDECL_MSABI cudaError_t ms___cudaPushCallConfiguration(dim3 gridDim,
                                                       dim3 blockDim,
                                                       size_t sharedMem,
                                                       void *stream) {
  const auto ret =
      __cudaPushCallConfiguration(gridDim, blockDim, sharedMem, stream);
  LOG_FUNC("ret = %d", ret);
  return ret;
}
