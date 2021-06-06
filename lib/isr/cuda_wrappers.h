#ifndef NGX_LINUX_CUDA_WRAPPER_H
#define NGX_LINUX_CUDA_WRAPPER_H

#include <cuda_runtime_api.h>
#include "macro_helpers.h"
#include "win_types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Name, Offset, Arguments
#define CUDA_HOOK_FUNCS_(FM, FM_VOID)                                          \
  FM(cudaEventCreateWithFlags, 0x0000000000001000, (cudaEvent_t *, event),     \
     (unsigned int, flags))                                                    \
  FM(cudaEventDestroy, 0x0000000000001140, (cudaEvent_t, event))               \
  FM(cudaEventQuery, 0x0000000000001270, (cudaEvent_t, event))                 \
  FM(cudaEventRecord, 0x00000000000013A0, (cudaEvent_t, event),                \
     (cudaStream_t, stream))                                                   \
  FM(cudaEventSynchronize, 0x0000000000001500, (cudaEvent_t, event))           \
  FM(cudaFree, 0x0000000000001630, (void **, a0))                              \
  FM(cudaFreeHost, 0x0000000000001750, (void **, a0))                          \
  FM_VOID(cudaGetLastError, 0x0000000000001870)                                \
  FM(cudaLaunchKernel, 0x0000000000001980, (const void *, func),               \
     (dim3, gridDim), (dim3, blockDim), (void **, args), (size_t, sharedMem),  \
     (cudaStream_t, stream))                                                   \
  FM(cudaMalloc, 0x0000000000001C00, (void **, a0), (size_t, a1))              \
  FM(cudaMallocHost, 0x0000000000001D40, (void **, ptr), (size_t, size))       \
  FM(cudaMemcpyAsync, 0x0000000000001E80, (void *, dst), (const void *, src),  \
     (size_t, count), (cudaMemcpyKind, kind), (cudaStream_t, stream))          \
  FM(cudaStreamCreate, 0x0000000000002030, (cudaStream_t *, pStream))          \
  FM(cudaStreamDestroy, 0x0000000000002160, (cudaStream_t, pStream))           \
  FM(cudaStreamSynchronize, 0x00000000000022B0, (cudaStream_t, pStream))

#define CUDA_HOOK_FUNCS(FM) CUDA_HOOK_FUNCS_(FM, FM)\

#define CUDA_DECL_WRAPPER(Name, Offset, ...)\
  CDECL_MSABI cudaError_t CAT(ms_,                                             \
                              Name)(FOR_EACH(VFUNC_param_decl, __VA_ARGS__));  \

#define CUDA_DECL_WRAPPER_VOID(Name, Offset)                                    \
  CDECL_MSABI cudaError_t CAT(ms_, Name)(); 

CUDA_HOOK_FUNCS_(CUDA_DECL_WRAPPER, CUDA_DECL_WRAPPER_VOID)

#undef CUDA_DECL_WRAPPER
#undef CUDA_DECL_WRAPPER_VOID


// Internal CUDA APIs
CDECL_MSABI void **ms___cudaRegisterFatBinary(void *fatCubin);
CDECL_MSABI void ms___cudaRegisterFunction(void **fatCubinHandle,
                                           const char *hostFun, char *deviceFun,
                                           const char *deviceName,
                                           int thread_limit, uint3 *tid,
                                           uint3 *bid, dim3 *bDim, dim3 *gDim,
                                           int *wSize);
CDECL_MSABI cudaError_t ms___cudaPopCallConfiguration(dim3 *gridDim,
                                                      dim3 *blockDim,
                                                      size_t *sharedMem,
                                                      void *stream);
CDECL_MSABI cudaError_t ms___cudaPushCallConfiguration(dim3 gridDim,
                                                       dim3 blockDim,
                                                       size_t sharedMem,
                                                       void *stream);

#define CUDA_INTERNAL_FUNCS(FM)\
  FM(0x00000000000030A0, ms___cudaRegisterFatBinary)\
  FM(0x00000000000030F0, ms___cudaRegisterFunction)\
  FM(0x0000000000002EE0, ms___cudaPopCallConfiguration)\
  FM(0x0000000000002FC0, ms___cudaPushCallConfiguration)

#define CUDA_WRAPPER_INIT_BEGIN 0x5B508
#define CUDA_WRAPPER_INIT_END   0x5B530

#ifdef __cplusplus
}
#endif

#endif
