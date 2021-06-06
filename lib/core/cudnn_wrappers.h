#ifndef NGX_LINUX_CDNN_WRAPPERS_H
#define NGX_LINUX_CDNN_WRAPPERS_H

#include "macro_helpers.h"
#include "win_types.h"

#include <cudnn.h>

#ifdef __cplusplus
extern "C" {
#endif

// cuDNN ABI wrappers
// Name, Arguments
#define CUDNN_WRAP_FUNCS(FM)                                                   \
  FM(cudnnCreate, (cudnnHandle_t *, handle))                                   \
  FM(cudnnCreateConvolutionDescriptor,                                         \
     (cudnnConvolutionDescriptor_t *, convDesc))                               \
  FM(cudnnCreateFilterDescriptor, (cudnnFilterDescriptor_t *, filterDesc))     \
  FM(cudnnCreateOpTensorDescriptor,                                            \
     (cudnnOpTensorDescriptor_t *, opTensorDesc))                              \
  FM(cudnnCreateTensorDescriptor, (cudnnTensorDescriptor_t *, tensorDesc))     \
  FM(cudnnDestroy, (cudnnHandle_t, handle))                                    \
  FM(cudnnGetConvolutionForwardWorkspaceSize, (cudnnHandle_t, handle),         \
     (const cudnnTensorDescriptor_t, xDesc),                                   \
     (const cudnnFilterDescriptor_t, wDesc),                                   \
     (const cudnnConvolutionDescriptor_t, convDesc),                           \
     (const cudnnTensorDescriptor_t, yDesc),                                   \
     (cudnnConvolutionFwdAlgo_t, algo), (size_t *, sizeInBytes))               \
  FM(cudnnGetTensorNdDescriptor, (const cudnnTensorDescriptor_t, tensorDesc),  \
     (int, nbDimsRequested), (cudnnDataType_t *, dataType), (int *, nbDims),   \
     (int *, dimA), (int *, strideA))                                          \
  FM(cudnnSetConvolutionNdDescriptor,                                          \
     (cudnnConvolutionDescriptor_t, convDesc), (int, arrayLength),             \
     (const int *, padA), (const int *, filterStrideA),                        \
     (const int *, dilationA), (cudnnConvolutionMode_t, mode),                 \
     (cudnnDataType_t, dataType))                                              \
  FM(cudnnSetFilterNdDescriptor, (cudnnFilterDescriptor_t, filterDesc),        \
     (cudnnDataType_t, dataType), (cudnnTensorFormat_t, format),               \
     (int, nbDims), (const int *, filterDimA))                                 \
  FM(cudnnSetOpTensorDescriptor, (cudnnOpTensorDescriptor_t, opTensorDesc),    \
     (cudnnOpTensorOp_t, opTensorOp), (cudnnDataType_t, opTensorCompType),     \
     (cudnnNanPropagation_t, opTensorNanOpt))                                  \
  FM(cudnnSetStream, (cudnnHandle_t, handle), (cudaStream_t, streamId))        \
  FM(cudnnConvolutionForward, (cudnnHandle_t, handle), (const void *, alpha),  \
     (const cudnnTensorDescriptor_t, xDesc), (const void *, x),                \
     (const cudnnFilterDescriptor_t, wDesc), (const void *, w),                \
     (const cudnnConvolutionDescriptor_t, convDesc),                           \
     (cudnnConvolutionFwdAlgo_t, algo), (void *, workSpace),                   \
     (size_t, workSpaceSizeInBytes), (const void *, beta),                     \
     (const cudnnTensorDescriptor_t, yDesc), (void *, y))                      \
  FM(cudnnSetTensorNdDescriptor, (cudnnTensorDescriptor_t, tensorDesc),        \
     (cudnnDataType_t, dataType), (int, nbDims), (const int *, dimA),          \
     (const int *, strideA))                                                   \
  FM(cudnnSetTensorNdDescriptorEx, (cudnnTensorDescriptor_t, tensorDesc),      \
     (cudnnTensorFormat_t, format), (cudnnDataType_t, dataType),               \
     (int, nbDims), (const int *, dimA))                                       \
  FM(cudnnSetConvolutionGroupCount, (cudnnConvolutionDescriptor_t, convDesc),  \
     (int, groupCount))                                                        \
  FM(cudnnSetTensor, (cudnnHandle_t, handle),                                  \
     (const cudnnTensorDescriptor_t, yDesc), (void *, y),                      \
     (const void *, valuePtr))                                                 \
  FM(cudnnGetConvolutionForwardAlgorithm, (cudnnHandle_t, handle),             \
     (const cudnnTensorDescriptor_t, xDesc),                                   \
     (const cudnnFilterDescriptor_t, wDesc),                                   \
     (const cudnnConvolutionDescriptor_t, convDesc),                           \
     (const cudnnTensorDescriptor_t, yDesc),                                   \
     (cudnnConvolutionFwdPreference_t, preference),                            \
     (size_t, memoryLimitInBytes), (cudnnConvolutionFwdAlgo_t *, algo))        \
  FM(cudnnOpTensor, (cudnnHandle_t, handle),                                   \
     (const cudnnOpTensorDescriptor_t, opTensorDesc), (const void *, alpha1),  \
     (const cudnnTensorDescriptor_t, aDesc), (const void *, A),                \
     (const void *, alpha2), (const cudnnTensorDescriptor_t, bDesc),           \
     (const void *, B), (const void *, beta),                                  \
     (const cudnnTensorDescriptor_t, cDesc), (void *, C))                      \
  FM(cudnnDestroyOpTensorDescriptor,                                           \
     (cudnnOpTensorDescriptor_t, opTensorDesc))                                \
  FM(cudnnDestroyConvolutionDescriptor,                                        \
     (cudnnConvolutionDescriptor_t, convDesc))                                 \
  FM(cudnnDestroyTensorDescriptor, (cudnnTensorDescriptor_t, tensorDesc))      \
  FM(cudnnDestroyFilterDescriptor, (cudnnFilterDescriptor_t, filterDesc))      \
  FM(cudnnTransformTensor, (cudnnHandle_t, handle), (const void *, alpha),     \
     (const cudnnTensorDescriptor_t, xDesc), (const void *, x),                \
     (const void *, beta), (const cudnnTensorDescriptor_t, yDesc),             \
     (void *, y))

#define CUDNN_DECL_WRAPPER(Name, ...)                                           \
  CDECL_MSABI cudnnStatus_t CAT(ms_, Name)(                                    \
      FOR_EACH(VFUNC_param_decl, __VA_ARGS__));

CUDNN_WRAP_FUNCS(CUDNN_DECL_WRAPPER)

#ifdef __cplusplus
}
#endif

#endif
