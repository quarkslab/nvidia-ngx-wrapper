/*
* Copyright (c) 2018 NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA Corporation and its licensors retain all intellectual property and proprietary
* rights in and to this software, related documentation and any modifications thereto.
* Any use, reproduction, disclosure or distribution of this software and related
* documentation without an express license agreement from NVIDIA Corporation is strictly
* prohibited.
*
* TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
* AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
* INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
* PARTICULAR PURPOSE.  IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS BE LIABLE FOR ANY
* SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
* LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
* BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR
* INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE POSSIBILITY OF
* SUCH DAMAGES.
*/

/*
*  HOW TO USE:
*
*  IMPORTANT: FOR DLSS/DLISP PLEASE SEE THE PROGRAMMING GUIDE
* 
*  IMPORTANT: Methods in this library are NOT thread safe. It is up to the
*  client to ensure that thread safety is enforced as needed.
*
*  1) Call NVSDK_CONV NVSDK_NGX_D3D11/D3D12/CUDA_Init and pass your app Id
*     and other parameters. This will initialize SDK or return an error code
*     if SDK cannot run on target machine. Depending on error user might
*     need to update drivers. Please note that application Id is provided
*     by NVIDIA so if you do not have one please contact us.
*
*  2) Call NVSDK_NGX_D3D11/D3D12/CUDA_GetParameters to obtain pointer to
*     interface used to pass parameters to SDK. Interface instance is
*     allocated and released by SDK so there is no need to do any memory
*     management on client side.
*
*  3) Set key parameters for the feature you want to use. For example,
*     width and height are required for all features and they can be
*     set like this:
*         Params->Set(NVSDK_NGX_Parameter_Width,MY_WIDTH);
*         Params->Set(NVSDK_NGX_Parameter_Height,MY_HEIGHT);
*
*     You can also provide hints like NVSDK_NGX_Parameter_Hint_HDR to tell
*     SDK that it should expect HDR color space is needed. Please refer to
*     samples since different features need different parameters and hints.
*
*  4) Call NVSDK_NGX_D3D11/D3D12/CUDA_GetScratchBufferSize to obtain size of
*     the scratch buffer needed by specific feature. This D3D or CUDA buffer
*     should be allocated by client and passed as:
*        Params->Set(NVSDK_NGX_Parameter_Scratch,MY_SCRATCH_POINTER)
*        Params->Set(NVSDK_NGX_Parameter_Scratch_SizeInBytes,MY_SCRATCH_SIZE_IN_BYTES)
*     NOTE: Returned size can be 0 if feature does not use any scratch buffer.
*     It is OK to use bigger buffer or reuse buffers across features as long
*     as minimum size requirement is met.
*
*  5) Call NVSDK_NGX_D3D11/D3D12/CUDA_CreateFeature to create feature you need.
*     On success SDK will return a handle which must be used in any successive
*     calls to SDK which require feature handle. SDK will use all parameters
*     and hints provided by client to generate feature. If feature with the same
*     parameters already exists and error code will be returned.
*
*  6) Call NVSDK_NGX_D3D11/D3D12/CUDA_EvaluateFeature to invoke execution of
*     specific feature. Before feature can be evaluated input parameters must
*     be specified (like for example color/albedo buffer, motion vectors etc)
*
*  6) Call NVSDK_NGX_D3D11/D3D12/CUDA_ReleaseFeature when feature is no longer
*     needed. After this call feature handle becomes invalid and cannot be used.
*
*  7) Call NVSDK_NGX_D3D11/D3D12/CUDA_Shutdown when SDK is no longer needed to
*     release all resources.

*  Contact: ngxsupport@nvidia.com
*/


#ifndef NVSDK_NGX_H
#define NVSDK_NGX_H

#include "nvsdk_ngx_defs.h"

struct IUnknown;

struct ID3D11Device;
struct ID3D11Resource;
struct ID3D11DeviceContext;
struct D3D11_TEXTURE2D_DESC;
struct D3D11_BUFFER_DESC;
struct ID3D11Buffer;
struct ID3D11Texture2D;

struct ID3D12Device;
struct ID3D12Resource;
struct ID3D12GraphicsCommandList;
struct D3D12_RESOURCE_DESC;
struct CD3DX12_HEAP_PROPERTIES;

typedef void (NVSDK_CONV *PFN_NVSDK_NGX_D3D12_ResourceAllocCallback)(D3D12_RESOURCE_DESC *InDesc, int InState, CD3DX12_HEAP_PROPERTIES *InHeap, ID3D12Resource **OutResource);
typedef void (NVSDK_CONV *PFN_NVSDK_NGX_D3D11_BufferAllocCallback)(D3D11_BUFFER_DESC *InDesc, ID3D11Buffer **OutResource);
typedef void (NVSDK_CONV *PFN_NVSDK_NGX_D3D11_Tex2DAllocCallback)(D3D11_TEXTURE2D_DESC *InDesc, ID3D11Texture2D **OutResource);
typedef void (NVSDK_CONV *PFN_NVSDK_NGX_ResourceReleaseCallback)(IUnknown *InResource);

struct NVSDK_NGX_Parameter
{
    virtual void Set(const char * InName, unsigned long long InValue) = 0;
    virtual void Set(const char * InName, float InValue) = 0;
    virtual void Set(const char * InName, double InValue) = 0;
    virtual void Set(const char * InName, unsigned int InValue) = 0;
    virtual void Set(const char * InName, int InValue) = 0;
    virtual void Set(const char * InName, ID3D11Resource *InValue) = 0;
    virtual void Set(const char * InName, ID3D12Resource *InValue) = 0;
    virtual void Set(const char * InName, void *InValue) = 0;

    virtual NVSDK_NGX_Result Get(const char * InName, unsigned long long *OutValue) = 0;
    virtual NVSDK_NGX_Result Get(const char * InName, float *OutValue) = 0;
    virtual NVSDK_NGX_Result Get(const char * InName, double *OutValue) = 0;
    virtual NVSDK_NGX_Result Get(const char * InName, unsigned int *OutValue) = 0;
    virtual NVSDK_NGX_Result Get(const char * InName, int *OutValue) = 0;
    virtual NVSDK_NGX_Result Get(const char * InName, ID3D11Resource **OutValue) = 0;
    virtual NVSDK_NGX_Result Get(const char * InName, ID3D12Resource **OutValue) = 0;
    virtual NVSDK_NGX_Result Get(const char * InName, void **OutValue) = 0;

    virtual void Reset() = 0;
};

///////////////////////////////////////////////////////////////////////////////////////////////////
// NVSDK_NGX_Init
// -------------------------------------
// 
// InApplicationId:
//      Unique Id provided by NVIDIA
//
// InApplicationDataPath:
//      Folder to store logs and other temporary files (write access required),
//      Normally this would be a location in Documents or ProgramData.
//
// InDevice: [d3d11/12 only]
//      DirectX device to use
//
// DESCRIPTION:
//      Initializes new SDK instance.
//
NVSDK_NGX_API NVSDK_NGX_Result  NVSDK_CONV NVSDK_NGX_D3D11_Init(unsigned long long InApplicationId, const wchar_t *InApplicationDataPath, ID3D11Device *InDevice, NVSDK_NGX_Version InSDKVersion = NVSDK_NGX_Version_API);
NVSDK_NGX_API NVSDK_NGX_Result  NVSDK_CONV NVSDK_NGX_D3D12_Init(unsigned long long InApplicationId, const wchar_t *InApplicationDataPath, ID3D12Device *InDevice, NVSDK_NGX_Version InSDKVersion = NVSDK_NGX_Version_API);
NVSDK_NGX_API NVSDK_NGX_Result  NVSDK_CONV NVSDK_NGX_CUDA_Init(unsigned long long InApplicationId, const wchar_t *InApplicationDataPath, NVSDK_NGX_Version InSDKVersion = NVSDK_NGX_Version_API);

////////////////////////////////////////////////////////////////////////////////////////////////////
// NVSDK_NGX_Shutdown
// -------------------------------------
// 
// DESCRIPTION:
//      Shuts down the current SDK instance and releases all resources.
//
NVSDK_NGX_API NVSDK_NGX_Result NVSDK_CONV NVSDK_NGX_D3D11_Shutdown();
NVSDK_NGX_API NVSDK_NGX_Result NVSDK_CONV NVSDK_NGX_D3D12_Shutdown();
NVSDK_NGX_API NVSDK_NGX_Result NVSDK_CONV NVSDK_NGX_CUDA_Shutdown();

////////////////////////////////////////////////////////////////////////////////////////////////////
// NVSDK_NGX_GetParameters
// ----------------------------------------------------------
//
// OutParameters:
//      Parameters interface used to set any parameter needed by the SDK
//
// DESCRIPTION:
//      This interface allows simple parameter setup using named fields.
//      For example one can set width by calling Set(NVSDK_NGX_Parameter_Denoiser_Width,100) or
//      provide CUDA buffer pointer by calling Set(NVSDK_NGX_Parameter_Denoiser_Color,cudaBuffer)
//      For more details please see sample code. Please note that allocated memory
//      will be freed by NGX so free/delete operator should NOT be called.
//
NVSDK_NGX_API NVSDK_NGX_Result NVSDK_CONV NVSDK_NGX_D3D11_GetParameters(NVSDK_NGX_Parameter **OutParameters);
NVSDK_NGX_API NVSDK_NGX_Result NVSDK_CONV NVSDK_NGX_D3D12_GetParameters(NVSDK_NGX_Parameter **OutParameters);
NVSDK_NGX_API NVSDK_NGX_Result NVSDK_CONV NVSDK_NGX_CUDA_GetParameters(NVSDK_NGX_Parameter **OutParameters);

////////////////////////////////////////////////////////////////////////////////////////////////////
// NVSDK_NGX_GetScratchBufferSize
// ----------------------------------------------------------
//
// InFeatureId:
//      AI feature in question
//
// InParameters:
//      Parameters used by the feature to help estimate scratch buffer size
//
// OutSizeInBytes:
//      Number of bytes needed for the scratch buffer for the specified feature.
//
// DESCRIPTION:
//      SDK needs a buffer of a certain size provided by the client in
//      order to initialize AI feature. Once feature is no longer
//      needed buffer can be released. It is safe to reuse the same
//      scratch buffer for different features as long as minimum size
//      requirement is met for all features. Please note that some
//      features might not need a scratch buffer so return size of 0
//      is completely valid.
//
NVSDK_NGX_API NVSDK_NGX_Result NVSDK_CONV NVSDK_NGX_D3D11_GetScratchBufferSize(NVSDK_NGX_Feature InFeatureId, const NVSDK_NGX_Parameter *InParameters, unsigned long long *OutSizeInBytes);
NVSDK_NGX_API NVSDK_NGX_Result NVSDK_CONV NVSDK_NGX_D3D12_GetScratchBufferSize(NVSDK_NGX_Feature InFeatureId, const NVSDK_NGX_Parameter *InParameters, unsigned long long *OutSizeInBytes);
NVSDK_NGX_API NVSDK_NGX_Result NVSDK_CONV NVSDK_NGX_CUDA_GetScratchBufferSize(NVSDK_NGX_Feature InFeatureId, const NVSDK_NGX_Parameter *InParameters, unsigned long long *OutSizeInBytes);

/////////////////////////////////////////////////////////////////////////
// NVSDK_NGX_CreateFeature
// -------------------------------------
//
// InCmdList:[d3d12 only]
//      Command list to use to execute GPU commands. Must be:
//      - Open and recording 
//      - With node mask including the device provided in NVSDK_NGX_D3D12_Init
//      - Execute on non-copy command queue.
// InDevCtx: [d3d11 only]
//      Device context to use to execute GPU commands
//
// InFeatureID:
//      AI feature to initialize
//
// InParameters:
//      List of parameters 
// 
// OutHandle:
//      Handle which uniquely identifies the feature. If feature with
//      provided parameters already exists the "already exists" error code is returned.
//
// DESCRIPTION:
//      Each feature needs to be created before it can be used. 
//      Refer to the sample code to find out which input parameters
//      are needed to create specific feature.
//
NVSDK_NGX_API NVSDK_NGX_Result NVSDK_CONV NVSDK_NGX_D3D11_CreateFeature(ID3D11DeviceContext *InDevCtx, NVSDK_NGX_Feature InFeatureID, const NVSDK_NGX_Parameter *InParameters, NVSDK_NGX_Handle **OutHandle);
NVSDK_NGX_API NVSDK_NGX_Result NVSDK_CONV NVSDK_NGX_D3D12_CreateFeature(ID3D12GraphicsCommandList *InCmdList, NVSDK_NGX_Feature InFeatureID, const NVSDK_NGX_Parameter *InParameters, NVSDK_NGX_Handle **OutHandle);
NVSDK_NGX_API NVSDK_NGX_Result NVSDK_CONV NVSDK_NGX_CUDA_CreateFeature(NVSDK_NGX_Feature InFeatureID, const NVSDK_NGX_Parameter *InParameters, NVSDK_NGX_Handle **OutHandle);

/////////////////////////////////////////////////////////////////////////
// NVSDK_NGX_Release
// -------------------------------------
// 
// InHandle:
//      Handle to feature to be released
//
// DESCRIPTION:
//      Releases feature with a given handle.
//      Handles are not reference counted so
//      after this call it is invalid to use provided handle.
//
NVSDK_NGX_API NVSDK_NGX_Result NVSDK_CONV NVSDK_NGX_D3D11_ReleaseFeature(NVSDK_NGX_Handle *InHandle);
NVSDK_NGX_API NVSDK_NGX_Result NVSDK_CONV NVSDK_NGX_D3D12_ReleaseFeature(NVSDK_NGX_Handle *InHandle);
NVSDK_NGX_API NVSDK_NGX_Result NVSDK_CONV NVSDK_NGX_CUDA_ReleaseFeature(NVSDK_NGX_Handle *InHandle);

/////////////////////////////////////////////////////////////////////////
// NVSDK_NGX_EvaluateFeature
// -------------------------------------
//
// InCmdList:[d3d12 only]
//      Command list to use to execute GPU commands. Must be:
//      - Open and recording 
//      - With node mask including the device provided in NVSDK_NGX_D3D12_Init
//      - Execute on non-copy command queue.
// InDevCtx: [d3d11 only]
//      Device context to use to execute GPU commands
//
// InFeatureHandle:
//      Handle representing feature to be evaluated
// 
// InParameters:
//      List of parameters required to evaluate feature
//
// InCallback:
//      Optional callback for features which might take longer
//      to execture. If specified SDK will call it with progress
//      values in range 0.0f - 1.0f
//
// DESCRIPTION:
//      Evaluates given feature using the provided parameters and
//      pre-trained NN. Please note that for most features
//      it can be benefitials to pass as many input buffers and parameters
//      as possible (for example provide all render targets like color, albedo, normals, depth etc)
//
typedef void (NVSDK_CONV *PFN_NVSDK_NGX_ProgressCallback)(float InCurrentProgress, bool &OutShouldCancel);
NVSDK_NGX_API NVSDK_NGX_Result NVSDK_CONV NVSDK_NGX_D3D11_EvaluateFeature(ID3D11DeviceContext *InDevCtx, const NVSDK_NGX_Handle *InFeatureHandle, const NVSDK_NGX_Parameter *InParameters, PFN_NVSDK_NGX_ProgressCallback InCallback = nullptr);
NVSDK_NGX_API NVSDK_NGX_Result NVSDK_CONV NVSDK_NGX_D3D12_EvaluateFeature(ID3D12GraphicsCommandList *InCmdList, const NVSDK_NGX_Handle *InFeatureHandle, const NVSDK_NGX_Parameter *InParameters, PFN_NVSDK_NGX_ProgressCallback InCallback = nullptr);
NVSDK_NGX_API NVSDK_NGX_Result NVSDK_CONV NVSDK_NGX_CUDA_EvaluateFeature(const NVSDK_NGX_Handle *InFeatureHandle, const NVSDK_NGX_Parameter *InParameters, PFN_NVSDK_NGX_ProgressCallback InCallback = nullptr);

#endif // #define NVSDK_NGX_H
