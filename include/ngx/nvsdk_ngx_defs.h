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

#ifndef NVSDK_NGX_DEFS_H
#define NVSDK_NGX_DEFS_H
#pragma once

#ifdef NVSDK_NGX
#define NVSDK_NGX_API extern "C" __declspec(dllexport)
#else
#define NVSDK_NGX_API extern "C"
#endif

#define NVSDK_CONV __cdecl

enum NVSDK_NGX_Version { NVSDK_NGX_Version_API = 0x0000012 }; // NGX_VERSION_DOT 1.2.13

enum NVSDK_NGX_Result
{
    NVSDK_NGX_Result_Success = 0x1,

    NVSDK_NGX_Result_Fail = 0xBAD00000,

    // Feature is not supported on current hardware
    NVSDK_NGX_Result_FAIL_FeatureNotSupported = NVSDK_NGX_Result_Fail | 1,

    // Platform error - for example - check d3d12 debug layer log for more information
    NVSDK_NGX_Result_FAIL_PlatformError = NVSDK_NGX_Result_Fail | 2,

    // Feature with given parameters already exists
    NVSDK_NGX_Result_FAIL_FeatureAlreadyExists = NVSDK_NGX_Result_Fail | 3,

    // Feature with provided handle does not exist
    NVSDK_NGX_Result_FAIL_FeatureNotFound = NVSDK_NGX_Result_Fail | 4,

    // Invalid parameter was provided
    NVSDK_NGX_Result_FAIL_InvalidParameter = NVSDK_NGX_Result_Fail | 5,

    // Provided buffer is too small, please use size provided by NVSDK_NGX_GetScratchBufferSize
    NVSDK_NGX_Result_FAIL_ScratchBufferTooSmall = NVSDK_NGX_Result_Fail | 6,

    // SDK was not initialized properly
    NVSDK_NGX_Result_FAIL_NotInitialized = NVSDK_NGX_Result_Fail | 7,

    //  Unsupported format used for input/output buffers
    NVSDK_NGX_Result_FAIL_UnsupportedInputFormat = NVSDK_NGX_Result_Fail | 8,

    // Feature input/output needs RW access (UAV) (d3d11/d3d12 specific)
    NVSDK_NGX_Result_FAIL_RWFlagMissing = NVSDK_NGX_Result_Fail | 9,

    // Feature was created with specific input but none is provided at evaluation
    NVSDK_NGX_Result_FAIL_MissingInput = NVSDK_NGX_Result_Fail | 10,

    // Feature is not available on the system
    NVSDK_NGX_Result_FAIL_UnableToInitializeFeature = NVSDK_NGX_Result_Fail | 11,

    // NGX system libraries are old and need an update
    NVSDK_NGX_Result_FAIL_OutOfDate = NVSDK_NGX_Result_Fail | 12,

    // Feature requires more GPU memory than it is available on system
    NVSDK_NGX_Result_FAIL_OutOfGPUMemory = NVSDK_NGX_Result_Fail | 13,

    // Format used in input buffer(s) is not supported by feature
    NVSDK_NGX_Result_FAIL_UnsupportedFormat = NVSDK_NGX_Result_Fail | 14,

    // Path provided in InApplicationDataPath cannot be written to
    NVSDK_NGX_Result_FAIL_UnableToWriteToAppDataPath = NVSDK_NGX_Result_Fail | 15,

    // Unsupported parameter was provided (e.g. specific scaling factor is unsupported)
    NVSDK_NGX_Result_FAIL_UnsupportedParameter = NVSDK_NGX_Result_Fail | 16
};

#define NVSDK_NGX_SUCCEED(value) (((value) & 0xFFF00000) != NVSDK_NGX_Result_Fail)
#define NVSDK_NGX_FAILED(value) (((value) & 0xFFF00000) == NVSDK_NGX_Result_Fail)

enum NVSDK_NGX_Feature
{
    NVSDK_NGX_Feature_Reserved0,

    NVSDK_NGX_Feature_SuperSampling,

    NVSDK_NGX_Feature_InPainting,

    NVSDK_NGX_Feature_ImageSuperResolution,

    NVSDK_NGX_Feature_SlowMotion,

    NVSDK_NGX_Feature_VideoSuperResolution,

    NVSDK_NGX_Feature_Reserved6,

    NVSDK_NGX_Feature_Reserved7,

    NVSDK_NGX_Feature_Reserved8,

    NVSDK_NGX_Feature_ImageSignalProcessing,

    // New features go here
    NVSDK_NGX_Feature_Count
};

//TODO create grayscale format (R32F?)
enum NVSDK_NGX_Buffer_Format
{
    NVSDK_NGX_Buffer_Format_Unknown,
    NVSDK_NGX_Buffer_Format_RGB8UI,
    NVSDK_NGX_Buffer_Format_RGB16F,
    NVSDK_NGX_Buffer_Format_RGB32F,
    NVSDK_NGX_Buffer_Format_RGBA8UI,
    NVSDK_NGX_Buffer_Format_RGBA16F,
    NVSDK_NGX_Buffer_Format_RGBA32F,
};

struct NVSDK_NGX_Handle { unsigned int Id;};

enum NVSDK_NGX_PerfQuality_Value
{
    NVSDK_NGX_PerfQuality_Value_MaxPerf,
    NVSDK_NGX_PerfQuality_Value_Balanced,
    NVSDK_NGX_PerfQuality_Value_MaxQuality
};

enum NVSDK_NGX_RTX_Value
{
    NVSDK_NGX_RTX_Value_Off,
    NVSDK_NGX_RTX_Value_On,
};

enum NVSDK_NGX_DLSS_Mode
{
    NVSDK_NGX_DLSS_Mode_Off,        // use existing in-engine AA + upscale solution
    NVSDK_NGX_DLSS_Mode_DLSS_DLISP,
    NVSDK_NGX_DLSS_Mode_DLISP_Only, // use existing in-engine AA solution
};


// Read-only parameters provided by NGX
#define NVSDK_NGX_EParameter_SuperSampling_Available              "#\x01"
#define NVSDK_NGX_EParameter_InPainting_Available                 "#\x02"
#define NVSDK_NGX_EParameter_ImageSuperResolution_Available       "#\x03"
#define NVSDK_NGX_EParameter_SlowMotion_Available                 "#\x04"
#define NVSDK_NGX_EParameter_VideoSuperResolution_Available       "#\x05"
#define NVSDK_NGX_EParameter_ImageSignalProcessing_Available      "#\x09"
#define NVSDK_NGX_EParameter_ImageSuperResolution_ScaleFactor_2_1 "#\x0a"
#define NVSDK_NGX_EParameter_ImageSuperResolution_ScaleFactor_3_1 "#\x0b"
#define NVSDK_NGX_EParameter_ImageSuperResolution_ScaleFactor_3_2 "#\x0c"
#define NVSDK_NGX_EParameter_ImageSuperResolution_ScaleFactor_4_3 "#\x0d"

#define NVSDK_NGX_Parameter_SuperSampling_ScaleFactor  "SuperSampling.ScaleFactor"
#define NVSDK_NGX_Parameter_ImageSignalProcessing_ScaleFactor "ImageSignalProcessing.ScaleFactor"

#define NVSDK_NGX_Parameter_SuperSampling_Available "SuperSampling.Available"
#define NVSDK_NGX_Parameter_InPainting_Available "InPainting.Available"
#define NVSDK_NGX_Parameter_ImageSuperResolution_Available "ImageSuperResolution.Available"
#define NVSDK_NGX_Parameter_SlowMotion_Available "SlowMotion.Available"
#define NVSDK_NGX_Parameter_VideoSuperResolution_Available "VideoSuperResolution.Available"
#define NVSDK_NGX_Parameter_ImageSignalProcessing_Available "ImageSignalProcessing.Available"

#define NVSDK_NGX_Parameter_ImageSuperResolution_ScaleFactor_2_1 "ImageSuperResolution.ScaleFactor.2.1"
#define NVSDK_NGX_Parameter_ImageSuperResolution_ScaleFactor_3_1 "ImageSuperResolution.ScaleFactor.3.1"
#define NVSDK_NGX_Parameter_ImageSuperResolution_ScaleFactor_3_2 "ImageSuperResolution.ScaleFactor.3.2"
#define NVSDK_NGX_Parameter_ImageSuperResolution_ScaleFactor_4_3 "ImageSuperResolution.ScaleFactor.4.3"

// Parameters provided by client application
#define NVSDK_NGX_EParameter_NumFrames           "#\x0e"
#define NVSDK_NGX_EParameter_Scale               "#\x0f"
#define NVSDK_NGX_EParameter_Width               "#\x10"
#define NVSDK_NGX_EParameter_Height              "#\x11"
#define NVSDK_NGX_EParameter_OutWidth            "#\x12"
#define NVSDK_NGX_EParameter_OutHeight           "#\x13"
#define NVSDK_NGX_EParameter_Sharpness           "#\x14"
#define NVSDK_NGX_EParameter_Scratch             "#\x15"
#define NVSDK_NGX_EParameter_Scratch_SizeInBytes "#\x16"
#define NVSDK_NGX_EParameter_Hint_HDR            "#\x17"
#define NVSDK_NGX_EParameter_Input1              "#\x18"
#define NVSDK_NGX_EParameter_Input1_Format       "#\x19"
#define NVSDK_NGX_EParameter_Input1_SizeInBytes  "#\x1a"
#define NVSDK_NGX_EParameter_Input2              "#\x1b"
#define NVSDK_NGX_EParameter_Input2_Format       "#\x1c"
#define NVSDK_NGX_EParameter_Input2_SizeInBytes  "#\x1d"
#define NVSDK_NGX_EParameter_Color               "#\x1e"
#define NVSDK_NGX_EParameter_Color_Format        "#\x1f"
#define NVSDK_NGX_EParameter_Color_SizeInBytes   "#\x20"
#define NVSDK_NGX_EParameter_Albedo              "#\x21"
#define NVSDK_NGX_EParameter_Output              "#\x22"
#define NVSDK_NGX_EParameter_Output_Format       "#\x23"
#define NVSDK_NGX_EParameter_Output_SizeInBytes  "#\x24"
#define NVSDK_NGX_EParameter_Reset               "#\x25"
#define NVSDK_NGX_EParameter_BlendFactor         "#\x26"
#define NVSDK_NGX_EParameter_MotionVectors       "#\x27"
#define NVSDK_NGX_EParameter_Rect_X              "#\x28"
#define NVSDK_NGX_EParameter_Rect_Y              "#\x29"
#define NVSDK_NGX_EParameter_Rect_W              "#\x2a"
#define NVSDK_NGX_EParameter_Rect_H              "#\x2b"
#define NVSDK_NGX_EParameter_MV_Scale_X          "#\x2c"
#define NVSDK_NGX_EParameter_MV_Scale_Y          "#\x2d"
#define NVSDK_NGX_EParameter_Model               "#\x2e"
#define NVSDK_NGX_EParameter_Format              "#\x2f"
#define NVSDK_NGX_EParameter_SizeInBytes         "#\x30"
#define NVSDK_NGX_EParameter_ResourceAllocCallback      "#\x31"
#define NVSDK_NGX_EParameter_BufferAllocCallback        "#\x32"
#define NVSDK_NGX_EParameter_Tex2DAllocCallback         "#\x33"
#define NVSDK_NGX_EParameter_ResourceReleaseCallback    "#\x34"
#define NVSDK_NGX_EParameter_CreationNodeMask           "#\x35"
#define NVSDK_NGX_EParameter_VisibilityNodeMask         "#\x36"
#define NVSDK_NGX_EParameter_PreviousOutput             "#\x37"
#define NVSDK_NGX_EParameter_MV_Offset_X                 "#\x38"
#define NVSDK_NGX_EParameter_MV_Offset_Y                 "#\x39"
#define NVSDK_NGX_EParameter_Hint_UseFireflySwatter      "#\x3a"
#define NVSDK_NGX_EParameter_Resource_Width              "#\x3b"
#define NVSDK_NGX_EParameter_Resource_Height             "#\x3c"
#define NVSDK_NGX_EParameter_Depth                       "#\x3d"
#define NVSDK_NGX_EParameter_DLSSOptimalSettingsCallback "#\x3e"
#define NVSDK_NGX_EParameter_PerfQualityValue            "#\x3f"
#define NVSDK_NGX_EParameter_RTXValue                    "#\x40"
#define NVSDK_NGX_EParameter_DLSSMode                    "#\x41"

#define NVSDK_NGX_Parameter_NumFrames "NumFrames"
#define NVSDK_NGX_Parameter_Scale "Scale"
#define NVSDK_NGX_Parameter_Width "Width"
#define NVSDK_NGX_Parameter_Height "Height"
#define NVSDK_NGX_Parameter_OutWidth "OutWidth"
#define NVSDK_NGX_Parameter_OutHeight "OutHeight"
#define NVSDK_NGX_Parameter_Sharpness "Sharpness"
#define NVSDK_NGX_Parameter_Scratch "Scratch"
#define NVSDK_NGX_Parameter_Scratch_SizeInBytes "Scratch.SizeInBytes"
#define NVSDK_NGX_Parameter_Hint_HDR "Hint.HDR"
#define NVSDK_NGX_Parameter_Input1 "Input1"
#define NVSDK_NGX_Parameter_Input1_Format "Input1.Format"
#define NVSDK_NGX_Parameter_Input1_SizeInBytes "Input1.SizeInBytes"
#define NVSDK_NGX_Parameter_Input2 "Input2"
#define NVSDK_NGX_Parameter_Input2_Format "Input2.Format"
#define NVSDK_NGX_Parameter_Input2_SizeInBytes "Input2.SizeInBytes"
#define NVSDK_NGX_Parameter_Color "Color"
#define NVSDK_NGX_Parameter_Color_Format "Color.Format"
#define NVSDK_NGX_Parameter_Color_SizeInBytes "Color.SizeInBytes"
#define NVSDK_NGX_Parameter_Albedo "Albedo"
#define NVSDK_NGX_Parameter_Output "Output"
#define NVSDK_NGX_Parameter_Output_Format "Output.Format"
#define NVSDK_NGX_Parameter_Output_SizeInBytes "Output.SizeInBytes"
#define NVSDK_NGX_Parameter_Reset "Reset"
#define NVSDK_NGX_Parameter_BlendFactor "BlendFactor"
#define NVSDK_NGX_Parameter_MotionVectors "MotionVectors"
#define NVSDK_NGX_Parameter_Rect_X "Rect.X"
#define NVSDK_NGX_Parameter_Rect_Y "Rect.Y"
#define NVSDK_NGX_Parameter_Rect_W "Rect.W"
#define NVSDK_NGX_Parameter_Rect_H "Rect.H"
#define NVSDK_NGX_Parameter_MV_Scale_X "MV.Scale.X"
#define NVSDK_NGX_Parameter_MV_Scale_Y "MV.Scale.Y"
#define NVSDK_NGX_Parameter_Model "Model"
#define NVSDK_NGX_Parameter_Format "Format"
#define NVSDK_NGX_Parameter_SizeInBytes "SizeInBytes"
#define NVSDK_NGX_Parameter_ResourceAllocCallback      "ResourceAllocCallback"
#define NVSDK_NGX_Parameter_BufferAllocCallback        "BufferAllocCallback"
#define NVSDK_NGX_Parameter_Tex2DAllocCallback         "Tex2DAllocCallback"
#define NVSDK_NGX_Parameter_ResourceReleaseCallback    "ResourceReleaseCallback"
#define NVSDK_NGX_Parameter_CreationNodeMask           "CreationNodeMask"
#define NVSDK_NGX_Parameter_VisibilityNodeMask         "VisibilityNodeMask"
#define NVSDK_NGX_Parameter_PreviousOutput             "PreviousOutput"
#define NVSDK_NGX_Parameter_MV_Offset_X "MV.Offset.X"
#define NVSDK_NGX_Parameter_MV_Offset_Y "MV.Offset.Y"
#define NVSDK_NGX_Parameter_Hint_UseFireflySwatter "Hint.UseFireflySwatter"
#define NVSDK_NGX_Parameter_Resource_Width "ResourceWidth"
#define NVSDK_NGX_Parameter_Resource_Height "ResourceHeight"
#define NVSDK_NGX_Parameter_Depth "Depth"
#define NVSDK_NGX_Parameter_DLSSOptimalSettingsCallback    "DLSSOptimalSettingsCallback"
#define NVSDK_NGX_Parameter_PerfQualityValue    "PerfQualityValue"
#define NVSDK_NGX_Parameter_RTXValue    "RTXValue"
#define NVSDK_NGX_Parameter_DLSSMode    "DLSSMode"

#endif
