/*
* Copyright 1993-2018 NVIDIA Corporation.  All rights reserved.
*
* NOTICE TO LICENSEE:
*
* This source code and/or documentation ("Licensed Deliverables") are
* subject to NVIDIA intellectual property rights under U.S. and
* international Copyright laws.
*
* These Licensed Deliverables contained herein is PROPRIETARY and
* CONFIDENTIAL to NVIDIA and is being provided under the terms and
* conditions of a form of NVIDIA software license agreement by and
* between NVIDIA and Licensee ("License Agreement") or electronically
* accepted by Licensee.  Notwithstanding any terms or conditions to
* the contrary in the License Agreement, reproduction or disclosure
* of the Licensed Deliverables to any third party without the express
* written consent of NVIDIA is prohibited.
*
* NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
* LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
* SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
* PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
* NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
* DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
* NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
* NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
* LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
* SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
* DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
* WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
* ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
* OF THESE LICENSED DELIVERABLES.
*
* U.S. Government End Users.  These Licensed Deliverables are a
* "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
* 1995), consisting of "commercial computer software" and "commercial
* computer software documentation" as such terms are used in 48
* C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
* only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
* 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
* U.S. Government End Users acquire the Licensed Deliverables with
* only those rights set forth herein.
*
* Any use of the Licensed Deliverables in individual and commercial
* software must include, in the user documentation and internal
* comments to the code, the above Disclaimer and U.S. Government End
* Users Notice.
*/

#include <iostream>

#include <cuda_runtime.h>

#include "CmdArgsMap.hpp"
#include "image_io_util.hpp"

#include <ngx/nvsdk_ngx.h>
#include <ngx_sign.h>

typedef struct _appParams {
    std::string wd;
    std::string input_image_filename;
    std::string output_image_filename;
    uint32_t uprez_factor;
} appParams;

NVSDK_NGX_Handle *DUHandle{ nullptr };
NVSDK_NGX_Parameter *params{ nullptr };

long long app_id = 0x0;

void NGXTestCallback(float InProgress, bool &OutShouldCancel)
{
    //Perform progress handling here.
    //For long running cases.
    //e.g. LOG("Progress callback %.2f%", InProgress * 100.0f);
    OutShouldCancel = false;
}

__attribute__((noinline)) static void test(NVSDK_NGX_Parameter* p)
{
  int v;
  p->Get(NVSDK_NGX_Parameter_Width, &v);
  printf("width: %llu\n", v);
}

int main(int argc, char *argv[])
{
  NGXDisableSigning();

  appParams myAppParams{"","","",0};

  std::cout << "\nNVIDIA NGX ISR Sample" << std::endl; 

  // Process command line arguments
  bool show_help = false;
  int uprez_factor_arg;
  CmdArgsMap cmdArgs = CmdArgsMap(argc, argv, "--")
    ("help", "Produce help message", &show_help)
    ("wd", "Base directory for image input and output files", &myAppParams.wd, myAppParams.wd)
    ("input", "Input image filename", &myAppParams.input_image_filename, myAppParams.input_image_filename)
    ("output", "Output image filename", &myAppParams.output_image_filename, myAppParams.output_image_filename)
    ("factor", "Super resolution factor (2, 4, 8)", &uprez_factor_arg, uprez_factor_arg);

  if (argc == 1 || show_help)
  {
    std::cout << cmdArgs.help();
    return 1;
  }

  // Verify that specified super resolution factor is valid
  if ((uprez_factor_arg != 2) && (uprez_factor_arg != 4) && (uprez_factor_arg != 8))
  {
    std::cerr << "Image super resolution factor (--factor) must be one of 2, 4 or 8." << std::endl;
    return 1;
  }
  else 
  {
    myAppParams.uprez_factor = uprez_factor_arg;
  }

  // Verify input image file specified.
  if (myAppParams.input_image_filename.empty())
  {
    std::cerr << "Input image filename must be specified." << std::endl;
    return 1;
  }

  // Verify output image file specified.
  if (myAppParams.output_image_filename.empty())
  {
    std::cerr << "Output image filename must be specified." << std::endl;
    return 1;
  }

  // Append trailing '/' to working directory if not specified to reduce user errors.
  if (!myAppParams.wd.empty())
  {
    switch (myAppParams.wd[myAppParams.wd.size() - 1])
    {
#ifdef _MSC_VER
      case '\\':
#endif // _MSC_VER
      case '/':
        break;
      default:
        myAppParams.wd += '/';
        break;
    }
  }

  // Read input image into host memory
  std::string input_image_file_path = myAppParams.wd + myAppParams.input_image_filename;

  int image_width, image_height;
  const auto rgba_bitmap_ptr = getRgbImage(input_image_file_path, image_width, image_height);
  if (nullptr == rgba_bitmap_ptr)
  {
    std::cerr << "Error reading Image " << input_image_file_path << std::endl;
    return 1;
  }

  // Copy input image to GPU device memory
  size_t in_image_row_bytes = image_width * 3;
  size_t in_image_width = image_width;
  size_t in_image_height = image_height;
  void *in_image_dev_ptr;

  if (cudaMalloc(&in_image_dev_ptr, in_image_row_bytes * in_image_height) != cudaSuccess)
  {
    std::cerr << "Error allocating output image CUDA buffer" << std::endl;
    return 1;
  }

  if (cudaMemcpy(in_image_dev_ptr, rgba_bitmap_ptr.get(), in_image_row_bytes * in_image_height,
        cudaMemcpyHostToDevice) != cudaSuccess)
  {
    std::cerr << "Error copying input RGBA image to CUDA buffer" << std::endl;
    return -1;
  }

  // Calculate output image paramters
  size_t out_image_row_bytes = image_width * myAppParams.uprez_factor * 3;
  size_t out_image_width = image_width * myAppParams.uprez_factor;
  size_t out_image_height = image_height * myAppParams.uprez_factor;
  void * out_image_dev_ptr;

  if (cudaMalloc(&out_image_dev_ptr, out_image_row_bytes * out_image_height) != cudaSuccess)
  {
    std::cout << "Error allocating output image CUDA buffer" << std::endl;
    return 1;
  }

  // Initialize NGX.
  NVSDK_NGX_Result rslt = NVSDK_NGX_Result_Success;
  std::cerr << "[INIT BEGIN]\n";
  rslt = NVSDK_NGX_CUDA_Init(app_id, L"./", NVSDK_NGX_Version_API);
  if (rslt != NVSDK_NGX_Result_Success) {
    std::cerr << "Error Initializing NGX. " << std::endl;
    return 1;
  }
  std::cerr << "[INIT END]\n";

  // Get the parameter block.
  NVSDK_NGX_CUDA_GetParameters(&params);

  // Verify feature is supported
  int Supported = 0;
  params->Get(NVSDK_NGX_Parameter_ImageSuperResolution_Available, &Supported);
  if (!Supported)
  {
    std::cerr << "NVSDK_NGX_Feature_ImageSuperResolution Unavailable on this System" << std::endl;
    return 1;
  }

  // Set the default hyperparameters for inferrence.
  params->Set(NVSDK_NGX_Parameter_Width, (unsigned long long)in_image_width);
  params->Set(NVSDK_NGX_Parameter_Height, (unsigned long long)in_image_height);
  params->Set(NVSDK_NGX_Parameter_Scale, myAppParams.uprez_factor);

  test(params);


  // Get the scratch buffer size and create the scratch allocation.
  // (if required)
  unsigned long long byteSize{ 0u };
  void *scratchBuffer{ nullptr };
  std::cerr << "Calling NVSDK_NGX_CUDA_GetScratchBufferSize...\n";
  rslt = NVSDK_NGX_CUDA_GetScratchBufferSize(NVSDK_NGX_Feature_ImageSuperResolution, params, &byteSize);
  if (rslt != NVSDK_NGX_Result_Success) {
    std::cerr << "Error Getting NGX Scratch Buffer Size: " << std::hex << rslt << "." << std::endl;
    return 1;
  }
  std::cerr << "Scratch Buffer Size: " << byteSize << "\n";
  cudaMalloc(&scratchBuffer, byteSize > 0u ? byteSize : 1u); //cudaMalloc, unlike malloc, fails on 0 size allocations

  // Update the parameter block with the scratch space metadata.:
  params->Set(NVSDK_NGX_Parameter_Scratch, scratchBuffer);
  params->Set(NVSDK_NGX_Parameter_Scratch_SizeInBytes, (uint32_t)byteSize);

  // Create the feature
  NVSDK_NGX_CUDA_CreateFeature(NVSDK_NGX_Feature_ImageSuperResolution, params, &DUHandle);

  // Pass the pointers to the GPU allocations to the
  // parameter block along with the format and size.
  params->Set(NVSDK_NGX_Parameter_Color_SizeInBytes, (unsigned long long) (in_image_row_bytes * in_image_height));
  params->Set(NVSDK_NGX_Parameter_Color_Format, NVSDK_NGX_Buffer_Format_RGB8UI);
  params->Set(NVSDK_NGX_Parameter_Color, in_image_dev_ptr);
  params->Set(NVSDK_NGX_Parameter_Output_SizeInBytes, (unsigned long long) (out_image_row_bytes * out_image_height));
  params->Set(NVSDK_NGX_Parameter_Output_Format, NVSDK_NGX_Buffer_Format_RGB8UI);
  params->Set(NVSDK_NGX_Parameter_Output, out_image_dev_ptr);

  //Synchronize the device.
  cudaDeviceSynchronize();

  //Execute the feature.
  std::cerr << "[+] Running EvaluateFeature\n";
  rslt = NVSDK_NGX_CUDA_EvaluateFeature(DUHandle, params, nullptr /*NGXTestCallback*/);
  if (rslt != NVSDK_NGX_Result_Success) {
    std::cerr << "Error running NVSDK_NGX_CUDA_EvaluateFeature: " << std::hex << rslt << "." << std::endl;
  }
  std::cerr << "[+] EvaluateFeature DONE \\o/\n";

  //Synchronize once more.
  cudaDeviceSynchronize();

  // Copy output image from GPU device memory
  std::unique_ptr<unsigned char[] > out_image{};
  out_image = std::unique_ptr<unsigned char[]>(new unsigned char[out_image_row_bytes * out_image_height]);

  if (cudaMemcpy(out_image.get(), out_image_dev_ptr, out_image_row_bytes * out_image_height, 
        cudaMemcpyDeviceToHost) != cudaSuccess)
  {
    std::cout << "Error copying output image from CUDA buffer" << std::endl;
    return 1;
  }

  // Write output image from host memory
  std::string output_image_file_path = myAppParams.wd + myAppParams.output_image_filename;
  putRgbImage(output_image_file_path, out_image.get(), (int)out_image_width, (int)out_image_height);

  // Tear down the feature.
  NVSDK_NGX_CUDA_ReleaseFeature(DUHandle);

  // Shutdown NGX
  NVSDK_NGX_CUDA_Shutdown();

  //Clean up device buffers.
  cudaFree(in_image_dev_ptr);
  cudaFree(out_image_dev_ptr);

  in_image_dev_ptr = NULL;
  out_image_dev_ptr = NULL;

  return 0;
}
