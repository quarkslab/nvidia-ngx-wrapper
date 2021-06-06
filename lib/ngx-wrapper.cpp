#include <LIEF/Abstract/Symbol.hpp>
#include <LIEF/PE/Binary.hpp>
#include <QBDL/Engine.hpp>
#include <QBDL/engines/Native.hpp>
#include <QBDL/loaders/PE.hpp>
#include <QBDL/log.hpp>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>

#include <dlfcn.h>

#include <ngx/nvsdk_ngx_defs.h>

#include "core/macro_helpers.h"
#include "core/log.h"
#include "core/win_types.h"
#include "core/ngx_wrapper.h"

#include "core/cudnn_wrappers.h"
#include "core/unicode.h"
#include "core/winapi.h"

#include "cuda_wrappers.h"
#include "crt_hooks.h"

extern "C" void unkFunc();

namespace {

struct NVNGXDll;
NVNGXDll &getDLL();

std::optional<uint64_t> getWinFunc(std::string const &name);

static const std::unordered_map<std::string, uint64_t> imp_funcs_ = {
  // Windows APIs
#define WIN_FUNC_DECL(Name)\
  {#Name, (uint64_t)Name},

  WIN_FUNC_NAMES(WIN_FUNC_DECL)
#undef WIN_FUNC_DECL

  // cuDNN
#define CUDNN_ENTRY(Name, ...) {#Name, (uint64_t)CAT(ms_, Name)},

    CUDNN_WRAP_FUNCS(CUDNN_ENTRY)
#undef CUDNN_ENTRY
};



std::optional<uint64_t> getWinFunc(std::string const &name) {
  auto it = imp_funcs_.find(name);
  if (it == imp_funcs_.end()) {
    return {};
  }
  return it->second;
}

struct WinTargetSystem : public QBDL::Engines::Native::TargetSystem {
  WinTargetSystem() : TargetSystem(mem_) {}

  uint64_t symlink(QBDL::Loader &loader, const LIEF::Symbol &sym) override {
    const std::string &name = sym.name();
    uint64_t ret = 0;
    auto maybe_addr = getWinFunc(name);
    if (maybe_addr) {
      ret = *maybe_addr;
      fprintf(stderr, "symbol %s resolves to %p\n", name.c_str(), (void *)ret);
    } else {
      ret = (uint64_t)unkFunc;
    }
    return ret;
  }

private:
  QBDL::Engines::Native::TargetMemory mem_;
};

CDECL_MSABI static void ms_atexit(void* ptr) { }
CDECL_MSABI static void ms_alloca_probe() { }

struct NVNGXDll {
  static std::string getNGXOrgDLLPath() {
    // Get the name of the current library
    Dl_info info;
    if (!dladdr((const void*)&getNGXOrgDLLPath, &info)) {
      fprintf(stderr, "Unable to get the name of the current NGX library: %s\n", dlerror());
      exit(1);
    }
    const char* name = info.dli_fname;

    const char* compstart = strrchr(name, '-');
    const char* dot = strrchr(name, '.');
    if (compstart == NULL || dot == NULL || dot < compstart) {
      fprintf(stderr, "Library name '%s' should follow the format libnvidia-ngx-XX.so\n", name);
      exit(1);
    }
    const std::string comp_name(compstart+1, std::distance(compstart,dot-1));

    std::string ret;
    const char* nvngx_dir = getenv("NGX_WIN_DLL_DIR");
    if (nvngx_dir != NULL) {
      ret = nvngx_dir;
    }
    else {
      const char* sep = strrchr(name, '/');
      if (sep != NULL) {
        ret = std::string(name, std::distance(name, sep));
      }
      else {
        ret = ".";
      }
    }
    ret += "/nvngx_" + comp_name + ".dll";
    fprintf(stderr, "[x] using original DLL at '%s'\n", ret.c_str());
    return ret;
  }

  NVNGXDll() {
    QBDL::setLogLevel(QBDL::LogLevel::warn);
    ngx_pe_ = QBDL::Loaders::PE::from_file(getNGXOrgDLLPath().c_str(), ts_);
    if (!ngx_pe_) {
      fprintf(stderr, "[-] Unable to load original DLL!\n");
      exit(1);
    }

    fprintf(stderr, "[+] PE base address: %p\n", (void *)base_address());

#define SET_EXPORT(FTy, Name)                                                  \
  Name = reinterpret_cast<FTy>(ngx_pe_->get_address(#Name));                   \
  if (!Name) {                                                                 \
    fprintf(stderr, "unable to find " #Name "\n");                             \
    exit(1);                                                                   \
  }

    SET_EXPORT(fty_U32Void, NVSDK_NGX_GetAPIVersion);
    SET_EXPORT(fty_U32Void, NVSDK_NGX_GetApplicationId);
    SET_EXPORT(fty_U32Void, NVSDK_NGX_GetDriverVersion);
    SET_EXPORT(fty_U32Void, NVSDK_NGX_GetGPUArchitecture);
    SET_EXPORT(fty_U32Void, NVSDK_NGX_GetSnippetVersion);
    SET_EXPORT(fty_U32VoidPtr, NVSDK_NGX_PopulateParameters);
    SET_EXPORT(fty_U32VoidPtr, NVSDK_NGX_SetInfoCallback);

    SET_EXPORT(fty_NGX_CUDA_Init, NVSDK_NGX_CUDA_Init);
    SET_EXPORT(fty_NGX_CUDA_GetScratchBufferSize,
               NVSDK_NGX_CUDA_GetScratchBufferSize);
    SET_EXPORT(fty_NGX_CUDA_CreateFeature, NVSDK_NGX_CUDA_CreateFeature);
    SET_EXPORT(fty_NGX_CUDA_EvaluateFeature, NVSDK_NGX_CUDA_EvaluateFeature);
    SET_EXPORT(fty_NGX_CUDA_ReleaseFeature, NVSDK_NGX_CUDA_ReleaseFeature);
    SET_EXPORT(fty_NGX_CUDA_Shutdown, NVSDK_NGX_CUDA_Shutdown);
  }

  void run() {
    // Setup hooks
#ifndef NGX_USE_NVCUDA_DLL
#define CUDA_HOOK(Name, Off, ...) hook(Off, (uint64_t)CAT(ms_, Name));

    CUDA_HOOK_FUNCS(CUDA_HOOK);
#undef CUDA_HOOK

#define CUDA_INTERNAL_HOOK(Off, Name)\
    hook(Off, (uint64_t)Name);

    CUDA_INTERNAL_FUNCS(CUDA_INTERNAL_HOOK)
#undef CUDA_INTERNAL_HOOK

#endif

    typedef void(CDECL_MSABI * fty_void)();
#ifdef NGX_USE_NVCUDA_DLL
    // Run constructors
    const auto &bin = ngx_pe_->get_binary();
    for (auto const &ctor : bin.ctor_functions()) {
      const uint64_t rva = ctor.address();
      const uint64_t ptr = ngx_pe_->get_address(rva);
      fprintf(stderr, "[x] run constructor at %p (rva 0x%08lX)\n", (void *)ptr,
              rva);
      const fty_void func = (fty_void)ptr;
      func();
    }

    // Ideally, we would just call the DllEntry function of the DLL, that would
    // call both the MSVCRT initializers and the CUDA ones. But we need more
    // Windows API to do so, and PEB emulation.
#if 0
    // Call DllEntryPoint
    typedef int(CDECL_MSABI * fty_dllentry)(void *, DWORD, void *);
    fty_dllentry mydllentry = (fty_dllentry)ngx_pe_->entrypoint();
    fprintf(stderr, "[x] run entry point at %p\n", mydllentry);
    mydllentry(NULL, 1 /* DLL_PROCESS_ATTACH */, NULL);
#endif

    fprintf(stderr, "[+] initialize heap\n");
    fty_void heap_init = (fty_void)ngx_pe_->get_address(0x0B210);
    heap_init();

    fprintf(stderr, "[+] initialize locks\n");
    fty_void locks_init = (fty_void)ngx_pe_->get_address(0x3ED60);
    locks_init();

    fprintf(stderr, "[+] initialize TLS\n");
    fty_void tls_init = (fty_void)ngx_pe_->get_address(0x45D30);
    tls_init();
#else
    hook(CRT_ATEXIT_OFF, (uint64_t)ms_atexit);
    hook(CRT_ALLOCA_PROBE, (uint64_t)ms_alloca_probe);

    fprintf(stderr, "[+] initialize cuda\n");
    fty_void *ptrs_begin = (fty_void *)ngx_pe_->get_address(CUDA_WRAPPER_INIT_BEGIN);
    fty_void *ptrs_end = (fty_void *)ngx_pe_->get_address(CUDA_WRAPPER_INIT_END);
    for (fty_void *ptr = ptrs_begin; ptr != ptrs_end; ++ptr) {
      (*ptr)();
    }
#endif
  }

  void hook(uint64_t offset, uint64_t addr) {
    void *dst = (void *)ngx_pe_->get_address(offset);
    // mov r11, 0x0001020304050607; jmp r11
    uint8_t stub[] = {0x49, 0xBB, 0x07, 0x06, 0x05, 0x04, 0x03,
                      0x02, 0x01, 0x00, 0x41, 0xFF, 0xE3};
    memcpy(&stub[2], &addr, sizeof(addr));
    memcpy(dst, stub, sizeof(stub));
  }

  typedef uint32_t(CDECL_MSABI *fty_U32Void)();
  typedef uint32_t(CDECL_MSABI *fty_U32VoidPtr)(void *);

  fty_U32Void NVSDK_NGX_GetAPIVersion;
  fty_U32Void NVSDK_NGX_GetApplicationId;
  fty_U32Void NVSDK_NGX_GetDriverVersion;
  fty_U32Void NVSDK_NGX_GetGPUArchitecture;
  fty_U32Void NVSDK_NGX_GetSnippetVersion;
  fty_U32VoidPtr NVSDK_NGX_PopulateParameters;
  fty_U32VoidPtr NVSDK_NGX_SetInfoCallback;

  typedef NVSDK_NGX_Result(CDECL_MSABI *fty_NGX_CUDA_Init)(unsigned long long,
                                                           const char16_t *,
                                                           NVSDK_NGX_Version);
  fty_NGX_CUDA_Init NVSDK_NGX_CUDA_Init;

  typedef NVSDK_NGX_Result(CDECL_MSABI *fty_NGX_CUDA_GetScratchBufferSize)(NVSDK_NGX_Feature, MS_NVSDK_NGX_Parameter*, unsigned long long*);
  fty_NGX_CUDA_GetScratchBufferSize NVSDK_NGX_CUDA_GetScratchBufferSize;

  typedef NVSDK_NGX_Result(CDECL_MSABI *fty_NGX_CUDA_CreateFeature)(
      NVSDK_NGX_Feature, MS_NVSDK_NGX_Parameter *, NVSDK_NGX_Handle **);
  fty_NGX_CUDA_CreateFeature NVSDK_NGX_CUDA_CreateFeature;

  typedef NVSDK_NGX_Result(CDECL_MSABI *fty_NGX_CUDA_EvaluateFeature)(
      const NVSDK_NGX_Handle *, MS_NVSDK_NGX_Parameter *, void *);
  fty_NGX_CUDA_EvaluateFeature NVSDK_NGX_CUDA_EvaluateFeature;

  typedef NVSDK_NGX_Result(CDECL_MSABI *fty_NGX_CUDA_ReleaseFeature)(
      NVSDK_NGX_Handle *);
  typedef NVSDK_NGX_Result(CDECL_MSABI *fty_NGX_CUDA_Shutdown)();
  fty_NGX_CUDA_ReleaseFeature NVSDK_NGX_CUDA_ReleaseFeature;
  fty_NGX_CUDA_Shutdown NVSDK_NGX_CUDA_Shutdown;

  uint64_t base_address() const { return ngx_pe_->base_address(); }

private:
  WinTargetSystem ts_;
  std::unique_ptr<QBDL::Loaders::PE> ngx_pe_;
};

std::optional<NVNGXDll> dll_;

NVNGXDll &getDLL() {
  if (!dll_) {
    dll_.emplace();
    dll_->run();
  }
  return *dll_;
}

} // namespace

void unkFunc() {
  void *ra = __builtin_return_address(0);
  uint64_t off = (uint64_t)ra - getDLL().base_address();
  fprintf(stderr,
          "Calling unimplemented function. RA = %p, offset = 0x%016lX\n", ra,
          off);
  fflush(stderr);
  exit(1);
}

uint64_t getBA() { return getDLL().base_address(); }

#define NGX_API_EXPORT __attribute__((visibility("default")))

extern "C" {

NGX_API_EXPORT uint32_t NVSDK_NGX_GetAPIVersion() {
  LOG_FUNC();
  return getDLL().NVSDK_NGX_GetAPIVersion();
}

NGX_API_EXPORT uint32_t NVSDK_NGX_GetApplicationId() {
  LOG_FUNC();
  return getDLL().NVSDK_NGX_GetApplicationId();
}

NGX_API_EXPORT uint32_t NVSDK_NGX_GetDriverVersion() {
  LOG_FUNC();
  return getDLL().NVSDK_NGX_GetDriverVersion();
}

NGX_API_EXPORT uint32_t NVSDK_NGX_GetGPUArchitecture() {
  LOG_FUNC();
  return getDLL().NVSDK_NGX_GetGPUArchitecture();
}

NGX_API_EXPORT uint32_t NVSDK_NGX_GetSnippetVersion() {
  LOG_FUNC();
  return getDLL().NVSDK_NGX_GetSnippetVersion();
}

NGX_API_EXPORT uint32_t NVSDK_NGX_PopulateParameters(void *p) {
  const auto ret = getDLL().NVSDK_NGX_PopulateParameters(p);
  LOG_FUNC(" = %d", ret);
  return ret;
}

NGX_API_EXPORT uint32_t NVSDK_NGX_SetInfoCallback(void *p) {
  LOG_FUNC();
  return getDLL().NVSDK_NGX_SetInfoCallback(p);
}

NGX_API_EXPORT NVSDK_NGX_Result NVSDK_NGX_CUDA_Init(
    unsigned long long InApplicationId, const char32_t *InApplicationDataPath,
    NVSDK_NGX_Version InSDKVersion) {
  const auto InApplicationDataPathU16 = u32tou16(InApplicationDataPath);
  const auto ret = getDLL().NVSDK_NGX_CUDA_Init(
      InApplicationId, &InApplicationDataPathU16[0], InSDKVersion);
  LOG_FUNC(" = %d", ret);
  return ret;
}

NGX_API_EXPORT NVSDK_NGX_Result
NVSDK_NGX_CUDA_GetScratchBufferSize(NVSDK_NGX_Feature InFeatureId, NVSDK_NGX_Parameter *InParameters, unsigned long long *OutSizeInBytes) {
  LOG_FUNC();
  MS_NVSDK_NGX_Parameter ms_params{InParameters};
  return getDLL().NVSDK_NGX_CUDA_GetScratchBufferSize(InFeatureId, &ms_params, OutSizeInBytes);
}

NGX_API_EXPORT NVSDK_NGX_Result NVSDK_NGX_CUDA_CreateFeature(
    NVSDK_NGX_Feature InFeatureID, NVSDK_NGX_Parameter *InParameters,
    NVSDK_NGX_Handle **OutHandle) {
  MS_NVSDK_NGX_Parameter ms_params{InParameters};
  const auto ret =
      getDLL().NVSDK_NGX_CUDA_CreateFeature(InFeatureID, &ms_params, OutHandle);
  LOG_FUNC("(InFeatureID=%X, InParameters=%p, OutHandle=%p) = %d",
         InFeatureID, InParameters, OutHandle, ret);
  return ret;
}

NGX_API_EXPORT NVSDK_NGX_Result NVSDK_NGX_CUDA_EvaluateFeature(
    const NVSDK_NGX_Handle *InFeatureHandle, NVSDK_NGX_Parameter *InParameters,
    void *InCallback) {
  LOG_FUNC();
  if (InCallback != nullptr) {
    fprintf(stderr,
            "  => InCallback != nullptr, a wrapper needs to be implemented\n");
    exit(1);
  }
  MS_NVSDK_NGX_Parameter ms_params{InParameters};
  return getDLL().NVSDK_NGX_CUDA_EvaluateFeature(InFeatureHandle, &ms_params,
                                                 nullptr);
}

NGX_API_EXPORT NVSDK_NGX_Result
NVSDK_NGX_CUDA_ReleaseFeature(NVSDK_NGX_Handle *InHandle) {
  LOG_FUNC();
  return getDLL().NVSDK_NGX_CUDA_ReleaseFeature(InHandle);
}

NGX_API_EXPORT NVSDK_NGX_Result NVSDK_NGX_CUDA_Shutdown() {
  LOG_FUNC();
  return getDLL().NVSDK_NGX_CUDA_Shutdown();
}

NGX_API_EXPORT bool NvOptimusEnablementCuda = 1;

} // extern "C
