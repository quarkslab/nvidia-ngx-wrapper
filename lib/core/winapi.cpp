#include <array>
#include <cstdlib>
#include <cstring>
#include <algorithm>

#include <dlfcn.h>
#include <malloc.h>
#include <sys/types.h>
#include <unistd.h>

#include "log.h"
#include "win_types.h"

#include "winapi.h"
#include "unicode.h"

extern "C" void unkFunc();

namespace {

constexpr LSTATUS MS_ERROR_SUCCESS = 0;
const HKEY HKEY_NGXCORE = (HKEY)0xAABBCCDD;
const HMODULE CALLING_MODULE = (HMODULE)0xEEFF0000;
constexpr DWORD ERROR_INSUFFICIENT_BUFFER = 0x7A;

thread_local DWORD lastError_ = 0;

thread_local DWORD cur_idx = 0;
thread_local std::array<void *, 16> values;

}


CDECL_MSABI LSTATUS RegOpenKeyExW(HKEY hKey, LPCWSTR subKey, uint32_t options,
                                  uint32_t samDesired, HKEY *ret) {
  const auto subKeyU8 = u16tou8(subKey);
  fprintf(stderr, "RegOpenKeyExW: subKey = '%s'\n", subKeyU8.c_str());
  if (subKeyU8 == "SOFTWARE\\NVIDIA Corporation\\Global\\NGXCore") {
    *ret = HKEY_NGXCORE;
  } else {
    fprintf(stderr, "RegOpenKeyExW: unknown subkey, abort!\n");
    exit(1);
  }
  return MS_ERROR_SUCCESS;
}

CDECL_MSABI LSTATUS RegQueryValueExW(HKEY hKey, LPCWSTR lpValueName,
                                     LPDWORD lpReserved, LPDWORD lpType,
                                     LPBYTE lpData, LPDWORD lpcbData) {
  const auto valU8 = u16tou8(lpValueName);
  if (valU8 == "LogLevel" && hKey == HKEY_NGXCORE) {
    DWORD val = 0;
    memcpy(lpData, &val, sizeof(val));
  } else {
    fprintf(stderr, "RegOpenKeyExW: unknown subkey, abort!\n");
    exit(1);
  }
  return MS_ERROR_SUCCESS;
}

CDECL_MSABI LSTATUS RegCloseKey(HKEY hKey) {
  LOG_FUNC();
  return MS_ERROR_SUCCESS;
}

CDECL_MSABI bool GetModuleHandleExA(DWORD dwFlags, LPCSTR lpModuleName,
                                    HMODULE *phModule) {
  LOG_FUNC();
  // It's only used to determine the calling module.
  *phModule = CALLING_MODULE;
  return true;
}

CDECL_MSABI DWORD GetModuleFileNameW(HMODULE hModule, LPWSTR lpFilename,
                                     DWORD nSize) {
  if (hModule != CALLING_MODULE) {
    fprintf(stderr, "GetModuleFileNameW: called with an unknown module\n");
    exit(1);
  }
  LOG_FUNC();
  const char16_t name[] = u"nvngx.dll";
  const size_t len = std::distance(std::begin(name), std::end(name));
  if (nSize < len) {
    return ERROR_INSUFFICIENT_BUFFER;
  }
  std::copy(std::begin(name), std::end(name), lpFilename);
  return len - 1;
}

CDECL_MSABI void *HeapAlloc(void *, DWORD, size_t len) {
  void *ret = malloc(len);
  return ret;
}

CDECL_MSABI bool HeapFree(void *, DWORD, void *buf) {
  free(buf);
  return true;
}


// In order to use nvcuda.dll (and avoid hooking "manually" cudart), we need
// more Windows API to try and get MSVCRT code working!
#ifdef NGX_USE_NVCUDA_DLL

CDECL_MSABI HANDLE HeapCreate(DWORD flOptions, size_t dwInitialSize,
                              size_t dwMaximumSize) {
  LOG_FUNC();
  return (HANDLE)'HEAP';
}

CDECL_MSABI void *LocalAlloc(unsigned flags, size_t len) {
  return HeapAlloc(NULL, 0, len);
}

CDECL_MSABI void *LocalFree(void *ptr) {
  HeapFree(NULL, 0, ptr);
  return NULL;
}

CDECL_MSABI size_t HeapSize(void *hHeap, DWORD dwFlags, void *lpMem) {
  printf("[x] heap_size %p\n", lpMem);
  return malloc_usable_size(lpMem);
}

CDECL_MSABI void *HeapReAlloc(void *hHeap, DWORD dwFlags, void *lpMem,
                              size_t dwBytes) {
  void *ret = realloc(lpMem, dwBytes);
  printf("[x] heap realloc %p for %lu bytes => %p\n", lpMem, dwBytes, ret);
  return ret;
}

struct FILETIME {
  DWORD dwLowDateTime;
  DWORD dwHighDateTime;
};
CDECL_MSABI void GetSystemTimeAsFileTime(FILETIME *time) {
  time->dwLowDateTime = 0xAAAAAAAAU;
  time->dwHighDateTime = 0xBBBBBBBBU;
}

CDECL_MSABI DWORD GetCurrentThreadId() { return gettid(); }

CDECL_MSABI DWORD GetCurrentProcessId() { return getpid(); }

CDECL_MSABI bool QueryPerformanceCounter(uint64_t *val) {
  *val = 10;
  return true;
}

CDECL_MSABI HMODULE LoadLibraryExW(LPCWSTR name, void *file, DWORD flags) {
  std::u16string_view nameview(name);
  fprintf(stderr, "[x] LoadLibraryExW(%s)\n", u16tou8(name).c_str());
  if (nameview == u"kernel32") {
    return (HMODULE)'KERN';
  }
  if (nameview.find(u"gdi32") != std::u16string_view::npos) {
    return (HMODULE)'GDII';
  }
  if (nameview.find(u"Shell32") != std::u16string_view::npos) {
    return (HMODULE)'SHEL';
  }
  if (nameview.find(u"nvcuda") != std::u16string_view::npos) {
#ifdef NGX_USE_NVCUDA_DLL
    return (HMODULE)'CUDA';
#else
    fprintf(stderr, "nvcuda support isn't enabled!\n");
    exit(1);
#endif
  }

  return NULL;
}

CDECL_MSABI bool FreeLibrary(HMODULE hMod) {
  char hModVal[9];
  memset(hModVal, 0, sizeof(hModVal));
  memcpy(hModVal, &hMod, sizeof(hMod));
  std::reverse(&hModVal[0], &hModVal[strlen(hModVal)]);
  fprintf(stderr, "[+] FreeLibrary(%s)\n", hModVal);
  return true;
}

#ifdef NGX_USE_NVCUDA_DLL
void *dl_cuda_;
__attribute__((constructor)) void dlopen_cuda() {
  dl_cuda_ =
      dlopen("/home/aguinet/dev/sandbox/ngx/build/libnvcuda.dll.so", RTLD_LAZY);
  if (dl_cuda_ == nullptr) {
    fprintf(stderr, "unable to dlopen libnvcuda.dll.so: %s\n", dlerror());
    exit(1);
  }
}

void *dlsym_cuda(LPCSTR func) {
  std::string_view sfunc(func);
  if (sfunc.find("cuD3D") != std::string_view::npos) {
    return nullptr;
  }
  void *ret = dlsym(dl_cuda_, (std::string{"wine_"} + func).c_str());
  if (ret == nullptr) {
    fprintf(stderr, "[-] unable to load %s from libnvcuda.dll.so\n", func);
    ret = (void *)unkFunc;
  }
  return ret;
}
#endif

void *GetProcAddressImpl(HMODULE hMod, LPCSTR func) {
  if (func[0] == 'F' && func[1] == 'l' && func[2] == 's') {
    return NULL;
  }

#ifdef NGX_USE_NVCUDA_DLL
  if (strcmp(func, "D3DKMTEnumAdapters2") == 0) {
    return NULL;
  }
  if (hMod == (HMODULE)'CUDA') {
    return dlsym_cuda(func);
  }
#endif

  auto maybe_addr = getWinFunc(func);
  if (maybe_addr) {
    return (void *)*maybe_addr;
  }
  return (void *)unkFunc;
}

CDECL_MSABI void *GetProcAddress(HMODULE hMod, LPCSTR func) {
  void *ret = GetProcAddressImpl(hMod, func);
  char hModVal[9];
  memset(hModVal, 0, sizeof(hModVal));
  memcpy(hModVal, &hMod, sizeof(hMod));
  std::reverse(&hModVal[0], &hModVal[strlen(hModVal)]);
  fprintf(stderr, "[x] GetProcAddress(%s, %s) = %p\n", hModVal, func, ret);
  return ret;
}

CDECL_MSABI DWORD GetFileAttributesW(LPCWSTR lpFileName) {
  auto name = u16tou8(lpFileName);
  fprintf(stderr, "[x] GetFileAttributesW('%s')\n", name.c_str());
  SetLastError(2 /* ERROR_FILE_NOT_FOUND */);
  return -1;
}

struct CRITICAL_SECTION {
  void *DebugInfo;
  LONG LockCount;
  LONG RecursionCount;
  HANDLE OwningThread;
  HANDLE LockSemaphore;
  ULONG_PTR SpinCount;
};

CDECL_MSABI bool
InitializeCriticalSectionEx(CRITICAL_SECTION *lpCriticalSection,
                            DWORD dwSpinCount, DWORD Flags) {
  LOG_FUNC("(%p)", lpCriticalSection);
  memset(lpCriticalSection, 0xEE, sizeof(CRITICAL_SECTION));
  lpCriticalSection->LockCount = 0;
  lpCriticalSection->SpinCount = (ULONG_PTR)malloc(sizeof(unsigned long long));
  *lpCriticalSection->SpinCount = 0;
  pthread_mutex_t *mutex = new pthread_mutex_t{};
  *mutex = PTHREAD_RECURSIVE_MUTEX_INITIALIZER_NP;
  lpCriticalSection->DebugInfo = mutex;
  return true;
}

CDECL_MSABI bool
InitializeCriticalSection(CRITICAL_SECTION *lpCriticalSection) {
  return InitializeCriticalSectionEx(lpCriticalSection, 0, 0);
}

CDECL_MSABI bool
InitializeCriticalSectionAndSpinCount(CRITICAL_SECTION *lpCriticalSection,
                                      DWORD) {
  return InitializeCriticalSectionEx(lpCriticalSection, 0, 0);
}

CDECL_MSABI void EnterCriticalSection(CRITICAL_SECTION *cs) {
  LOG_FUNC("(%p)", cs);
  pthread_mutex_t *mutex = (pthread_mutex_t *)cs->DebugInfo;
  pthread_mutex_lock(mutex);
}

CDECL_MSABI void LeaveCriticalSection(CRITICAL_SECTION *cs) {
  LOG_FUNC("(%p)", cs);
  pthread_mutex_t *mutex = (pthread_mutex_t *)cs->DebugInfo;
  pthread_mutex_unlock(mutex);
}

CDECL_MSABI ULONGLONG VerSetConditionMask(ULONGLONG ConditionMask,
                                          DWORD TypeMask, BYTE Condition) {
  LOG_FUNC();
  return 0;
}

CDECL_MSABI bool VerifyVersionInfoW(void *lpVersionInformation,
                                    DWORD dwTypeMask,
                                    uint64_t dwlConditionMask) {
  LOG_FUNC();
  return true;
}

constexpr char16_t sysDir[] = u"C:\\Windows\\System32";
CDECL_MSABI unsigned GetSystemDirectoryW(LPWSTR lpBuffer, unsigned uSize) {
  LOG_FUNC();
  if (uSize < sizeof(sysDir) / 2) {
    return sizeof(sysDir) / 2;
  }
  memcpy(lpBuffer, sysDir, sizeof(sysDir));
  return (sizeof(sysDir) / 2) - 1;
}

CDECL_MSABI DWORD TlsAlloc() {
  LOG_FUNC();
  if (cur_idx >= 16) {
    fprintf(stderr, "TlsAlloc: not enough slot\n");
    exit(1);
  }
  SetLastError(0);
  values[cur_idx] = nullptr;
  return cur_idx++;
}
CDECL_MSABI bool TlsSetValue(DWORD idx, void *val) {
  if (idx >= cur_idx) {
    LOG_FUNC("idx %u is out of bounds!", idx);
    SetLastError(0xC /*ERROR_INVALID_DATA*/);
    return false;
  }
  LOG_FUNC("(%u, %p)", idx, val);
  values[idx] = val;
  return true;
}
CDECL_MSABI void *TlsGetValue(DWORD idx) {
  if (idx >= cur_idx) {
    fprintf(stderr, "TlsGetValue: idx is out of bound\n");
    SetLastError(0xC /*ERROR_INVALID_DATA*/);
    return NULL;
  }
  SetLastError(0);
  void *const ret = values[idx];
  LOG_FUNC("(%u) = %p", idx, ret);
  return ret;
}

CDECL_MSABI HANDLE GetProcessHeap() { return (HANDLE)0xDEADBEEF00000000ULL; }

CDECL_MSABI DWORD GetFullPathNameW(LPCWSTR lpFileName, DWORD nBufferLength,
                                   LPWSTR lpBuffer, LPWSTR *lpFilePart) {
  fprintf(stderr, "[x] GetFullPathNameW('%s')\n", u16tou8(lpFileName).c_str());
  std::u16string_view name(lpFileName);
  if (nBufferLength <= name.size()) {
    return name.size() + 1;
  }
  memcpy(lpBuffer, lpFileName, name.size() * 2 + 1);
  if (lpFilePart) {
    *lpFilePart = lpBuffer;
  }
  return name.size();
}

CDECL_MSABI bool TryEnterCriticalSection(void *) { return true; }

CDECL_MSABI void DeleteCriticalSection(void *lpCriticalSection) {}

CDECL_MSABI void SetLastError(DWORD val) {
  LOG_FUNC("(%u)", val);
  lastError_ = val;
}

CDECL_MSABI DWORD GetLastError() {
  const DWORD ret = lastError_;
  LOG_FUNC("(%u)", ret);
  return ret;
}

#endif
