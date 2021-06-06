#ifndef NGX_LINUX_WINAPI_H
#define NGX_LINUX_WINAPI_H

#include <optional>
#include <cstdint>
#include <string>

#include "win_types.h"

// To be implemented by the final DLL
std::optional<uint64_t> getWinFunc(std::string const &name);

#ifdef __cplusplus
extern "C" {
#endif

struct FILETIME;
struct CRITICAL_SECTION;

CDECL_MSABI void SetLastError(DWORD val);
CDECL_MSABI DWORD GetLastError();
CDECL_MSABI LSTATUS RegOpenKeyExW(HKEY hKey, LPCWSTR subKey, uint32_t options,
                                  uint32_t samDesired, HKEY *ret);
CDECL_MSABI LSTATUS RegQueryValueExW(HKEY hKey, LPCWSTR lpValueName,
                                     LPDWORD lpReserved, LPDWORD lpType,
                                     LPBYTE lpData, LPDWORD lpcbData);
CDECL_MSABI LSTATUS RegCloseKey(HKEY hKey);
CDECL_MSABI bool GetModuleHandleExA(DWORD dwFlags, LPCSTR lpModuleName,
                                    HMODULE *phModule);
CDECL_MSABI DWORD GetModuleFileNameW(HMODULE hModule, LPWSTR lpFilename,
                                     DWORD nSize);
CDECL_MSABI HANDLE HeapCreate(DWORD flOptions, size_t dwInitialSize,
                              size_t dwMaximumSize);
CDECL_MSABI void *HeapAlloc(void *, DWORD, size_t len);
CDECL_MSABI void *LocalAlloc(unsigned flags, size_t len);
CDECL_MSABI bool HeapFree(void *, DWORD, void *buf);
CDECL_MSABI void *LocalFree(void *ptr);
CDECL_MSABI size_t HeapSize(void *hHeap, DWORD dwFlags, void *lpMem);
CDECL_MSABI void *HeapReAlloc(void *hHeap, DWORD dwFlags, void *lpMem,
                              size_t dwBytes);
CDECL_MSABI void GetSystemTimeAsFileTime(FILETIME *time);
CDECL_MSABI DWORD GetCurrentThreadId();
CDECL_MSABI DWORD GetCurrentProcessId();
CDECL_MSABI bool QueryPerformanceCounter(uint64_t *val);
CDECL_MSABI HMODULE LoadLibraryExW(LPCWSTR name, void *file, DWORD flags);
CDECL_MSABI bool FreeLibrary(HMODULE hMod);
void *GetProcAddressImpl(HMODULE hMod, LPCSTR func);
CDECL_MSABI void *GetProcAddress(HMODULE hMod, LPCSTR func);
CDECL_MSABI DWORD GetFileAttributesW(LPCWSTR lpFileName);
CDECL_MSABI bool
InitializeCriticalSectionEx(CRITICAL_SECTION *lpCriticalSection,
                            DWORD dwSpinCount, DWORD Flags);
CDECL_MSABI bool
InitializeCriticalSection(CRITICAL_SECTION *lpCriticalSection);
CDECL_MSABI bool
InitializeCriticalSectionAndSpinCount(CRITICAL_SECTION *lpCriticalSection,
                                      DWORD);
CDECL_MSABI void EnterCriticalSection(CRITICAL_SECTION *cs);
CDECL_MSABI void LeaveCriticalSection(CRITICAL_SECTION *cs);
CDECL_MSABI ULONGLONG VerSetConditionMask(ULONGLONG ConditionMask,
                                          DWORD TypeMask, BYTE Condition);
CDECL_MSABI bool VerifyVersionInfoW(void *lpVersionInformation,
                                    DWORD dwTypeMask,
                                    uint64_t dwlConditionMask);
CDECL_MSABI unsigned GetSystemDirectoryW(LPWSTR lpBuffer, unsigned uSize);
CDECL_MSABI DWORD TlsAlloc();
CDECL_MSABI bool TlsSetValue(DWORD idx, void *val);
CDECL_MSABI void *TlsGetValue(DWORD idx);
CDECL_MSABI HANDLE GetProcessHeap();
CDECL_MSABI DWORD GetFullPathNameW(LPCWSTR lpFileName, DWORD nBufferLength,
                                   LPWSTR lpBuffer, LPWSTR *lpFilePart);
CDECL_MSABI bool TryEnterCriticalSection(void *);
CDECL_MSABI void DeleteCriticalSection(void *lpCriticalSection);

// Windows functions used by the NGX code tiself
#define WIN_FUNC_NAMES_NGX(FM)\
  FM(RegOpenKeyExW) \
  FM(RegQueryValueExW) \
  FM(RegCloseKey) \
  FM(GetModuleHandleExA) \
  FM(GetModuleFileNameW) \
  FM(HeapAlloc) \
  FM(HeapFree) \

// Windows functions used by MSVCRT
#define WIN_FUNC_NAMES_MSVCRT(FM)\
  FM(LocalAlloc) \
  FM(LocalFree) \
  FM(HeapCreate) \
  FM(HeapReAlloc) \
  FM(HeapSize) \
  FM(EnterCriticalSection) \
  FM(LeaveCriticalSection) \
  FM(GetSystemTimeAsFileTime) \
  FM(GetCurrentThreadId) \
  FM(GetCurrentProcessId) \
  FM(QueryPerformanceCounter) \
  FM(LoadLibraryExW) \
  FM(FreeLibrary) \
  FM(GetProcAddress) \
  FM(InitializeCriticalSection) \
  FM(InitializeCriticalSectionEx) \
  FM(InitializeCriticalSectionAndSpinCount) \
  FM(SetLastError) \
  FM(GetLastError) \
  FM(TlsAlloc) \
  FM(TlsSetValue) \
  FM(TlsGetValue) \
  FM(GetProcessHeap) \
  FM(VerSetConditionMask) \
  FM(VerifyVersionInfoW) \
  FM(GetSystemDirectoryW) \
  FM(GetFileAttributesW) \
  FM(GetFullPathNameW) \
  FM(TryEnterCriticalSection) \
  FM(DeleteCriticalSection)

#ifdef NGX_USE_NVCUDA_DLL
#define WIN_FUNC_NAMES(FM)\
  WIN_FUNC_NAMES_NGX(FM)\
  WIN_FUNC_NAMES_MSVCRT(FM)

#else
#define WIN_FUNC_NAMES(FM)\
  WIN_FUNC_NAMES_NGX(FM)

#endif

#ifdef __cplusplus
}
#endif

#endif
