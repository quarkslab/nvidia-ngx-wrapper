#ifndef NGX_WRAPPER_WIN_TYPES_H
#define NGX_WRAPPER_WIN_TYPES_H

#include <stdint.h>

#define CDECL_MSABI __attribute__((ms_abi))

// Simulate some windows type we need
typedef void *HKEY;
typedef int LSTATUS;
typedef const char *LPCSTR;
typedef const char16_t *LPCWSTR;
typedef char16_t *LPWSTR;
typedef uint32_t DWORD;
typedef DWORD *LPDWORD;
typedef uint8_t BYTE;
typedef BYTE *LPBYTE;
typedef void *HMODULE;
typedef void *HANDLE;
typedef long LONG;
typedef unsigned long *ULONG_PTR;
typedef unsigned long long ULONGLONG;

#endif
