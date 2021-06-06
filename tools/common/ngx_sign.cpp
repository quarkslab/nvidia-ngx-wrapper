#include "ngx_sign.h"

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>

#include <dlfcn.h>
#include <unistd.h>
#include <sys/mman.h>

void NGXDisableSigning()
{
  // Get the base address of libnvidia-ngx.so
  // We don't use the "C function pointer" NVSDK_NGX_CUDA_Init here directly,
  // because it will be the pointer to the PLT stub, which belongs to the final
  // binary (and not libnvidia-ngx.so). Hence this dlsym call.
  void* nvptr = dlsym(NULL, "NVSDK_NGX_CUDA_Init");
  Dl_info info;
  if (!dladdr(nvptr, &info)) {
    fprintf(stderr, "[-] Unable to get the base address of libnvidia-ngx.so: '%s'. Is it loaded in the current binary?\n", dlerror());
    exit(1);
  }
  puts(info.dli_fname);

  uint8_t* const instr_addr = (uint8_t*)((uintptr_t)(info.dli_fbase) + 0x555EF);
  const uintptr_t page_size = sysconf(_SC_PAGE_SIZE);
  void* instr_addr_aligned = (void*)((uintptr_t)(instr_addr) & ~(page_size-1));
  printf("%p %p\n", instr_addr, instr_addr_aligned);
  // page_size*2 in the unlikely case where instr_addr and instr_addr+3 overlap on two pages
  if (mprotect(instr_addr_aligned, page_size*2, PROT_READ|PROT_WRITE) != 0) {
    perror("mprotect");
    exit(1);
  }

  // Verify that we patch 'mov esi, 0x80001'
  const uint8_t org_data[] = {0xBE, 0x01, 0x00, 0x08, 0x00};
  if (memcmp(instr_addr, org_data, sizeof(org_data)) != 0) {
    fprintf(stderr, "[-] unsupported version of libnvidia-ngx.so\n");
    exit(1);
  }
  instr_addr[3] = 0;
  if (mprotect(instr_addr_aligned, page_size*2, PROT_EXEC) != 0) {
    perror("mprotect");
    exit(1);
  }
}
