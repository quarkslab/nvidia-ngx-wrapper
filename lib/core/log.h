#ifndef NGX_WRAPPER_LOG_H
#define NGX_WRAPPER_LOG_H

#include <stdio.h>
#include <stdint.h>

extern uint64_t getBA();

// Logging
#define _CAT_STR(A, B) A B
#define CAT_STR(A, B) _CAT_STR(A, B)
#define LOG_VA_ARGS(...) , ##__VA_ARGS__

#define LOG_FUNC(format, ...)                                                  \
  fprintf(stderr, CAT_STR("[+] %s, RA off: 0x%08lX ", CAT_STR(format, "\n")),  \
          __FUNCTION__,                                                        \
          (uint64_t)__builtin_return_address(0) - getBA()                      \
                                                      LOG_VA_ARGS(__VA_ARGS__))

#endif
