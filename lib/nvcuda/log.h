#define _CAT(A,B) A B
#define CAT_STR(A,B) _CAT(A,B)
#define VA_ARGS(...) , ##__VA_ARGS__
#define TRACE(format, ...) fprintf(stderr, CAT_STR("[CUDA] (%p) %s", format), __builtin_return_address(0), __FUNCTION__ VA_ARGS(__VA_ARGS__))
#define FIXME(format, ...) fprintf(stderr, CAT_STR("[CUDA FIXME] (%p) %s", format), __builtin_return_address(0), __FUNCTION__ VA_ARGS(__VA_ARGS__))
