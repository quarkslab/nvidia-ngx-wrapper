#include "log.h"
#include "cudnn_wrappers.h"

#define CUDNN_DEF_WRAPPER(Name, ...)                                           \
  CDECL_MSABI cudnnStatus_t CAT(ms_, Name)(                                    \
      FOR_EACH(VFUNC_param_decl, __VA_ARGS__)) {                               \
    const auto ret = Name(FOR_EACH(VFUNC_param_use, __VA_ARGS__));             \
    LOG_FUNC(" = %d", ret);                                                    \
    return ret;                                                                \
  }

CUDNN_WRAP_FUNCS(CUDNN_DEF_WRAPPER)
#undef CUDNN_DEF_WRAPPER
