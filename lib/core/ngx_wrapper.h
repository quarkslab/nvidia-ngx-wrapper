#ifndef NGX_LINUX_NGX_WRAPPER_H
#define NGX_LINUX_NGX_WRAPPER_H

#include <ngx/nvsdk_ngx_defs.h>
#include "win_types.h"
#include "log.h"

// NVSDK_NGX_Parameter ABI wrapper
struct ID3D11Resource;
struct ID3D12Resource;
struct NVSDK_NGX_Parameter {
  virtual void Set(const char *InName, unsigned long long InValue) = 0;
  virtual void Set(const char *InName, float InValue) = 0;
  virtual void Set(const char *InName, double InValue) = 0;
  virtual void Set(const char *InName, unsigned int InValue) = 0;
  virtual void Set(const char *InName, int InValue) = 0;
  virtual void Set(const char *InName, ID3D11Resource *InValue) = 0;
  virtual void Set(const char *InName, ID3D12Resource *InValue) = 0;
  virtual void Set(const char *InName, void *InValue) = 0;

  virtual NVSDK_NGX_Result Get(const char *InName,
                               unsigned long long *OutValue) = 0;
  virtual NVSDK_NGX_Result Get(const char *InName, float *OutValue) = 0;
  virtual NVSDK_NGX_Result Get(const char *InName, double *OutValue) = 0;
  virtual NVSDK_NGX_Result Get(const char *InName, unsigned int *OutValue) = 0;
  virtual NVSDK_NGX_Result Get(const char *InName, int *OutValue) = 0;
  virtual NVSDK_NGX_Result Get(const char *InName,
                               ID3D11Resource **OutValue) = 0;
  virtual NVSDK_NGX_Result Get(const char *InName,
                               ID3D12Resource **OutValue) = 0;
  virtual NVSDK_NGX_Result Get(const char *InName, void **OutValue) = 0;

  virtual void Reset() = 0;
};

// This is the order that MSVC generates for the NVSDK_NGX_Parameter vtable
// (From https://godbolt.org/z/7q577z6x1)
// DQ  FLAT:virtual void NVSDK_NGX_ParameterImpl::Set(char const *,void *)
// DQ  FLAT:virtual void NVSDK_NGX_ParameterImpl::Set(char const
// *,ID3D12Resource *) DQ  FLAT:virtual void NVSDK_NGX_ParameterImpl::Set(char
// const *,ID3D11Resource *) DQ  FLAT:virtual void
// NVSDK_NGX_ParameterImpl::Set(char const *,int) DQ  FLAT:virtual void
// NVSDK_NGX_ParameterImpl::Set(char const *,unsigned int) DQ  FLAT:virtual void
// NVSDK_NGX_ParameterImpl::Set(char const *,double) DQ  FLAT:virtual void
// NVSDK_NGX_ParameterImpl::Set(char const *,float) DQ  FLAT:virtual void
// NVSDK_NGX_ParameterImpl::Set(char const *,unsigned __int64) DQ  FLAT:virtual
// int NVSDK_NGX_ParameterImpl::Get(char const *,void * *) DQ  FLAT:virtual int
// NVSDK_NGX_ParameterImpl::Get(char const *,ID3D12Resource * *) DQ FLAT:virtual
// int NVSDK_NGX_ParameterImpl::Get(char const *,ID3D11Resource * *) DQ
// FLAT:virtual int NVSDK_NGX_ParameterImpl::Get(char const *,int *) DQ
// FLAT:virtual int NVSDK_NGX_ParameterImpl::Get(char const *,unsigned int *) DQ
// FLAT:virtual int NVSDK_NGX_ParameterImpl::Get(char const *,double *) DQ
// FLAT:virtual int NVSDK_NGX_ParameterImpl::Get(char const *,float *) DQ
// FLAT:virtual int NVSDK_NGX_ParameterImpl::Get(char const *,unsigned __int64
// *) DQ  FLAT:virtual void NVSDK_NGX_ParameterImpl::Reset(void)
//
// We thus create by hand this layout, as unfortunately the ms_abi attribute
// has no effects on classes.

struct MS_NVSDK_NGX_Parameter {
  MS_NVSDK_NGX_Parameter(NVSDK_NGX_Parameter *org)
      : vtable_(the_vtable), org_(org) {}

  void *const *vtable_;
  NVSDK_NGX_Parameter *org_;

private:
  static CDECL_MSABI void Reset(MS_NVSDK_NGX_Parameter *obj) { obj->org_->Reset(); }

#define GET_IMPL(Name, Ty)                                                     \
  static CDECL_MSABI NVSDK_NGX_Result CAT(Get_, Name)(                         \
      const MS_NVSDK_NGX_Parameter *obj, const char *name, Ty *val) {          \
    fprintf(stderr, "[+] NGX Parameter get '%s'\n", name); \
    return obj->org_->Get(name, val);                                          \
  }

#define SET_IMPL(Name, Ty)                                                     \
  static CDECL_MSABI void CAT(Set_, Name)(MS_NVSDK_NGX_Parameter * obj,        \
                                          const char *name, Ty val) {          \
    fprintf(stderr, "[+] NGX Parameter set '%s'\n", name); \
    obj->org_->Set(name, val);                                                 \
  }

  // WARNING: order here is important, as it is tied to the order in the vtable
#define FUNCS_IMPL(WHAT)                                                       \
  WHAT(voidptr, void *)                                                        \
  WHAT(ID3D12, ID3D12Resource *)                                               \
  WHAT(ID3D11, ID3D11Resource *)                                               \
  WHAT(int, int)                                                               \
  WHAT(unsigned, unsigned)                                                     \
  WHAT(double, double)                                                         \
  WHAT(float, float)                                                           \
  WHAT(ull, unsigned long long)

  FUNCS_IMPL(GET_IMPL)
  FUNCS_IMPL(SET_IMPL)
#undef GET_IMPL
#undef SET_IMPL

  static constexpr void *the_vtable[] = {
#define SET_PTR(Name, Ty) (void *)MS_NVSDK_NGX_Parameter::CAT(Set_, Name),
#define GET_PTR(Name, Ty) (void *)MS_NVSDK_NGX_Parameter::CAT(Get_, Name),

      FUNCS_IMPL(SET_PTR)
      FUNCS_IMPL(GET_PTR)
      (void *) Reset
#undef SET_PTR
#undef GET_PTR
#undef FUNCS_IMPL
  };
};

#endif
