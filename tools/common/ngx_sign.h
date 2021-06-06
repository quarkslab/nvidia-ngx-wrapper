#ifndef NGX_LINUX_SIGN_H
#define NGX_LINUX_SIGN_H

// Patch libnvidia-ngx.so to disable signature verification on
// libnvidia-ngx-dl*.so shared libraries.
extern void NGXDisableSigning();

#endif
