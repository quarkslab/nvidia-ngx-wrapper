#ifndef NGX_LINUX_UNICODE_H
#define NGX_LINUX_UNICODE_H

#include <string>
#include <vector>

std::vector<char16_t> u32tou16(char32_t const *str);
std::string u16tou8(char16_t const *str);

#endif
