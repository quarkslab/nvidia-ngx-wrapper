#include "unicode.h"

// Credits to Jeremy Demeule
template <class OutputIt> OutputIt u32tou16(OutputIt out, char32_t codepoint) {
  static constexpr char32_t last_code_point = 0x10FFFF;

  if (codepoint < 0x10000) {
    *out++ = static_cast<char16_t>(codepoint);
  } else if (codepoint < last_code_point) {
    codepoint -= 0x10000;
    *out++ = static_cast<char16_t>(0xD800 + (codepoint >> 10));
    *out++ = static_cast<char16_t>(0xDC00 + (codepoint & 0x3FF));
  } else {
    // TODO: error management here (replacement char, exception...)
  }

  return out;
}

std::vector<char16_t> u32tou16(char32_t const *str) {
  std::vector<char16_t> ret;
  while (true) {
    const char32_t v = *str++;
    if (v == 0) {
      break;
    }
    u32tou16(std::back_inserter(ret), v);
  }
  ret.push_back(0);
  return ret;
}

std::string u16tou8(char16_t const *str) {
  std::string ret;
  while (true) {
    const char16_t v = *str++;
    if (v == 0) {
      break;
    }
    ret += (char)(v & 0xFF);
  }
  return ret;
}
