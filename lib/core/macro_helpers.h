#ifndef NGX_WRAPPER_MACRO_HELPERS_H
#define NGX_WRAPPER_MACRO_HELPERS_H

#define _CAT(x, y) x##y
#define CAT(x, y) _CAT(x, y)

// Inspired by https://stackoverflow.com/questions/57400549/creating-a-c-macro-that-allows-for-easy-creation-of-virtual-function-wrapper-f
#define VA_COUNT(...)                                                          \
  VA_COUNT_(__VA_ARGS__, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, )
#define VA_COUNT_(p14, p13, p12, p11, p10, p9, p8, p7, p6, p5, p4, p3, p2, p1, \
                  x, ...)                                                      \
  x

#define FOR_EACH(macro, ...)                                                   \
  CAT(FOR_EACH_, VA_COUNT(__VA_ARGS__))(macro, __VA_ARGS__)
#define FOR_EACH_1(m, p1) m p1
#define FOR_EACH_2(m, p1, p2) m p1, m p2
#define FOR_EACH_3(m, p1, p2, p3) m p1, m p2, m p3
#define FOR_EACH_4(m, p1, p2, p3, p4) m p1, m p2, m p3, m p4
#define FOR_EACH_5(m, p1, p2, p3, p4, p5) m p1, m p2, m p3, m p4, m p5
#define FOR_EACH_6(m, p1, p2, p3, p4, p5, p6) m p1, m p2, m p3, m p4, m p5, m p6
#define FOR_EACH_7(m, p1, p2, p3, p4, p5, p6, p7)                              \
  m p1, m p2, m p3, m p4, m p5, m p6, m p7
#define FOR_EACH_8(m, p1, p2, p3, p4, p5, p6, p7, p8)                          \
  m p1, m p2, m p3, m p4, m p5, m p6, m p7, m p8
#define FOR_EACH_9(m, p1, p2, p3, p4, p5, p6, p7, p8, p9)                      \
  m p1, m p2, m p3, m p4, m p5, m p6, m p7, m p8, m p9
#define FOR_EACH_10(m, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10)                \
  m p1, m p2, m p3, m p4, m p5, m p6, m p7, m p8, m p9, m p10
#define FOR_EACH_11(m, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11)           \
  m p1, m p2, m p3, m p4, m p5, m p6, m p7, m p8, m p9, m p10, m p11
#define FOR_EACH_12(m, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12)      \
  m p1, m p2, m p3, m p4, m p5, m p6, m p7, m p8, m p9, m p10, m p11, m p12
#define FOR_EACH_13(m, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13) \
  m p1, m p2, m p3, m p4, m p5, m p6, m p7, m p8, m p9, m p10, m p11, m p12,   \
      m p13
#define FOR_EACH_14(m, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, \
                    p14)                                                       \
  m p1, m p2, m p3, m p4, m p5, m p6, m p7, m p8, m p9, m p10, m p11, m p12,   \
      m p13, m p14

#define VFUNC_param_decl(type_, name_) type_ name_
#define VFUNC_param_use(type_, name_) name_

#endif
