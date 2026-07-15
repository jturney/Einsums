/*
 * Minimal complex-multiply builtins for the windows-cross leg.
 *
 * clang lowers C99 _Complex multiplication to __mulsc3/__muldc3 from
 * compiler-rt. Real clang-cl installs auto-link clang_rt.builtins; this
 * cross image's lld-link has no Windows builtins library, and conda-forge's
 * compiler-rt packaging is a chain of metapackages that is annoying to
 * unwind. Since only these two symbols are needed (direct_prod.c), compile
 * them here. Algorithm follows C11 Annex G ยง6.2 (the same recovery steps
 * compiler-rt implements): naive product first, then reclassify if it came
 * out NaN+NaN.
 */

#include <math.h>

#define DEFINE_MULC3(name, T, FABS, ISINF, ISNAN, COPYSIGN)                                                                                \
    T _Complex name(T a, T b, T c, T d) {                                                                                                  \
        T ac = a * c, bd = b * d, ad = a * d, bc = b * c;                                                                                  \
        T re = ac - bd;                                                                                                                    \
        T im = ad + bc;                                                                                                                    \
        if (ISNAN(re) && ISNAN(im)) {                                                                                                      \
            int recalc = 0;                                                                                                                \
            if (ISINF(a) || ISINF(b)) {                                                                                                    \
                a      = COPYSIGN(ISINF(a) ? (T)1 : (T)0, a);                                                                              \
                b      = COPYSIGN(ISINF(b) ? (T)1 : (T)0, b);                                                                              \
                if (ISNAN(c))                                                                                                              \
                    c = COPYSIGN((T)0, c);                                                                                                 \
                if (ISNAN(d))                                                                                                              \
                    d = COPYSIGN((T)0, d);                                                                                                 \
                recalc = 1;                                                                                                                \
            }                                                                                                                              \
            if (ISINF(c) || ISINF(d)) {                                                                                                    \
                c      = COPYSIGN(ISINF(c) ? (T)1 : (T)0, c);                                                                              \
                d      = COPYSIGN(ISINF(d) ? (T)1 : (T)0, d);                                                                              \
                if (ISNAN(a))                                                                                                              \
                    a = COPYSIGN((T)0, a);                                                                                                 \
                if (ISNAN(b))                                                                                                              \
                    b = COPYSIGN((T)0, b);                                                                                                 \
                recalc = 1;                                                                                                                \
            }                                                                                                                              \
            if (!recalc && (ISINF(ac) || ISINF(bd) || ISINF(ad) || ISINF(bc))) {                                                           \
                if (ISNAN(a))                                                                                                              \
                    a = COPYSIGN((T)0, a);                                                                                                 \
                if (ISNAN(b))                                                                                                              \
                    b = COPYSIGN((T)0, b);                                                                                                 \
                if (ISNAN(c))                                                                                                              \
                    c = COPYSIGN((T)0, c);                                                                                                 \
                if (ISNAN(d))                                                                                                              \
                    d = COPYSIGN((T)0, d);                                                                                                 \
                recalc = 1;                                                                                                                \
            }                                                                                                                              \
            if (recalc) {                                                                                                                  \
                re = INFINITY * (a * c - b * d);                                                                                           \
                im = INFINITY * (a * d + b * c);                                                                                           \
            }                                                                                                                              \
        }                                                                                                                                  \
        T _Complex z;                                                                                                                      \
        __real__ z = re;                                                                                                                   \
        __imag__ z = im;                                                                                                                   \
        return z;                                                                                                                          \
    }

DEFINE_MULC3(__mulsc3, float, fabsf, isinf, isnan, copysignf)
DEFINE_MULC3(__muldc3, double, fabs, isinf, isnan, copysign)
