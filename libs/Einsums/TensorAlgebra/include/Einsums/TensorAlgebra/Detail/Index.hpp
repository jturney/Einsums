//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Print.hpp>

#include <cstdarg>
#include <ostream>
#include <vector>

// Including complex header defines "I" to be used with complex numbers. If we allow that then
// we cannot use "I" as an indexing tab to einsum.
#if defined(I)
#    undef I
#endif

namespace einsums::index {
/// Base struct for index tags. It might not be technically needed but it will allow
/// compile-time checks to be performed.
struct LabelBase {};
} // namespace einsums::index

/*! \def MAKE_INDEX(x)
    Macro that defines new index tags that can be used with einsums. Also includes code
    for easy printing using fmtlib.
*/
#define MAKE_INDEX(x)                                                                                                                      \
    namespace einsums::index {                                                                                                             \
    struct x : public LabelBase {                                                                                                          \
        static constexpr const char *letter = #x;                                                                                          \
        constexpr x()                       = default;                                                                                     \
        size_t operator()(std::va_list args) const {                                                                                       \
            return va_arg(args, size_t);                                                                                                   \
        }                                                                                                                                  \
                                                                                                                                           \
        size_t operator()(size_t index) const {                                                                                            \
            return index;                                                                                                                  \
        }                                                                                                                                  \
                                                                                                                                           \
        template <typename T, typename Alloc>                                                                                              \
        size_t operator()(std::vector<T, Alloc> *args) const {                                                                             \
            size_t out = args->at(0);                                                                                                      \
            args->erase(args->begin());                                                                                                    \
                                                                                                                                           \
            return out;                                                                                                                    \
        }                                                                                                                                  \
    };                                                                                                                                     \
    static constexpr struct x x;                                                                                                           \
                                                                                                                                           \
    inline auto operator<<(std::ostream &os, const struct x &) -> std::ostream & {                                                         \
        os << x::letter;                                                                                                                   \
        return os;                                                                                                                         \
    }                                                                                                                                      \
    }                                                                                                                                      \
    template <>                                                                                                                            \
    struct fmt::formatter<struct ::einsums::index::x> : fmt::formatter<const char *> {                                                     \
        template <typename FormatContext>                                                                                                  \
        auto format(const struct ::einsums::index::x &, FormatContext &ctx) const {                                                        \
            return formatter<const char *>::format(::einsums::index::x::letter, ctx);                                                      \
        }                                                                                                                                  \
    };

#if !defined(DOXYGEN)
MAKE_INDEX(A); // NOLINT
MAKE_INDEX(a); // NOLINT
MAKE_INDEX(B); // NOLINT
MAKE_INDEX(b); // NOLINT
MAKE_INDEX(C); // NOLINT
MAKE_INDEX(c); // NOLINT
MAKE_INDEX(D); // NOLINT
MAKE_INDEX(d); // NOLINT
MAKE_INDEX(E); // NOLINT
MAKE_INDEX(e); // NOLINT
MAKE_INDEX(F); // NOLINT
MAKE_INDEX(f); // NOLINT
MAKE_INDEX(G); // NOLINT
MAKE_INDEX(g); // NOLINT
MAKE_INDEX(H); // NOLINT
MAKE_INDEX(h); // NOLINT
MAKE_INDEX(I); // NOLINT
MAKE_INDEX(i); // NOLINT
MAKE_INDEX(J); // NOLINT
MAKE_INDEX(j); // NOLINT
MAKE_INDEX(K); // NOLINT
MAKE_INDEX(k); // NOLINT
MAKE_INDEX(L); // NOLINT
MAKE_INDEX(l); // NOLINT
MAKE_INDEX(M); // NOLINT
MAKE_INDEX(m); // NOLINT
MAKE_INDEX(N); // NOLINT
MAKE_INDEX(n); // NOLINT
MAKE_INDEX(O); // NOLINT
MAKE_INDEX(o); // NOLINT
MAKE_INDEX(P); // NOLINT
MAKE_INDEX(p); // NOLINT
MAKE_INDEX(Q); // NOLINT
MAKE_INDEX(q); // NOLINT
MAKE_INDEX(R); // NOLINT
MAKE_INDEX(r); // NOLINT
MAKE_INDEX(S); // NOLINT
MAKE_INDEX(s); // NOLINT
MAKE_INDEX(T); // NOLINT
MAKE_INDEX(t); // NOLINT
MAKE_INDEX(U); // NOLINT
MAKE_INDEX(u); // NOLINT
MAKE_INDEX(V); // NOLINT
MAKE_INDEX(v); // NOLINT
MAKE_INDEX(w); // NOLINT
MAKE_INDEX(W); // NOLINT
MAKE_INDEX(x); // NOLINT
MAKE_INDEX(X); // NOLINT
MAKE_INDEX(y); // NOLINT
MAKE_INDEX(Y); // NOLINT
MAKE_INDEX(z); // NOLINT

// Z is a special index used internally. Unless you know what you're doing, DO NOT USE.
MAKE_INDEX(Z); // NOLINT

// Greek indices are useful too!
MAKE_INDEX(alpha)
MAKE_INDEX(beta)
MAKE_INDEX(gamma)
MAKE_INDEX(delta)
MAKE_INDEX(epsilon)
MAKE_INDEX(zeta)
MAKE_INDEX(eta)
MAKE_INDEX(theta)
MAKE_INDEX(iota)
MAKE_INDEX(kappa)
MAKE_INDEX(lambda)
MAKE_INDEX(mu)
MAKE_INDEX(nu)
MAKE_INDEX(xi)
MAKE_INDEX(omicron)
MAKE_INDEX(pi)
MAKE_INDEX(rho)
MAKE_INDEX(sigma)
MAKE_INDEX(tau)
MAKE_INDEX(upsilon)
MAKE_INDEX(phi)
MAKE_INDEX(chi)
MAKE_INDEX(psi)
MAKE_INDEX(omega)

MAKE_INDEX(Alpha)
MAKE_INDEX(Beta)
MAKE_INDEX(Gamma)
MAKE_INDEX(Delta)
MAKE_INDEX(Epsilon)
MAKE_INDEX(Zeta)
MAKE_INDEX(Eta)
MAKE_INDEX(Theta)
MAKE_INDEX(Iota)
MAKE_INDEX(Kappa)
MAKE_INDEX(Lambda)
MAKE_INDEX(Mu)
MAKE_INDEX(Nu)
MAKE_INDEX(Xi)
MAKE_INDEX(Omicron)
MAKE_INDEX(Pi)
MAKE_INDEX(Rho)
MAKE_INDEX(Sigma)
MAKE_INDEX(Tau)
MAKE_INDEX(Upsilon)
MAKE_INDEX(Phi)
MAKE_INDEX(Chi)
MAKE_INDEX(Psi)
MAKE_INDEX(Omega)
#endif

#undef MAKE_INDEX

namespace einsums {

namespace index {

#if !defined(DOXYGEN)
constexpr auto list = std::make_tuple(i, j, k, l, m, n, a, b, c, d, e, f, p, q, r, s);
#endif

} // namespace index

/**
 * @struct Indices
 *
 * @brief Identifier for providing index labels to the the einsum function.
 *
 * If a variable name is passed to this class, it will raise a compiler error. If this is the case,
 * please double check that you don't have any name clashes. Try to prefix the indices with @c einsums::index
 * to see if that fixes your issues.
 *
 * @tparam Args The indices to pass.
 */
template <typename... Args>
    requires(std::is_base_of_v<index::LabelBase, Args> && ... && true)
struct Indices : std::tuple<Args...> {
    /**
     * Construct a new Indices object using the given indices.
     */
    Indices(Args... args) : std::tuple<Args...>(args...){};
};

} // namespace einsums