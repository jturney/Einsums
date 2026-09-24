//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/Detail/ErasedEinsum.hpp>
#include <Einsums/ComputeGraph/Detail/MixedPrecision.hpp>
#include <Einsums/ComputeGraph/StringDispatch.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>

#include <complex>
#include <stdexcept>
#include <type_traits>

EINSUMS_NAMESPACE_BEGIN(compute_graph::dispatch)

namespace {

template <typename T>
using Impl = einsums::detail::TensorImpl<T>;

/// Views re-seated onto each call's operands. Constructing a view copies an operand's TensorImpl,
/// whose dims and strides are heap vectors, and three of those cost about 126 ns, a tenth of a small
/// contraction's whole eager call. Assigning into a view reuses the vectors it already holds, which
/// is how the replay executor has always done it.
template <typename TC, typename TA, typename TB>
struct Seats {
    RuntimeTensorView<TC> c;
    RuntimeTensorView<TA> a;
    RuntimeTensorView<TB> b;
    bool                  in_use = false;
};

/// Call @p fn with views over @p C, @p A and @p B. The views are this thread's reused ones, unless a
/// call on this thread is already using them, in which case fresh ones are built so the outer call's
/// are not re-seated under it.
template <typename TC, typename TA, typename TB, typename Fn>
void with_views(Impl<TC> &C, Impl<TA> const &A, Impl<TB> const &B, Fn &&fn) {
    thread_local Seats<TC, TA, TB> seats;
    if (seats.in_use) {
        RuntimeTensorView<TC> c(C);
        RuntimeTensorView<TA> a(A);
        RuntimeTensorView<TB> b(B);
        fn(c, a, b);
        return;
    }

    struct Release {
        bool &flag;
        ~Release() { flag = false; }
    };
    seats.in_use = true;
    Release const release{seats.in_use};

    seats.c.impl() = C;
    seats.a.impl() = A;
    seats.b.impl() = B;
    fn(seats.c, seats.a, seats.b);
}

} // namespace

template <typename T>
void erased_string_einsum(ParsedEinsumSpec const &parsed, T c_pf, Impl<T> &C, T ab_pf, Impl<T> const &A, Impl<T> const &B, bool conj_a,
                          bool conj_b) {
    with_views(C, A, B, [&](auto &c, auto &a, auto &b) { string_einsum(parsed, c_pf, &c, ab_pf, a, b, conj_a, conj_b); });
}

template <typename TC, typename TA, typename TB>
void erased_mixed_string_einsum(ParsedEinsumSpec const &parsed, TC c_pf, Impl<TC> &C, detail::PromoteT<TA, TB> ab_pf, Impl<TA> const &A,
                                Impl<TB> const &B, bool conj_a, bool conj_b) {
    if constexpr ((std::is_same_v<TC, TA> && std::is_same_v<TA, TB>) || !detail::storable_v<detail::PromoteT<TA, TB>, TC>) {
        // cg::einsum sends a uniform triple to erased_string_einsum and rejects a complex product
        // into a real output before it gets here, so reaching this is a caller's bug.
        (void)parsed, (void)c_pf, (void)C, (void)ab_pf, (void)A, (void)B, (void)conj_a, (void)conj_b;
        EINSUMS_THROW_EXCEPTION(std::logic_error, "erased_mixed_string_einsum: this operand type triple is not a mixed-precision einsum");
    } else {
        with_views(C, A, B, [&](auto &c, auto &a, auto &b) { mixed_string_einsum(parsed, c_pf, &c, ab_pf, a, b, conj_a, conj_b); });
    }
}

// The four element types, and every triple of them for the mixed-precision entry.
#define EINSUMS_ERASED_EINSUM(T)                                                                                                           \
    template EINSUMS_EXPORT void erased_string_einsum<T>(ParsedEinsumSpec const &, T, Impl<T> &, T, Impl<T> const &, Impl<T> const &,      \
                                                         bool, bool);
EINSUMS_ERASED_EINSUM(float)
EINSUMS_ERASED_EINSUM(double)
EINSUMS_ERASED_EINSUM(std::complex<float>)
EINSUMS_ERASED_EINSUM(std::complex<double>)
#undef EINSUMS_ERASED_EINSUM

#define EINSUMS_ERASED_MIXED(TC, TA, TB)                                                                                                   \
    template EINSUMS_EXPORT void erased_mixed_string_einsum<TC, TA, TB>(                                                                   \
        ParsedEinsumSpec const &, TC, Impl<TC> &, detail::PromoteT<TA, TB>, Impl<TA> const &, Impl<TB> const &, bool, bool);
#define EINSUMS_ERASED_MIXED_B(TC, TA)                                                                                                     \
    EINSUMS_ERASED_MIXED(TC, TA, float)                                                                                                    \
    EINSUMS_ERASED_MIXED(TC, TA, double)                                                                                                   \
    EINSUMS_ERASED_MIXED(TC, TA, std::complex<float>)                                                                                      \
    EINSUMS_ERASED_MIXED(TC, TA, std::complex<double>)
#define EINSUMS_ERASED_MIXED_A(TC)                                                                                                         \
    EINSUMS_ERASED_MIXED_B(TC, float)                                                                                                      \
    EINSUMS_ERASED_MIXED_B(TC, double)                                                                                                     \
    EINSUMS_ERASED_MIXED_B(TC, std::complex<float>)                                                                                        \
    EINSUMS_ERASED_MIXED_B(TC, std::complex<double>)
EINSUMS_ERASED_MIXED_A(float)
EINSUMS_ERASED_MIXED_A(double)
EINSUMS_ERASED_MIXED_A(std::complex<float>)
EINSUMS_ERASED_MIXED_A(std::complex<double>)
#undef EINSUMS_ERASED_MIXED_A
#undef EINSUMS_ERASED_MIXED_B
#undef EINSUMS_ERASED_MIXED

EINSUMS_NAMESPACE_END(compute_graph::dispatch)
