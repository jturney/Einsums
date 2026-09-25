//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/Detail/ErasedEinsum.hpp>
#include <Einsums/ComputeGraph/Detail/MixedPrecision.hpp>
#include <Einsums/ComputeGraph/Operations.hpp>
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

#define EINSUMS_STRING_PERMUTE(T)                                                                                                          \
    template EINSUMS_EXPORT void string_permute_impl<T>(ParsedPermuteSpec const &, T, Impl<T> *, T, Impl<T> const &);
EINSUMS_STRING_PERMUTE(float)
EINSUMS_STRING_PERMUTE(double)
EINSUMS_STRING_PERMUTE(std::complex<float>)
EINSUMS_STRING_PERMUTE(std::complex<double>)
#undef EINSUMS_STRING_PERMUTE

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

EINSUMS_NAMESPACE_BEGIN(compute_graph::detail)

namespace {

/// Record one einsum node. Every capture comes here, whatever its operand types: the descriptor's
/// snapshot and live blocks, the letter-space binding, and an executor from build_executor, which
/// is also what a pass that rebuilds the node and the loader reach, so all three run one lowering.
template <typename TC, typename TA, typename TB>
void record_einsum(CaptureContext &ctx, ParsedEinsumSpec const &parsed, PrefactorScalar c_pf, PrefactorScalar ab_pf, bool conj_a,
                   bool conj_b, CapturedOperand<TA> const &a, CapturedOperand<TB> const &b, CapturedOperand<TC> const &c) {
    auto desc = detail::build_einsum_descriptor(parsed, c_pf, ab_pf, conj_a, conj_b);
    detail::attach_live_state(desc, parsed.raw);

    // Index-space binding (design part 1.3). A space is a property of the SLOT an index occupies,
    // not of the letter globally, so the letters of this one contraction are resolved against the
    // operands' annotations here and the result is stored per node. Costs nothing for a program that
    // annotates nothing: every operand's `spaces` is empty and the map comes back empty.
    desc.letter_spaces =
        detail::bind_einsum_spaces(*ctx.graph(), a.id, b.id, c.id, parsed.a_indices, parsed.b_indices, parsed.c_indices, "cg::einsum");

    // BLAS-level batching hint, derived by the one function capture and Graph::make_einsum_node
    // share. A mixed-precision contraction runs the generic loop and is never batched.
    if constexpr (std::is_same_v<TA, TC> && std::is_same_v<TB, TC>) {
        desc.gemm_hint = derive_gemm_hint(packed_gemm::get_scalar_type<TC>(), desc.spec, *ctx.graph(), a.id, b.id, c.id);
    }

    auto label = fmt::format("einsum: C[{}] = A[{}] * B[{}]", fmt::join(parsed.c_indices, ","), fmt::join(parsed.a_indices, ","),
                             fmt::join(parsed.b_indices, ","));

    // The operand lists in the order the builder reads them: A, B from the inputs and C from the
    // outputs. Capture records the two inputs only; the RMW repeat of an accumulating destination is
    // Graph::make_einsum_node's convention, and the builder ignores that trailing position either way.
    std::vector<TensorId> const node_inputs{a.id, b.id};
    std::vector<TensorId> const node_outputs{c.id};

    // The node's dtype is C's, as for every einsum node. The descriptor is handed over as the OpData
    // the node will carry, so the live blocks the builder reads are the very ones the node records.
    OpData op_data{std::move(desc)};
    auto   executor = build_executor(OpKind::Einsum, packed_gemm::get_scalar_type<TC>(), c.impl->rank(), op_data, *ctx.graph(),
                                     std::span<TensorId const>{node_inputs}, std::span<TensorId const>{node_outputs});
    ctx.record(OpKind::Einsum, std::move(label), node_inputs, node_outputs, std::move(executor), std::move(op_data));
}

} // namespace

template <typename T>
void capture_string_einsum(CaptureContext &ctx, ParsedEinsumSpec const &parsed, T c_pf, T ab_pf, bool conj_a, bool conj_b,
                           CapturedOperand<T> const &a, CapturedOperand<T> const &b, CapturedOperand<T> const &c) {
    record_einsum<T, T, T>(ctx, parsed, c_pf, ab_pf, conj_a, conj_b, a, b, c);
}

template <typename TC, typename TA, typename TB>
void capture_mixed_string_einsum(CaptureContext &ctx, ParsedEinsumSpec const &parsed, TC c_pf, PromoteT<TA, TB> ab_pf, bool conj_a,
                                 bool conj_b, CapturedOperand<TA> const &a, CapturedOperand<TB> const &b, CapturedOperand<TC> const &c) {
    if constexpr ((std::is_same_v<TC, TA> && std::is_same_v<TA, TB>) || !storable_v<PromoteT<TA, TB>, TC>) {
        // As erased_mixed_string_einsum: cg::einsum never sends these here.
        (void)ctx, (void)parsed, (void)c_pf, (void)ab_pf, (void)conj_a, (void)conj_b, (void)a, (void)b, (void)c;
        EINSUMS_THROW_EXCEPTION(std::logic_error, "capture_mixed_string_einsum: this operand type triple is not a mixed-precision einsum");
    } else {
        record_einsum<TC, TA, TB>(ctx, parsed, c_pf, ab_pf, conj_a, conj_b, a, b, c);
    }
}

#define EINSUMS_CAPTURE_EINSUM(T)                                                                                                          \
    template EINSUMS_EXPORT void capture_string_einsum<T>(CaptureContext &, ParsedEinsumSpec const &, T, T, bool, bool,                    \
                                                          CapturedOperand<T> const &, CapturedOperand<T> const &,                          \
                                                          CapturedOperand<T> const &);
EINSUMS_CAPTURE_EINSUM(float)
EINSUMS_CAPTURE_EINSUM(double)
EINSUMS_CAPTURE_EINSUM(std::complex<float>)
EINSUMS_CAPTURE_EINSUM(std::complex<double>)
#undef EINSUMS_CAPTURE_EINSUM

#define EINSUMS_CAPTURE_MIXED(TC, TA, TB)                                                                                                  \
    template EINSUMS_EXPORT void capture_mixed_string_einsum<TC, TA, TB>(CaptureContext &, ParsedEinsumSpec const &, TC, PromoteT<TA, TB>, \
                                                                         bool, bool, CapturedOperand<TA> const &,                          \
                                                                         CapturedOperand<TB> const &, CapturedOperand<TC> const &);
#define EINSUMS_CAPTURE_MIXED_B(TC, TA)                                                                                                    \
    EINSUMS_CAPTURE_MIXED(TC, TA, float)                                                                                                   \
    EINSUMS_CAPTURE_MIXED(TC, TA, double)                                                                                                  \
    EINSUMS_CAPTURE_MIXED(TC, TA, std::complex<float>)                                                                                     \
    EINSUMS_CAPTURE_MIXED(TC, TA, std::complex<double>)
#define EINSUMS_CAPTURE_MIXED_A(TC)                                                                                                        \
    EINSUMS_CAPTURE_MIXED_B(TC, float)                                                                                                     \
    EINSUMS_CAPTURE_MIXED_B(TC, double)                                                                                                    \
    EINSUMS_CAPTURE_MIXED_B(TC, std::complex<float>)                                                                                       \
    EINSUMS_CAPTURE_MIXED_B(TC, std::complex<double>)
EINSUMS_CAPTURE_MIXED_A(float)
EINSUMS_CAPTURE_MIXED_A(double)
EINSUMS_CAPTURE_MIXED_A(std::complex<float>)
EINSUMS_CAPTURE_MIXED_A(std::complex<double>)
#undef EINSUMS_CAPTURE_MIXED_A
#undef EINSUMS_CAPTURE_MIXED_B
#undef EINSUMS_CAPTURE_MIXED

EINSUMS_NAMESPACE_END(compute_graph::detail)
