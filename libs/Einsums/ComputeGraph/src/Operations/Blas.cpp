//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/BLAS/ThreadControl.hpp>
#include <Einsums/ComputeGraph/CaptureContext.hpp>
#include <Einsums/ComputeGraph/Detail/ErasedOperations.hpp>
#include <Einsums/ComputeGraph/ExecutorBuilder.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/Prefactor.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/LinearAlgebra/Base.hpp>
#include <Einsums/LinearAlgebra/SymmetryDispatch.hpp>
#include <Einsums/PackedGemm/ContractionKey.hpp>
#include <Einsums/Profile.hpp>

#include <fmt/format.h>

#include <complex>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

#include "Record.hpp"

EINSUMS_NAMESPACE_BEGIN(compute_graph::detail)

// ── gemm ──────────────────────────────────────────────────────────────────────

template <typename T>
void eager_gemm(char ta, char tb, T alpha, Impl<T> const &A, SymmetryDescriptor const *desc_a, Impl<T> const &B,
                SymmetryDescriptor const *desc_b, T beta, Impl<T> &C) {
    LabeledSection("gemm eager");
    // Symmetry-aware fast path, as the typed linear_algebra::gemm takes it: a
    // declared symmetric or Hermitian operand goes to symm / hemm.
    if (desc_a != nullptr || desc_b != nullptr) {
        if (linear_algebra::detail::try_symmetric_gemm<T>(ta, tb, alpha, A, desc_a, B, desc_b, beta, &C)) {
            return;
        }
    }
    linear_algebra::detail::gemm(ta, tb, alpha, A, B, beta, &C);
}

template <typename T>
void capture_gemm(CaptureContext &ctx, char ta, char tb, bool from_flags, T alpha, T beta, bool reads_c, TensorId a_id, TensorId b_id,
                  TensorId c_id) {
    LabeledSection("gemm capture");

    auto label = from_flags ? fmt::format("gemm<{},{}>", ta == 't' ? "T" : "N", tb == 't' ? "T" : "N") : fmt::format("gemm({},{})", ta, tb);

    // beta != 0 → the gemm reads C as well as writing it (``C = α·A·B + β·C``);
    // list C as an input so loop-invariance and scheduling see the read.
    std::vector<TensorId> inputs = {a_id, b_id};
    if (reads_c) {
        inputs.push_back(c_id);
    }

    // The chars go in as given, conjugate transpose included: the runtime-flag
    // overload exists to reach BLAS 'c', and the descriptor records chars
    // rather than bools so it can.
    ctx.record_built(OpKind::Gemm, std::move(label), packed_gemm::get_scalar_type<T>(), 2,
                     GemmDescriptor{.alpha = PrefactorScalar{alpha}, .beta = PrefactorScalar{beta}, .trans_a = ta, .trans_b = tb}, inputs,
                     std::span<TensorId const>{&c_id, 1}, inputs, {c_id});
}

// ── dot / trace ───────────────────────────────────────────────────────────────

template <typename T>
T eager_dot(Impl<T> const &A, Impl<T> const &B, bool conjugated) {
    LabeledSection(conjugated ? "dotc eager" : "dot eager");
    // A reduction's summation order is its thread count's, so the fence is what makes this a function of the operands alone.
    blas::SerialVendorScope const serial;
    return conjugated ? linear_algebra::detail::true_dot(A, B) : linear_algebra::detail::dot(A, B);
}

template <typename T>
void capture_dot(CaptureContext &ctx, bool conjugated, TensorId a_id, TensorId b_id, TensorId r_id, std::size_t rank) {
    LabeledSection(conjugated ? "dotc capture" : "dot capture");
    // The conjugation is the descriptor's one field, and it is the whole
    // difference between dot and dotc: the builder picks true_dot over dot.
    std::vector<TensorId> const inputs{a_id, b_id};
    ctx.record_built(OpKind::Dot, conjugated ? "dotc" : "dot", packed_gemm::get_scalar_type<T>(), rank,
                     DotDescriptor{.conjugated = conjugated}, inputs, std::span<TensorId const>{&r_id, 1}, inputs, {r_id});
}

template <typename T>
T eager_trace(Impl<T> const &A) {
    LabeledSection("trace eager");
    // The same walk as the replay executor's (build_trace): sequential and in
    // index order, so eager and replay agree bit for bit.
    if (A.rank() != 2) {
        EINSUMS_THROW_EXCEPTION(RankError, "cg::trace: input must be rank-2; got rank {}.", A.rank());
    }
    if (A.dim(0) != A.dim(1)) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "cg::trace: input must be square");
    }
    T sum{};
    for (size_t i = 0; i < A.dim(0); ++i) {
        sum += A.subscript(i, i);
    }
    return sum;
}

template <typename T>
void capture_trace(CaptureContext &ctx, TensorId a_id, TensorId r_id, std::size_t rank) {
    LabeledSection("trace capture");
    ctx.record_built(OpKind::Trace, "trace", packed_gemm::get_scalar_type<T>(), rank, TraceDescriptor{},
                     std::span<TensorId const>{&a_id, 1}, std::span<TensorId const>{&r_id, 1}, {a_id}, {r_id});
}

// ── gemv / ger / gerc ─────────────────────────────────────────────────────────

template <typename T>
void eager_gemv(char ta, T alpha, Impl<T> const &A, Impl<T> const &x, T beta, Impl<T> &y) {
    LabeledSection("gemv eager");
    linear_algebra::detail::gemv(ta, alpha, A, x, beta, &y);
}

template <typename T>
void capture_gemv(CaptureContext &ctx, char ta, bool from_flags, T alpha, T beta, bool reads_y, SlotRef a, SlotRef x, SlotRef y) {
    LabeledSection("gemv capture");
    constexpr auto dtype = packed_gemm::get_scalar_type<T>();

    auto label = from_flags ? fmt::format("gemv<{}>", ta == 't' ? "T" : "N") : fmt::format("gemv({})", ta);

    // The operands are re-read through their slots on every run, so the node
    // follows rebind() and the memory planner moving a tensor's storage.
    OperandAccessor const a_access(a.second, dtype);
    OperandAccessor const x_access(x.second, dtype);
    OperandAccessor const y_access(y.second, dtype);
    auto                  executor = [alpha, beta, ta, from_flags, a_access, x_access, y_access]() {
        LabeledSection("gemv execute");
        auto const *A = a_access.impl<T>();
        if (from_flags) {
            ProfileAnnotate("trans", ta == 't' ? "T" : "N");
            ProfileAnnotate("m", static_cast<int64_t>(A->dim(0)));
            ProfileAnnotate("n", static_cast<int64_t>(A->dim(1)));
        }
        linear_algebra::detail::gemv(ta, alpha, *A, *x_access.impl<T>(), beta, y_access.impl<T>());
    };

    // beta != 0 → gemv reads y as well as writing it; list it as an input so
    // loop-invariance and scheduling see the read (see capture_gemm).
    std::vector<TensorId> inputs = {a.first, x.first};
    if (reads_y) {
        inputs.push_back(y.first);
    }
    ctx.record(OpKind::Gemv, std::move(label), std::move(inputs), {y.first}, std::move(executor));
}

template <typename T>
void eager_ger(bool conjugated, T alpha, Impl<T> const &x, Impl<T> const &y, Impl<T> &A) {
    LabeledSection(conjugated ? "gerc eager" : "ger eager");
    if (conjugated) {
        linear_algebra::detail::gerc(alpha, x, y, &A);
    } else {
        linear_algebra::detail::ger(alpha, x, y, &A);
    }
}

template <typename T>
void capture_ger(CaptureContext &ctx, bool conjugated, T alpha, SlotRef x, SlotRef y, SlotRef a) {
    LabeledSection(conjugated ? "gerc capture" : "ger capture");
    constexpr auto dtype = packed_gemm::get_scalar_type<T>();

    OperandAccessor const x_access(x.second, dtype);
    OperandAccessor const y_access(y.second, dtype);
    OperandAccessor const a_access(a.second, dtype);
    auto                  executor = [alpha, conjugated, x_access, y_access, a_access]() {
        LabeledSection(conjugated ? "gerc execute" : "ger execute");
        auto const *X = x_access.impl<T>();
        auto const *Y = y_access.impl<T>();
        if (conjugated) {
            linear_algebra::detail::gerc(alpha, *X, *Y, a_access.impl<T>());
        } else {
            ProfileAnnotate("m", static_cast<int64_t>(X->dim(0)));
            ProfileAnnotate("n", static_cast<int64_t>(Y->dim(0)));
            linear_algebra::detail::ger(alpha, *X, *Y, a_access.impl<T>());
        }
    };

    // ger always accumulates (``A += α·X·Y^T``), so it reads A as well as
    // writing it: list A as an input so loop-invariance and scheduling see the
    // read-modify-write.
    ctx.record(OpKind::Ger, conjugated ? "gerc" : "ger", {x.first, y.first, a.first}, {a.first}, std::move(executor));
}

#define EINSUMS_BLAS_OPERATIONS(T)                                                                                                         \
    template EINSUMS_EXPORT void eager_gemm<T>(char, char, T, Impl<T> const &, SymmetryDescriptor const *, Impl<T> const &,                \
                                               SymmetryDescriptor const *, T, Impl<T> &);                                                  \
    template EINSUMS_EXPORT void capture_gemm<T>(CaptureContext &, char, char, bool, T, T, bool, TensorId, TensorId, TensorId);            \
    template EINSUMS_EXPORT T    eager_dot<T>(Impl<T> const &, Impl<T> const &, bool);                                                     \
    template EINSUMS_EXPORT void capture_dot<T>(CaptureContext &, bool, TensorId, TensorId, TensorId, std::size_t);                        \
    template EINSUMS_EXPORT T    eager_trace<T>(Impl<T> const &);                                                                          \
    template EINSUMS_EXPORT void capture_trace<T>(CaptureContext &, TensorId, TensorId, std::size_t);                                      \
    template EINSUMS_EXPORT void eager_gemv<T>(char, T, Impl<T> const &, Impl<T> const &, T, Impl<T> &);                                   \
    template EINSUMS_EXPORT void capture_gemv<T>(CaptureContext &, char, bool, T, T, bool, SlotRef, SlotRef, SlotRef);                     \
    template EINSUMS_EXPORT void eager_ger<T>(bool, T, Impl<T> const &, Impl<T> const &, Impl<T> &);                                       \
    template EINSUMS_EXPORT void capture_ger<T>(CaptureContext &, bool, T, SlotRef, SlotRef, SlotRef);

EINSUMS_CG_ELEMENT_TYPES(EINSUMS_BLAS_OPERATIONS)
#undef EINSUMS_BLAS_OPERATIONS

EINSUMS_NAMESPACE_END(compute_graph::detail)
