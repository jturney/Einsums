//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/CaptureContext.hpp>
#include <Einsums/ComputeGraph/Detail/ErasedOperations.hpp>
#include <Einsums/ComputeGraph/ExecutorBuilder.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/LinearAlgebra/Base.hpp>
#include <Einsums/PackedGemm/ContractionKey.hpp>
#include <Einsums/Profile.hpp>

#include <algorithm>
#include <complex>
#include <cstdint>
#include <tuple>
#include <vector>

#include "Record.hpp"

EINSUMS_NAMESPACE_BEGIN(compute_graph::detail)

// ── syev / heev ───────────────────────────────────────────────────────────────

namespace {
// The typed syev and heev both land here: the rank-erased syev handles a complex
// A with its real W. (The rank-erased heev overload drops the eigenvector flag,
// so it is not the one to call.)
template <typename T>
void run_syev(bool compute_eigenvectors, Impl<T> *A, Impl<RemoveComplexT<T>> *W) {
    if (compute_eigenvectors) {
        linear_algebra::detail::syev<true>(A, W);
    } else {
        linear_algebra::detail::syev<false>(A, W);
    }
}
} // namespace

template <typename T>
void eager_syev(bool compute_eigenvectors, Impl<T> &A, Impl<RemoveComplexT<T>> &W) {
    WAGGLE_ZONE(IsComplexV<T> ? "heev eager" : "syev eager");
    run_syev<T>(compute_eigenvectors, &A, &W);
}

template <typename T>
void capture_syev(CaptureContext &ctx, bool compute_eigenvectors, SlotRef a, SlotRef w) {
    constexpr bool hermitian = IsComplexV<T>;
    WAGGLE_ZONE(hermitian ? "heev capture" : "syev capture");

    OperandAccessor const a_access(a.second, packed_gemm::get_scalar_type<T>());
    OperandAccessor const w_access(w.second, packed_gemm::get_scalar_type<RemoveComplexT<T>>());
    auto                  executor = [compute_eigenvectors, a_access, w_access]() {
        WAGGLE_ZONE(hermitian ? "heev execute" : "syev execute");
        auto *A = a_access.impl<T>();
        WAGGLE_ANNOTATE("n", static_cast<int64_t>(A->dim(0)));
        run_syev<T>(compute_eigenvectors, A, w_access.impl<RemoveComplexT<T>>());
    };

    // A is BOTH an input and an output: the decomposition overwrites it. Listing it only as
    // an output would let a reader of the original matrix be ordered after this node, and
    // listing it only as an input would leave the overwrite unordered against a later writer.
    if constexpr (hermitian) {
        ctx.record(OpKind::Heev, "heev", {a.first}, {a.first, w.first}, std::move(executor));
    } else {
        // The descriptor carries the one piece of state a saved file cannot recover from the
        // operand lists, because it is a template argument rather than a value. See SyevDescriptor.
        ctx.record(OpKind::Syev, "syev", {a.first}, {a.first, w.first}, std::move(executor),
                   SyevDescriptor{.compute_eigenvectors = compute_eigenvectors});
    }
}

// ── gesv ──────────────────────────────────────────────────────────────────────

template <typename T>
int eager_gesv(Impl<T> &A, Impl<T> &B) {
    WAGGLE_ZONE("gesv eager");
    return linear_algebra::detail::gesv(&A, &B);
}

template <typename T>
void capture_gesv(CaptureContext &ctx, SlotRef a, SlotRef b) {
    WAGGLE_ZONE("gesv capture");
    constexpr auto        dtype = packed_gemm::get_scalar_type<T>();
    OperandAccessor const a_access(a.second, dtype);
    OperandAccessor const b_access(b.second, dtype);
    auto                  executor = [a_access, b_access]() {
        WAGGLE_ZONE("gesv execute");
        auto *A = a_access.impl<T>();
        auto *B = b_access.impl<T>();
        WAGGLE_ANNOTATE("n", static_cast<int64_t>(A->dim(0)));
        // A rank-1 right-hand side is one column; dim(1) would be out of range.
        WAGGLE_ANNOTATE("nrhs", static_cast<int64_t>(B->rank() > 1 ? B->dim(1) : 1));
        std::ignore = linear_algebra::detail::gesv(A, B);
    };
    ctx.record(OpKind::Gesv, "gesv", {a.first, b.first}, {a.first, b.first}, std::move(executor));
}

// ── invert ────────────────────────────────────────────────────────────────────

template <typename T>
void eager_invert(Impl<T> &A) {
    WAGGLE_ZONE("invert eager");
    linear_algebra::detail::invert(&A);
}

template <typename T>
void capture_invert(CaptureContext &ctx, SlotRef a) {
    WAGGLE_ZONE("invert capture");
    OperandAccessor const a_access(a.second, packed_gemm::get_scalar_type<T>());
    auto                  executor = [a_access]() {
        WAGGLE_ZONE("invert execute");
        auto *A = a_access.impl<T>();
        WAGGLE_ANNOTATE("n", static_cast<int64_t>(A->dim(0)));
        linear_algebra::detail::invert(A);
    };
    ctx.record(OpKind::Invert, "invert", {a.first}, {a.first}, std::move(executor));
}

// ── getrf / getrs ─────────────────────────────────────────────────────────────

namespace {
template <typename T>
int run_getrf(Impl<T> *A, std::vector<blas::int_t> *pivots) {
    // The typed linear_algebra::getrf grows a resizable pivot buffer to fit.
    auto const pivot_size = std::min(A->dim(0), A->dim(1));
    if (pivots->size() < pivot_size) {
        pivots->resize(pivot_size);
    }
    return linear_algebra::detail::getrf(A, pivots);
}
} // namespace

template <typename T>
int eager_getrf(Impl<T> &A, LuPivots &pivots) {
    WAGGLE_ZONE("getrf eager");
    return run_getrf<T>(&A, pivots.buffer().get());
}

template <typename T>
void capture_getrf(CaptureContext &ctx, SlotRef a, LuPivots const &pivots) {
    WAGGLE_ZONE("getrf capture");
    OperandAccessor const a_access(a.second, packed_gemm::get_scalar_type<T>());
    auto                  executor = [a_access, buffer = pivots.buffer()]() {
        WAGGLE_ZONE("getrf execute");
        auto *A = a_access.impl<T>();
        WAGGLE_ANNOTATE("n", static_cast<int64_t>(A->dim(0)));
        std::ignore = run_getrf<T>(A, buffer.get());
    };
    ctx.record(OpKind::Getrf, "getrf", {a.first}, {a.first}, std::move(executor));
}

template <typename T>
int eager_getrs(Impl<T> const &A, LuPivots const &pivots, Impl<T> &B) {
    WAGGLE_ZONE("getrs eager");
    return linear_algebra::detail::getrs(A, *pivots.buffer(), &B);
}

template <typename T>
void capture_getrs(CaptureContext &ctx, SlotRef a, LuPivots const &pivots, SlotRef b) {
    WAGGLE_ZONE("getrs capture");
    constexpr auto        dtype = packed_gemm::get_scalar_type<T>();
    OperandAccessor const a_access(a.second, dtype);
    OperandAccessor const b_access(b.second, dtype);
    auto                  executor = [a_access, b_access, buffer = pivots.buffer()]() {
        WAGGLE_ZONE("getrs execute");
        auto const *A = a_access.impl<T>();
        WAGGLE_ANNOTATE("n", static_cast<int64_t>(A->dim(0)));
        std::ignore = linear_algebra::detail::getrs(*A, *buffer, b_access.impl<T>());
    };
    ctx.record(OpKind::Getrs, "getrs", {a.first, b.first}, {b.first}, std::move(executor));
}

#define EINSUMS_LAPACK_OPERATIONS(T)                                                                                                       \
    template EINSUMS_EXPORT void eager_syev<T>(bool, Impl<T> &, Impl<RemoveComplexT<T>> &);                                                \
    template EINSUMS_EXPORT void capture_syev<T>(CaptureContext &, bool, SlotRef, SlotRef);                                                \
    template EINSUMS_EXPORT int  eager_gesv<T>(Impl<T> &, Impl<T> &);                                                                      \
    template EINSUMS_EXPORT void capture_gesv<T>(CaptureContext &, SlotRef, SlotRef);                                                      \
    template EINSUMS_EXPORT void eager_invert<T>(Impl<T> &);                                                                               \
    template EINSUMS_EXPORT void capture_invert<T>(CaptureContext &, SlotRef);                                                             \
    template EINSUMS_EXPORT int  eager_getrf<T>(Impl<T> &, LuPivots &);                                                                    \
    template EINSUMS_EXPORT void capture_getrf<T>(CaptureContext &, SlotRef, LuPivots const &);                                            \
    template EINSUMS_EXPORT int  eager_getrs<T>(Impl<T> const &, LuPivots const &, Impl<T> &);                                             \
    template EINSUMS_EXPORT void capture_getrs<T>(CaptureContext &, SlotRef, LuPivots const &, SlotRef);

EINSUMS_CG_ELEMENT_TYPES(EINSUMS_LAPACK_OPERATIONS)
#undef EINSUMS_LAPACK_OPERATIONS

EINSUMS_NAMESPACE_END(compute_graph::detail)
