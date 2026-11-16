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
#include <span>
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

    // A is BOTH an input and an output: the decomposition overwrites it. Listing it only as
    // an output would let a reader of the original matrix be ordered after this node, and
    // listing it only as an input would leave the overwrite unordered against a later writer.
    //
    // The job flag is the one piece of state a saved file cannot recover from the operand lists,
    // because it is a template argument rather than a value. See SyevDescriptor and HeevDescriptor.
    std::vector<TensorId> const inputs{a.first};
    std::vector<TensorId> const outputs{a.first, w.first};
    constexpr auto              dtype = packed_gemm::get_scalar_type<T>();
    if constexpr (hermitian) {
        ctx.record_built(OpKind::Heev, "heev", dtype, 2, HeevDescriptor{.compute_eigenvectors = compute_eigenvectors}, inputs, outputs,
                         inputs, outputs);
    } else {
        ctx.record_built(OpKind::Syev, "syev", dtype, 2, SyevDescriptor{.compute_eigenvectors = compute_eigenvectors}, inputs, outputs,
                         inputs, outputs);
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
    // Both operands are overwritten (B with X, A with its LU factors), so each is an input and
    // an output for the reason capture_syev gives for A.
    std::vector<TensorId> const ids{a.first, b.first};
    ctx.record_built(OpKind::Gesv, "gesv", packed_gemm::get_scalar_type<T>(), 2, GesvDescriptor{}, ids, ids, ids, ids);
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
    ctx.record_built(OpKind::Invert, "invert", packed_gemm::get_scalar_type<T>(), 2, InvertDescriptor{},
                     std::span<TensorId const>{&a.first, 1}, std::span<TensorId const>{&a.first, 1}, {a.first}, {a.first});
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
    // The pivots are the node's second output: an operand, so a getrs reading them is ordered
    // after this node through the node lists. A is an input and an output, as for gesv.
    std::size_t const           order = a.second->dims.size() == 2 ? std::min(a.second->dims[0], a.second->dims[1]) : 0;
    TensorId const              p_id  = ctx.get_or_register_pivots(pivots, order);
    std::vector<TensorId> const inputs{a.first};
    std::vector<TensorId> const outputs{a.first, p_id};
    ctx.record_built(OpKind::Getrf, "getrf", packed_gemm::get_scalar_type<T>(), 2, GetrfDescriptor{}, inputs, outputs, inputs, outputs);
}

template <typename T>
int eager_getrs(Impl<T> const &A, LuPivots const &pivots, Impl<T> &B) {
    WAGGLE_ZONE("getrs eager");
    return linear_algebra::detail::getrs(A, *pivots.buffer(), &B);
}

template <typename T>
void capture_getrs(CaptureContext &ctx, SlotRef a, LuPivots const &pivots, SlotRef b) {
    WAGGLE_ZONE("getrs capture");
    // A and the pivots are read, and B is read and overwritten, so B is the trailing input.
    std::size_t const           order = a.second->dims.empty() ? 0 : a.second->dims[0];
    TensorId const              p_id  = ctx.get_or_register_pivots(pivots, order);
    std::vector<TensorId> const operands{a.first, p_id};
    ctx.record_built(OpKind::Getrs, "getrs", packed_gemm::get_scalar_type<T>(), b.second->rank, GetrsDescriptor{}, operands,
                     std::span<TensorId const>{&b.first, 1}, {a.first, p_id, b.first}, {b.first});
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
