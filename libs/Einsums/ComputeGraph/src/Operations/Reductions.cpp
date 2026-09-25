//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/BLAS/ThreadControl.hpp>
#include <Einsums/ComputeGraph/CaptureContext.hpp>
#include <Einsums/ComputeGraph/Detail/ErasedOperations.hpp>
#include <Einsums/ComputeGraph/ExecutorBuilder.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/LinearAlgebra/Base.hpp>
#include <Einsums/PackedGemm/ContractionKey.hpp>
#include <Einsums/Profile.hpp>

#include <cmath>
#include <complex>
#include <limits>
#include <vector>

#include "Record.hpp"

EINSUMS_NAMESPACE_BEGIN(compute_graph::detail)

namespace {
/// Stride-correct fold over every element of a dense tensor or view.
///
/// Walks the logical index space with an odometer (multi-index -> offset via
/// strides), so non-contiguous views (slices, transposes) reduce correctly,
/// not just contiguous storage. O(size * rank).
template <typename T, typename Acc, typename Op>
Acc reduce_elements(Impl<T> const &A, Acc init, Op op) {
    size_t const rank = A.rank();
    size_t const n    = A.size();
    T const     *base = A.data();
    if (base == nullptr || n == 0)
        return init;
    std::vector<size_t> dims(rank), strides(rank);
    for (size_t a = 0; a < rank; ++a) {
        dims[a]    = A.dim(a);
        strides[a] = A.stride(a);
    }
    Acc acc = init;
    // Last axis fastest: the fold's order, and so its rounding, is part of what it computes.
    for_each_index<true>(dims, [&](std::vector<size_t> const &idx) { acc = op(acc, base[offset_of(idx, strides)]); });
    return acc;
}

template <typename T>
RemoveComplexT<T> run_norm(char norm_type, Impl<T> const &A) {
    // A reduction's summation order is its thread count's, so the fence is what makes this a function of the operands alone.
    blas::SerialVendorScope const serial;
    return linear_algebra::detail::norm(norm_type, A);
}

template <typename T>
T run_sum(Impl<T> const &A) {
    return reduce_elements(A, T{0}, [](T acc, T x) { return acc + x; });
}

template <typename T>
T run_max(Impl<T> const &A) {
    // Propagate NaN like numpy.max: a plain ``x > acc`` comparison is false for
    // a NaN x, so NaN would be silently dropped, and an all-NaN reduction would
    // leak the ``lowest()`` seed. Test ``isnan(x)`` so a NaN poisons the
    // accumulator (and ``acc`` stays NaN thereafter, since ``x > NaN`` is false).
    return reduce_elements(A, std::numeric_limits<T>::lowest(), [](T acc, T x) { return (std::isnan(x) || x > acc) ? x : acc; });
}

} // namespace

// ── norm ──────────────────────────────────────────────────────────────────────

template <typename T>
RemoveComplexT<T> eager_norm(char norm_type, Impl<T> const &A) {
    LabeledSection("norm eager");
    return run_norm<T>(norm_type, A);
}

template <typename T>
void capture_norm(CaptureContext &ctx, char norm_type, SlotRef a, TensorId r_id, RemoveComplexT<T> *result) {
    LabeledSection("norm capture");
    OperandAccessor const a_access(a.second, packed_gemm::get_scalar_type<T>());
    auto                  executor = [result, norm_type, a_access]() {
        LabeledSection("norm execute");
        *result = run_norm<T>(norm_type, *a_access.impl<T>());
    };
    ctx.record(OpKind::Norm, "norm", {a.first}, {r_id}, std::move(executor));
}

template <typename T>
void capture_norm_into(CaptureContext &ctx, char norm_type, SlotRef a, SlotRef r) {
    LabeledSection("norm_python capture");
    using R = RemoveComplexT<T>;
    OperandAccessor const a_access(a.second, packed_gemm::get_scalar_type<T>());
    OperandAccessor const r_access(r.second, packed_gemm::get_scalar_type<R>());
    auto                  executor = [norm_type, a_access, r_access]() {
        LabeledSection("norm_python execute");
        r_access.impl<R>()->data()[0] = run_norm<T>(norm_type, *a_access.impl<T>());
    };
    ctx.record(OpKind::Norm, "norm", {a.first}, {r.first}, std::move(executor));
}

// ── sum / max ─────────────────────────────────────────────────────────────────

template <typename T>
T eager_sum(Impl<T> const &A) {
    LabeledSection("sum_python eager");
    return run_sum<T>(A);
}

template <typename T>
void capture_sum(CaptureContext &ctx, SlotRef r, SlotRef a) {
    LabeledSection("sum_python capture");
    // The reduction writes one element, so it fits the same one-in one-out node
    // every other elementwise op records.
    record_reduction<T, T>(ctx, "sum", "sum_python execute", r, a, [](Impl<T> const &src) { return run_sum<T>(src); });
}

template <typename T>
T eager_max(Impl<T> const &A) {
    LabeledSection("max_python eager");
    return run_max<T>(A);
}

template <typename T>
void capture_max(CaptureContext &ctx, SlotRef r, SlotRef a) {
    LabeledSection("max_python capture");
    record_reduction<T, T>(ctx, "max", "max_python execute", r, a, [](Impl<T> const &src) { return run_max<T>(src); });
}

#define EINSUMS_REDUCTION_OPERATIONS(T)                                                                                                    \
    template EINSUMS_EXPORT RemoveComplexT<T> eager_norm<T>(char, Impl<T> const &);                                                        \
    template EINSUMS_EXPORT void              capture_norm<T>(CaptureContext &, char, SlotRef, TensorId, RemoveComplexT<T> *);             \
    template EINSUMS_EXPORT void              capture_norm_into<T>(CaptureContext &, char, SlotRef, SlotRef);                              \
    template EINSUMS_EXPORT T                 eager_sum<T>(Impl<T> const &);                                                               \
    template EINSUMS_EXPORT void              capture_sum<T>(CaptureContext &, SlotRef, SlotRef);

EINSUMS_CG_ELEMENT_TYPES(EINSUMS_REDUCTION_OPERATIONS)
#undef EINSUMS_REDUCTION_OPERATIONS

// max orders its elements, so it is real-only.
#define EINSUMS_REAL_REDUCTION_OPERATIONS(T)                                                                                               \
    template EINSUMS_EXPORT T    eager_max<T>(Impl<T> const &);                                                                            \
    template EINSUMS_EXPORT void capture_max<T>(CaptureContext &, SlotRef, SlotRef);

EINSUMS_CG_REAL_ELEMENT_TYPES(EINSUMS_REAL_REDUCTION_OPERATIONS)
#undef EINSUMS_REAL_REDUCTION_OPERATIONS

EINSUMS_NAMESPACE_END(compute_graph::detail)
