//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/BLAS.hpp>
#include <Einsums/ComputeGraph/CaptureContext.hpp>
#include <Einsums/ComputeGraph/EinsumSpec.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/StringDispatch.hpp>
#include <Einsums/ComputeGraph/TensorRank.hpp>
#include <Einsums/Concepts/TensorConcepts.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/LinearAlgebra.hpp>
#include <Einsums/Python/Annotations.hpp>
#include <Einsums/TaskPool/TaskPool.hpp>
#include <Einsums/TensorAlgebra/Backends/ElementTransform.hpp>
#include <Einsums/TensorAlgebra/Permute.hpp>
#include <Einsums/TensorAlgebra/TensorAlgebra.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace einsums::compute_graph {

// ─────────────────────────────────────────────────────────────────────────────
// einsum — graph-aware, runtime-string contraction spec
// ─────────────────────────────────────────────────────────────────────────────
//
// The tuple-indexed overloads (`einsum(Indices{i,j}, ...)`) were removed
// in favour of the runtime-string form. Index tuples are compile-time
// types and can't be rewritten by optimization passes; strings are data
// that live on the graph and can be mutated in place. See the string
// overloads further down in this file for the public entry points.

// ─────────────────────────────────────────────────────────────────────────────
// scale
// ─────────────────────────────────────────────────────────────────────────────

/// Graph-aware scale: A *= factor.
template <TensorConcept AType>
// clang-format off
EINSUMS_PYBIND_EXPOSE
EINSUMS_PYBIND_MODULE("linalg")
EINSUMS_PYBIND_INSTANTIATE_AS("scale", einsums::GeneralRuntimeTensor<float, std::allocator<float>>)
EINSUMS_PYBIND_INSTANTIATE_AS("scale", einsums::GeneralRuntimeTensor<double, std::allocator<double>>)
EINSUMS_PYBIND_INSTANTIATE_AS("scale", einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>)
EINSUMS_PYBIND_INSTANTIATE_AS("scale", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
    // clang-format on
    void scale(typename AType::ValueType factor, AType *A) {
    auto &ctx = CaptureContext::current();
    if (!ctx.is_capturing()) {
        linear_algebra::scale(factor, A);
        return;
    }

    auto [a_id, a_slot] = ctx.get_slot(*A);

    ScaleDescriptor desc;
    if constexpr (IsComplexV<typename AType::ValueType>) {
        desc.factor = static_cast<double>(factor.real());
    } else {
        desc.factor = static_cast<double>(factor);
    }

    auto factor_str = [&]() -> std::string {
        if constexpr (IsComplexV<typename AType::ValueType>) {
            return fmt::format("({},{})", factor.real(), factor.imag());
        } else {
            return fmt::format("{}", factor);
        }
    }();
    auto label    = fmt::format("scale({}, {})", factor_str, A->name());
    auto executor = [factor, a_slot]() {
        auto *a_ptr = static_cast<AType *>(a_slot->ptr);
        linear_algebra::scale(factor, a_ptr);
    };

    ctx.record(OpKind::Scale, std::move(label), {a_id}, {a_id}, std::move(executor), std::move(desc));
}

// ─────────────────────────────────────────────────────────────────────────────
// permute
// ─────────────────────────────────────────────────────────────────────────────

/// String-based graph-aware permute with prefactors.
///
/// @code
/// cg::permute("ji <- ij", 0.0, &C, 1.0, A);       // C = A^T
/// cg::permute("kji <- ijk", 0.0, &D, 1.0, T);      // rank-3 transpose
/// cg::permute("mu,nu <- nu,mu", 0.0, &C, 1.0, A);  // multi-char indices
/// @endcode
template <BasicTensorConcept AType, BasicTensorConcept CType>
    requires std::is_same_v<typename AType::ValueType, typename CType::ValueType>
void permute(PermuteFormatString spec, typename CType::ValueType beta, CType *C, typename AType::ValueType alpha, AType const &A) {
    using T = typename AType::ValueType;

    auto parse_result = parse_permute_spec(static_cast<std::string_view>(spec));
    if (!parse_result) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "{}", parse_result.error().message);
    }
    auto &parsed = parse_result.value();

    auto &ctx = CaptureContext::current();
    if (!ctx.is_capturing()) {
        dispatch::string_permute(parsed, beta, C, alpha, A);
        return;
    }

    // Capture mode with slots
    auto [a_id, a_slot] = ctx.get_slot(A);
    auto [c_id, c_slot] = ctx.get_slot(*C);

    PermuteDescriptor desc;
    if constexpr (IsComplexV<T>) {
        desc.alpha = static_cast<double>(alpha.real());
        desc.beta  = static_cast<double>(beta.real());
    } else {
        desc.alpha = static_cast<double>(alpha);
        desc.beta  = static_cast<double>(beta);
    }
    desc.c_indices = parsed.c_indices;
    desc.a_indices = parsed.a_indices;

    auto label = fmt::format("permute: C[{}] = A[{}]", fmt::join(parsed.c_indices, ","), fmt::join(parsed.a_indices, ","));

    auto executor = [parsed, beta, alpha, a_slot, c_slot]() {
        dispatch::string_permute<AType, CType>(parsed, static_cast<T>(beta), static_cast<CType *>(c_slot->ptr), static_cast<T>(alpha),
                                               *static_cast<AType const *>(a_slot->ptr));
    };

    ctx.record(OpKind::Permute, std::move(label), {a_id}, {c_id}, std::move(executor), std::move(desc));
}

/// String-based permute with default prefactors (beta=0, alpha=1): C = permute(A).
template <BasicTensorConcept AType, BasicTensorConcept CType>
    requires std::is_same_v<typename AType::ValueType, typename CType::ValueType>
void permute(PermuteFormatString spec, CType *C, AType const &A) {
    using T = typename AType::ValueType;
    permute(spec, T{0}, C, T{1}, A);
}

/// Graph-aware permute with explicit prefactors.
///
/// ``spec`` is a permutation pattern such as ``"ji <- ij"`` (transpose)
/// or ``"kji <- ijk"`` (rank-3 reorder). Computes ``C = c_pf * C + a_pf
/// * permute(A)`` according to ``spec``. ``c_pf`` defaults to 0 and
/// ``a_pf`` to 1, i.e. ``C = permute(A)``.
template <BasicTensorConcept AType, BasicTensorConcept CType>
    requires std::is_same_v<typename AType::ValueType, typename CType::ValueType>
// clang-format off
EINSUMS_PYBIND_EXPOSE
EINSUMS_PYBIND_INSTANTIATE_AS("permute", einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::GeneralRuntimeTensor<float, std::allocator<float>>)
EINSUMS_PYBIND_INSTANTIATE_AS("permute", einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::GeneralRuntimeTensor<double, std::allocator<double>>)
EINSUMS_PYBIND_INSTANTIATE_AS("permute", einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>)
EINSUMS_PYBIND_INSTANTIATE_AS("permute", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
    // clang-format on
    void string_permute(std::string const &spec, CType *C, AType const &A, typename CType::ValueType c_pf = typename CType::ValueType{0},
                        typename AType::ValueType a_pf = typename AType::ValueType{1}) {
    permute(PermuteFormatString{spec}, c_pf, C, a_pf, A);
}

// ─────────────────────────────────────────────────────────────────────────────
// transpose
// ─────────────────────────────────────────────────────────────────────────────

/// Graph-aware transpose.
template <TensorConcept CType, TensorConcept AType>
void transpose(CType *C, AType const &A) {
    auto &ctx = CaptureContext::current();
    if (!ctx.is_capturing()) {
        tensor_algebra::transpose(C, A);
        return;
    }

    auto [a_id, a_slot] = ctx.get_slot(A);
    auto [c_id, c_slot] = ctx.get_slot(*C);

    auto executor = [c_slot, a_slot]() {
        tensor_algebra::transpose(static_cast<CType *>(c_slot->ptr), *static_cast<AType const *>(a_slot->ptr));
    };

    ctx.record(OpKind::Transpose, "transpose", {a_id}, {c_id}, std::move(executor));
}

// ─────────────────────────────────────────────────────────────────────────────
// element_transform
// ─────────────────────────────────────────────────────────────────────────────

/// Graph-aware element_transform: apply unary operator element-wise.
template <CoreTensorConcept CType, typename UnaryOperator>
    requires requires {
        requires BasicTensorConcept<CType>;
        requires RankTensorConcept<CType>;
    }
void element_transform(CType *C, UnaryOperator unary_op) {
    auto &ctx = CaptureContext::current();
    if (!ctx.is_capturing()) {
        tensor_algebra::element_transform(C, unary_op);
        return;
    }

    auto [c_id, c_slot] = ctx.get_slot(*C);

    auto executor = [c_slot, unary_op]() { tensor_algebra::element_transform(static_cast<CType *>(c_slot->ptr), unary_op); };

    ctx.record(OpKind::ElementTransform, "element_transform", {c_id}, {c_id}, std::move(executor));
}

/// Python-friendly element_transform wrapper.
///
/// The generic ``element_transform`` template requires ``RankTensorConcept``
/// (compile-time rank), which ``GeneralRuntimeTensor`` doesn't satisfy, so it
/// can't be reused here. This overload walks the contiguous underlying storage
/// directly and accepts ``std::function<T(T)>`` so pybind11's caster can wrap a
/// Python callable. A serial loop (rather than the OMP-parallel path used by
/// ``tensor_algebra::element_transform``) keeps the per-call GIL acquire from
/// causing thread contention — fine for the small unary maps typical of
/// SCF/MP2 (eigenvalues, denominators).
template <CoreBasicTensorConcept TensorType>
// clang-format off
EINSUMS_PYBIND_EXPOSE
EINSUMS_PYBIND_MODULE("linalg")
EINSUMS_PYBIND_INSTANTIATE_AS("element_transform", einsums::GeneralRuntimeTensor<float, std::allocator<float>>)
EINSUMS_PYBIND_INSTANTIATE_AS("element_transform", einsums::GeneralRuntimeTensor<double, std::allocator<double>>)
EINSUMS_PYBIND_INSTANTIATE_AS("element_transform", einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>)
EINSUMS_PYBIND_INSTANTIATE_AS("element_transform", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
    // clang-format on
    void element_transform_python(TensorType *C, std::function<typename TensorType::ValueType(typename TensorType::ValueType)> unary_op) {
    using T = typename TensorType::ValueType;

    auto apply = [unary_op](TensorType *target) {
        T           *data = target->data();
        size_t const n    = target->size();
        for (size_t i = 0; i < n; ++i) {
            data[i] = unary_op(data[i]);
        }
    };

    auto &ctx = CaptureContext::current();
    if (!ctx.is_capturing()) {
        apply(C);
        return;
    }

    auto [c_id, c_slot] = ctx.get_slot(*C);
    auto executor       = [c_slot, apply]() { apply(static_cast<TensorType *>(c_slot->ptr)); };
    ctx.record(OpKind::ElementTransform, "element_transform", {c_id}, {c_id}, std::move(executor));
}

// ─────────────────────────────────────────────────────────────────────────────
// axpy: Y += alpha * X
// ─────────────────────────────────────────────────────────────────────────────

/// Graph-aware AXPY: ``Y += alpha * X`` (BLAS level-1).
///
/// X and Y must have the same dtype and shape; the operation is
/// element-wise. Eager outside graph capture; recorded as a node when
/// inside a capture context.
template <TensorConcept XType>
// clang-format off
EINSUMS_PYBIND_EXPOSE
EINSUMS_PYBIND_MODULE("linalg")
EINSUMS_PYBIND_INSTANTIATE_AS("axpy", einsums::GeneralRuntimeTensor<float, std::allocator<float>>)
EINSUMS_PYBIND_INSTANTIATE_AS("axpy", einsums::GeneralRuntimeTensor<double, std::allocator<double>>)
EINSUMS_PYBIND_INSTANTIATE_AS("axpy", einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>)
EINSUMS_PYBIND_INSTANTIATE_AS("axpy", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
    // clang-format on
    void axpy(typename XType::ValueType alpha, XType const &X, XType *Y) {
    auto &ctx = CaptureContext::current();
    if (!ctx.is_capturing()) {
        linear_algebra::axpy(alpha, X, Y);
        return;
    }

    auto [x_id, x_slot] = ctx.get_slot(X);
    auto [y_id, y_slot] = ctx.get_slot(*Y);

    auto label    = fmt::format("axpy(alpha={}, {}, {})", alpha, X.name(), Y->name());
    auto executor = [alpha, x_slot, y_slot]() {
        linear_algebra::axpy(alpha, *static_cast<XType const *>(x_slot->ptr), static_cast<XType *>(y_slot->ptr));
    };

    ctx.record(OpKind::Axpy, std::move(label), {x_id}, {y_id}, std::move(executor));
}

// ─────────────────────────────────────────────────────────────────────────────
// axpby: Y = alpha * X + beta * Y
// ─────────────────────────────────────────────────────────────────────────────

/// Graph-aware AXPBY: ``Y = alpha * X + beta * Y`` (extended BLAS level-1).
///
/// Like ``axpy`` but also scales the destination by ``beta`` first.
/// X and Y must have the same dtype and shape.
template <TensorConcept T>
// clang-format off
EINSUMS_PYBIND_EXPOSE
EINSUMS_PYBIND_MODULE("linalg")
EINSUMS_PYBIND_INSTANTIATE_AS("axpby", einsums::GeneralRuntimeTensor<float, std::allocator<float>>)
EINSUMS_PYBIND_INSTANTIATE_AS("axpby", einsums::GeneralRuntimeTensor<double, std::allocator<double>>)
EINSUMS_PYBIND_INSTANTIATE_AS("axpby", einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>)
EINSUMS_PYBIND_INSTANTIATE_AS("axpby", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
    // clang-format on
    void axpby(typename T::ValueType alpha, T const &X, typename T::ValueType beta, T *Y) {
    auto &ctx = CaptureContext::current();
    if (!ctx.is_capturing()) {
        linear_algebra::axpby(alpha, X, beta, Y);
        return;
    }

    auto [x_id, x_slot] = ctx.get_slot(X);
    auto [y_id, y_slot] = ctx.get_slot(*Y);

    auto label    = fmt::format("axpby(alpha={}, beta={})", alpha, beta);
    auto executor = [alpha, x_slot, beta, y_slot]() {
        linear_algebra::axpby(alpha, *static_cast<T const *>(x_slot->ptr), beta, static_cast<T *>(y_slot->ptr));
    };

    ctx.record(OpKind::Axpby, std::move(label), {x_id}, {y_id}, std::move(executor));
}

// ─────────────────────────────────────────────────────────────────────────────
// gemm: C = alpha * op(A) * op(B) + beta * C
// ─────────────────────────────────────────────────────────────────────────────

template <bool TransA, bool TransB, MatrixConcept T, typename U>
    requires requires { requires std::convertible_to<U, typename T::ValueType>; }
void gemm(U const alpha, T const &A, T const &B, U const beta, T *C) {
    auto &ctx = CaptureContext::current();
    if (!ctx.is_capturing()) {
        linear_algebra::gemm<TransA, TransB>(alpha, A, B, beta, C);
        return;
    }

    auto [a_id, a_slot] = ctx.get_slot(A);
    auto [b_id, b_slot] = ctx.get_slot(B);
    auto [c_id, c_slot] = ctx.get_slot(*C);

    auto label    = fmt::format("gemm<{},{}>", TransA ? "T" : "N", TransB ? "T" : "N");
    auto executor = [alpha, a_slot, b_slot, beta, c_slot]() {
        linear_algebra::gemm<TransA, TransB>(alpha, *static_cast<T const *>(a_slot->ptr), *static_cast<T const *>(b_slot->ptr), beta,
                                             static_cast<T *>(c_slot->ptr));
    };

    ctx.record(OpKind::Gemm, std::move(label), {a_id, b_id}, {c_id}, std::move(executor));
}

/// Graph-aware GEMM: ``C = alpha * op(A) * op(B) + beta * C``.
///
/// ``trans_a`` and ``trans_b`` (Python kwargs, default ``False``) request
/// the transpose of the corresponding matrix. All three tensors must be
/// rank 2; a clear ``rank_error`` is raised up front otherwise rather
/// than letting the BLAS kernel fail mid-pipeline.
template <bool TransA, bool TransB, RuntimeRankTensorConcept T, typename U>
    requires std::convertible_to<U, typename T::ValueType>
// clang-format off
EINSUMS_PYBIND_EXPOSE
EINSUMS_PYBIND_MODULE("linalg")
EINSUMS_PYBIND_TEMPLATE_KWARGS("trans_a", "trans_b")
EINSUMS_PYBIND_INSTANTIATE_BOOLS("gemm", einsums::GeneralRuntimeTensor<float, std::allocator<float>>, float)
EINSUMS_PYBIND_INSTANTIATE_BOOLS("gemm", einsums::GeneralRuntimeTensor<double, std::allocator<double>>, double)
EINSUMS_PYBIND_INSTANTIATE_BOOLS("gemm", einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, std::complex<float>)
EINSUMS_PYBIND_INSTANTIATE_BOOLS("gemm", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, std::complex<double>)
    // clang-format on
    void gemm(U const alpha, T const &A, T const &B, U const beta, T *C) {
    if (A.rank() != 2 || B.rank() != 2 || C->rank() != 2) {
        EINSUMS_THROW_EXCEPTION(rank_error, "cg::gemm requires rank-2 tensors; got ranks {}, {}, {}.", A.rank(), B.rank(), C->rank());
    }

    auto &ctx = CaptureContext::current();
    if (!ctx.is_capturing()) {
        linear_algebra::gemm<TransA, TransB>(alpha, A, B, beta, C);
        return;
    }

    auto [a_id, a_slot] = ctx.get_slot(A);
    auto [b_id, b_slot] = ctx.get_slot(B);
    auto [c_id, c_slot] = ctx.get_slot(*C);

    auto label    = fmt::format("gemm<{},{}>", TransA ? "T" : "N", TransB ? "T" : "N");
    auto executor = [alpha, a_slot, b_slot, beta, c_slot]() {
        linear_algebra::gemm<TransA, TransB>(alpha, *static_cast<T const *>(a_slot->ptr), *static_cast<T const *>(b_slot->ptr), beta,
                                             static_cast<T *>(c_slot->ptr));
    };

    ctx.record(OpKind::Gemm, std::move(label), {a_id, b_id}, {c_id}, std::move(executor));
}

// ─────────────────────────────────────────────────────────────────────────────
// gemv: y = alpha * op(A) * z + beta * y
// ─────────────────────────────────────────────────────────────────────────────

template <bool TransA, MatrixConcept AType, VectorConcept XType, VectorConcept YType, typename U>
    requires requires {
        requires SameUnderlying<AType, XType, YType>;
        requires std::convertible_to<U, typename AType::ValueType>;
    }
void gemv(U const alpha, AType const &A, XType const &z, U const beta, YType *y) {
    auto &ctx = CaptureContext::current();
    if (!ctx.is_capturing()) {
        linear_algebra::gemv<TransA>(alpha, A, z, beta, y);
        return;
    }

    auto [a_id, a_slot] = ctx.get_slot(A);
    auto [z_id, z_slot] = ctx.get_slot(z);
    auto [y_id, y_slot] = ctx.get_slot(*y);

    auto label    = fmt::format("gemv<{}>", TransA ? "T" : "N");
    auto executor = [alpha, a_slot, z_slot, beta, y_slot]() {
        linear_algebra::gemv<TransA>(alpha, *static_cast<AType const *>(a_slot->ptr), *static_cast<XType const *>(z_slot->ptr), beta,
                                     static_cast<YType *>(y_slot->ptr));
    };

    ctx.record(OpKind::Gemv, std::move(label), {a_id, z_id}, {y_id}, std::move(executor));
}

/// Graph-aware GEMV: ``y = alpha * op(A) * z + beta * y``.
///
/// ``trans_a`` (Python kwarg, default ``False``) transposes A. A must be
/// rank 2 and z, y must be rank 1; a ``rank_error`` is raised otherwise.
template <bool TransA, RuntimeRankTensorConcept AType, RuntimeRankTensorConcept XType, RuntimeRankTensorConcept YType, typename U>
    requires(SameUnderlying<AType, XType, YType> && std::convertible_to<U, typename AType::ValueType>)
// clang-format off
EINSUMS_PYBIND_EXPOSE
EINSUMS_PYBIND_MODULE("linalg")
EINSUMS_PYBIND_TEMPLATE_KWARGS("trans_a")
EINSUMS_PYBIND_INSTANTIATE_BOOLS("gemv", einsums::GeneralRuntimeTensor<float,                std::allocator<float>>,                einsums::GeneralRuntimeTensor<float,                std::allocator<float>>,                einsums::GeneralRuntimeTensor<float,                std::allocator<float>>,                float)
EINSUMS_PYBIND_INSTANTIATE_BOOLS("gemv", einsums::GeneralRuntimeTensor<double,               std::allocator<double>>,               einsums::GeneralRuntimeTensor<double,               std::allocator<double>>,               einsums::GeneralRuntimeTensor<double,               std::allocator<double>>,               double)
EINSUMS_PYBIND_INSTANTIATE_BOOLS("gemv", einsums::GeneralRuntimeTensor<std::complex<float>,  std::allocator<std::complex<float>>>,  einsums::GeneralRuntimeTensor<std::complex<float>,  std::allocator<std::complex<float>>>,  einsums::GeneralRuntimeTensor<std::complex<float>,  std::allocator<std::complex<float>>>,  std::complex<float>)
EINSUMS_PYBIND_INSTANTIATE_BOOLS("gemv", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, std::complex<double>)
    // clang-format on
    void gemv(U const alpha, AType const &A, XType const &z, U const beta, YType *y) {
    if (A.rank() != 2 || z.rank() != 1 || y->rank() != 1) {
        EINSUMS_THROW_EXCEPTION(rank_error, "cg::gemv requires A rank-2 and x/y rank-1; got {}, {}, {}.", A.rank(), z.rank(), y->rank());
    }

    auto &ctx = CaptureContext::current();
    if (!ctx.is_capturing()) {
        linear_algebra::gemv<TransA>(alpha, A, z, beta, y);
        return;
    }

    auto [a_id, a_slot] = ctx.get_slot(A);
    auto [z_id, z_slot] = ctx.get_slot(z);
    auto [y_id, y_slot] = ctx.get_slot(*y);

    auto label    = fmt::format("gemv<{}>", TransA ? "T" : "N");
    auto executor = [alpha, a_slot, z_slot, beta, y_slot]() {
        linear_algebra::gemv<TransA>(alpha, *static_cast<AType const *>(a_slot->ptr), *static_cast<XType const *>(z_slot->ptr), beta,
                                     static_cast<YType *>(y_slot->ptr));
    };

    ctx.record(OpKind::Gemv, std::move(label), {a_id, z_id}, {y_id}, std::move(executor));
}

// ─────────────────────────────────────────────────────────────────────────────
// ger: A += alpha * X * Y^T
// ─────────────────────────────────────────────────────────────────────────────

template <MatrixConcept AType, VectorConcept XType, VectorConcept YType>
    requires SameUnderlying<AType, XType, YType>
void ger(typename AType::ValueType alpha, XType const &X, YType const &Y, AType *A) {
    auto &ctx = CaptureContext::current();
    if (!ctx.is_capturing()) {
        linear_algebra::ger(alpha, X, Y, A);
        return;
    }

    auto [x_id, x_slot] = ctx.get_slot(X);
    auto [y_id, y_slot] = ctx.get_slot(Y);
    auto [a_id, a_slot] = ctx.get_slot(*A);

    auto executor = [alpha, x_slot, y_slot, a_slot]() {
        linear_algebra::ger(alpha, *static_cast<XType const *>(x_slot->ptr), *static_cast<YType const *>(y_slot->ptr),
                            static_cast<AType *>(a_slot->ptr));
    };

    ctx.record(OpKind::Ger, "ger", {x_id, y_id}, {a_id}, std::move(executor));
}

/// Graph-aware GER (rank-1 update): ``A += alpha * X * Y^T``.
///
/// Outer product of vectors X and Y added to matrix A. X and Y must be
/// rank 1; A must be rank 2.
template <RuntimeRankTensorConcept AType, RuntimeRankTensorConcept XType, RuntimeRankTensorConcept YType>
    requires SameUnderlying<AType, XType, YType>
// clang-format off
EINSUMS_PYBIND_EXPOSE
EINSUMS_PYBIND_MODULE("linalg")
EINSUMS_PYBIND_INSTANTIATE_AS("ger", einsums::GeneralRuntimeTensor<float,                std::allocator<float>>,                einsums::GeneralRuntimeTensor<float,                std::allocator<float>>,                einsums::GeneralRuntimeTensor<float,                std::allocator<float>>)
EINSUMS_PYBIND_INSTANTIATE_AS("ger", einsums::GeneralRuntimeTensor<double,               std::allocator<double>>,               einsums::GeneralRuntimeTensor<double,               std::allocator<double>>,               einsums::GeneralRuntimeTensor<double,               std::allocator<double>>)
EINSUMS_PYBIND_INSTANTIATE_AS("ger", einsums::GeneralRuntimeTensor<std::complex<float>,  std::allocator<std::complex<float>>>,  einsums::GeneralRuntimeTensor<std::complex<float>,  std::allocator<std::complex<float>>>,  einsums::GeneralRuntimeTensor<std::complex<float>,  std::allocator<std::complex<float>>>)
EINSUMS_PYBIND_INSTANTIATE_AS("ger", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
    // clang-format on
    void ger(typename AType::ValueType alpha, XType const &X, YType const &Y, AType *A) {
    if (X.rank() != 1 || Y.rank() != 1 || A->rank() != 2) {
        EINSUMS_THROW_EXCEPTION(rank_error, "cg::ger requires X/Y rank-1 and A rank-2; got {}, {}, {}.", X.rank(), Y.rank(), A->rank());
    }

    auto &ctx = CaptureContext::current();
    if (!ctx.is_capturing()) {
        linear_algebra::ger(alpha, X, Y, A);
        return;
    }

    auto [x_id, x_slot] = ctx.get_slot(X);
    auto [y_id, y_slot] = ctx.get_slot(Y);
    auto [a_id, a_slot] = ctx.get_slot(*A);

    auto executor = [alpha, x_slot, y_slot, a_slot]() {
        linear_algebra::ger(alpha, *static_cast<XType const *>(x_slot->ptr), *static_cast<YType const *>(y_slot->ptr),
                            static_cast<AType *>(a_slot->ptr));
    };

    ctx.record(OpKind::Ger, "ger", {x_id, y_id}, {a_id}, std::move(executor));
}

// ─────────────────────────────────────────────────────────────────────────────
// dot: result = sum(A * B)
// ─────────────────────────────────────────────────────────────────────────────

template <TensorConcept AType, TensorConcept BType>
    requires requires {
        requires SameRank<AType, BType>;
        requires InSamePlace<AType, BType>;
    }
// clang-format off
EINSUMS_PYBIND_EXPOSE
EINSUMS_PYBIND_MODULE("linalg")
EINSUMS_PYBIND_INSTANTIATE_AS("dot", einsums::GeneralRuntimeTensor<float,                std::allocator<float>>,                einsums::GeneralRuntimeTensor<float,                std::allocator<float>>)
EINSUMS_PYBIND_INSTANTIATE_AS("dot", einsums::GeneralRuntimeTensor<double,               std::allocator<double>>,               einsums::GeneralRuntimeTensor<double,               std::allocator<double>>)
EINSUMS_PYBIND_INSTANTIATE_AS("dot", einsums::GeneralRuntimeTensor<std::complex<float>,  std::allocator<std::complex<float>>>,  einsums::GeneralRuntimeTensor<std::complex<float>,  std::allocator<std::complex<float>>>)
EINSUMS_PYBIND_INSTANTIATE_AS("dot", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
    // clang-format on
    auto dot(AType const &A, BType const &B) -> BiggestTypeT<typename AType::ValueType, typename BType::ValueType> {
    if (CaptureContext::current().is_capturing()) {
        EINSUMS_THROW_EXCEPTION(std::logic_error, "cg::dot(A, B) returning scalar cannot be used during graph capture. "
                                                  "Use cg::einsum(\" <- i ; i\", &result, A, B) instead.");
    }
    return linear_algebra::dot(A, B);
}

/// Graph-aware dot product writing result to a pre-allocated scalar.
/// Unlike dot(A, B) which throws during capture, this overload records the
/// operation into the graph and can be used with distributed tensors.
template <TensorConcept AType, TensorConcept BType>
    requires requires {
        requires SameRank<AType, BType>;
        requires InSamePlace<AType, BType>;
    }
void dot(BiggestTypeT<typename AType::ValueType, typename BType::ValueType> *result, AType const &A, BType const &B) {
    using ResultT = BiggestTypeT<typename AType::ValueType, typename BType::ValueType>;

    auto &ctx = CaptureContext::current();
    if (!ctx.is_capturing()) {
        *result = linear_algebra::dot(A, B);
        return;
    }

    auto [a_id, a_slot] = ctx.get_slot(A);
    auto [b_id, b_slot] = ctx.get_slot(B);
    TensorId r_id       = ctx.get_or_register_scalar(result, "dot_result");

    auto executor = [result, a_slot, b_slot]() {
        *result = linear_algebra::dot(*static_cast<AType const *>(a_slot->ptr), *static_cast<BType const *>(b_slot->ptr));
    };

    ctx.record(OpKind::Dot, "dot", {a_id, b_id}, {r_id}, std::move(executor));
}

/// Python-friendly graph-aware dot: writes the result into ``result->data()[0]``.
///
/// ``result`` is a pre-allocated rank-1 (or higher, but only element 0 is
/// touched) tensor that gives Python users a graph-native scalar handle —
/// SCF energy patterns like ``e = ½ Σ D · (H+F)`` can be captured.
template <CoreBasicTensorConcept ResultType, CoreBasicTensorConcept AType, CoreBasicTensorConcept BType>
    requires requires {
        requires std::is_same_v<typename ResultType::ValueType, typename AType::ValueType>;
        requires std::is_same_v<typename AType::ValueType, typename BType::ValueType>;
    }
// clang-format off
EINSUMS_PYBIND_EXPOSE
EINSUMS_PYBIND_MODULE("linalg")
EINSUMS_PYBIND_INSTANTIATE_AS("dot", einsums::GeneralRuntimeTensor<float,                std::allocator<float>>,                einsums::GeneralRuntimeTensor<float,                std::allocator<float>>,                einsums::GeneralRuntimeTensor<float,                std::allocator<float>>)
EINSUMS_PYBIND_INSTANTIATE_AS("dot", einsums::GeneralRuntimeTensor<double,               std::allocator<double>>,               einsums::GeneralRuntimeTensor<double,               std::allocator<double>>,               einsums::GeneralRuntimeTensor<double,               std::allocator<double>>)
EINSUMS_PYBIND_INSTANTIATE_AS("dot", einsums::GeneralRuntimeTensor<std::complex<float>,  std::allocator<std::complex<float>>>,  einsums::GeneralRuntimeTensor<std::complex<float>,  std::allocator<std::complex<float>>>,  einsums::GeneralRuntimeTensor<std::complex<float>,  std::allocator<std::complex<float>>>)
EINSUMS_PYBIND_INSTANTIATE_AS("dot", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
    // clang-format on
    void dot_python(ResultType *result, AType const &A, BType const &B) {
    if (result->size() < 1) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "cg::dot: result tensor must have at least one element");
    }

    auto &ctx = CaptureContext::current();
    if (!ctx.is_capturing()) {
        result->data()[0] = linear_algebra::dot(A, B);
        return;
    }

    // Register the result as a normal tensor slot (not a scalar handle) so
    // downstream tensor ops (scale, axpy, ...) on the same tensor see the
    // same slot id — get_or_register_scalar would key by data()[0] and
    // collide with get_slot(*result), giving rank-0 metadata to the scale.
    auto [a_id, a_slot] = ctx.get_slot(A);
    auto [b_id, b_slot] = ctx.get_slot(B);
    auto [r_id, r_slot] = ctx.get_slot(*result);

    auto executor = [a_slot, b_slot, r_slot]() {
        auto *r_ptr      = static_cast<ResultType *>(r_slot->ptr);
        r_ptr->data()[0] = linear_algebra::dot(*static_cast<AType const *>(a_slot->ptr), *static_cast<BType const *>(b_slot->ptr));
    };
    ctx.record(OpKind::Dot, "dot", {a_id, b_id}, {r_id}, std::move(executor));
}

// ─────────────────────────────────────────────────────────────────────────────
// direct_product: C = alpha * (A ⊙ B) + beta * C
// ─────────────────────────────────────────────────────────────────────────────

template <typename T, TensorConcept AType, TensorConcept BType, TensorConcept CType>
// clang-format off
EINSUMS_PYBIND_EXPOSE
EINSUMS_PYBIND_MODULE("linalg")
EINSUMS_PYBIND_INSTANTIATE_AS("direct_product", float,                einsums::GeneralRuntimeTensor<float,                std::allocator<float>>,                einsums::GeneralRuntimeTensor<float,                std::allocator<float>>,                einsums::GeneralRuntimeTensor<float,                std::allocator<float>>)
EINSUMS_PYBIND_INSTANTIATE_AS("direct_product", double,               einsums::GeneralRuntimeTensor<double,               std::allocator<double>>,               einsums::GeneralRuntimeTensor<double,               std::allocator<double>>,               einsums::GeneralRuntimeTensor<double,               std::allocator<double>>)
EINSUMS_PYBIND_INSTANTIATE_AS("direct_product", std::complex<float>,  einsums::GeneralRuntimeTensor<std::complex<float>,  std::allocator<std::complex<float>>>,  einsums::GeneralRuntimeTensor<std::complex<float>,  std::allocator<std::complex<float>>>,  einsums::GeneralRuntimeTensor<std::complex<float>,  std::allocator<std::complex<float>>>)
EINSUMS_PYBIND_INSTANTIATE_AS("direct_product", std::complex<double>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
    // clang-format on
    void direct_product(T alpha, AType const &A, BType const &B, T beta, CType *C) {
    auto &ctx = CaptureContext::current();
    if (!ctx.is_capturing()) {
        linear_algebra::direct_product(alpha, A, B, beta, C);
        return;
    }

    auto [a_id, a_slot] = ctx.get_slot(A);
    auto [b_id, b_slot] = ctx.get_slot(B);
    auto [c_id, c_slot] = ctx.get_slot(*C);

    auto executor = [alpha, a_slot, b_slot, beta, c_slot]() {
        linear_algebra::direct_product(alpha, *static_cast<AType const *>(a_slot->ptr), *static_cast<BType const *>(b_slot->ptr), beta,
                                       static_cast<CType *>(c_slot->ptr));
    };

    ctx.record(OpKind::DirectProduct, "direct_product", {a_id, b_id}, {c_id}, std::move(executor));
}

// ─────────────────────────────────────────────────────────────────────────────
// outer_sum: rank-N result(i_0,...,i_{N-1}) = Σ_k c_k * v_k(i_k)
// ─────────────────────────────────────────────────────────────────────────────

/// Outer sum of N rank-1 vectors with per-axis coefficients.
///
/// Fills ``result`` with
///   ``result(i_0, i_1, ..., i_{N-1}) = Σ_k coefficients[k] * vectors[k](i_k)``.
///
/// Canonical use case is the MP2/CC energy denominator:
///   Δ(i,j,a,b) = ε_i + ε_j − ε_a − ε_b
///     ↪ outer_sum(&Δ, {ε_occ, ε_occ, ε_virt, ε_virt}, {+1, +1, -1, -1})
///
/// If ``coefficients`` is empty, defaults to all +1.
///
/// Capture-aware: outside capture executes immediately; inside capture
/// records a Custom node with each input vector and the result as
/// dependencies. ``result->rank()`` must equal ``vectors.size()`` and
/// ``result->dim(k)`` must equal ``vectors[k]->dim(0)``.
template <CoreBasicTensorConcept ResultType, CoreBasicTensorConcept VectorType>
    requires std::is_same_v<typename ResultType::ValueType, typename VectorType::ValueType>
// clang-format off
EINSUMS_PYBIND_EXPOSE
EINSUMS_PYBIND_MODULE("linalg")
EINSUMS_PYBIND_INSTANTIATE_AS("outer_sum", einsums::GeneralRuntimeTensor<float,                std::allocator<float>>,                einsums::GeneralRuntimeTensor<float,                std::allocator<float>>)
EINSUMS_PYBIND_INSTANTIATE_AS("outer_sum", einsums::GeneralRuntimeTensor<double,               std::allocator<double>>,               einsums::GeneralRuntimeTensor<double,               std::allocator<double>>)
EINSUMS_PYBIND_INSTANTIATE_AS("outer_sum", einsums::GeneralRuntimeTensor<std::complex<float>,  std::allocator<std::complex<float>>>,  einsums::GeneralRuntimeTensor<std::complex<float>,  std::allocator<std::complex<float>>>)
EINSUMS_PYBIND_INSTANTIATE_AS("outer_sum", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
    // clang-format on
    void outer_sum(ResultType *result, std::vector<VectorType const *> vectors, std::vector<double> coefficients) {
    using T = typename ResultType::ValueType;

    size_t const N = vectors.size();
    if (N == 0) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "cg::outer_sum: must provide at least one vector");
    }
    if (result->rank() != N) {
        EINSUMS_THROW_EXCEPTION(rank_error, "cg::outer_sum: result rank ({}) must equal number of vectors ({})", result->rank(), N);
    }
    for (size_t k = 0; k < N; ++k) {
        if (vectors[k] == nullptr) {
            EINSUMS_THROW_EXCEPTION(std::invalid_argument, "cg::outer_sum: vector[{}] is null", k);
        }
        if (vectors[k]->rank() != 1) {
            EINSUMS_THROW_EXCEPTION(rank_error, "cg::outer_sum: vector[{}] must be rank-1; got rank {}", k, vectors[k]->rank());
        }
        if (vectors[k]->dim(0) != result->dim(k)) {
            EINSUMS_THROW_EXCEPTION(std::invalid_argument, "cg::outer_sum: vector[{}] length ({}) doesn't match result dim {} ({})", k,
                                    vectors[k]->dim(0), k, result->dim(k));
        }
    }

    std::vector<T> effective_coeffs(N, T{1});
    if (!coefficients.empty()) {
        if (coefficients.size() != N) {
            EINSUMS_THROW_EXCEPTION(std::invalid_argument, "cg::outer_sum: coefficients length ({}) must equal number of vectors ({})",
                                    coefficients.size(), N);
        }
        for (size_t k = 0; k < N; ++k)
            effective_coeffs[k] = static_cast<T>(coefficients[k]);
    }

    auto apply = [vectors, effective_coeffs, N](ResultType *r) {
        size_t const        total = r->size();
        std::vector<size_t> idx(N, 0);
        std::vector<size_t> dims(N), strides(N);
        for (size_t k = 0; k < N; ++k) {
            dims[k]    = r->dim(k);
            strides[k] = r->stride(k);
        }
        T *out = r->data();
        for (size_t count = 0; count < total; ++count) {
            T sum{};
            for (size_t k = 0; k < N; ++k) {
                sum += effective_coeffs[k] * vectors[k]->data()[idx[k]];
            }
            size_t offset = 0;
            for (size_t k = 0; k < N; ++k)
                offset += idx[k] * strides[k];
            out[offset] = sum;
            // Increment multi-index (axis 0 fastest — direction is irrelevant for correctness).
            for (size_t k = 0; k < N; ++k) {
                if (++idx[k] < dims[k])
                    break;
                idx[k] = 0;
            }
        }
    };

    auto &ctx = CaptureContext::current();
    if (!ctx.is_capturing()) {
        apply(result);
        return;
    }

    std::vector<TensorId> in_ids;
    in_ids.reserve(N);
    std::vector<TensorSlot const *> v_slots;
    v_slots.reserve(N);
    for (size_t k = 0; k < N; ++k) {
        auto [vid, vslot] = ctx.get_slot(*vectors[k]);
        in_ids.push_back(vid);
        v_slots.push_back(vslot);
    }
    auto [r_id, r_slot] = ctx.get_slot(*result);

    auto executor = [v_slots, r_slot, effective_coeffs, N]() {
        auto                           *r_ptr = static_cast<ResultType *>(r_slot->ptr);
        std::vector<VectorType const *> rebound(N);
        for (size_t k = 0; k < N; ++k)
            rebound[k] = static_cast<VectorType const *>(v_slots[k]->ptr);

        size_t const        total = r_ptr->size();
        std::vector<size_t> idx(N, 0);
        std::vector<size_t> dims(N), strides(N);
        for (size_t k = 0; k < N; ++k) {
            dims[k]    = r_ptr->dim(k);
            strides[k] = r_ptr->stride(k);
        }
        T *out = r_ptr->data();
        for (size_t count = 0; count < total; ++count) {
            T sum{};
            for (size_t k = 0; k < N; ++k) {
                sum += effective_coeffs[k] * rebound[k]->data()[idx[k]];
            }
            size_t offset = 0;
            for (size_t k = 0; k < N; ++k)
                offset += idx[k] * strides[k];
            out[offset] = sum;
            for (size_t k = 0; k < N; ++k) {
                if (++idx[k] < dims[k])
                    break;
                idx[k] = 0;
            }
        }
    };

    ctx.record(OpKind::Custom, "outer_sum", in_ids, {r_id}, std::move(executor));
}

// ─────────────────────────────────────────────────────────────────────────────
// norm
// ─────────────────────────────────────────────────────────────────────────────

template <TensorConcept AType>
// clang-format off
EINSUMS_PYBIND_EXPOSE
EINSUMS_PYBIND_MODULE("linalg")
EINSUMS_PYBIND_INSTANTIATE_AS("norm", einsums::GeneralRuntimeTensor<float,                std::allocator<float>>)
EINSUMS_PYBIND_INSTANTIATE_AS("norm", einsums::GeneralRuntimeTensor<double,               std::allocator<double>>)
EINSUMS_PYBIND_INSTANTIATE_AS("norm", einsums::GeneralRuntimeTensor<std::complex<float>,  std::allocator<std::complex<float>>>)
EINSUMS_PYBIND_INSTANTIATE_AS("norm", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
    // clang-format on
    auto norm(linear_algebra::Norm norm_type, AType const &A) -> RemoveComplexT<typename AType::ValueType> {
    if (CaptureContext::current().is_capturing()) {
        EINSUMS_THROW_EXCEPTION(std::logic_error, "cg::norm() returning scalar cannot be used during graph capture.");
    }
    return linear_algebra::norm(norm_type, A);
}

/// Graph-aware norm writing result to a pre-allocated scalar.
template <TensorConcept AType>
void norm(RemoveComplexT<typename AType::ValueType> *result, linear_algebra::Norm norm_type, AType const &A) {
    auto &ctx = CaptureContext::current();
    if (!ctx.is_capturing()) {
        *result = linear_algebra::norm(norm_type, A);
        return;
    }

    auto [a_id, a_slot] = ctx.get_slot(A);
    TensorId r_id       = ctx.get_or_register_scalar(result, "norm_result");

    auto executor = [result, norm_type, a_slot]() { *result = linear_algebra::norm(norm_type, *static_cast<AType const *>(a_slot->ptr)); };

    ctx.record(OpKind::Norm, "norm", {a_id}, {r_id}, std::move(executor));
}

/// Python-friendly graph-aware norm: writes the result into ``result->data()[0]``.
///
/// For complex inputs the result is real-valued (e.g. complex<double> input
/// requires a ``double`` result tensor). Use ``Norm::ONE``, ``Norm::TWO``,
/// ``Norm::INFINITY_``, ``Norm::FROBENIUS``, etc.
template <CoreBasicTensorConcept ResultType, CoreBasicTensorConcept AType>
    requires requires { requires std::is_same_v<typename ResultType::ValueType, RemoveComplexT<typename AType::ValueType>>; }
// clang-format off
EINSUMS_PYBIND_EXPOSE
EINSUMS_PYBIND_MODULE("linalg")
EINSUMS_PYBIND_INSTANTIATE_AS("norm", einsums::GeneralRuntimeTensor<float,  std::allocator<float>>,  einsums::GeneralRuntimeTensor<float,                std::allocator<float>>)
EINSUMS_PYBIND_INSTANTIATE_AS("norm", einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::GeneralRuntimeTensor<double,               std::allocator<double>>)
EINSUMS_PYBIND_INSTANTIATE_AS("norm", einsums::GeneralRuntimeTensor<float,  std::allocator<float>>,  einsums::GeneralRuntimeTensor<std::complex<float>,  std::allocator<std::complex<float>>>)
EINSUMS_PYBIND_INSTANTIATE_AS("norm", einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
    // clang-format on
    void norm_python(ResultType *result, linear_algebra::Norm norm_type, AType const &A) {
    if (result->size() < 1) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "cg::norm: result tensor must have at least one element");
    }

    auto &ctx = CaptureContext::current();
    if (!ctx.is_capturing()) {
        result->data()[0] = linear_algebra::norm(norm_type, A);
        return;
    }

    auto [a_id, a_slot] = ctx.get_slot(A);
    auto [r_id, r_slot] = ctx.get_slot(*result);

    auto executor = [norm_type, a_slot, r_slot]() {
        auto *r_ptr      = static_cast<ResultType *>(r_slot->ptr);
        r_ptr->data()[0] = linear_algebra::norm(norm_type, *static_cast<AType const *>(a_slot->ptr));
    };
    ctx.record(OpKind::Norm, "norm", {a_id}, {r_id}, std::move(executor));
}

// ─────────────────────────────────────────────────────────────────────────────
// trace
// ─────────────────────────────────────────────────────────────────────────────
//
// ``cg::trace`` records the diagonal sum of a square rank-2 tensor:
//   tr(A) = Σᵢ A(i,i)
//
// Two forms, mirroring the dot/norm pattern:
//   - eager:    ``auto t = cg::trace(A);`` — executes immediately. Throws if
//               called during graph capture (capture has no scalar return).
//   - recorded: ``cg::trace(&t, A);`` — records into the active graph. ``t``
//               is read at execute time and is the destination scalar.
//
// Trace doesn't need a dedicated OpKind — it lowers to a one-liner inside an
// ``OpKind::Custom`` node. Passes that want to recognize the pattern can
// pattern-match by node label (``"trace"``).

template <MatrixConcept AType>
auto trace(AType const &A) -> typename AType::ValueType {
    if (CaptureContext::current().is_capturing()) {
        EINSUMS_THROW_EXCEPTION(std::logic_error, "cg::trace(A) returning scalar cannot be used during graph capture. "
                                                  "Use cg::trace(&result, A) instead.");
    }
    if (A.dim(0) != A.dim(1))
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "cg::trace: input must be square");
    using T = typename AType::ValueType;
    T sum   = T{};
    for (size_t i = 0; i < A.dim(0); ++i)
        sum += A(i, i);
    return sum;
}

/// Trace of a square matrix: ``sum(A_ii)``. Returns the scalar.
///
/// Cannot be used during graph capture (returns by value, so there is no
/// destination slot). For the in-graph form use the four-argument
/// ``cg::trace(&result, A)`` with a pre-allocated scalar.
template <RuntimeRankTensorConcept AType>
// clang-format off
EINSUMS_PYBIND_EXPOSE
EINSUMS_PYBIND_MODULE("linalg")
EINSUMS_PYBIND_INSTANTIATE_AS("trace", einsums::GeneralRuntimeTensor<float,                std::allocator<float>>)
EINSUMS_PYBIND_INSTANTIATE_AS("trace", einsums::GeneralRuntimeTensor<double,               std::allocator<double>>)
EINSUMS_PYBIND_INSTANTIATE_AS("trace", einsums::GeneralRuntimeTensor<std::complex<float>,  std::allocator<std::complex<float>>>)
EINSUMS_PYBIND_INSTANTIATE_AS("trace", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
    // clang-format on
    auto trace(AType const &A) -> typename AType::ValueType {
    if (CaptureContext::current().is_capturing()) {
        EINSUMS_THROW_EXCEPTION(std::logic_error, "cg::trace(A) returning scalar cannot be used during graph capture. "
                                                  "Use cg::trace(&result, A) instead.");
    }
    if (A.rank() != 2) {
        EINSUMS_THROW_EXCEPTION(rank_error, "cg::trace: input must be rank-2; got rank {}.", A.rank());
    }
    if (A.dim(0) != A.dim(1)) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "cg::trace: input must be square");
    }
    using T = typename AType::ValueType;
    T sum   = T{};
    for (size_t i = 0; i < A.dim(0); ++i)
        sum += A(i, i);
    return sum;
}

/// Graph-aware trace writing the result to a pre-allocated scalar.
///
/// @tparam AType A square rank-2 tensor type satisfying @c MatrixConcept.
/// @param[out] result Pointer to the destination scalar; populated at execute.
/// @param[in]  A      Source tensor; ``A.dim(0) == A.dim(1)`` is required.
template <MatrixConcept AType>
void trace(typename AType::ValueType *result, AType const &A) {
    using T = typename AType::ValueType;

    auto &ctx = CaptureContext::current();
    if (!ctx.is_capturing()) {
        if (A.dim(0) != A.dim(1))
            EINSUMS_THROW_EXCEPTION(std::invalid_argument, "cg::trace: input must be square");
        T sum = T{};
        for (size_t i = 0; i < A.dim(0); ++i)
            sum += A(i, i);
        *result = sum;
        return;
    }

    auto [a_id, a_slot] = ctx.get_slot(A);
    TensorId r_id       = ctx.get_or_register_scalar(result, "trace_result");

    auto executor = [result, a_slot]() {
        auto const &a = *static_cast<AType const *>(a_slot->ptr);
        if (a.dim(0) != a.dim(1))
            EINSUMS_THROW_EXCEPTION(std::invalid_argument, "cg::trace: input must be square");
        T sum = T{};
        for (size_t i = 0; i < a.dim(0); ++i)
            sum += a(i, i);
        *result = sum;
    };

    ctx.record(OpKind::Trace, "trace", {a_id}, {r_id}, std::move(executor));
}

/// Python-friendly graph-aware trace: writes the diagonal sum into
/// ``result->data()[0]``. Runtime-rank input must be a square rank-2 tensor.
template <CoreBasicTensorConcept ResultType, CoreBasicTensorConcept AType>
    requires requires { requires std::is_same_v<typename ResultType::ValueType, typename AType::ValueType>; }
// clang-format off
EINSUMS_PYBIND_EXPOSE
EINSUMS_PYBIND_MODULE("linalg")
EINSUMS_PYBIND_INSTANTIATE_AS("trace", einsums::GeneralRuntimeTensor<float,                std::allocator<float>>,                einsums::GeneralRuntimeTensor<float,                std::allocator<float>>)
EINSUMS_PYBIND_INSTANTIATE_AS("trace", einsums::GeneralRuntimeTensor<double,               std::allocator<double>>,               einsums::GeneralRuntimeTensor<double,               std::allocator<double>>)
EINSUMS_PYBIND_INSTANTIATE_AS("trace", einsums::GeneralRuntimeTensor<std::complex<float>,  std::allocator<std::complex<float>>>,  einsums::GeneralRuntimeTensor<std::complex<float>,  std::allocator<std::complex<float>>>)
EINSUMS_PYBIND_INSTANTIATE_AS("trace", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
    // clang-format on
    void trace_python(ResultType *result, AType const &A) {
    using T = typename AType::ValueType;

    if (result->size() < 1) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "cg::trace: result tensor must have at least one element");
    }
    if (A.rank() != 2) {
        EINSUMS_THROW_EXCEPTION(rank_error, "cg::trace: input must be rank-2; got rank {}.", A.rank());
    }
    if (A.dim(0) != A.dim(1)) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "cg::trace: input must be square");
    }

    auto compute = [](AType const &a) -> T {
        T sum = T{};
        for (size_t i = 0; i < a.dim(0); ++i)
            sum += a(i, i);
        return sum;
    };

    auto &ctx = CaptureContext::current();
    if (!ctx.is_capturing()) {
        result->data()[0] = compute(A);
        return;
    }

    auto [a_id, a_slot] = ctx.get_slot(A);
    auto [r_id, r_slot] = ctx.get_slot(*result);

    auto executor = [a_slot, r_slot, compute]() {
        auto *r_ptr      = static_cast<ResultType *>(r_slot->ptr);
        r_ptr->data()[0] = compute(*static_cast<AType const *>(a_slot->ptr));
    };

    ctx.record(OpKind::Trace, "trace", {a_id}, {r_id}, std::move(executor));
}

// ─────────────────────────────────────────────────────────────────────────────
// symm_gemm: C = op(B)^T * op(A) * op(B)
// ─────────────────────────────────────────────────────────────────────────────

template <bool TransA, bool TransB, RuntimeRankTensorConcept AType, RuntimeRankTensorConcept BType, RuntimeRankTensorConcept CType>
    requires requires {
        requires InSamePlace<AType, BType, CType>;
        requires SameUnderlying<AType, BType, CType>;
    }
// clang-format off
EINSUMS_PYBIND_EXPOSE
EINSUMS_PYBIND_MODULE("linalg")
EINSUMS_PYBIND_TEMPLATE_KWARGS("trans_a", "trans_b")
EINSUMS_PYBIND_INSTANTIATE_BOOLS("symm_gemm", einsums::GeneralRuntimeTensor<float,                std::allocator<float>>,                einsums::GeneralRuntimeTensor<float,                std::allocator<float>>,                einsums::GeneralRuntimeTensor<float,                std::allocator<float>>)
EINSUMS_PYBIND_INSTANTIATE_BOOLS("symm_gemm", einsums::GeneralRuntimeTensor<double,               std::allocator<double>>,               einsums::GeneralRuntimeTensor<double,               std::allocator<double>>,               einsums::GeneralRuntimeTensor<double,               std::allocator<double>>)
EINSUMS_PYBIND_INSTANTIATE_BOOLS("symm_gemm", einsums::GeneralRuntimeTensor<std::complex<float>,  std::allocator<std::complex<float>>>,  einsums::GeneralRuntimeTensor<std::complex<float>,  std::allocator<std::complex<float>>>,  einsums::GeneralRuntimeTensor<std::complex<float>,  std::allocator<std::complex<float>>>)
EINSUMS_PYBIND_INSTANTIATE_BOOLS("symm_gemm", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
    // clang-format on
    void symm_gemm(AType const &A, BType const &B, CType *C) {
    if (A.rank() != 2 || B.rank() != 2 || C->rank() != 2) {
        EINSUMS_THROW_EXCEPTION(rank_error, "cg::symm_gemm requires rank-2 tensors; got ranks {}, {}, {}.", A.rank(), B.rank(), C->rank());
    }
    auto &ctx = CaptureContext::current();
    if (!ctx.is_capturing()) {
        linear_algebra::symm_gemm<TransA, TransB>(A, B, C);
        return;
    }
    auto [a_id, a_slot] = ctx.get_slot(A);
    auto [b_id, b_slot] = ctx.get_slot(B);
    auto [c_id, c_slot] = ctx.get_slot(*C);
    auto executor       = [a_slot, b_slot, c_slot]() {
        linear_algebra::symm_gemm<TransA, TransB>(*static_cast<AType const *>(a_slot->ptr), *static_cast<BType const *>(b_slot->ptr),
                                                  static_cast<CType *>(c_slot->ptr));
    };
    ctx.record(OpKind::SymmGemm, "symm_gemm", {a_id, b_id}, {c_id}, std::move(executor));
}

// Original compile-time-rank symm_gemm — kept under a different signature
// for legacy C++ callers; the runtime-rank form above is what the Python
// bindings target.
template <bool TransA, bool TransB, MatrixConcept AType, MatrixConcept BType, MatrixConcept CType>
    requires requires {
        requires InSamePlace<AType, BType, CType>;
        requires SameUnderlying<AType, BType, CType>;
    }
void symm_gemm(AType const &A, BType const &B, CType *C) {
    auto &ctx = CaptureContext::current();
    if (!ctx.is_capturing()) {
        linear_algebra::symm_gemm<TransA, TransB>(A, B, C);
        return;
    }

    auto [a_id, a_slot] = ctx.get_slot(A);
    auto [b_id, b_slot] = ctx.get_slot(B);
    auto [c_id, c_slot] = ctx.get_slot(*C);

    auto executor = [a_slot, b_slot, c_slot]() {
        linear_algebra::symm_gemm<TransA, TransB>(*static_cast<AType const *>(a_slot->ptr), *static_cast<BType const *>(b_slot->ptr),
                                                  static_cast<CType *>(c_slot->ptr));
    };

    ctx.record(OpKind::SymmGemm, "symm_gemm", {a_id, b_id}, {c_id}, std::move(executor));
}

// ═══════════════════════════════════════════════════════════════════════════
// LAPACK-level operations (return-value)
//
// These eagerly execute during capture so that returned tensors exist for
// subsequent captured operations to reference. They are recorded as nodes
// so that on replay they re-execute.
// ═══════════════════════════════════════════════════════════════════════════

// ─────────────────────────────────────────────────────────────────────────────
// syev (in-place form): eigendecompose A, store eigenvalues in W
// ─────────────────────────────────────────────────────────────────────────────

template <bool ComputeEigenvectors = true, MatrixConcept AType, VectorConcept WType>
    requires requires {
        requires InSamePlace<AType, WType>;
        requires SameUnderlying<AType, WType>;
        requires !Complex<AType>;
    }
void syev(AType *A, WType *W) {
    auto &ctx = CaptureContext::current();
    if (!ctx.is_capturing()) {
        linear_algebra::syev<ComputeEigenvectors>(A, W);
        return;
    }

    auto [a_id, a_slot] = ctx.get_slot(*A);
    auto [w_id, w_slot] = ctx.get_slot(*W);

    auto executor = [a_slot, w_slot]() {
        linear_algebra::syev<ComputeEigenvectors>(static_cast<AType *>(a_slot->ptr), static_cast<WType *>(w_slot->ptr));
    };

    ctx.record(OpKind::Syev, "syev", {a_id}, {a_id, w_id}, std::move(executor));
}

/// Real symmetric eigendecomposition (in-place): ``A = V * diag(W) * V^T``.
///
/// On return, when ``compute_eigenvectors=True`` (the default), ``A``
/// holds the eigenvectors as columns; ``W`` holds the eigenvalues in
/// ascending order. ``A`` must be rank 2 and square; ``W`` must be
/// rank 1 with size ``A.dim(0)``. For the returning form (allocates
/// fresh outputs) see ``syev_eig``.
template <bool ComputeEigenvectors = true, RuntimeRankTensorConcept AType, RuntimeRankTensorConcept WType>
    requires(InSamePlace<AType, WType> && SameUnderlying<AType, WType> && !Complex<AType>)
// clang-format off
EINSUMS_PYBIND_EXPOSE
EINSUMS_PYBIND_MODULE("linalg")
EINSUMS_PYBIND_TEMPLATE_KWARGS("compute_eigenvectors")
EINSUMS_PYBIND_INSTANTIATE_BOOLS("syev", einsums::GeneralRuntimeTensor<float,  std::allocator<float>>,  einsums::GeneralRuntimeTensor<float,  std::allocator<float>>)
EINSUMS_PYBIND_INSTANTIATE_BOOLS("syev", einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::GeneralRuntimeTensor<double, std::allocator<double>>)
    // clang-format on
    void syev(AType *A, WType *W) {
    if (A->rank() != 2 || W->rank() != 1) {
        EINSUMS_THROW_EXCEPTION(rank_error, "cg::syev requires A rank-2 and W rank-1; got {}, {}.", A->rank(), W->rank());
    }

    auto &ctx = CaptureContext::current();
    if (!ctx.is_capturing()) {
        linear_algebra::syev<ComputeEigenvectors>(A, W);
        return;
    }

    auto [a_id, a_slot] = ctx.get_slot(*A);
    auto [w_id, w_slot] = ctx.get_slot(*W);

    auto executor = [a_slot, w_slot]() {
        linear_algebra::syev<ComputeEigenvectors>(static_cast<AType *>(a_slot->ptr), static_cast<WType *>(w_slot->ptr));
    };

    ctx.record(OpKind::Syev, "syev", {a_id}, {a_id, w_id}, std::move(executor));
}

// ─────────────────────────────────────────────────────────────────────────────
// syev (returning form): returns (eigenvectors, eigenvalues)
// NOTE: Not supported during capture. Use the in-place form syev(&A, &W) instead.
// ─────────────────────────────────────────────────────────────────────────────

template <bool ComputeEigenvectors = true, MatrixConcept AType>
    requires(NotComplex<AType>)
auto syev(AType const &A) -> std::tuple<RemoveViewT<AType>, BasicTensorLike<AType, typename AType::ValueType, 1>> {
    if (CaptureContext::current().is_capturing()) {
        EINSUMS_THROW_EXCEPTION(std::logic_error, "cg::syev(A) returning form cannot be used during graph capture. "
                                                  "Use the in-place form cg::syev(&A, &W) with pre-allocated tensors instead.");
    }
    return linear_algebra::syev<ComputeEigenvectors>(A);
}

/// Real symmetric eigendecomposition (returning): ``(V, W) = syev_eig(A)``.
///
/// Allocates fresh tensors for the eigenvectors and eigenvalues and
/// returns them as a tuple. Cannot be used during graph capture; for
/// the in-graph form use the in-place ``cg::syev(A, W)`` with
/// pre-allocated outputs. ``A`` is left unmodified.
template <bool ComputeEigenvectors = true, RuntimeRankTensorConcept AType>
    requires NotComplex<AType>
// clang-format off
EINSUMS_PYBIND_EXPOSE
EINSUMS_PYBIND_MODULE("linalg")
EINSUMS_PYBIND_TEMPLATE_KWARGS("compute_eigenvectors")
EINSUMS_PYBIND_INSTANTIATE_BOOLS("syev_eig", einsums::GeneralRuntimeTensor<float,  std::allocator<float>>)
EINSUMS_PYBIND_INSTANTIATE_BOOLS("syev_eig", einsums::GeneralRuntimeTensor<double, std::allocator<double>>)
    // clang-format on
    std::tuple<einsums::GeneralRuntimeTensor<typename AType::ValueType, std::allocator<typename AType::ValueType>>,
               einsums::GeneralRuntimeTensor<typename AType::ValueType, std::allocator<typename AType::ValueType>>> syev_eig(AType const
                                                                                                                                 &A) {
    if (CaptureContext::current().is_capturing()) {
        EINSUMS_THROW_EXCEPTION(std::logic_error, "cg::syev(A) returning form cannot be used during graph capture. "
                                                  "Use the in-place form cg::syev(&A, &W) instead.");
    }
    if (A.rank() != 2 || A.dim(0) != A.dim(1)) {
        EINSUMS_THROW_EXCEPTION(rank_error, "cg::syev requires square rank-2 input; got dims ({}, {}).", A.rank() >= 1 ? A.dim(0) : 0,
                                A.rank() >= 2 ? A.dim(1) : 0);
    }

    using T  = typename AType::ValueType;
    using RT = einsums::GeneralRuntimeTensor<T, std::allocator<T>>;
    RT a     = A;
    RT w{"eigenvalues", std::vector<size_t>{A.dim(0)}};
    syev<ComputeEigenvectors>(&a, &w);
    return std::make_tuple(std::move(a), std::move(w));
}

// ─────────────────────────────────────────────────────────────────────────────
// heev: Hermitian eigendecomposition (in-place)
// ─────────────────────────────────────────────────────────────────────────────

template <bool ComputeEigenvectors = true, MatrixConcept AType, VectorConcept WType>
    requires requires {
        requires InSamePlace<AType, WType>;
        requires Complex<AType>;
        requires NotComplex<WType>;
        requires std::is_same_v<typename WType::ValueType, RemoveComplexT<typename AType::ValueType>>;
    }
void heev(AType *A, WType *W) {
    auto &ctx = CaptureContext::current();
    if (!ctx.is_capturing()) {
        linear_algebra::heev<ComputeEigenvectors>(A, W);
        return;
    }

    auto [a_id, a_slot] = ctx.get_slot(*A);
    auto [w_id, w_slot] = ctx.get_slot(*W);

    auto executor = [a_slot, w_slot]() {
        linear_algebra::heev<ComputeEigenvectors>(static_cast<AType *>(a_slot->ptr), static_cast<WType *>(w_slot->ptr));
    };
    ctx.record(OpKind::Heev, "heev", {a_id}, {a_id, w_id}, std::move(executor));
}

/// Hermitian eigendecomposition (in-place): ``A = V * diag(W) * V^H``.
///
/// Complex analogue of ``syev``. On return ``A`` holds eigenvectors as
/// columns (when ``compute_eigenvectors=True``); ``W`` holds the
/// eigenvalues in ascending order. ``W`` is real even though ``A`` is
/// complex.
template <bool ComputeEigenvectors = true, RuntimeRankTensorConcept AType, RuntimeRankTensorConcept WType>
    requires(InSamePlace<AType, WType> && Complex<AType> && NotComplex<WType> &&
             std::is_same_v<typename WType::ValueType, RemoveComplexT<typename AType::ValueType>>)
// clang-format off
EINSUMS_PYBIND_EXPOSE
EINSUMS_PYBIND_MODULE("linalg")
EINSUMS_PYBIND_TEMPLATE_KWARGS("compute_eigenvectors")
EINSUMS_PYBIND_INSTANTIATE_BOOLS("heev", einsums::GeneralRuntimeTensor<std::complex<float>,  std::allocator<std::complex<float>>>,  einsums::GeneralRuntimeTensor<float,  std::allocator<float>>)
EINSUMS_PYBIND_INSTANTIATE_BOOLS("heev", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::GeneralRuntimeTensor<double, std::allocator<double>>)
    // clang-format on
    void heev(AType *A, WType *W) {
    if (A->rank() != 2 || W->rank() != 1) {
        EINSUMS_THROW_EXCEPTION(rank_error, "cg::heev requires A rank-2 and W rank-1; got {}, {}.", A->rank(), W->rank());
    }

    auto &ctx = CaptureContext::current();
    if (!ctx.is_capturing()) {
        linear_algebra::heev<ComputeEigenvectors>(A, W);
        return;
    }

    auto [a_id, a_slot] = ctx.get_slot(*A);
    auto [w_id, w_slot] = ctx.get_slot(*W);

    auto executor = [a_slot, w_slot]() {
        linear_algebra::heev<ComputeEigenvectors>(static_cast<AType *>(a_slot->ptr), static_cast<WType *>(w_slot->ptr));
    };
    ctx.record(OpKind::Heev, "heev", {a_id}, {a_id, w_id}, std::move(executor));
}

// ─────────────────────────────────────────────────────────────────────────────
// gesv: solve AX = B
// ─────────────────────────────────────────────────────────────────────────────

template <MatrixConcept AType, TensorConcept BType>
    requires requires {
        requires SameUnderlying<AType, BType>;
        requires MatrixConcept<BType> || VectorConcept<BType>;
    }
auto gesv(AType *A, BType *B) -> int {
    auto &ctx = CaptureContext::current();
    if (!ctx.is_capturing()) {
        return linear_algebra::gesv(A, B);
    }

    auto [a_id, a_slot] = ctx.get_slot(*A);
    auto [b_id, b_slot] = ctx.get_slot(*B);

    auto executor = [a_slot, b_slot]() {
        std::ignore = linear_algebra::gesv(static_cast<AType *>(a_slot->ptr), static_cast<BType *>(b_slot->ptr));
    };
    ctx.record(OpKind::Gesv, "gesv", {a_id, b_id}, {a_id, b_id}, std::move(executor));

    return 0; // Return value not meaningful during capture
}

/// Solve the linear system ``A * X = B`` in place.
///
/// On return ``B`` holds ``X`` and ``A`` holds its LU factorization.
/// ``B`` may be rank 1 (single right-hand side) or rank 2 (multiple
/// right-hand-side columns). Returns the LAPACK info code: 0 on
/// success, positive ``i`` if ``A`` is singular at row ``i``.
template <RuntimeRankTensorConcept AType, RuntimeRankTensorConcept BType>
    requires SameUnderlying<AType, BType>
// clang-format off
EINSUMS_PYBIND_EXPOSE
EINSUMS_PYBIND_MODULE("linalg")
EINSUMS_PYBIND_INSTANTIATE_AS("gesv", einsums::GeneralRuntimeTensor<float,                std::allocator<float>>,                einsums::GeneralRuntimeTensor<float,                std::allocator<float>>)
EINSUMS_PYBIND_INSTANTIATE_AS("gesv", einsums::GeneralRuntimeTensor<double,               std::allocator<double>>,               einsums::GeneralRuntimeTensor<double,               std::allocator<double>>)
EINSUMS_PYBIND_INSTANTIATE_AS("gesv", einsums::GeneralRuntimeTensor<std::complex<float>,  std::allocator<std::complex<float>>>,  einsums::GeneralRuntimeTensor<std::complex<float>,  std::allocator<std::complex<float>>>)
EINSUMS_PYBIND_INSTANTIATE_AS("gesv", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
    // clang-format on
    auto gesv(AType *A, BType *B) -> int {
    if (A->rank() != 2 || (B->rank() != 1 && B->rank() != 2)) {
        EINSUMS_THROW_EXCEPTION(rank_error, "cg::gesv requires A rank-2 and B rank-1 or rank-2; got {}, {}.", A->rank(), B->rank());
    }

    auto &ctx = CaptureContext::current();
    if (!ctx.is_capturing()) {
        return linear_algebra::gesv(A, B);
    }

    auto [a_id, a_slot] = ctx.get_slot(*A);
    auto [b_id, b_slot] = ctx.get_slot(*B);

    auto executor = [a_slot, b_slot]() {
        std::ignore = linear_algebra::gesv(static_cast<AType *>(a_slot->ptr), static_cast<BType *>(b_slot->ptr));
    };
    ctx.record(OpKind::Gesv, "gesv", {a_id, b_id}, {a_id, b_id}, std::move(executor));

    return 0;
}

// ─────────────────────────────────────────────────────────────────────────────
// invert: in-place matrix inverse
// ─────────────────────────────────────────────────────────────────────────────

template <MatrixConcept AType>
void invert(AType *A) {
    auto &ctx = CaptureContext::current();
    if (!ctx.is_capturing()) {
        linear_algebra::invert(A);
        return;
    }

    auto [a_id, a_slot] = ctx.get_slot(*A);

    auto executor = [a_slot]() { linear_algebra::invert(static_cast<AType *>(a_slot->ptr)); };
    ctx.record(OpKind::Invert, "invert", {a_id}, {a_id}, std::move(executor));
}

/// In-place matrix inverse: ``A := A^-1``.
///
/// ``A`` must be rank 2 and square. Internally calls ``getrf`` followed
/// by ``getri``; raises ``rank_error`` if the input rank is wrong and
/// the LAPACK kernel raises if ``A`` is singular.
template <RuntimeRankTensorConcept AType>
// clang-format off
EINSUMS_PYBIND_EXPOSE
EINSUMS_PYBIND_MODULE("linalg")
EINSUMS_PYBIND_INSTANTIATE_AS("invert", einsums::GeneralRuntimeTensor<float,                std::allocator<float>>)
EINSUMS_PYBIND_INSTANTIATE_AS("invert", einsums::GeneralRuntimeTensor<double,               std::allocator<double>>)
EINSUMS_PYBIND_INSTANTIATE_AS("invert", einsums::GeneralRuntimeTensor<std::complex<float>,  std::allocator<std::complex<float>>>)
EINSUMS_PYBIND_INSTANTIATE_AS("invert", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
    // clang-format on
    void invert(AType *A) {
    if (A->rank() != 2) {
        EINSUMS_THROW_EXCEPTION(rank_error, "cg::invert requires rank-2 tensor; got rank {}.", A->rank());
    }

    auto &ctx = CaptureContext::current();
    if (!ctx.is_capturing()) {
        linear_algebra::invert(A);
        return;
    }

    auto [a_id, a_slot] = ctx.get_slot(*A);

    auto executor = [a_slot]() { linear_algebra::invert(static_cast<AType *>(a_slot->ptr)); };
    ctx.record(OpKind::Invert, "invert", {a_id}, {a_id}, std::move(executor));
}

// ─────────────────────────────────────────────────────────────────────────────
// svd: singular value decomposition (returning)
// ─────────────────────────────────────────────────────────────────────────────

template <MatrixConcept AType>
auto svd(AType const &A) {
    if (CaptureContext::current().is_capturing()) {
        EINSUMS_THROW_EXCEPTION(std::logic_error, "cg::svd(A) returning form cannot be used during graph capture.");
    }
    return linear_algebra::svd(A);
}

/// Singular value decomposition (returning): ``A = U * diag(S) * Vt``.
///
/// Returns the tuple ``(U, S, Vt)``. ``S`` is real even for complex
/// inputs. ``U`` and ``Vt`` are always present — the user can post-
/// filter for the "no vectors" case if needed. Cannot be used during
/// graph capture (returns by value); ``A`` is left unmodified.
template <RuntimeRankTensorConcept AType>
// clang-format off
EINSUMS_PYBIND_EXPOSE
EINSUMS_PYBIND_MODULE("linalg")
EINSUMS_PYBIND_INSTANTIATE_AS("svd", einsums::GeneralRuntimeTensor<float,                std::allocator<float>>)
EINSUMS_PYBIND_INSTANTIATE_AS("svd", einsums::GeneralRuntimeTensor<double,               std::allocator<double>>)
EINSUMS_PYBIND_INSTANTIATE_AS("svd", einsums::GeneralRuntimeTensor<std::complex<float>,  std::allocator<std::complex<float>>>)
EINSUMS_PYBIND_INSTANTIATE_AS("svd", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
    // clang-format on
    auto svd(AType const &A) -> std::tuple<
        einsums::GeneralRuntimeTensor<typename AType::ValueType, std::allocator<typename AType::ValueType>>,
        einsums::GeneralRuntimeTensor<RemoveComplexT<typename AType::ValueType>, std::allocator<RemoveComplexT<typename AType::ValueType>>>,
        einsums::GeneralRuntimeTensor<typename AType::ValueType, std::allocator<typename AType::ValueType>>> {
    if (CaptureContext::current().is_capturing()) {
        EINSUMS_THROW_EXCEPTION(std::logic_error, "cg::svd(A) returning form cannot be used during graph capture.");
    }
    if (A.rank() != 2) {
        EINSUMS_THROW_EXCEPTION(rank_error, "cg::svd requires rank-2 input; got rank {}.", A.rank());
    }

    using T  = typename AType::ValueType;
    using R  = RemoveComplexT<T>;
    using RT = einsums::GeneralRuntimeTensor<T, std::allocator<T>>;
    using RR = einsums::GeneralRuntimeTensor<R, std::allocator<R>>;

    Tensor<T, 2> a_static{A.name(), A.dim(0), A.dim(1)};
    std::memcpy(a_static.data(), A.data(), A.size() * sizeof(T));

    auto [U_opt, S_static, Vt_opt] = linear_algebra::svd(a_static);
    if (!U_opt || !Vt_opt) {
        EINSUMS_THROW_EXCEPTION(std::runtime_error, "cg::svd: SVD computation did not return U and V^T as expected.");
    }
    return std::make_tuple(RT{std::move(*U_opt)}, RR{std::move(S_static)}, RT{std::move(*Vt_opt)});
}

// ─────────────────────────────────────────────────────────────────────────────
// svd_dd: SVD with divide-and-conquer (returning)
// ─────────────────────────────────────────────────────────────────────────────

template <MatrixConcept AType>
auto svd_dd(AType const &A, linear_algebra::Vectors job = linear_algebra::Vectors::ALL) {
    if (CaptureContext::current().is_capturing()) {
        EINSUMS_THROW_EXCEPTION(std::logic_error, "cg::svd_dd(A) returning form cannot be used during graph capture.");
    }
    return linear_algebra::svd_dd(A, job);
}

/// Divide-and-conquer singular value decomposition (returning): ``A = U * diag(S) * Vt``.
///
/// Faster than ``svd`` for large matrices; uses LAPACK's ``gesdd`` driver.
/// Returns ``(U, S, Vt)`` like ``svd``. ``job`` controls whether the
/// singular vectors are computed (``ALL``, ``SOME``, or ``NONE``).
/// Cannot be used during graph capture (returns by value); ``A`` is left
/// unmodified.
template <RuntimeRankTensorConcept AType>
// clang-format off
EINSUMS_PYBIND_EXPOSE
EINSUMS_PYBIND_MODULE("linalg")
EINSUMS_PYBIND_INSTANTIATE_AS("svd_dd", einsums::GeneralRuntimeTensor<float,                std::allocator<float>>)
EINSUMS_PYBIND_INSTANTIATE_AS("svd_dd", einsums::GeneralRuntimeTensor<double,               std::allocator<double>>)
EINSUMS_PYBIND_INSTANTIATE_AS("svd_dd", einsums::GeneralRuntimeTensor<std::complex<float>,  std::allocator<std::complex<float>>>)
EINSUMS_PYBIND_INSTANTIATE_AS("svd_dd", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
    // clang-format on
    auto svd_dd(AType const &A, linear_algebra::Vectors job = linear_algebra::Vectors::ALL) -> std::tuple<
        einsums::GeneralRuntimeTensor<typename AType::ValueType, std::allocator<typename AType::ValueType>>,
        einsums::GeneralRuntimeTensor<RemoveComplexT<typename AType::ValueType>, std::allocator<RemoveComplexT<typename AType::ValueType>>>,
        einsums::GeneralRuntimeTensor<typename AType::ValueType, std::allocator<typename AType::ValueType>>> {
    if (CaptureContext::current().is_capturing()) {
        EINSUMS_THROW_EXCEPTION(std::logic_error, "cg::svd_dd(A) returning form cannot be used during graph capture.");
    }
    if (A.rank() != 2) {
        EINSUMS_THROW_EXCEPTION(rank_error, "cg::svd_dd requires rank-2 input; got rank {}.", A.rank());
    }

    using T  = typename AType::ValueType;
    using R  = RemoveComplexT<T>;
    using RT = einsums::GeneralRuntimeTensor<T, std::allocator<T>>;
    using RR = einsums::GeneralRuntimeTensor<R, std::allocator<R>>;

    Tensor<T, 2> a_static{A.name(), A.dim(0), A.dim(1)};
    std::memcpy(a_static.data(), A.data(), A.size() * sizeof(T));

    auto [U_opt, S_static, Vt_opt] = linear_algebra::svd_dd(a_static, job);
    if (!U_opt || !Vt_opt) {
        EINSUMS_THROW_EXCEPTION(std::runtime_error, "cg::svd_dd: SVD computation did not return U and V^T as expected.");
    }
    return std::make_tuple(RT{std::move(*U_opt)}, RR{std::move(S_static)}, RT{std::move(*Vt_opt)});
}

// ─────────────────────────────────────────────────────────────────────────────
// truncated_svd
// ─────────────────────────────────────────────────────────────────────────────

template <MatrixConcept AType>
auto truncated_svd(AType const &A, size_t k) {
    if (CaptureContext::current().is_capturing()) {
        EINSUMS_THROW_EXCEPTION(std::logic_error, "cg::truncated_svd(A, k) returning form cannot be used during graph capture.");
    }
    return linear_algebra::truncated_svd(A, k);
}

/// Rank-``k`` truncated SVD (returning): keeps the top ``k`` singular
/// triples of ``A``. Returns ``(U_k, S_k, Vt_k)`` with ``U_k`` of shape
/// ``(m, k)``, ``S_k`` of shape ``(k,)``, and ``Vt_k`` of shape ``(k, n)``.
///
/// Randomized algorithm with over-sampling factor 5 — requires
/// ``A.dim(0) >= k + 5``. Smaller inputs raise ``IndexError`` from the
/// projection step. Results are approximate; expect small drift versus a
/// full ``svd``.
///
/// Cannot be used during graph capture (returns by value); ``A`` is left
/// unmodified.
template <RuntimeRankTensorConcept AType>
// clang-format off
EINSUMS_PYBIND_EXPOSE
EINSUMS_PYBIND_MODULE("linalg")
EINSUMS_PYBIND_INSTANTIATE_AS("truncated_svd", einsums::GeneralRuntimeTensor<float,                std::allocator<float>>)
EINSUMS_PYBIND_INSTANTIATE_AS("truncated_svd", einsums::GeneralRuntimeTensor<double,               std::allocator<double>>)
EINSUMS_PYBIND_INSTANTIATE_AS("truncated_svd", einsums::GeneralRuntimeTensor<std::complex<float>,  std::allocator<std::complex<float>>>)
EINSUMS_PYBIND_INSTANTIATE_AS("truncated_svd", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
    // clang-format on
    auto truncated_svd(AType const &A, size_t k) -> std::tuple<
        einsums::GeneralRuntimeTensor<typename AType::ValueType, std::allocator<typename AType::ValueType>>,
        einsums::GeneralRuntimeTensor<RemoveComplexT<typename AType::ValueType>, std::allocator<RemoveComplexT<typename AType::ValueType>>>,
        einsums::GeneralRuntimeTensor<typename AType::ValueType, std::allocator<typename AType::ValueType>>> {
    if (CaptureContext::current().is_capturing()) {
        EINSUMS_THROW_EXCEPTION(std::logic_error, "cg::truncated_svd(A, k) returning form cannot be used during graph capture.");
    }
    if (A.rank() != 2) {
        EINSUMS_THROW_EXCEPTION(rank_error, "cg::truncated_svd requires rank-2 input; got rank {}.", A.rank());
    }

    using T  = typename AType::ValueType;
    using R  = RemoveComplexT<T>;
    using RT = einsums::GeneralRuntimeTensor<T, std::allocator<T>>;
    using RR = einsums::GeneralRuntimeTensor<R, std::allocator<R>>;

    Tensor<T, 2> a_static{A.name(), A.dim(0), A.dim(1)};
    std::memcpy(a_static.data(), A.data(), A.size() * sizeof(T));

    auto [U_static, S_static, Vt_static] = linear_algebra::truncated_svd(a_static, k);
    return std::make_tuple(RT{std::move(U_static)}, RR{std::move(S_static)}, RT{std::move(Vt_static)});
}

// ─────────────────────────────────────────────────────────────────────────────
// truncated_syev
// ─────────────────────────────────────────────────────────────────────────────

template <MatrixConcept AType>
    requires(NotComplex<AType>)
auto truncated_syev(AType const &A, size_t k) {
    if (CaptureContext::current().is_capturing()) {
        EINSUMS_THROW_EXCEPTION(std::logic_error, "cg::truncated_syev(A, k) returning form cannot be used during graph capture.");
    }
    return linear_algebra::truncated_syev(A, k);
}

/// Rank-``k`` truncated symmetric eigendecomposition (returning): keeps the
/// top ``k`` eigenpairs of a real symmetric ``A``. Returns
/// ``(eigenvectors, eigenvalues)`` where eigenvectors has shape ``(n, k)``
/// and eigenvalues has shape ``(k,)``.
///
/// Randomized algorithm with over-sampling factor 5 — requires
/// ``A.dim(0) >= k + 5``. Smaller inputs raise ``IndexError`` from the
/// projection step. Results are approximate top-``k`` eigenpairs; expect
/// small drift versus a full ``syev``, especially for tightly clustered
/// eigenvalues.
///
/// Cannot be used during graph capture (returns by value); ``A`` is left
/// unmodified.
template <RuntimeRankTensorConcept AType>
    requires(NotComplex<AType>)
// clang-format off
EINSUMS_PYBIND_EXPOSE
EINSUMS_PYBIND_MODULE("linalg")
EINSUMS_PYBIND_INSTANTIATE_AS("truncated_syev", einsums::GeneralRuntimeTensor<float,  std::allocator<float>>)
EINSUMS_PYBIND_INSTANTIATE_AS("truncated_syev", einsums::GeneralRuntimeTensor<double, std::allocator<double>>)
    // clang-format on
    auto truncated_syev(AType const &A, size_t k)
        -> std::tuple<einsums::GeneralRuntimeTensor<typename AType::ValueType, std::allocator<typename AType::ValueType>>,
                      einsums::GeneralRuntimeTensor<typename AType::ValueType, std::allocator<typename AType::ValueType>>> {
    if (CaptureContext::current().is_capturing()) {
        EINSUMS_THROW_EXCEPTION(std::logic_error, "cg::truncated_syev(A, k) returning form cannot be used during graph capture.");
    }
    if (A.rank() != 2) {
        EINSUMS_THROW_EXCEPTION(rank_error, "cg::truncated_syev requires rank-2 input; got rank {}.", A.rank());
    }

    using T  = typename AType::ValueType;
    using RT = einsums::GeneralRuntimeTensor<T, std::allocator<T>>;

    Tensor<T, 2> a_static{A.name(), A.dim(0), A.dim(1)};
    std::memcpy(a_static.data(), A.data(), A.size() * sizeof(T));

    auto [V_static, W_static] = linear_algebra::truncated_syev(a_static, k);
    return std::make_tuple(RT{std::move(V_static)}, RT{std::move(W_static)});
}

// ─────────────────────────────────────────────────────────────────────────────
// qr: QR decomposition (returning)
// ─────────────────────────────────────────────────────────────────────────────

template <MatrixConcept AType>
auto qr(AType const &A) {
    if (CaptureContext::current().is_capturing()) {
        EINSUMS_THROW_EXCEPTION(std::logic_error, "cg::qr(A) returning form cannot be used during graph capture.");
    }
    return linear_algebra::qr(A);
}

/// QR decomposition (returning): ``A = Q * R``.
///
/// Returns the tuple ``(Q, R)`` where ``Q`` is orthogonal (or unitary
/// for complex inputs) and ``R`` is upper-triangular. Cannot be used
/// during graph capture; ``A`` is left unmodified.
template <RuntimeRankTensorConcept AType>
// clang-format off
EINSUMS_PYBIND_EXPOSE
EINSUMS_PYBIND_MODULE("linalg")
EINSUMS_PYBIND_INSTANTIATE_AS("qr", einsums::GeneralRuntimeTensor<float,                std::allocator<float>>)
EINSUMS_PYBIND_INSTANTIATE_AS("qr", einsums::GeneralRuntimeTensor<double,               std::allocator<double>>)
EINSUMS_PYBIND_INSTANTIATE_AS("qr", einsums::GeneralRuntimeTensor<std::complex<float>,  std::allocator<std::complex<float>>>)
EINSUMS_PYBIND_INSTANTIATE_AS("qr", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
    // clang-format on
    auto qr(AType const &A)
        -> std::tuple<einsums::GeneralRuntimeTensor<typename AType::ValueType, std::allocator<typename AType::ValueType>>,
                      einsums::GeneralRuntimeTensor<typename AType::ValueType, std::allocator<typename AType::ValueType>>> {
    if (CaptureContext::current().is_capturing()) {
        EINSUMS_THROW_EXCEPTION(std::logic_error, "cg::qr(A) returning form cannot be used during graph capture.");
    }
    if (A.rank() != 2) {
        EINSUMS_THROW_EXCEPTION(rank_error, "cg::qr requires rank-2 input; got rank {}.", A.rank());
    }

    using T  = typename AType::ValueType;
    using RT = einsums::GeneralRuntimeTensor<T, std::allocator<T>>;

    Tensor<T, 2> a_static{A.name(), A.dim(0), A.dim(1)};
    std::memcpy(a_static.data(), A.data(), A.size() * sizeof(T));

    auto [Q_static, R_static] = linear_algebra::qr(a_static);
    return std::make_tuple(RT{std::move(Q_static)}, RT{std::move(R_static)});
}

// ─────────────────────────────────────────────────────────────────────────────
// pow: matrix power (returning)
// ─────────────────────────────────────────────────────────────────────────────

template <MatrixConcept AType>
auto pow(AType const &A, typename AType::ValueType alpha,
         typename AType::ValueType cutoff = std::numeric_limits<typename AType::ValueType>::epsilon()) -> RemoveViewT<AType> {
    if (CaptureContext::current().is_capturing()) {
        EINSUMS_THROW_EXCEPTION(std::logic_error, "cg::pow(A, alpha) returning form cannot be used during graph capture.");
    }
    return linear_algebra::pow(A, alpha, cutoff);
}

/// Matrix power: ``A^alpha`` via eigendecomposition.
///
/// Returns a freshly-allocated square matrix. ``alpha`` may be negative; an
/// optional ``cutoff`` zeros out eigenvalues whose magnitude is below
/// ``cutoff * max|eig|`` before exponentiation (guards against near-singular
/// inputs when ``alpha < 0``). Cannot be used during graph capture (returns
/// by value); ``A`` is left unmodified.
template <RuntimeRankTensorConcept AType>
    requires(NotComplex<AType>)
// clang-format off
EINSUMS_PYBIND_EXPOSE
EINSUMS_PYBIND_MODULE("linalg")
EINSUMS_PYBIND_INSTANTIATE_AS("pow", einsums::GeneralRuntimeTensor<float,  std::allocator<float>>)
EINSUMS_PYBIND_INSTANTIATE_AS("pow", einsums::GeneralRuntimeTensor<double, std::allocator<double>>)
    // clang-format on
    auto pow(AType const &A, typename AType::ValueType alpha,
             typename AType::ValueType cutoff = std::numeric_limits<typename AType::ValueType>::epsilon())
        -> einsums::GeneralRuntimeTensor<typename AType::ValueType, std::allocator<typename AType::ValueType>> {
    if (CaptureContext::current().is_capturing()) {
        EINSUMS_THROW_EXCEPTION(std::logic_error, "cg::pow(A, alpha) returning form cannot be used during graph capture.");
    }
    if (A.rank() != 2) {
        EINSUMS_THROW_EXCEPTION(rank_error, "cg::pow requires rank-2 input; got rank {}.", A.rank());
    }

    using T  = typename AType::ValueType;
    using RT = einsums::GeneralRuntimeTensor<T, std::allocator<T>>;

    Tensor<T, 2> a_static{A.name(), A.dim(0), A.dim(1)};
    std::memcpy(a_static.data(), A.data(), A.size() * sizeof(T));

    auto result_static = linear_algebra::pow(a_static, alpha, cutoff);
    return RT{std::move(result_static)};
}

// ─────────────────────────────────────────────────────────────────────────────
// det: matrix determinant (returning scalar)
// ─────────────────────────────────────────────────────────────────────────────

template <MatrixConcept AType>
auto det(AType const &A) -> typename AType::ValueType {
    if (CaptureContext::current().is_capturing()) {
        EINSUMS_THROW_EXCEPTION(std::logic_error, "cg::det(A) returning scalar cannot be used during graph capture.");
    }
    return linear_algebra::det(A);
}

/// Determinant of a square matrix (returning scalar).
///
/// Cannot be used during graph capture (returns by value, so there is
/// no destination slot). For complex inputs the determinant is itself
/// complex.
template <RuntimeRankTensorConcept AType>
// clang-format off
EINSUMS_PYBIND_EXPOSE
EINSUMS_PYBIND_MODULE("linalg")
EINSUMS_PYBIND_INSTANTIATE_AS("det", einsums::GeneralRuntimeTensor<float,                std::allocator<float>>)
EINSUMS_PYBIND_INSTANTIATE_AS("det", einsums::GeneralRuntimeTensor<double,               std::allocator<double>>)
EINSUMS_PYBIND_INSTANTIATE_AS("det", einsums::GeneralRuntimeTensor<std::complex<float>,  std::allocator<std::complex<float>>>)
EINSUMS_PYBIND_INSTANTIATE_AS("det", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
    // clang-format on
    auto det(AType const &A) -> typename AType::ValueType {
    if (CaptureContext::current().is_capturing()) {
        EINSUMS_THROW_EXCEPTION(std::logic_error, "cg::det(A) returning scalar cannot be used during graph capture.");
    }
    if (A.rank() != 2 || A.dim(0) != A.dim(1)) {
        EINSUMS_THROW_EXCEPTION(rank_error, "cg::det requires square rank-2 input; got rank {}.", A.rank());
    }

    using T                        = typename AType::ValueType;
    AType                     temp = A;
    BufferVector<blas::int_t> pivots;
    int const                 singular = linear_algebra::getrf(&temp, &pivots);
    if (singular > 0) {
        return T{0.0};
    }

    T   ret{1.0};
    int parity = 0;
    for (size_t i = 0; i < A.dim(0); ++i) {
        if (pivots[i] != static_cast<blas::int_t>(i + 1)) {
            ++parity;
        }
    }
    for (size_t i = 0; i < A.dim(0); ++i) {
        ret *= temp(i, i);
    }
    if ((parity & 1) != 0) {
        ret = -ret;
    }
    return ret;
}

// ═══════════════════════════════════════════════════════════════════════════
// String-based einsum API
//
// Uses string notation instead of compile-time index types.
// Supports both "ij <- ik ; kj" (arrow) and "ik;kj -> ij" (NumPy) formats.
// Single-char and multi-char indices auto-detected.
// Dispatches to DOT, GER, GEMV, GEMM, direct product, or generic fallback.
// ═══════════════════════════════════════════════════════════════════════════

namespace detail {

/// Build an EinsumDescriptor from a ParsedEinsumSpec.
inline EinsumDescriptor build_einsum_descriptor(ParsedEinsumSpec const &parsed, PrefactorScalar c_pf, PrefactorScalar ab_pf) {
    EinsumDescriptor desc;
    desc.c_prefactor         = c_pf;
    desc.ab_prefactor        = ab_pf;
    desc.spec.c_indices      = parsed.c_indices;
    desc.spec.a_indices      = parsed.a_indices;
    desc.spec.b_indices      = parsed.b_indices;
    desc.spec.link_indices   = parsed.link_indices();
    desc.spec.target_indices = parsed.target_indices();
    desc.spec.all_indices    = desc.spec.target_indices;
    desc.spec.all_indices.insert(desc.spec.all_indices.end(), desc.spec.link_indices.begin(), desc.spec.link_indices.end());
    return desc;
}

} // namespace detail

/**
 * @brief String-based einsum with explicit prefactors.
 *
 * Uses string notation instead of compile-time index types:
 * @code
 * cg::einsum("ij <- ik ; kj", 0.0, &C, 1.0, A, B);     // GEMM
 * cg::einsum("ij <- ki ; kj", 0.0, &C, 1.0, A, B);     // GEMM with transposed A
 * cg::einsum("i <- ij ; j", 0.0, &y, 1.0, A, x);       // GEMV
 * cg::einsum(" <- i ; i", 0.0, &dot, 1.0, x, y);        // DOT product
 * cg::einsum("ij <- i ; j", 0.0, &C, 1.0, x, y);       // GER (outer product)
 * cg::einsum("ij <- ij ; ij", 0.0, &C, 1.0, A, B);     // Element-wise (direct product)
 * cg::einsum("mu,nu <- mu,rho ; rho,nu", 0.0, &C, 1.0, A, B);  // Multi-char indices
 * @endcode
 *
 * During graph capture, records the operation as a node. Outside capture,
 * dispatches to the appropriate BLAS routine based on the contraction pattern.
 *
 * @tparam AType First input tensor type (must satisfy BasicTensorConcept).
 * @tparam BType Second input tensor type.
 * @tparam CType Output tensor type.
 * @param[in] spec Einsum specification string.
 * @param[in] c_pf Scalar prefactor for C accumulation: C = c_pf * C + ab_pf * contract(A, B).
 * @param[in,out] C Output tensor.
 * @param[in] ab_pf Scalar prefactor for the A*B contraction.
 * @param[in] A First input tensor.
 * @param[in] B Second input tensor.
 */
template <BasicTensorConcept AType, BasicTensorConcept BType, BasicTensorConcept CType>
    requires requires {
        requires std::is_same_v<typename AType::ValueType, typename BType::ValueType>;
        requires std::is_same_v<typename AType::ValueType, typename CType::ValueType>;
    }
void einsum(EinsumFormatString spec, typename AType::ValueType c_pf, CType *C, typename AType::ValueType ab_pf, AType const &A,
            BType const &B) {
    using T = typename AType::ValueType;

    // Operand rank ↔ spec consistency check. When the spec is a literal,
    // ``spec.counts`` is populated at consteval time and folds to compile-
    // time constants here; for typed tensors with a static ::Rank the whole
    // condition is a constant comparison and the throw-branch is dead-code-
    // eliminated. For runtime-rank tensors (RuntimeTensor) the check fires
    // against ``tensor.rank()``. Spec strings built at runtime — ``Python``
    // bindings, user input — leave ``counts.known == false`` and skip the
    // check entirely (matching the "compile-time when possible, silent
    // otherwise" policy).
    if (spec.counts.known) {
        std::size_t const a_rank = detail::tensor_rank(A);
        std::size_t const b_rank = detail::tensor_rank(B);
        std::size_t const c_rank = detail::tensor_rank(*C);
        if (a_rank != spec.counts.a) {
            EINSUMS_THROW_EXCEPTION(std::invalid_argument, "cg::einsum: operand A has rank {} but spec expects {} indices for A", a_rank,
                                    spec.counts.a);
        }
        if (b_rank != spec.counts.b) {
            EINSUMS_THROW_EXCEPTION(std::invalid_argument, "cg::einsum: operand B has rank {} but spec expects {} indices for B", b_rank,
                                    spec.counts.b);
        }
        // Scalar-output convention: an empty C operand in the spec
        // (e.g. ``" <- i ; i"`` for DOT) accepts either rank-0 or a rank-1
        // single-element tensor. Otherwise C's rank must equal the index count.
        bool const c_ok = (spec.counts.c == 0) ? (c_rank <= 1) : (c_rank == spec.counts.c);
        if (!c_ok) {
            EINSUMS_THROW_EXCEPTION(std::invalid_argument, "cg::einsum: operand C has rank {} but spec expects {} indices for C", c_rank,
                                    spec.counts.c);
        }
    }

    auto parse_result = parse_einsum_spec(static_cast<std::string_view>(spec));
    if (!parse_result) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "{}", parse_result.error().message);
    }
    auto &parsed = parse_result.value();

    auto &ctx = CaptureContext::current();
    if (!ctx.is_capturing()) {
        dispatch::string_einsum(parsed, c_pf, C, ab_pf, A, B);
        return;
    }

    // Capture mode with slots
    auto [a_id, a_slot] = ctx.get_slot(A);
    auto [b_id, b_slot] = ctx.get_slot(B);
    auto [c_id, c_slot] = ctx.get_slot(*C);

    auto params = ctx.graph()->create_params(c_pf, ab_pf);
    auto desc   = detail::build_einsum_descriptor(parsed, params->c_pf, params->ab_pf);

    // Runtime-mutable index state. Created once per einsum capture and
    // shared between the descriptor (for pass introspection / rewrite)
    // and the executor lambda. Seeded with the parsed indices and the
    // link set that `detail::build_einsum_descriptor` computes from
    // them — avoids recomputing link indices on every execute().
    auto indices = ctx.graph()->create_indices(parsed.a_indices, parsed.b_indices, parsed.c_indices, desc.spec.link_indices);
    desc.indices = indices;

    // BLAS-level batching hint. Only populated when the contraction is
    // a pure 2D GEMM pattern (rank-2 inputs/output, one link index) —
    // that's the shape `blas::gemm_batch` accepts. For other shapes the
    // hint stays null and GEMMBatching falls through. `trans_a`/`trans_b`
    // follow string_gemm's convention: 'T' when the link is the first
    // index of A (resp. last index of B), 'N' otherwise.
    if (detail::tensor_rank(A) == 2 && detail::tensor_rank(B) == 2 && detail::tensor_rank(*C) == 2) {
        if (parsed.a_indices.size() == 2 && parsed.b_indices.size() == 2 && parsed.c_indices.size() == 2 &&
            desc.spec.link_indices.size() == 1) {
            auto hint = std::make_shared<GemmHint>();
            if constexpr (std::is_same_v<T, float>)
                hint->scalar = BlasScalar::Float;
            else if constexpr (std::is_same_v<T, double>)
                hint->scalar = BlasScalar::Double;
            else if constexpr (std::is_same_v<T, std::complex<float>>)
                hint->scalar = BlasScalar::ComplexFloat;
            else if constexpr (std::is_same_v<T, std::complex<double>>)
                hint->scalar = BlasScalar::ComplexDouble;

            std::string const &link = desc.spec.link_indices[0];
            hint->trans_a           = (parsed.a_indices[0] == link) ? 'T' : 'N';
            hint->trans_b           = (parsed.b_indices[1] == link) ? 'T' : 'N';
            // m = rows of C, n = cols of C, k = link dim (taken from A)
            hint->m = static_cast<int>(C->dim(0));
            hint->n = static_cast<int>(C->dim(1));
            hint->k = static_cast<int>(hint->trans_a == 'N' ? A.dim(1) : A.dim(0));

            // Extractors capture AType/BType/CType so at call time they
            // can read .data() + .impl().get_lda() off the live tensor
            // (handles graph.rebind() — the slot's ptr points at the
            // current tensor). get_lda is on TensorImpl, not GeneralTensor.
            hint->extract_a = [a_slot]() -> std::pair<void const *, int> {
                auto const &a_ref = *static_cast<AType const *>(a_slot->ptr);
                return {static_cast<void const *>(a_ref.data()), static_cast<int>(a_ref.impl().get_lda())};
            };
            hint->extract_b = [b_slot]() -> std::pair<void const *, int> {
                auto const &b_ref = *static_cast<BType const *>(b_slot->ptr);
                return {static_cast<void const *>(b_ref.data()), static_cast<int>(b_ref.impl().get_lda())};
            };
            hint->extract_c = [c_slot]() -> std::pair<void *, int> {
                auto *c_ptr = static_cast<CType *>(c_slot->ptr);
                return {static_cast<void *>(c_ptr->data()), static_cast<int>(c_ptr->impl().get_lda())};
            };
            desc.gemm_hint = std::move(hint);
        }
    }

    // ────────────────────────────────────────────────────────────────────
    // Strided-batched GEMM fast path for 3D×3D→3D with a batch index.
    // ────────────────────────────────────────────────────────────────────
    //
    // If the einsum expresses a batched matrix multiply — a 3D tensor
    // where one index appears in A, B, AND C (the batch) and the other
    // three indices form a standard 2D GEMM pattern (target-A, link,
    // target-B) — we can collapse N per-batch 2D gemms into a single
    // `blas::gemm_batch` call at execute time. This is the same layout
    // `cublasDgemmStridedBatched` expects on GPU, so the descriptor
    // carries enough info to dispatch there too once the GPU backend is
    // wired up.
    //
    // Both conventions are supported so users don't have to transpose
    // their data to match some arbitrary choice:
    //   - Row-major tensors with batch(es) at the FIRST axes
    //     (e.g. "bij;bjk->bik" shape (B, M, K); or "abij;abjk->abik"
    //     shape (A, B, M, K)) — the ML/CUDA convention
    //   - Column-major tensors with batch(es) at the LAST axes
    //     (e.g. "ijb;jkb->ikb" shape (M, K, B); or "ijab;jkab->ikab"
    //     shape (M, K, A, B)) — Einsums's default layout
    //
    // Multiple batch indices (rank 4+) are flattened: N batch dims with
    // sizes (d1, d2, ..., dN) become a single effective batch of size
    // prod(di) with uniform stride equal to the product of the per-slice
    // 2D dims. This works as long as all batch indices appear in the
    // outermost contiguous region in memory, in the same relative order
    // across A, B, C — typical of how tensors carry "free" axes like
    // (head, layer, sample, fragment) through to the matmul.
    //
    // In either case the 2D slice at each flat batch index is a
    // contiguous block of memory. Arrangements that don't match (batches
    // at the wrong end for the layout, or reordered across operands)
    // interleave batches in memory; those fall through to the generic
    // string_einsum executor.
    if (auto const ar = detail::tensor_rank(A); ar >= 3 && ar == detail::tensor_rank(B) && ar == detail::tensor_rank(*C)) {
        std::size_t const Rank = ar;
        if (parsed.a_indices.size() == Rank && parsed.b_indices.size() == Rank && parsed.c_indices.size() == Rank &&
            desc.spec.link_indices.size() == 1) {

            std::string const &link = desc.spec.link_indices[0];

            auto find_pos = [](std::vector<std::string> const &idx, std::string const &name) -> int {
                for (int i = 0; std::cmp_less(i, idx.size()); ++i)
                    if (idx[i] == name)
                        return i;
                return -1;
            };

            // Collect batch indices: those appearing in A, B, AND C
            // (and not being the link). Preserve A's order so
            // "abij;abjk->abik" gives batch_names = [a, b] and we can
            // enforce matching positions across operands.
            std::vector<std::string> batch_names;
            for (auto const &idx : parsed.a_indices) {
                auto in_b = std::ranges::find(parsed.b_indices, idx) != parsed.b_indices.end();
                auto in_c = std::ranges::find(parsed.c_indices, idx) != parsed.c_indices.end();
                if (in_b && in_c && idx != link)
                    batch_names.push_back(idx);
            }

            // Each tensor has batch indices + 1 link + 1 target (per A or B),
            // or batch indices + 2 targets (for C). So num_batch == Rank - 2.
            bool const shape_ok = batch_names.size() == Rank - 2;

            // Batch indices must appear at the same positions in all three
            // operands — otherwise flattening the batch doesn't produce
            // consistent strides. Collect those positions (same for A, B, C).
            std::vector<int> batch_positions;
            batch_positions.reserve(batch_names.size());
            bool positions_match = shape_ok;
            for (auto const &bname : batch_names) {
                int pa = find_pos(parsed.a_indices, bname);
                int pb = find_pos(parsed.b_indices, bname);
                int pc = find_pos(parsed.c_indices, bname);
                if (pa < 0 || pa != pb || pa != pc) {
                    positions_match = false;
                    break;
                }
                batch_positions.push_back(pa);
            }

            // Mode selection: for stride math to work with a single
            // batch_stride, the batch axes must form a contiguous
            // outermost block. Row-major outermost = [0..num_batch-1];
            // col-major outermost = [rank-num_batch..rank-1].
            bool const all_contig    = A.impl().is_contiguous() && B.impl().is_contiguous() && C->impl().is_contiguous();
            bool const all_row_major = A.impl().is_row_major() && B.impl().is_row_major() && C->impl().is_row_major();
            bool const all_col_major = A.impl().is_column_major() && B.impl().is_column_major() && C->impl().is_column_major();

            auto is_prefix_range = [&](std::vector<int> const &positions, size_t count) {
                if (positions.size() != count)
                    return false;
                for (size_t i = 0; i < count; ++i)
                    if (std::cmp_not_equal(positions[i], i))
                        return false;
                return true;
            };
            auto is_suffix_range = [&](std::vector<int> const &positions, size_t count, size_t rank) {
                if (positions.size() != count)
                    return false;
                for (size_t i = 0; i < count; ++i)
                    if (std::cmp_not_equal(positions[i], rank - count + i))
                        return false;
                return true;
            };

            bool const row_mode = positions_match && all_row_major && is_prefix_range(batch_positions, batch_names.size());
            bool const col_mode = positions_match && all_col_major && is_suffix_range(batch_positions, batch_names.size(), Rank);

            if (shape_ok && positions_match && all_contig && (row_mode || col_mode)) {
                // Non-batch indices in original order: strip the batch positions.
                // For row_mode they're the LAST 2 positions; for col_mode the FIRST 2.
                std::vector<std::string> a_rest, b_rest;
                if (row_mode) {
                    a_rest = {parsed.a_indices[Rank - 2], parsed.a_indices[Rank - 1]};
                    b_rest = {parsed.b_indices[Rank - 2], parsed.b_indices[Rank - 1]};
                } else {
                    a_rest = {parsed.a_indices[0], parsed.a_indices[1]};
                    b_rest = {parsed.b_indices[0], parsed.b_indices[1]};
                }

                // 2D slice dim lookups: positions of the non-batch axes in the
                // original tensor. row_mode: positions (Rank-2, Rank-1);
                // col_mode: positions (0, 1).
                auto a_slice_dim = [&](int local_pos) -> int {
                    int orig = row_mode ? static_cast<int>(Rank - 2) + local_pos : local_pos;
                    return static_cast<int>(A.dim(orig));
                };
                auto b_slice_dim = [&](int local_pos) -> int {
                    int orig = row_mode ? static_cast<int>(Rank - 2) + local_pos : local_pos;
                    return static_cast<int>(B.dim(orig));
                };
                auto c_slice_dim = [&](int local_pos) -> int {
                    int orig = row_mode ? static_cast<int>(Rank - 2) + local_pos : local_pos;
                    return static_cast<int>(C->dim(orig));
                };

                // Flat batch count = product of each batch dim's size. Same
                // answer whether we read from A, B, or C since the sizes
                // must agree at construction time (shape compatibility).
                std::int64_t flat_batch = 1;
                for (int p : batch_positions)
                    flat_batch *= static_cast<std::int64_t>(A.dim(p));

                BatchedGemmDescriptor d;
                if constexpr (std::is_same_v<T, float>)
                    d.scalar = BlasScalar::Float;
                else if constexpr (std::is_same_v<T, double>)
                    d.scalar = BlasScalar::Double;
                else if constexpr (std::is_same_v<T, std::complex<float>>)
                    d.scalar = BlasScalar::ComplexFloat;
                else if constexpr (std::is_same_v<T, std::complex<double>>)
                    d.scalar = BlasScalar::ComplexDouble;

                char natural_trans_a = (a_rest[0] == link) ? 'T' : 'N';
                char natural_trans_b = (b_rest[1] == link) ? 'T' : 'N';

                if (col_mode) {
                    d.trans_a = natural_trans_a;
                    d.trans_b = natural_trans_b;
                    d.m       = c_slice_dim(0);
                    d.n       = c_slice_dim(1);
                    d.k       = (natural_trans_a == 'N') ? a_slice_dim(1) : a_slice_dim(0);
                    d.lda     = a_slice_dim(0);
                    d.ldb     = b_slice_dim(0);
                    d.ldc     = c_slice_dim(0);
                } else {
                    d.trans_a = natural_trans_b;
                    d.trans_b = natural_trans_a;
                    d.m       = c_slice_dim(1);
                    d.n       = c_slice_dim(0);
                    d.k       = (natural_trans_a == 'N') ? a_slice_dim(1) : a_slice_dim(0);
                    d.lda     = b_slice_dim(1);
                    d.ldb     = a_slice_dim(1);
                    d.ldc     = c_slice_dim(1);
                }

                d.alpha          = as_real<double>(params->ab_pf);
                d.beta           = as_real<double>(params->c_pf);
                d.batch_count    = static_cast<int>(flat_batch);
                d.strided        = true;
                d.batch_stride_a = static_cast<std::int64_t>(a_slice_dim(0)) * static_cast<std::int64_t>(a_slice_dim(1));
                d.batch_stride_b = static_cast<std::int64_t>(b_slice_dim(0)) * static_cast<std::int64_t>(b_slice_dim(1));
                d.batch_stride_c = static_cast<std::int64_t>(c_slice_dim(0)) * static_cast<std::int64_t>(c_slice_dim(1));

                bool const swap_ab  = row_mode;
                auto       executor = [d, swap_ab, a_slot, b_slot, c_slot]() {
                    auto const *base_a = static_cast<T const *>(static_cast<AType const *>(a_slot->ptr)->data());
                    auto const *base_b = static_cast<T const *>(static_cast<BType const *>(b_slot->ptr)->data());
                    auto       *base_c = static_cast<T *>(static_cast<CType *>(c_slot->ptr)->data());

                    std::vector<T const *> a_arr(d.batch_count);
                    std::vector<T const *> b_arr(d.batch_count);
                    std::vector<T *>       c_arr(d.batch_count);
                    for (int i = 0; i < d.batch_count; ++i) {
                        a_arr[i] = base_a + i * d.batch_stride_a;
                        b_arr[i] = base_b + i * d.batch_stride_b;
                        c_arr[i] = base_c + i * d.batch_stride_c;
                    }

                    T const **blas_a = swap_ab ? b_arr.data() : a_arr.data();
                    T const **blas_b = swap_ab ? a_arr.data() : b_arr.data();

                    if constexpr (std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>) {
                        T alpha{static_cast<typename T::value_type>(d.alpha), typename T::value_type{0}};
                        T beta{static_cast<typename T::value_type>(d.beta), typename T::value_type{0}};
                        blas::gemm_batch<T>(d.trans_a, d.trans_b, d.m, d.n, d.k, alpha, blas_a, d.lda, blas_b, d.ldb, beta, c_arr.data(),
                                            d.ldc, d.batch_count);
                    } else {
                        blas::gemm_batch<T>(d.trans_a, d.trans_b, d.m, d.n, d.k, static_cast<T>(d.alpha), blas_a, d.lda, blas_b, d.ldb,
                                            static_cast<T>(d.beta), c_arr.data(), d.ldc, d.batch_count);
                    }
                };

                auto label = fmt::format("gemm_batch_strided x{} ({}-major, batch={}, M={}, K={}, N={})", d.batch_count,
                                         col_mode ? "col" : "row", fmt::join(batch_names, ","), d.m, d.k, d.n);
                ctx.record(OpKind::BatchedGemm, std::move(label), {a_id, b_id}, {c_id}, std::move(executor), std::move(d));
                return;
            }
        }
    }

    auto label = fmt::format("einsum: C[{}] = A[{}] * B[{}]", fmt::join(parsed.c_indices, ","), fmt::join(parsed.a_indices, ","),
                             fmt::join(parsed.b_indices, ","));

    // Capture the shared indices + params by shared_ptr; dispatch
    // constructs a ParsedEinsumSpec on every call from the current
    // (live, possibly rewritten) index state. The ParsedEinsumSpec is
    // tiny (three vector<string> copies by reference + one std::string)
    // so this is cheap compared to the contraction itself.
    auto executor = [indices, params, a_slot, b_slot, c_slot]() {
        ParsedEinsumSpec parsed_live{indices->c_indices, indices->a_indices, indices->b_indices, /*raw*/ std::string{}};
        dispatch::string_einsum(parsed_live, as<T>(params->c_pf), static_cast<CType *>(c_slot->ptr), as<T>(params->ab_pf),
                                *static_cast<AType const *>(a_slot->ptr), *static_cast<BType const *>(b_slot->ptr));
    };

    ctx.record(OpKind::Einsum, std::move(label), {a_id, b_id}, {c_id}, std::move(executor), std::move(desc));
}

/**
 * @brief String-based einsum with default prefactors (c_pf=0, ab_pf=1).
 *
 * String literals are validated at compile time. For runtime strings,
 * use `EinsumFormatString(string_view)` explicitly.
 *
 * @code
 * cg::einsum("ij <- ik ; kj", &C, A, B);  // Compile-time validated
 *
 * std::string spec = build_spec();
 * cg::einsum(EinsumFormatString(spec), &C, A, B);  // Runtime string
 * @endcode
 */
template <BasicTensorConcept AType, BasicTensorConcept BType, BasicTensorConcept CType>
    requires requires {
        requires std::is_same_v<typename AType::ValueType, typename BType::ValueType>;
        requires std::is_same_v<typename AType::ValueType, typename CType::ValueType>;
    }
void einsum(EinsumFormatString spec, CType *C, AType const &A, BType const &B) {
    using T = typename AType::ValueType;
    einsum(spec, T{0}, C, T{1}, A, B);
}

/// Graph-aware einsum: contract A and B according to ``spec``.
///
/// ``spec`` is a string of the form ``"<output> <- <a> ; <b>"``; e.g.
/// ``"ij <- ik ; kj"`` (matrix multiply), ``"i <- ij ; j"`` (matrix-
/// vector), ``" <- i ; i"`` (dot product). ``c_pf`` and ``ab_pf``
/// default to 0 and 1 — i.e. ``C = A op B`` — but can be set to
/// accumulate (``c_pf=1``) or scale.
///
/// Complex dtypes (``RuntimeTensor<complex<T>>``) accept only the
/// 4-argument form (no explicit prefactors); Graph::create_params still
/// stores prefactors as doubles internally, which would narrow the
/// imaginary part out of a complex ``c_pf``/``ab_pf``.
template <BasicTensorConcept AType, BasicTensorConcept BType, BasicTensorConcept CType>
    requires(std::is_same_v<typename AType::ValueType, typename BType::ValueType> &&
             std::is_same_v<typename AType::ValueType, typename CType::ValueType>)
// clang-format off
EINSUMS_PYBIND_EXPOSE
EINSUMS_PYBIND_INSTANTIATE_AS("einsum", einsums::GeneralRuntimeTensor<float, std::allocator<float>>,
                              einsums::GeneralRuntimeTensor<float, std::allocator<float>>,
                              einsums::GeneralRuntimeTensor<float, std::allocator<float>>)
EINSUMS_PYBIND_INSTANTIATE_AS("einsum", einsums::GeneralRuntimeTensor<double, std::allocator<double>>,
                              einsums::GeneralRuntimeTensor<double, std::allocator<double>>,
                              einsums::GeneralRuntimeTensor<double, std::allocator<double>>)
EINSUMS_PYBIND_INSTANTIATE_AS("einsum", einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>,
                              einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>,
                              einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>)
EINSUMS_PYBIND_INSTANTIATE_AS("einsum", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>,
                              einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>,
                              einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
    // clang-format on
    void einsum_python(std::string const &spec, CType *C, AType const &A, BType const &B,
                       typename CType::ValueType c_pf  = typename CType::ValueType{0},
                       typename AType::ValueType ab_pf = typename AType::ValueType{1}) {
    einsum(EinsumFormatString(std::string_view{spec}), c_pf, C, ab_pf, A, B);
}

// ─────────────────────────────────────────────────────────────────────────────
// parallel_for — graph-capturable data-parallel loop via TaskPool
// ─────────────────────────────────────────────────────────────────────────────

/// @brief Graph-aware parallel_for that captures into the computation graph.
///
/// During capture, records a ParallelFor node with the specified input/output
/// tensor dependencies. At execution time, delegates to TaskPool::parallel_for().
///
/// The user must declare which tensors the body reads (inputs) and writes (outputs)
/// so that the graph's topological sort can order this node correctly relative to
/// other operations.
///
/// @param name   Label for profiling and debugging.
/// @param begin  Start of the index range [begin, end).
/// @param end    End of the index range.
/// @param body   Callable with signature void(size_t index).
/// @param reads  Tensors read by the body (used for dependency tracking).
/// @param writes Tensors written by the body (used for dependency tracking).
///
/// @code
/// cg::CaptureGuard guard(graph);
/// cg::parallel_for("J_build", 0, n_pairs,
///     [&](size_t pair) { compute_integrals(pair, J_mat, K_mat); },
///     {&D_mat, &eri},    // reads
///     {&J_mat, &K_mat}   // writes
/// );
/// cg::einsum("...", &F, ...);  // Automatically ordered after J_build
/// @endcode
template <typename F, CoreBasicTensorConcept... ReadTensors, CoreBasicTensorConcept... WriteTensors>
void parallel_for(std::string name, size_t begin, size_t end, F &&body, std::tuple<ReadTensors const *...> reads,
                  std::tuple<WriteTensors *...> writes) {
    auto &ctx = CaptureContext::current();
    if (!ctx.is_capturing()) {
        task_pool::TaskPool::get_singleton().parallel_for(name, begin, end, std::forward<F>(body));
        return;
    }

    // Collect input tensor IDs
    std::vector<TensorId> input_ids;
    std::apply([&](auto *...ptrs) { (input_ids.push_back(ctx.get_or_register(*ptrs)), ...); }, reads);

    // Collect output tensor IDs
    std::vector<TensorId> output_ids;
    std::apply([&](auto *...ptrs) { (output_ids.push_back(ctx.get_or_register(*ptrs)), ...); }, writes);

    auto executor = [name, begin, end, body = std::forward<F>(body)]() mutable {
        task_pool::TaskPool::get_singleton().parallel_for(name, begin, end, body);
    };

    ctx.record(OpKind::ParallelFor, std::move(name), std::move(input_ids), std::move(output_ids), std::move(executor));
}

/// @brief Simplified parallel_for with variadic output tensors.
///
/// Convenience overload: all listed tensors are treated as both inputs AND outputs.
/// This is the common case for accumulation patterns where the body both reads and
/// writes the same tensors.
///
/// @code
/// cg::parallel_for("J_build", 0, n_pairs,
///     [&](size_t pair) { compute_integrals(pair); },
///     &J_mat, &K_mat   // Both read and written
/// );
/// @endcode
template <typename F, CoreBasicTensorConcept... TensorTypes>
void parallel_for(std::string name, size_t begin, size_t end, F &&body, TensorTypes *...tensors) {
    auto &ctx = CaptureContext::current();
    if (!ctx.is_capturing()) {
        task_pool::TaskPool::get_singleton().parallel_for(name, begin, end, std::forward<F>(body));
        return;
    }

    // All listed tensors are both inputs and outputs
    std::vector<TensorId> tensor_ids;
    (tensor_ids.push_back(ctx.get_or_register(*tensors)), ...);

    auto executor = [name, begin, end, body = std::forward<F>(body)]() mutable {
        task_pool::TaskPool::get_singleton().parallel_for(name, begin, end, body);
    };

    ctx.record(OpKind::ParallelFor, std::move(name), tensor_ids, tensor_ids, std::move(executor));
}

/// @brief Graph-aware parallel_reduce that captures into the computation graph.
///
/// During capture, records a ParallelReduce node. At execution time, delegates
/// to TaskPool::parallel_reduce() and stores the result.
///
/// @code
/// double energy = 0.0;
/// cg::CaptureGuard guard(graph);
/// cg::parallel_reduce<double>("energy", 0, N, &energy,
///     []() { return 0.0; },
///     [&](size_t i, double &acc) { acc += contrib(i); },
///     [](double &g, double const &l) { g += l; },
///     &D_mat, &F_mat  // Tensors read during the reduction
/// );
/// @endcode
template <typename Acc, typename InitFactory, typename Body, typename Combiner, CoreBasicTensorConcept... TensorTypes>
void parallel_reduce(std::string name, size_t begin, size_t end, Acc *result, InitFactory &&init, Body &&body, Combiner &&combine,
                     TensorTypes *...tensors) {
    auto &ctx = CaptureContext::current();
    if (!ctx.is_capturing()) {
        *result = task_pool::TaskPool::get_singleton().parallel_reduce<Acc>(name, begin, end, std::forward<InitFactory>(init),
                                                                            std::forward<Body>(body), std::forward<Combiner>(combine));
        return;
    }

    // Input tensors
    std::vector<TensorId> tensor_ids;
    (tensor_ids.push_back(ctx.get_or_register(*tensors)), ...);

    auto executor = [name, begin, end, result, init = std::forward<InitFactory>(init), body = std::forward<Body>(body),
                     combine = std::forward<Combiner>(combine)]() mutable {
        *result = task_pool::TaskPool::get_singleton().parallel_reduce<Acc>(name, begin, end, init, body, combine);
    };

    // Result scalar is not a tensor, so no output TensorId.
    // The input tensors establish the dependency ordering.
    ctx.record(OpKind::ParallelReduce, std::move(name), std::move(tensor_ids), {}, std::move(executor));
}

// ===========================================================================
// Custom operations and disk I/O
// ===========================================================================

/**
 * @brief Record a custom (user-defined) operation in the graph.
 *
 * Use this for operations that don't have a built-in graph wrapper,
 * such as computing integrals, applying custom transformations, etc.
 *
 * @param label     Human-readable name for profiling and debugging.
 * @param inputs    Tensors read by this operation.
 * @param outputs   Tensors written by this operation.
 * @param executor  Lambda that performs the computation.
 *
 * @code
 * cg::Graph graph("scf");
 * {
 *     cg::CaptureGuard guard(graph);
 *
 *     cg::custom("compute_ERI", {}, {&ERI}, [&]() {
 *         compute_two_electron_integrals(basis, ERI);
 *     });
 *
 *     cg::einsum("ijkl;kl->ij", 0.0, &F, 1.0, ERI, D);
 * }
 * @endcode
 */
template <typename F, CoreBasicTensorConcept... InputTensors, CoreBasicTensorConcept... OutputTensors>
void custom(std::string label, std::initializer_list<void const *> input_ptrs, std::initializer_list<void *> output_ptrs, F &&executor) {
    // This overload takes raw pointers — prefer the typed overload below.
    // Provided for cases where tensor types are heterogeneous.
    (void)input_ptrs;
    (void)output_ptrs;

    auto &ctx = CaptureContext::current();
    ctx.record(OpKind::Custom, std::move(label), {}, {}, std::forward<F>(executor));
}

/// @brief Record a custom operation with typed tensor inputs/outputs.
template <typename F, CoreBasicTensorConcept... Outputs>
void custom(std::string label, F &&executor, Outputs *...outputs) {
    auto &ctx = CaptureContext::current();

    std::vector<TensorId> output_ids;
    (output_ids.push_back(ctx.get_or_register(*outputs)), ...);

    ctx.record(OpKind::Custom, std::move(label), {}, std::move(output_ids), std::forward<F>(executor));
}

/// @brief Record a custom operation with typed input and output tensors.
template <typename F, CoreBasicTensorConcept... Inputs, CoreBasicTensorConcept... Outputs>
void custom(std::string label, std::tuple<Inputs const &...> inputs, std::tuple<Outputs &...> outputs, F &&executor) {
    auto &ctx = CaptureContext::current();

    std::vector<TensorId> input_ids;
    std::apply([&](auto const &...ts) { (input_ids.push_back(ctx.get_or_register(ts)), ...); }, inputs);

    std::vector<TensorId> output_ids;
    std::apply([&](auto &...ts) { (output_ids.push_back(ctx.get_or_register(ts)), ...); }, outputs);

    ctx.record(OpKind::Custom, std::move(label), std::move(input_ids), std::move(output_ids), std::forward<F>(executor));
}

/**
 * @brief No-dependency custom node — Python-friendly overload.
 *
 * Records a graph node that runs ``executor`` with no declared
 * input/output tensor dependencies. Outside a capture context the
 * executor runs immediately. This is the simplest way to splice a
 * Python (or arbitrary C++) callable into a graph as a compute step;
 * for read-modify-write patterns prefer the tensor-tagged overload
 * below so the optimizer can see the dependency.
 *
 * @code
 * cg::custom("debug_print", []() { fmt::print("hello\n"); });
 * @endcode
 */
/**
 * @brief No-dependency custom node — Python-friendly overload.
 *
 * Records a graph node that runs ``executor`` with no declared
 * input/output tensor dependencies. Outside a capture context the
 * executor runs immediately. Use the tensor-tagged overload below for
 * read-modify-write patterns where the optimizer should see the
 * dependency.
 *
 * @code
 * cg::custom("debug_print", []() { fmt::print("hello\n"); });
 * @endcode
 */
EINSUMS_PYBIND_EXPOSE EINSUMS_PYBIND_MODULE("graph") inline void custom(std::string label, std::function<void()> executor) {
    auto &ctx = CaptureContext::current();
    if (!ctx.is_capturing()) {
        executor();
        return;
    }
    ctx.record(OpKind::Custom, std::move(label), {}, {}, std::move(executor));
}

/**
 * @brief Single-tensor read-modify-write custom node — Python-friendly.
 *
 * Records a graph node whose executor mutates @p target in place.
 * @p target is registered as both an input and an output, so the
 * optimizer treats this node as a read-modify-write barrier on that
 * tensor. Outside a capture context the executor runs immediately.
 *
 * @code
 * // Body of a graph-driven loop: read slab, transform, write slab.
 * einsums::tensor_io::read_slice_etn(path, "A", slab, &block);
 * cg::custom("scale_x10", [&]() { block *= 10.0; }, &block);
 * einsums::tensor_io::write_slice_etn(path, "A", slab, &block);
 * @endcode
 */
// clang-format off
template <CoreBasicTensorConcept TensorType>
EINSUMS_PYBIND_EXPOSE
EINSUMS_PYBIND_MODULE("graph")
EINSUMS_PYBIND_INSTANTIATE_AS("custom", einsums::GeneralRuntimeTensor<float, std::allocator<float>>)
EINSUMS_PYBIND_INSTANTIATE_AS("custom", einsums::GeneralRuntimeTensor<double, std::allocator<double>>)
EINSUMS_PYBIND_INSTANTIATE_AS("custom", einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>)
EINSUMS_PYBIND_INSTANTIATE_AS("custom", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
    // clang-format on
    void custom(std::string label, std::function<void()> executor, TensorType *target) {
    auto &ctx = CaptureContext::current();
    if (!ctx.is_capturing()) {
        executor();
        return;
    }
    auto id = ctx.get_or_register(*target);
    ctx.record(OpKind::Custom, std::move(label), {id}, {id}, std::move(executor));
}

/**
 * @brief Record a disk read operation: load tensor data from a file.
 *
 * The executor should read data from the specified file into the tensor.
 * The graph tracks the output dependency so downstream operations wait
 * for the read to complete.
 *
 * @param label      Human-readable name.
 * @param file_path  Path to the file.
 * @param dataset    Dataset/key name within the file (for HDF5).
 * @param output     Tensor to populate with data from disk.
 * @param executor   Lambda that performs the actual read.
 *
 * @code
 * cg::read("load integrals", "integrals.h5", "/eri", &ERI, [&]() {
 *     einsums::read(ERI, "integrals.h5", "/eri");
 * });
 * @endcode
 */
template <CoreBasicTensorConcept TensorType, typename F>
void read(std::string label, std::string file_path, std::string dataset, TensorType *output, F &&executor) {
    auto &ctx    = CaptureContext::current();
    auto  out_id = ctx.get_or_register(*output);

    DiskIODescriptor desc;
    desc.file_path    = std::move(file_path);
    desc.dataset_name = std::move(dataset);
    desc.tensor_id    = out_id;
    desc.size_bytes   = output->size() * sizeof(typename std::remove_cvref_t<TensorType>::ValueType);

    ctx.record(OpKind::DiskRead, std::move(label), {}, {out_id}, std::forward<F>(executor), std::move(desc));
}

/**
 * @brief Record a disk write operation: save tensor data to a file.
 *
 * The executor should write the tensor data to the specified file.
 * The graph tracks the input dependency so the write waits for the
 * tensor to be computed.
 *
 * @param label      Human-readable name.
 * @param file_path  Path to the file.
 * @param dataset    Dataset/key name within the file (for HDF5).
 * @param input      Tensor to write to disk.
 * @param executor   Lambda that performs the actual write.
 *
 * @code
 * cg::write("checkpoint F", "checkpoint.h5", "/fock", &F, [&]() {
 *     einsums::write(F, "checkpoint.h5", "/fock");
 * });
 * @endcode
 */
template <CoreBasicTensorConcept TensorType, typename F>
void write(std::string label, std::string file_path, std::string dataset, TensorType const *input, F &&executor) {
    auto &ctx   = CaptureContext::current();
    auto  in_id = ctx.get_or_register(*input);

    DiskIODescriptor desc;
    desc.file_path    = std::move(file_path);
    desc.dataset_name = std::move(dataset);
    desc.tensor_id    = in_id;
    desc.size_bytes   = input->size() * sizeof(typename std::remove_cvref_t<TensorType>::ValueType);

    ctx.record(OpKind::DiskWrite, std::move(label), {in_id}, {}, std::forward<F>(executor), std::move(desc));
}

/**
 * @brief Record an async-capable disk read operation.
 *
 * The DataflowExecutor calls @p start_fn to begin the read as soon as
 * predecessors complete, then calls @p finish_fn before any consumer runs.
 * Independent compute nodes can execute between start and finish, overlapping
 * I/O with computation.
 *
 * SequentialExecutor and OpenMPExecutor call @p sync_fn (the synchronous
 * fallback) and ignore the async lambdas.
 *
 * @param label      Human-readable name.
 * @param file_path  Path to the file.
 * @param dataset    Dataset/key name within the file.
 * @param output     Tensor to populate with data from disk.
 * @param start_fn   Lambda that begins the async read (should return quickly).
 * @param finish_fn  Lambda that waits for the read to complete and finalizes the tensor.
 * @param sync_fn    Lambda that performs the full synchronous read (fallback).
 *
 * @code
 * std::future<void> io_future;
 * cg::read_async("load ERI", "integrals.h5", "/eri", &ERI,
 *     [&]() { io_future = std::async(std::launch::async, [&]{ load(ERI); }); },
 *     [&]() { io_future.get(); },
 *     [&]() { load(ERI); }
 * );
 * @endcode
 */
template <CoreBasicTensorConcept TensorType, typename StartFn, typename FinishFn, typename SyncFn>
void read_async(std::string label, std::string file_path, std::string dataset, TensorType *output, StartFn &&start_fn, FinishFn &&finish_fn,
                SyncFn &&sync_fn) {
    auto &ctx    = CaptureContext::current();
    auto  out_id = ctx.get_or_register(*output);

    DiskIODescriptor desc;
    desc.file_path    = std::move(file_path);
    desc.dataset_name = std::move(dataset);
    desc.tensor_id    = out_id;
    desc.size_bytes   = output->size() * sizeof(typename std::remove_cvref_t<TensorType>::ValueType);

    ctx.record_async(OpKind::DiskRead, std::move(label), {}, {out_id}, std::forward<SyncFn>(sync_fn), std::forward<StartFn>(start_fn),
                     std::forward<FinishFn>(finish_fn), std::move(desc));
}

/**
 * @brief Record an async-capable disk write operation.
 *
 * Same async semantics as read_async() but for writing tensor data to disk.
 *
 * @param label      Human-readable name.
 * @param file_path  Path to the file.
 * @param dataset    Dataset/key name within the file.
 * @param input      Tensor to write to disk.
 * @param start_fn   Lambda that begins the async write (should return quickly).
 * @param finish_fn  Lambda that waits for the write to complete.
 * @param sync_fn    Lambda that performs the full synchronous write (fallback).
 */
template <CoreBasicTensorConcept TensorType, typename StartFn, typename FinishFn, typename SyncFn>
void write_async(std::string label, std::string file_path, std::string dataset, TensorType const *input, StartFn &&start_fn,
                 FinishFn &&finish_fn, SyncFn &&sync_fn) {
    auto &ctx   = CaptureContext::current();
    auto  in_id = ctx.get_or_register(*input);

    DiskIODescriptor desc;
    desc.file_path    = std::move(file_path);
    desc.dataset_name = std::move(dataset);
    desc.tensor_id    = in_id;
    desc.size_bytes   = input->size() * sizeof(typename std::remove_cvref_t<TensorType>::ValueType);

    ctx.record_async(OpKind::DiskWrite, std::move(label), {in_id}, {}, std::forward<SyncFn>(sync_fn), std::forward<StartFn>(start_fn),
                     std::forward<FinishFn>(finish_fn), std::move(desc));
}

} // namespace einsums::compute_graph
