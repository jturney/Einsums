//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/BLAS.hpp>
#include <Einsums/ComputeGraph/CaptureContext.hpp>
#include <Einsums/ComputeGraph/Detail/TiledRuntimeEinsum.hpp>
#include <Einsums/ComputeGraph/Detail/TiledRuntimeElementwise.hpp>
#include <Einsums/ComputeGraph/EinsumSpec.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/StringDispatch.hpp>
#include <Einsums/ComputeGraph/TensorRank.hpp>
#include <Einsums/Concepts/TensorConcepts.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/LinearAlgebra.hpp>
#include <Einsums/Profile.hpp>
#include <Einsums/Python/Annotations.hpp>
#include <Einsums/TaskPool/TaskPool.hpp>
#include <Einsums/TensorAlgebra/Backends/ElementTransform.hpp>
#include <Einsums/TensorAlgebra/Permute.hpp>
#include <Einsums/TensorAlgebra/TensorAlgebra.hpp>

#include <fmt/format.h>

// When this header is compiled into the Python bindings (PyEinsums target),
// pull in pybind11 so element_transform_python can translate a Python callback's
// exception under the GIL. The generated binding TU includes pybind11 only after
// this header, so we can't rely on PYBIND11_VERSION_MAJOR being defined here.
#if defined(PyEinsums_EXPORTS)
#    include <pybind11/pybind11.h>
#endif

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace einsums::compute_graph {

// ─────────────────────────────────────────────────────────────────────────────
// einsum: graph-aware, runtime-string contraction spec
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

/// Graph-aware scale: multiplies @p A in place by the scalar @p factor.
template <TensorConcept AType>
// clang-format off
APIARY_EXPOSE
APIARY_MODULE("linalg")
APIARY_INSTANTIATE_AS("scale", einsums::GeneralRuntimeTensor<float, std::allocator<float>>)
APIARY_INSTANTIATE_AS("scale", einsums::GeneralRuntimeTensor<double, std::allocator<double>>)
APIARY_INSTANTIATE_AS("scale", einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>)
APIARY_INSTANTIATE_AS("scale", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
APIARY_INSTANTIATE_AS("scale", einsums::RuntimeTensorView<float>)
APIARY_INSTANTIATE_AS("scale", einsums::RuntimeTensorView<double>)
APIARY_INSTANTIATE_AS("scale", einsums::RuntimeTensorView<std::complex<float>>)
APIARY_INSTANTIATE_AS("scale", einsums::RuntimeTensorView<std::complex<double>>)
APIARY_INSTANTIATE_AS("scale", einsums::TiledRuntimeTensor<float>)
APIARY_INSTANTIATE_AS("scale", einsums::TiledRuntimeTensor<double>)
APIARY_INSTANTIATE_AS("scale", einsums::TiledRuntimeTensor<std::complex<float>>)
APIARY_INSTANTIATE_AS("scale", einsums::TiledRuntimeTensor<std::complex<double>>)
    // clang-format on
    void scale(typename AType::ValueType factor, AType *A) {
    if constexpr (IsTiledTensorV<std::remove_cvref_t<AType>>) {
        // Tiled: scale every populated tile. Eager, or an opaque Custom node
        // (the einsum-rewriting passes don't apply to a tiled per-tile op).
        using T   = typename AType::ValueType;
        auto &ctx = CaptureContext::current();
        if (!ctx.is_capturing()) {
            LabeledSection("scale eager");
            detail::tiled_scale<T>(factor, A);
            return;
        }
        LabeledSection("scale capture");
        auto [a_id, a_slot] = ctx.get_slot(*A);
        auto label          = fmt::format("tiled scale({})", A->name());
        auto executor       = [factor, a_slot]() {
            LabeledSection("scale execute");
            detail::tiled_scale<T>(factor, static_cast<AType *>(a_slot->ptr));
        };
        ctx.record(OpKind::Custom, std::move(label), {a_id}, {a_id}, std::move(executor));
    } else {
        auto &ctx = CaptureContext::current();
        if (!ctx.is_capturing()) {
            LabeledSection("scale eager");
            linear_algebra::scale(factor, A);
            return;
        }

        LabeledSection("scale capture");
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
            LabeledSection("scale execute");
            auto *a_ptr = static_cast<AType *>(a_slot->ptr);
            linear_algebra::scale(factor, a_ptr);
        };

        ctx.record(OpKind::Scale, std::move(label), {a_id}, {a_id}, std::move(executor), std::move(desc));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// conj / real / imag / abs  (complex-conjugation primitives)
// ─────────────────────────────────────────────────────────────────────────────

/// Graph-aware in-place complex conjugate: ``A := conj(A)``. No-op for real
/// dtypes (matches numpy.conj on a real array). Captured as an opaque Custom
/// node — the einsum-rewriting passes don't reason about conjugation.
template <TensorConcept AType>
// clang-format off
APIARY_EXPOSE
APIARY_MODULE("linalg")
APIARY_INSTANTIATE_AS("conj", einsums::GeneralRuntimeTensor<float, std::allocator<float>>)
APIARY_INSTANTIATE_AS("conj", einsums::GeneralRuntimeTensor<double, std::allocator<double>>)
APIARY_INSTANTIATE_AS("conj", einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>)
APIARY_INSTANTIATE_AS("conj", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
APIARY_INSTANTIATE_AS("conj", einsums::RuntimeTensorView<float>)
APIARY_INSTANTIATE_AS("conj", einsums::RuntimeTensorView<double>)
APIARY_INSTANTIATE_AS("conj", einsums::RuntimeTensorView<std::complex<float>>)
APIARY_INSTANTIATE_AS("conj", einsums::RuntimeTensorView<std::complex<double>>)
    // clang-format on
    void conj(AType *A) {
    auto &ctx = CaptureContext::current();
    if (!ctx.is_capturing()) {
        LabeledSection("conj eager");
        einsums::detail::impl_conj(A->impl());
        return;
    }
    LabeledSection("conj capture");
    auto [a_id, a_slot] = ctx.get_slot(*A);
    auto label          = fmt::format("conj({})", A->name());
    auto executor       = [a_slot]() {
        LabeledSection("conj execute");
        einsums::detail::impl_conj(static_cast<AType *>(a_slot->ptr)->impl());
    };
    ctx.record(OpKind::Custom, std::move(label), {a_id}, {a_id}, std::move(executor));
}

/// Graph-aware real part: ``out := Re(A)`` (complex ``A`` -> real ``out``).
template <CoreBasicTensorConcept ResultType, CoreBasicTensorConcept AType>
// clang-format off
APIARY_EXPOSE
APIARY_MODULE("linalg")
APIARY_INSTANTIATE_AS("real", einsums::GeneralRuntimeTensor<float, std::allocator<float>>,   einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>)
APIARY_INSTANTIATE_AS("real", einsums::GeneralRuntimeTensor<float, std::allocator<float>>,   einsums::RuntimeTensorView<std::complex<float>>)
APIARY_INSTANTIATE_AS("real", einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
APIARY_INSTANTIATE_AS("real", einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::RuntimeTensorView<std::complex<double>>)
    // clang-format on
    void real(ResultType *out, AType const &A) {
    auto &ctx = CaptureContext::current();
    if (!ctx.is_capturing()) {
        LabeledSection("real eager");
        einsums::detail::impl_real(A.impl(), out->impl());
        return;
    }
    LabeledSection("real capture");
    auto [a_id, a_slot] = ctx.get_slot(A);
    auto [r_id, r_slot] = ctx.get_slot(*out);
    auto executor       = [a_slot, r_slot]() {
        LabeledSection("real execute");
        einsums::detail::impl_real(static_cast<AType const *>(a_slot->ptr)->impl(), static_cast<ResultType *>(r_slot->ptr)->impl());
    };
    ctx.record(OpKind::Custom, "real", {a_id}, {r_id}, std::move(executor));
}

/// Graph-aware imaginary part: ``out := Im(A)`` (complex ``A`` -> real ``out``).
template <CoreBasicTensorConcept ResultType, CoreBasicTensorConcept AType>
// clang-format off
APIARY_EXPOSE
APIARY_MODULE("linalg")
APIARY_INSTANTIATE_AS("imag", einsums::GeneralRuntimeTensor<float, std::allocator<float>>,   einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>)
APIARY_INSTANTIATE_AS("imag", einsums::GeneralRuntimeTensor<float, std::allocator<float>>,   einsums::RuntimeTensorView<std::complex<float>>)
APIARY_INSTANTIATE_AS("imag", einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
APIARY_INSTANTIATE_AS("imag", einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::RuntimeTensorView<std::complex<double>>)
    // clang-format on
    void imag(ResultType *out, AType const &A) {
    auto &ctx = CaptureContext::current();
    if (!ctx.is_capturing()) {
        LabeledSection("imag eager");
        einsums::detail::impl_imag(A.impl(), out->impl());
        return;
    }
    LabeledSection("imag capture");
    auto [a_id, a_slot] = ctx.get_slot(A);
    auto [r_id, r_slot] = ctx.get_slot(*out);
    auto executor       = [a_slot, r_slot]() {
        LabeledSection("imag execute");
        einsums::detail::impl_imag(static_cast<AType const *>(a_slot->ptr)->impl(), static_cast<ResultType *>(r_slot->ptr)->impl());
    };
    ctx.record(OpKind::Custom, "imag", {a_id}, {r_id}, std::move(executor));
}

/// Graph-aware magnitude: ``out := |A|`` (real or complex ``A`` -> real ``out``).
template <CoreBasicTensorConcept ResultType, CoreBasicTensorConcept AType>
// clang-format off
APIARY_EXPOSE
APIARY_MODULE("linalg")
APIARY_INSTANTIATE_AS("abs", einsums::GeneralRuntimeTensor<float, std::allocator<float>>,   einsums::GeneralRuntimeTensor<float, std::allocator<float>>)
APIARY_INSTANTIATE_AS("abs", einsums::GeneralRuntimeTensor<float, std::allocator<float>>,   einsums::RuntimeTensorView<float>)
APIARY_INSTANTIATE_AS("abs", einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::GeneralRuntimeTensor<double, std::allocator<double>>)
APIARY_INSTANTIATE_AS("abs", einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::RuntimeTensorView<double>)
APIARY_INSTANTIATE_AS("abs", einsums::GeneralRuntimeTensor<float, std::allocator<float>>,   einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>)
APIARY_INSTANTIATE_AS("abs", einsums::GeneralRuntimeTensor<float, std::allocator<float>>,   einsums::RuntimeTensorView<std::complex<float>>)
APIARY_INSTANTIATE_AS("abs", einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
APIARY_INSTANTIATE_AS("abs", einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::RuntimeTensorView<std::complex<double>>)
    // clang-format on
    void abs(ResultType *out, AType const &A) {
    auto &ctx = CaptureContext::current();
    if (!ctx.is_capturing()) {
        LabeledSection("abs eager");
        einsums::detail::impl_abs(A.impl(), out->impl());
        return;
    }
    LabeledSection("abs capture");
    auto [a_id, a_slot] = ctx.get_slot(A);
    auto [r_id, r_slot] = ctx.get_slot(*out);
    auto executor       = [a_slot, r_slot]() {
        LabeledSection("abs execute");
        einsums::detail::impl_abs(static_cast<AType const *>(a_slot->ptr)->impl(), static_cast<ResultType *>(r_slot->ptr)->impl());
    };
    ctx.record(OpKind::Custom, "abs", {a_id}, {r_id}, std::move(executor));
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
        LabeledSection("permute eager");
        dispatch::string_permute(parsed, beta, C, alpha, A);
        return;
    }

    LabeledSection("permute capture");
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
        LabeledSection("permute execute");
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
APIARY_EXPOSE
APIARY_INSTANTIATE_AS("permute", einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::GeneralRuntimeTensor<float, std::allocator<float>>)
APIARY_INSTANTIATE_AS("permute", einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::GeneralRuntimeTensor<double, std::allocator<double>>)
APIARY_INSTANTIATE_AS("permute", einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>)
APIARY_INSTANTIATE_AS("permute", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
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
        LabeledSection("transpose eager");
        tensor_algebra::transpose(C, A);
        return;
    }

    LabeledSection("transpose capture");
    auto [a_id, a_slot] = ctx.get_slot(A);
    auto [c_id, c_slot] = ctx.get_slot(*C);

    auto executor = [c_slot, a_slot]() {
        LabeledSection("transpose execute");
        tensor_algebra::transpose(static_cast<CType *>(c_slot->ptr), *static_cast<AType const *>(a_slot->ptr));
    };

    ctx.record(OpKind::Transpose, "transpose", {a_id}, {c_id}, std::move(executor));
}

// ─────────────────────────────────────────────────────────────────────────────
// block_copy: slab copy between same-rank tensors
// ─────────────────────────────────────────────────────────────────────────────

/// Copy a contiguous N-dimensional sub-region of @p src into @p dst.
///
/// For each axis k, copies @p extents[k] elements starting at @p src_offsets[k]
/// in src into @p dst_offsets[k] in dst:
///
///   dst[dst_offsets + i] = src[src_offsets + i]   for all i in extents
///
/// Both tensors must have the same rank and dtype. Uses per-axis strides so
/// arbitrary memory layouts are handled correctly. Capture-aware: outside
/// capture, runs immediately; inside capture, records a node with src as an
/// input dependency and dst as an output.
///
/// Common patterns:
///   * Extract occupied MO block:  block_copy(&C_occ, C, {0,0}, {0,0}, {nbf, nocc})
///   * Extract (ia|jb) ERI block:  block_copy(&iajb, eri_mo, {0,0,0,0},
///                                             {0, nocc, 0, nocc},
///                                             {nocc, nvirt, nocc, nvirt})
template <CoreBasicTensorConcept DstType, CoreBasicTensorConcept SrcType>
    requires std::is_same_v<typename DstType::ValueType, typename SrcType::ValueType>
// clang-format off
APIARY_EXPOSE
APIARY_MODULE("linalg")
APIARY_INSTANTIATE_AS("block_copy", einsums::GeneralRuntimeTensor<float,                std::allocator<float>>,                einsums::GeneralRuntimeTensor<float,                std::allocator<float>>)
APIARY_INSTANTIATE_AS("block_copy", einsums::GeneralRuntimeTensor<double,               std::allocator<double>>,               einsums::GeneralRuntimeTensor<double,               std::allocator<double>>)
APIARY_INSTANTIATE_AS("block_copy", einsums::GeneralRuntimeTensor<std::complex<float>,  std::allocator<std::complex<float>>>,  einsums::GeneralRuntimeTensor<std::complex<float>,  std::allocator<std::complex<float>>>)
APIARY_INSTANTIATE_AS("block_copy", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
// dst is a view, src is an owning tensor, common when writing a tensor into a slab of a larger destination.
APIARY_INSTANTIATE_AS("block_copy", einsums::RuntimeTensorView<float>,                                                   einsums::GeneralRuntimeTensor<float,                std::allocator<float>>)
APIARY_INSTANTIATE_AS("block_copy", einsums::RuntimeTensorView<double>,                                                  einsums::GeneralRuntimeTensor<double,               std::allocator<double>>)
APIARY_INSTANTIATE_AS("block_copy", einsums::RuntimeTensorView<std::complex<float>>,                                     einsums::GeneralRuntimeTensor<std::complex<float>,  std::allocator<std::complex<float>>>)
APIARY_INSTANTIATE_AS("block_copy", einsums::RuntimeTensorView<std::complex<double>>,                                    einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
// dst is an owning tensor, src is a view, common when extracting a slab into a freshly-allocated dst.
APIARY_INSTANTIATE_AS("block_copy", einsums::GeneralRuntimeTensor<float,                std::allocator<float>>,                einsums::RuntimeTensorView<float>)
APIARY_INSTANTIATE_AS("block_copy", einsums::GeneralRuntimeTensor<double,               std::allocator<double>>,               einsums::RuntimeTensorView<double>)
APIARY_INSTANTIATE_AS("block_copy", einsums::GeneralRuntimeTensor<std::complex<float>,  std::allocator<std::complex<float>>>,  einsums::RuntimeTensorView<std::complex<float>>)
APIARY_INSTANTIATE_AS("block_copy", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::RuntimeTensorView<std::complex<double>>)
// Both view, copying between two captured view slabs.
APIARY_INSTANTIATE_AS("block_copy", einsums::RuntimeTensorView<float>,                                                   einsums::RuntimeTensorView<float>)
APIARY_INSTANTIATE_AS("block_copy", einsums::RuntimeTensorView<double>,                                                  einsums::RuntimeTensorView<double>)
APIARY_INSTANTIATE_AS("block_copy", einsums::RuntimeTensorView<std::complex<float>>,                                     einsums::RuntimeTensorView<std::complex<float>>)
APIARY_INSTANTIATE_AS("block_copy", einsums::RuntimeTensorView<std::complex<double>>,                                    einsums::RuntimeTensorView<std::complex<double>>)
    // clang-format on
    void block_copy(DstType *dst, SrcType const &src, std::vector<size_t> dst_offsets, std::vector<size_t> src_offsets,
                    std::vector<size_t> extents) {
    size_t const N = extents.size();
    if (N == 0) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "cg::block_copy: extents must be non-empty");
    }
    if (dst->rank() != N || src.rank() != N) {
        EINSUMS_THROW_EXCEPTION(rank_error, "cg::block_copy: rank mismatch — dst rank={}, src rank={}, extents.size()={}", dst->rank(),
                                src.rank(), N);
    }
    if (dst_offsets.size() != N || src_offsets.size() != N) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "cg::block_copy: dst_offsets ({}) and src_offsets ({}) must match extents ({})",
                                dst_offsets.size(), src_offsets.size(), N);
    }
    for (size_t k = 0; k < N; ++k) {
        if (dst_offsets[k] + extents[k] > dst->dim(k)) {
            EINSUMS_THROW_EXCEPTION(std::out_of_range, "cg::block_copy: dst axis {} — offset {} + extent {} exceeds dim {}", k,
                                    dst_offsets[k], extents[k], dst->dim(k));
        }
        if (src_offsets[k] + extents[k] > src.dim(k)) {
            EINSUMS_THROW_EXCEPTION(std::out_of_range, "cg::block_copy: src axis {} — offset {} + extent {} exceeds dim {}", k,
                                    src_offsets[k], extents[k], src.dim(k));
        }
    }

    auto apply = [dst_offsets, src_offsets, extents, N](DstType *d, SrcType const *s) {
        using T      = typename DstType::ValueType;
        size_t total = 1;
        for (size_t k = 0; k < N; ++k)
            total *= extents[k];

        std::vector<size_t> idx(N, 0);
        std::vector<size_t> d_str(N), s_str(N);
        for (size_t k = 0; k < N; ++k) {
            d_str[k] = d->stride(k);
            s_str[k] = s->stride(k);
        }
        T       *d_data = d->data();
        T const *s_data = s->data();

        for (size_t count = 0; count < total; ++count) {
            size_t d_off = 0, s_off = 0;
            for (size_t k = 0; k < N; ++k) {
                d_off += (dst_offsets[k] + idx[k]) * d_str[k];
                s_off += (src_offsets[k] + idx[k]) * s_str[k];
            }
            d_data[d_off] = s_data[s_off];
            // Axis 0 fastest, correctness-only, not cache-aware.
            for (size_t k = 0; k < N; ++k) {
                if (++idx[k] < extents[k])
                    break;
                idx[k] = 0;
            }
        }
    };

    auto &ctx = CaptureContext::current();
    if (!ctx.is_capturing()) {
        LabeledSection("block_copy eager");
        apply(dst, &src);
        return;
    }

    LabeledSection("block_copy capture");
    auto [s_id, s_slot] = ctx.get_slot(src);
    auto [d_id, d_slot] = ctx.get_slot(*dst);

    auto executor = [s_slot, d_slot, apply]() {
        LabeledSection("block_copy execute");
        apply(static_cast<DstType *>(d_slot->ptr), static_cast<SrcType const *>(s_slot->ptr));
    };
    ctx.record(OpKind::Custom, "block_copy", {s_id}, {d_id}, std::move(executor));
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
        LabeledSection("element_transform eager");
        tensor_algebra::element_transform(C, unary_op);
        return;
    }

    LabeledSection("element_transform capture");
    auto [c_id, c_slot] = ctx.get_slot(*C);

    auto executor = [c_slot, unary_op]() {
        LabeledSection("element_transform execute");
        tensor_algebra::element_transform(static_cast<CType *>(c_slot->ptr), unary_op);
    };

    ctx.record(OpKind::ElementTransform, "element_transform", {c_id}, {c_id}, std::move(executor));
}

/// Tiled element_transform: apply @p unary_op to every stored tile. (The generic
/// overload requires BasicTensorConcept, which a tiled tensor no longer
/// satisfies, so this is selected unambiguously for tiled operands.)
template <TiledTensorConcept CType, typename UnaryOperator>
void element_transform(CType *C, UnaryOperator unary_op) {
    auto &ctx = CaptureContext::current();
    if (!ctx.is_capturing()) {
        LabeledSection("element_transform eager");
        detail::tiled_element_transform(C, unary_op);
        return;
    }
    LabeledSection("element_transform capture");
    auto [c_id, c_slot] = ctx.get_slot(*C);
    auto executor       = [c_slot, unary_op]() {
        LabeledSection("element_transform execute");
        detail::tiled_element_transform(static_cast<CType *>(c_slot->ptr), unary_op);
    };
    ctx.record(OpKind::Custom, "tiled element_transform", {c_id}, {c_id}, std::move(executor));
}

/// Python-friendly element_transform wrapper.
///
/// The generic ``element_transform`` template requires ``RankTensorConcept``
/// (compile-time rank), which ``GeneralRuntimeTensor`` doesn't satisfy, so it
/// can't be reused here. This overload walks the contiguous underlying storage
/// directly and accepts ``std::function<T(T)>`` so pybind11's caster can wrap a
/// Python callable. A serial loop (rather than the OMP-parallel path used by
/// ``tensor_algebra::element_transform``) keeps the per-call GIL acquire from
/// causing thread contention, which is fine for the small unary maps typical of
/// SCF/MP2 (eigenvalues, denominators).
template <typename TensorType>
    requires(CoreBasicTensorConcept<TensorType> || IsTiledTensorV<std::remove_cvref_t<TensorType>>)
// clang-format off
APIARY_EXPOSE
APIARY_MODULE("linalg")
APIARY_INSTANTIATE_AS("element_transform", einsums::GeneralRuntimeTensor<float, std::allocator<float>>)
APIARY_INSTANTIATE_AS("element_transform", einsums::GeneralRuntimeTensor<double, std::allocator<double>>)
APIARY_INSTANTIATE_AS("element_transform", einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>)
APIARY_INSTANTIATE_AS("element_transform", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
APIARY_INSTANTIATE_AS("element_transform", einsums::RuntimeTensorView<float>)
APIARY_INSTANTIATE_AS("element_transform", einsums::RuntimeTensorView<double>)
APIARY_INSTANTIATE_AS("element_transform", einsums::RuntimeTensorView<std::complex<float>>)
APIARY_INSTANTIATE_AS("element_transform", einsums::RuntimeTensorView<std::complex<double>>)
APIARY_INSTANTIATE_AS("element_transform", einsums::TiledRuntimeTensor<float>)
APIARY_INSTANTIATE_AS("element_transform", einsums::TiledRuntimeTensor<double>)
APIARY_INSTANTIATE_AS("element_transform", einsums::TiledRuntimeTensor<std::complex<float>>)
APIARY_INSTANTIATE_AS("element_transform", einsums::TiledRuntimeTensor<std::complex<double>>)
    // clang-format on
    void element_transform_python(TensorType *C, std::function<typename TensorType::ValueType(typename TensorType::ValueType)> unary_op) {
    using T = typename TensorType::ValueType;

    // Operate on one data-bearing unit (the whole dense tensor, or a single
    // tile). Generic so it accepts both TensorType* and RuntimeTensor<T>*.
    auto apply = [unary_op](auto *target) {
        T           *data = target->data();
        size_t const n    = target->size();
#if defined(PyEinsums_EXPORTS)
        // The callback is a Python callable. Hold the GIL across the whole loop
        // (cheaper than pybind's per-call acquire) and, crucially, translate any
        // Python exception to a plain C++ exception *while the GIL is held*. If a
        // pybind11::error_already_set were allowed to escape onto a parallel
        // executor's worker thread, its later off-GIL destruction corrupts the
        // CPython thread state and crashes at interpreter finalization. A
        // std::runtime_error carries safely across threads (the executors then
        // propagate it to the waiter, which re-raises it as a Python RuntimeError).
        pybind11::gil_scoped_acquire const gil;
        try {
            for (size_t i = 0; i < n; ++i) {
                data[i] = unary_op(data[i]);
            }
        } catch (pybind11::error_already_set const &e) {
            throw std::runtime_error(std::string("element_transform callback raised: ") + e.what());
        }
#else
        for (size_t i = 0; i < n; ++i) {
            data[i] = unary_op(data[i]);
        }
#endif
    };

    // Apply across the whole tensor: one call for dense, once per (materialized)
    // tile for tiled. Absent tiles are zero and left untouched.
    auto run = [apply](TensorType *target) {
        if constexpr (IsTiledTensorV<std::remove_cvref_t<TensorType>>) {
            for (auto &kv : target->tiles()) {
                kv.second.materialize();
                apply(&kv.second);
            }
        } else {
            apply(target);
        }
    };

    auto &ctx = CaptureContext::current();
    if (!ctx.is_capturing()) {
        LabeledSection("element_transform_python eager");
        run(C);
        return;
    }

    LabeledSection("element_transform_python capture");
    auto [c_id, c_slot] = ctx.get_slot(*C);
    auto executor       = [c_slot, run]() {
        LabeledSection("element_transform_python execute");
        run(static_cast<TensorType *>(c_slot->ptr));
    };
    ctx.record(OpKind::ElementTransform, "element_transform", {c_id}, {c_id}, std::move(executor));
}

// ─────────────────────────────────────────────────────────────────────────────
// shift: A += beta (add a scalar to every element)
// ─────────────────────────────────────────────────────────────────────────────

/// Graph-aware in-place scalar shift: ``A += beta``.
///
/// The additive complement of @ref scale. Unlike the Python-callable
/// @ref element_transform_python (a Python call per element), this is a tight
/// compiled loop, so it's the fast backing for the numpy-style scalar ``+`` /
/// ``-`` operators (e.g. ``A + 1.0``, ``A += c``). It works in place with no
/// allocation, which is what lets a denominator scratch be reused across a loop
/// instead of allocating a fresh tensor per iteration. Records an opaque @ref
/// OpKind::Custom node when capturing (no einsum pass rewrites it), or runs
/// eagerly otherwise.
///
/// Walks the contiguous backing storage (``data()[0..size())``), matching
/// @ref element_transform_python's convention. This is correct for dense tensors
/// and the contiguous views the operators produce.
template <CoreBasicTensorConcept AType>
// clang-format off
APIARY_EXPOSE
APIARY_MODULE("linalg")
APIARY_INSTANTIATE_AS("shift", einsums::GeneralRuntimeTensor<float, std::allocator<float>>)
APIARY_INSTANTIATE_AS("shift", einsums::GeneralRuntimeTensor<double, std::allocator<double>>)
APIARY_INSTANTIATE_AS("shift", einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>)
APIARY_INSTANTIATE_AS("shift", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
APIARY_INSTANTIATE_AS("shift", einsums::RuntimeTensorView<float>)
APIARY_INSTANTIATE_AS("shift", einsums::RuntimeTensorView<double>)
APIARY_INSTANTIATE_AS("shift", einsums::RuntimeTensorView<std::complex<float>>)
APIARY_INSTANTIATE_AS("shift", einsums::RuntimeTensorView<std::complex<double>>)
    // clang-format on
    void shift(typename AType::ValueType beta, AType *A) {
    using T = typename AType::ValueType;

    auto run = [beta](AType *target) {
        T           *data = target->data();
        size_t const n    = target->size();
        for (size_t i = 0; i < n; ++i) {
            data[i] += beta;
        }
    };

    auto &ctx = CaptureContext::current();
    if (!ctx.is_capturing()) {
        LabeledSection("shift eager");
        run(A);
        return;
    }

    LabeledSection("shift capture");
    auto [a_id, a_slot] = ctx.get_slot(*A);
    auto label          = fmt::format("shift({})", A->name());
    auto executor       = [a_slot, run]() {
        LabeledSection("shift execute");
        run(static_cast<AType *>(a_slot->ptr));
    };
    ctx.record(OpKind::Custom, std::move(label), {a_id}, {a_id}, std::move(executor));
}

// ─────────────────────────────────────────────────────────────────────────────
// axpy: Y += alpha * X
// ─────────────────────────────────────────────────────────────────────────────

/// Graph-aware AXPY: ``Y += alpha * X`` (BLAS level-1).
///
/// X and Y must have the same dtype and shape; the operation is
/// element-wise. Eager outside graph capture; recorded as a node when
/// inside a capture context.
template <TensorConcept XType, TensorConcept YType>
    requires SameUnderlying<XType, YType>
// clang-format off
APIARY_EXPOSE
APIARY_MODULE("linalg")
// All 4 combinations of (X, Y) x (owning, view), per dtype. Same-dtype
// across operands is enforced by SameUnderlying above.
//
// float
APIARY_INSTANTIATE_AS("axpy", einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::GeneralRuntimeTensor<float, std::allocator<float>>)
APIARY_INSTANTIATE_AS("axpy", einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::RuntimeTensorView<float>)
APIARY_INSTANTIATE_AS("axpy", einsums::RuntimeTensorView<float>,                          einsums::GeneralRuntimeTensor<float, std::allocator<float>>)
APIARY_INSTANTIATE_AS("axpy", einsums::RuntimeTensorView<float>,                          einsums::RuntimeTensorView<float>)
// double
APIARY_INSTANTIATE_AS("axpy", einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::GeneralRuntimeTensor<double, std::allocator<double>>)
APIARY_INSTANTIATE_AS("axpy", einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::RuntimeTensorView<double>)
APIARY_INSTANTIATE_AS("axpy", einsums::RuntimeTensorView<double>,                          einsums::GeneralRuntimeTensor<double, std::allocator<double>>)
APIARY_INSTANTIATE_AS("axpy", einsums::RuntimeTensorView<double>,                          einsums::RuntimeTensorView<double>)
// complex<float>
APIARY_INSTANTIATE_AS("axpy", einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>)
APIARY_INSTANTIATE_AS("axpy", einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::RuntimeTensorView<std::complex<float>>)
APIARY_INSTANTIATE_AS("axpy", einsums::RuntimeTensorView<std::complex<float>>,                                            einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>)
APIARY_INSTANTIATE_AS("axpy", einsums::RuntimeTensorView<std::complex<float>>,                                            einsums::RuntimeTensorView<std::complex<float>>)
// complex<double>
APIARY_INSTANTIATE_AS("axpy", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
APIARY_INSTANTIATE_AS("axpy", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::RuntimeTensorView<std::complex<double>>)
APIARY_INSTANTIATE_AS("axpy", einsums::RuntimeTensorView<std::complex<double>>,                                          einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
APIARY_INSTANTIATE_AS("axpy", einsums::RuntimeTensorView<std::complex<double>>,                                          einsums::RuntimeTensorView<std::complex<double>>)
APIARY_INSTANTIATE_AS("axpy", einsums::TiledRuntimeTensor<float>, einsums::TiledRuntimeTensor<float>)
APIARY_INSTANTIATE_AS("axpy", einsums::TiledRuntimeTensor<double>, einsums::TiledRuntimeTensor<double>)
APIARY_INSTANTIATE_AS("axpy", einsums::TiledRuntimeTensor<std::complex<float>>, einsums::TiledRuntimeTensor<std::complex<float>>)
APIARY_INSTANTIATE_AS("axpy", einsums::TiledRuntimeTensor<std::complex<double>>, einsums::TiledRuntimeTensor<std::complex<double>>)
    // clang-format on
    void axpy(typename XType::ValueType alpha, XType const &X, YType *Y) {
    if constexpr (IsTiledTensorV<std::remove_cvref_t<XType>> || IsTiledTensorV<std::remove_cvref_t<YType>>) {
        static_assert(IsTiledTensorV<std::remove_cvref_t<XType>> && IsTiledTensorV<std::remove_cvref_t<YType>>,
                      "cg::axpy with a tiled operand requires both X and Y to be TiledRuntimeTensor");
        using T   = typename XType::ValueType;
        auto &ctx = CaptureContext::current();
        if (!ctx.is_capturing()) {
            LabeledSection("axpy eager");
            detail::tiled_axpy<T>(alpha, X, Y);
            return;
        }
        LabeledSection("axpy capture");
        auto [x_id, x_slot] = ctx.get_slot(X);
        auto [y_id, y_slot] = ctx.get_slot(*Y);
        auto label          = fmt::format("tiled axpy({}, {})", X.name(), Y->name());
        auto executor       = [alpha, x_slot, y_slot]() {
            LabeledSection("axpy execute");
            detail::tiled_axpy<T>(alpha, *static_cast<XType const *>(x_slot->ptr), static_cast<YType *>(y_slot->ptr));
        };
        ctx.record(OpKind::Custom, std::move(label), {x_id}, {y_id}, std::move(executor));
    } else {
        auto &ctx = CaptureContext::current();
        if (!ctx.is_capturing()) {
            LabeledSection("axpy eager");
            linear_algebra::axpy(alpha, X, Y);
            return;
        }

        LabeledSection("axpy capture");
        auto [x_id, x_slot] = ctx.get_slot(X);
        auto [y_id, y_slot] = ctx.get_slot(*Y);

        auto label    = fmt::format("axpy(alpha={}, {}, {})", alpha, X.name(), Y->name());
        auto executor = [alpha, x_slot, y_slot]() {
            LabeledSection("axpy execute");
            linear_algebra::axpy(alpha, *static_cast<XType const *>(x_slot->ptr), static_cast<YType *>(y_slot->ptr));
        };

        ctx.record(OpKind::Axpy, std::move(label), {x_id}, {y_id}, std::move(executor));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// axpby: Y = alpha * X + beta * Y
// ─────────────────────────────────────────────────────────────────────────────

/// Graph-aware AXPBY: ``Y = alpha * X + beta * Y`` (extended BLAS level-1).
///
/// Like ``axpy`` but also scales the destination by ``beta`` first.
/// X and Y must have the same dtype and shape.
template <TensorConcept XType, TensorConcept YType>
    requires SameUnderlying<XType, YType>
// clang-format off
APIARY_EXPOSE
APIARY_MODULE("linalg")
// All 4 combinations of (X, Y) x (owning, view), per dtype.
//
// float
APIARY_INSTANTIATE_AS("axpby", einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::GeneralRuntimeTensor<float, std::allocator<float>>)
APIARY_INSTANTIATE_AS("axpby", einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::RuntimeTensorView<float>)
APIARY_INSTANTIATE_AS("axpby", einsums::RuntimeTensorView<float>,                          einsums::GeneralRuntimeTensor<float, std::allocator<float>>)
APIARY_INSTANTIATE_AS("axpby", einsums::RuntimeTensorView<float>,                          einsums::RuntimeTensorView<float>)
// double
APIARY_INSTANTIATE_AS("axpby", einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::GeneralRuntimeTensor<double, std::allocator<double>>)
APIARY_INSTANTIATE_AS("axpby", einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::RuntimeTensorView<double>)
APIARY_INSTANTIATE_AS("axpby", einsums::RuntimeTensorView<double>,                          einsums::GeneralRuntimeTensor<double, std::allocator<double>>)
APIARY_INSTANTIATE_AS("axpby", einsums::RuntimeTensorView<double>,                          einsums::RuntimeTensorView<double>)
// complex<float>
APIARY_INSTANTIATE_AS("axpby", einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>)
APIARY_INSTANTIATE_AS("axpby", einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::RuntimeTensorView<std::complex<float>>)
APIARY_INSTANTIATE_AS("axpby", einsums::RuntimeTensorView<std::complex<float>>,                                            einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>)
APIARY_INSTANTIATE_AS("axpby", einsums::RuntimeTensorView<std::complex<float>>,                                            einsums::RuntimeTensorView<std::complex<float>>)
// complex<double>
APIARY_INSTANTIATE_AS("axpby", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
APIARY_INSTANTIATE_AS("axpby", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::RuntimeTensorView<std::complex<double>>)
APIARY_INSTANTIATE_AS("axpby", einsums::RuntimeTensorView<std::complex<double>>,                                          einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
APIARY_INSTANTIATE_AS("axpby", einsums::RuntimeTensorView<std::complex<double>>,                                          einsums::RuntimeTensorView<std::complex<double>>)
    // clang-format on
    void axpby(typename XType::ValueType alpha, XType const &X, typename XType::ValueType beta, YType *Y) {
    auto &ctx = CaptureContext::current();
    if (!ctx.is_capturing()) {
        LabeledSection("axpby eager");
        linear_algebra::axpby(alpha, X, beta, Y);
        return;
    }

    LabeledSection("axpby capture");
    auto [x_id, x_slot] = ctx.get_slot(X);
    auto [y_id, y_slot] = ctx.get_slot(*Y);

    auto label    = fmt::format("axpby(alpha={}, beta={})", alpha, beta);
    auto executor = [alpha, x_slot, beta, y_slot]() {
        LabeledSection("axpby execute");
        linear_algebra::axpby(alpha, *static_cast<XType const *>(x_slot->ptr), beta, static_cast<YType *>(y_slot->ptr));
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
        LabeledSection("gemm eager");
        linear_algebra::gemm<TransA, TransB>(alpha, A, B, beta, C);
        return;
    }

    LabeledSection("gemm capture");
    auto [a_id, a_slot] = ctx.get_slot(A);
    auto [b_id, b_slot] = ctx.get_slot(B);
    auto [c_id, c_slot] = ctx.get_slot(*C);

    auto label    = fmt::format("gemm<{},{}>", TransA ? "T" : "N", TransB ? "T" : "N");
    auto executor = [alpha, a_slot, b_slot, beta, c_slot]() {
        LabeledSection("gemm execute");
        ProfileAnnotate("trans", TransA ? (TransB ? "TT" : "TN") : (TransB ? "NT" : "NN"));
        ProfileAnnotate("m", static_cast<int64_t>(static_cast<T *>(c_slot->ptr)->dim(0)));
        ProfileAnnotate("n", static_cast<int64_t>(static_cast<T *>(c_slot->ptr)->dim(1)));
        ProfileAnnotate(
            "k", static_cast<int64_t>(TransA ? static_cast<T const *>(a_slot->ptr)->dim(0) : static_cast<T const *>(a_slot->ptr)->dim(1)));
        linear_algebra::gemm<TransA, TransB>(alpha, *static_cast<T const *>(a_slot->ptr), *static_cast<T const *>(b_slot->ptr), beta,
                                             static_cast<T *>(c_slot->ptr));
    };

    // When beta != 0 the gemm accumulates into C (``C = α·A·B + β·C``), so it
    // *reads* C as well as writing it. List C as an input in that case so the
    // scheduler and loop-invariance analysis see the read-modify-write, without
    // it, an accumulating gemm looks like a pure producer and can be wrongly
    // hoisted out of a loop or reordered. A pure overwrite (beta == 0) keeps the
    // two-input form and stays eligible for those optimizations.
    std::vector<TensorId> inputs = {a_id, b_id};
    if (beta != U{}) {
        inputs.push_back(c_id);
    }
    ctx.record(OpKind::Gemm, std::move(label), std::move(inputs), {c_id}, std::move(executor));
}

/// Graph-aware GEMM: ``C = alpha * op(A) * op(B) + beta * C``.
///
/// ``trans_a`` and ``trans_b`` (Python kwargs, default ``False``) request
/// the transpose of the corresponding matrix. All three tensors must be
/// rank 2; a clear ``rank_error`` is raised up front otherwise rather
/// than letting the BLAS kernel fail mid-pipeline.
///
/// A, B, C may be any combination of owning ``RuntimeTensor`` and
/// ``RuntimeTensorView`` so long as they share an underlying element type.
/// Views alias their parents so writes through C-view land in the parent
/// and the optimization passes see the dependency via ``TensorHandle::aliases``.
template <bool TransA, bool TransB, RuntimeRankTensorConcept AType, RuntimeRankTensorConcept BType, RuntimeRankTensorConcept CType,
          typename U>
    requires requires {
        requires std::convertible_to<U, typename AType::ValueType>;
        requires SameUnderlying<AType, BType, CType>;
    }
// clang-format off
APIARY_EXPOSE
APIARY_MODULE("linalg")
APIARY_TEMPLATE_KWARGS("trans_a", "trans_b")
// All 8 combinations of (A, B, C) x (owning, view), per dtype, per (TransA, TransB)
// bool pair. INSTANTIATE_BOOLS expands each line to 4 entries (T/F)x(T/F).
//
// float: AAA/AAV/AVA/AVV/VAA/VAV/VVA/VVV (A = owning, V = view)
APIARY_INSTANTIATE_BOOLS("gemm", einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::GeneralRuntimeTensor<float, std::allocator<float>>, float)
APIARY_INSTANTIATE_BOOLS("gemm", einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::RuntimeTensorView<float>,                          float)
APIARY_INSTANTIATE_BOOLS("gemm", einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::RuntimeTensorView<float>,                          einsums::GeneralRuntimeTensor<float, std::allocator<float>>, float)
APIARY_INSTANTIATE_BOOLS("gemm", einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::RuntimeTensorView<float>,                          einsums::RuntimeTensorView<float>,                          float)
APIARY_INSTANTIATE_BOOLS("gemm", einsums::RuntimeTensorView<float>,                          einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::GeneralRuntimeTensor<float, std::allocator<float>>, float)
APIARY_INSTANTIATE_BOOLS("gemm", einsums::RuntimeTensorView<float>,                          einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::RuntimeTensorView<float>,                          float)
APIARY_INSTANTIATE_BOOLS("gemm", einsums::RuntimeTensorView<float>,                          einsums::RuntimeTensorView<float>,                          einsums::GeneralRuntimeTensor<float, std::allocator<float>>, float)
APIARY_INSTANTIATE_BOOLS("gemm", einsums::RuntimeTensorView<float>,                          einsums::RuntimeTensorView<float>,                          einsums::RuntimeTensorView<float>,                          float)
// double
APIARY_INSTANTIATE_BOOLS("gemm", einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::GeneralRuntimeTensor<double, std::allocator<double>>, double)
APIARY_INSTANTIATE_BOOLS("gemm", einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::RuntimeTensorView<double>,                          double)
APIARY_INSTANTIATE_BOOLS("gemm", einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::RuntimeTensorView<double>,                          einsums::GeneralRuntimeTensor<double, std::allocator<double>>, double)
APIARY_INSTANTIATE_BOOLS("gemm", einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::RuntimeTensorView<double>,                          einsums::RuntimeTensorView<double>,                          double)
APIARY_INSTANTIATE_BOOLS("gemm", einsums::RuntimeTensorView<double>,                          einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::GeneralRuntimeTensor<double, std::allocator<double>>, double)
APIARY_INSTANTIATE_BOOLS("gemm", einsums::RuntimeTensorView<double>,                          einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::RuntimeTensorView<double>,                          double)
APIARY_INSTANTIATE_BOOLS("gemm", einsums::RuntimeTensorView<double>,                          einsums::RuntimeTensorView<double>,                          einsums::GeneralRuntimeTensor<double, std::allocator<double>>, double)
APIARY_INSTANTIATE_BOOLS("gemm", einsums::RuntimeTensorView<double>,                          einsums::RuntimeTensorView<double>,                          einsums::RuntimeTensorView<double>,                          double)
// complex<float>
APIARY_INSTANTIATE_BOOLS("gemm", einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, std::complex<float>)
APIARY_INSTANTIATE_BOOLS("gemm", einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::RuntimeTensorView<std::complex<float>>,                                            std::complex<float>)
APIARY_INSTANTIATE_BOOLS("gemm", einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::RuntimeTensorView<std::complex<float>>,                                            einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, std::complex<float>)
APIARY_INSTANTIATE_BOOLS("gemm", einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::RuntimeTensorView<std::complex<float>>,                                            einsums::RuntimeTensorView<std::complex<float>>,                                            std::complex<float>)
APIARY_INSTANTIATE_BOOLS("gemm", einsums::RuntimeTensorView<std::complex<float>>,                                            einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, std::complex<float>)
APIARY_INSTANTIATE_BOOLS("gemm", einsums::RuntimeTensorView<std::complex<float>>,                                            einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::RuntimeTensorView<std::complex<float>>,                                            std::complex<float>)
APIARY_INSTANTIATE_BOOLS("gemm", einsums::RuntimeTensorView<std::complex<float>>,                                            einsums::RuntimeTensorView<std::complex<float>>,                                            einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, std::complex<float>)
APIARY_INSTANTIATE_BOOLS("gemm", einsums::RuntimeTensorView<std::complex<float>>,                                            einsums::RuntimeTensorView<std::complex<float>>,                                            einsums::RuntimeTensorView<std::complex<float>>,                                            std::complex<float>)
// complex<double>
APIARY_INSTANTIATE_BOOLS("gemm", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, std::complex<double>)
APIARY_INSTANTIATE_BOOLS("gemm", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::RuntimeTensorView<std::complex<double>>,                                          std::complex<double>)
APIARY_INSTANTIATE_BOOLS("gemm", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::RuntimeTensorView<std::complex<double>>,                                          einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, std::complex<double>)
APIARY_INSTANTIATE_BOOLS("gemm", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::RuntimeTensorView<std::complex<double>>,                                          einsums::RuntimeTensorView<std::complex<double>>,                                          std::complex<double>)
APIARY_INSTANTIATE_BOOLS("gemm", einsums::RuntimeTensorView<std::complex<double>>,                                          einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, std::complex<double>)
APIARY_INSTANTIATE_BOOLS("gemm", einsums::RuntimeTensorView<std::complex<double>>,                                          einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::RuntimeTensorView<std::complex<double>>,                                          std::complex<double>)
APIARY_INSTANTIATE_BOOLS("gemm", einsums::RuntimeTensorView<std::complex<double>>,                                          einsums::RuntimeTensorView<std::complex<double>>,                                          einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, std::complex<double>)
APIARY_INSTANTIATE_BOOLS("gemm", einsums::RuntimeTensorView<std::complex<double>>,                                          einsums::RuntimeTensorView<std::complex<double>>,                                          einsums::RuntimeTensorView<std::complex<double>>,                                          std::complex<double>)
    // clang-format on
    void gemm(U const alpha, AType const &A, BType const &B, U const beta, CType *C) {
    if (A.rank() != 2 || B.rank() != 2 || C->rank() != 2) {
        EINSUMS_THROW_EXCEPTION(rank_error, "cg::gemm requires rank-2 tensors; got ranks {}, {}, {}.", A.rank(), B.rank(), C->rank());
    }

    auto &ctx = CaptureContext::current();
    if (!ctx.is_capturing()) {
        LabeledSection("gemm eager");
        linear_algebra::gemm<TransA, TransB>(alpha, A, B, beta, C);
        return;
    }

    LabeledSection("gemm capture");
    auto [a_id, a_slot] = ctx.get_slot(A);
    auto [b_id, b_slot] = ctx.get_slot(B);
    auto [c_id, c_slot] = ctx.get_slot(*C);

    auto label    = fmt::format("gemm<{},{}>", TransA ? "T" : "N", TransB ? "T" : "N");
    auto executor = [alpha, a_slot, b_slot, beta, c_slot]() {
        LabeledSection("gemm execute");
        ProfileAnnotate("trans", TransA ? (TransB ? "TT" : "TN") : (TransB ? "NT" : "NN"));
        ProfileAnnotate("m", static_cast<int64_t>(static_cast<CType *>(c_slot->ptr)->dim(0)));
        ProfileAnnotate("n", static_cast<int64_t>(static_cast<CType *>(c_slot->ptr)->dim(1)));
        ProfileAnnotate("k", static_cast<int64_t>(TransA ? static_cast<AType const *>(a_slot->ptr)->dim(0)
                                                         : static_cast<AType const *>(a_slot->ptr)->dim(1)));
        linear_algebra::gemm<TransA, TransB>(alpha, *static_cast<AType const *>(a_slot->ptr), *static_cast<BType const *>(b_slot->ptr),
                                             beta, static_cast<CType *>(c_slot->ptr));
    };

    // beta != 0 → the gemm reads C as well as writing it (``C = α·A·B + β·C``);
    // list C as an input so loop-invariance and scheduling see the read. See the
    // matching note on the TransA/TransB overload above.
    std::vector<TensorId> inputs = {a_id, b_id};
    if (beta != U{}) {
        inputs.push_back(c_id);
    }
    ctx.record(OpKind::Gemm, std::move(label), std::move(inputs), {c_id}, std::move(executor));
}

/// Graph-aware GEMM with runtime ``Transpose`` op flags (N / T / C).
///
/// The companion of the bool ``trans_a``/``trans_b`` overload above; this one
/// adds conjugate-transpose (``Transpose::C``) for complex operands by routing
/// through the runtime-char ``linear_algebra::gemm`` (BLAS 'c'). The bool
/// overload still resolves ``trans_a=True/False``; pass ``trans_a=Transpose.C``
/// to reach this one. Defaults are ``Transpose::N`` so an untransposed call is
/// unambiguous.
template <RuntimeRankTensorConcept AType, RuntimeRankTensorConcept BType, RuntimeRankTensorConcept CType, typename U>
    requires requires {
        requires std::convertible_to<U, typename AType::ValueType>;
        requires SameUnderlying<AType, BType, CType>;
    }
// clang-format off
APIARY_EXPOSE
APIARY_MODULE("linalg")
// float — AAA/AAV/AVA/AVV/VAA/VAV/VVA/VVV (A = owning, V = view)
APIARY_INSTANTIATE_AS("gemm", einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::GeneralRuntimeTensor<float, std::allocator<float>>, float)
APIARY_INSTANTIATE_AS("gemm", einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::RuntimeTensorView<float>,                          float)
APIARY_INSTANTIATE_AS("gemm", einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::RuntimeTensorView<float>,                          einsums::GeneralRuntimeTensor<float, std::allocator<float>>, float)
APIARY_INSTANTIATE_AS("gemm", einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::RuntimeTensorView<float>,                          einsums::RuntimeTensorView<float>,                          float)
APIARY_INSTANTIATE_AS("gemm", einsums::RuntimeTensorView<float>,                          einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::GeneralRuntimeTensor<float, std::allocator<float>>, float)
APIARY_INSTANTIATE_AS("gemm", einsums::RuntimeTensorView<float>,                          einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::RuntimeTensorView<float>,                          float)
APIARY_INSTANTIATE_AS("gemm", einsums::RuntimeTensorView<float>,                          einsums::RuntimeTensorView<float>,                          einsums::GeneralRuntimeTensor<float, std::allocator<float>>, float)
APIARY_INSTANTIATE_AS("gemm", einsums::RuntimeTensorView<float>,                          einsums::RuntimeTensorView<float>,                          einsums::RuntimeTensorView<float>,                          float)
// double
APIARY_INSTANTIATE_AS("gemm", einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::GeneralRuntimeTensor<double, std::allocator<double>>, double)
APIARY_INSTANTIATE_AS("gemm", einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::RuntimeTensorView<double>,                          double)
APIARY_INSTANTIATE_AS("gemm", einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::RuntimeTensorView<double>,                          einsums::GeneralRuntimeTensor<double, std::allocator<double>>, double)
APIARY_INSTANTIATE_AS("gemm", einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::RuntimeTensorView<double>,                          einsums::RuntimeTensorView<double>,                          double)
APIARY_INSTANTIATE_AS("gemm", einsums::RuntimeTensorView<double>,                          einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::GeneralRuntimeTensor<double, std::allocator<double>>, double)
APIARY_INSTANTIATE_AS("gemm", einsums::RuntimeTensorView<double>,                          einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::RuntimeTensorView<double>,                          double)
APIARY_INSTANTIATE_AS("gemm", einsums::RuntimeTensorView<double>,                          einsums::RuntimeTensorView<double>,                          einsums::GeneralRuntimeTensor<double, std::allocator<double>>, double)
APIARY_INSTANTIATE_AS("gemm", einsums::RuntimeTensorView<double>,                          einsums::RuntimeTensorView<double>,                          einsums::RuntimeTensorView<double>,                          double)
// complex<float>
APIARY_INSTANTIATE_AS("gemm", einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, std::complex<float>)
APIARY_INSTANTIATE_AS("gemm", einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::RuntimeTensorView<std::complex<float>>,                                            std::complex<float>)
APIARY_INSTANTIATE_AS("gemm", einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::RuntimeTensorView<std::complex<float>>,                                            einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, std::complex<float>)
APIARY_INSTANTIATE_AS("gemm", einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::RuntimeTensorView<std::complex<float>>,                                            einsums::RuntimeTensorView<std::complex<float>>,                                            std::complex<float>)
APIARY_INSTANTIATE_AS("gemm", einsums::RuntimeTensorView<std::complex<float>>,                                            einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, std::complex<float>)
APIARY_INSTANTIATE_AS("gemm", einsums::RuntimeTensorView<std::complex<float>>,                                            einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::RuntimeTensorView<std::complex<float>>,                                            std::complex<float>)
APIARY_INSTANTIATE_AS("gemm", einsums::RuntimeTensorView<std::complex<float>>,                                            einsums::RuntimeTensorView<std::complex<float>>,                                            einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, std::complex<float>)
APIARY_INSTANTIATE_AS("gemm", einsums::RuntimeTensorView<std::complex<float>>,                                            einsums::RuntimeTensorView<std::complex<float>>,                                            einsums::RuntimeTensorView<std::complex<float>>,                                            std::complex<float>)
// complex<double>
APIARY_INSTANTIATE_AS("gemm", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, std::complex<double>)
APIARY_INSTANTIATE_AS("gemm", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::RuntimeTensorView<std::complex<double>>,                                          std::complex<double>)
APIARY_INSTANTIATE_AS("gemm", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::RuntimeTensorView<std::complex<double>>,                                          einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, std::complex<double>)
APIARY_INSTANTIATE_AS("gemm", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::RuntimeTensorView<std::complex<double>>,                                          einsums::RuntimeTensorView<std::complex<double>>,                                          std::complex<double>)
APIARY_INSTANTIATE_AS("gemm", einsums::RuntimeTensorView<std::complex<double>>,                                          einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, std::complex<double>)
APIARY_INSTANTIATE_AS("gemm", einsums::RuntimeTensorView<std::complex<double>>,                                          einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::RuntimeTensorView<std::complex<double>>,                                          std::complex<double>)
APIARY_INSTANTIATE_AS("gemm", einsums::RuntimeTensorView<std::complex<double>>,                                          einsums::RuntimeTensorView<std::complex<double>>,                                          einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, std::complex<double>)
APIARY_INSTANTIATE_AS("gemm", einsums::RuntimeTensorView<std::complex<double>>,                                          einsums::RuntimeTensorView<std::complex<double>>,                                          einsums::RuntimeTensorView<std::complex<double>>,                                          std::complex<double>)
    // clang-format on
    void gemm(U const alpha, AType const &A, BType const &B, U const beta, CType *C,
              linear_algebra::Transpose trans_a = linear_algebra::Transpose::N,
              linear_algebra::Transpose trans_b = linear_algebra::Transpose::N) {
    if (A.rank() != 2 || B.rank() != 2 || C->rank() != 2) {
        EINSUMS_THROW_EXCEPTION(rank_error, "cg::gemm requires rank-2 tensors; got ranks {}, {}, {}.", A.rank(), B.rank(), C->rank());
    }
    char const ta = static_cast<char>(trans_a), tb = static_cast<char>(trans_b);

    auto &ctx = CaptureContext::current();
    if (!ctx.is_capturing()) {
        LabeledSection("gemm eager");
        linear_algebra::gemm(ta, tb, alpha, A, B, beta, C);
        return;
    }

    LabeledSection("gemm capture");
    auto [a_id, a_slot] = ctx.get_slot(A);
    auto [b_id, b_slot] = ctx.get_slot(B);
    auto [c_id, c_slot] = ctx.get_slot(*C);

    auto label    = fmt::format("gemm({},{})", static_cast<char>(trans_a), static_cast<char>(trans_b));
    auto executor = [alpha, a_slot, b_slot, beta, c_slot, ta, tb]() {
        LabeledSection("gemm execute");
        linear_algebra::gemm(ta, tb, alpha, *static_cast<AType const *>(a_slot->ptr), *static_cast<BType const *>(b_slot->ptr), beta,
                             static_cast<CType *>(c_slot->ptr));
    };

    // beta != 0 → reads C as well as writing it; list C as input (see the bool overload's note).
    std::vector<TensorId> inputs = {a_id, b_id};
    if (beta != U{}) {
        inputs.push_back(c_id);
    }
    ctx.record(OpKind::Gemm, std::move(label), std::move(inputs), {c_id}, std::move(executor));
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
        LabeledSection("gemv eager");
        linear_algebra::gemv<TransA>(alpha, A, z, beta, y);
        return;
    }

    LabeledSection("gemv capture");
    auto [a_id, a_slot] = ctx.get_slot(A);
    auto [z_id, z_slot] = ctx.get_slot(z);
    auto [y_id, y_slot] = ctx.get_slot(*y);

    auto label    = fmt::format("gemv<{}>", TransA ? "T" : "N");
    auto executor = [alpha, a_slot, z_slot, beta, y_slot]() {
        LabeledSection("gemv execute");
        ProfileAnnotate("trans", TransA ? "T" : "N");
        ProfileAnnotate("m", static_cast<int64_t>(static_cast<AType const *>(a_slot->ptr)->dim(0)));
        ProfileAnnotate("n", static_cast<int64_t>(static_cast<AType const *>(a_slot->ptr)->dim(1)));
        linear_algebra::gemv<TransA>(alpha, *static_cast<AType const *>(a_slot->ptr), *static_cast<XType const *>(z_slot->ptr), beta,
                                     static_cast<YType *>(y_slot->ptr));
    };

    // beta != 0 → gemv reads y as well as writing it; list it as an input so
    // loop-invariance and scheduling see the read (see the gemm note above).
    std::vector<TensorId> inputs = {a_id, z_id};
    if (beta != U{}) {
        inputs.push_back(y_id);
    }
    ctx.record(OpKind::Gemv, std::move(label), std::move(inputs), {y_id}, std::move(executor));
}

/// Graph-aware GEMV: ``y = alpha * op(A) * z + beta * y``.
///
/// ``trans_a`` (Python kwarg, default ``False``) transposes A. A must be
/// rank 2 and z, y must be rank 1; a ``rank_error`` is raised otherwise.
template <bool TransA, RuntimeRankTensorConcept AType, RuntimeRankTensorConcept XType, RuntimeRankTensorConcept YType, typename U>
    requires(SameUnderlying<AType, XType, YType> && std::convertible_to<U, typename AType::ValueType>)
// clang-format off
APIARY_EXPOSE
APIARY_MODULE("linalg")
APIARY_TEMPLATE_KWARGS("trans_a")
// All 8 combinations of (A, X, Y) x (owning, view), per dtype. Same-dtype
// across operands is enforced by SameUnderlying above.
//
// float
APIARY_INSTANTIATE_BOOLS("gemv", einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::GeneralRuntimeTensor<float, std::allocator<float>>, float)
APIARY_INSTANTIATE_BOOLS("gemv", einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::RuntimeTensorView<float>,                          float)
APIARY_INSTANTIATE_BOOLS("gemv", einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::RuntimeTensorView<float>,                          einsums::GeneralRuntimeTensor<float, std::allocator<float>>, float)
APIARY_INSTANTIATE_BOOLS("gemv", einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::RuntimeTensorView<float>,                          einsums::RuntimeTensorView<float>,                          float)
APIARY_INSTANTIATE_BOOLS("gemv", einsums::RuntimeTensorView<float>,                          einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::GeneralRuntimeTensor<float, std::allocator<float>>, float)
APIARY_INSTANTIATE_BOOLS("gemv", einsums::RuntimeTensorView<float>,                          einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::RuntimeTensorView<float>,                          float)
APIARY_INSTANTIATE_BOOLS("gemv", einsums::RuntimeTensorView<float>,                          einsums::RuntimeTensorView<float>,                          einsums::GeneralRuntimeTensor<float, std::allocator<float>>, float)
APIARY_INSTANTIATE_BOOLS("gemv", einsums::RuntimeTensorView<float>,                          einsums::RuntimeTensorView<float>,                          einsums::RuntimeTensorView<float>,                          float)
// double
APIARY_INSTANTIATE_BOOLS("gemv", einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::GeneralRuntimeTensor<double, std::allocator<double>>, double)
APIARY_INSTANTIATE_BOOLS("gemv", einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::RuntimeTensorView<double>,                          double)
APIARY_INSTANTIATE_BOOLS("gemv", einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::RuntimeTensorView<double>,                          einsums::GeneralRuntimeTensor<double, std::allocator<double>>, double)
APIARY_INSTANTIATE_BOOLS("gemv", einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::RuntimeTensorView<double>,                          einsums::RuntimeTensorView<double>,                          double)
APIARY_INSTANTIATE_BOOLS("gemv", einsums::RuntimeTensorView<double>,                          einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::GeneralRuntimeTensor<double, std::allocator<double>>, double)
APIARY_INSTANTIATE_BOOLS("gemv", einsums::RuntimeTensorView<double>,                          einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::RuntimeTensorView<double>,                          double)
APIARY_INSTANTIATE_BOOLS("gemv", einsums::RuntimeTensorView<double>,                          einsums::RuntimeTensorView<double>,                          einsums::GeneralRuntimeTensor<double, std::allocator<double>>, double)
APIARY_INSTANTIATE_BOOLS("gemv", einsums::RuntimeTensorView<double>,                          einsums::RuntimeTensorView<double>,                          einsums::RuntimeTensorView<double>,                          double)
// complex<float>
APIARY_INSTANTIATE_BOOLS("gemv", einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, std::complex<float>)
APIARY_INSTANTIATE_BOOLS("gemv", einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::RuntimeTensorView<std::complex<float>>,                                            std::complex<float>)
APIARY_INSTANTIATE_BOOLS("gemv", einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::RuntimeTensorView<std::complex<float>>,                                            einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, std::complex<float>)
APIARY_INSTANTIATE_BOOLS("gemv", einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::RuntimeTensorView<std::complex<float>>,                                            einsums::RuntimeTensorView<std::complex<float>>,                                            std::complex<float>)
APIARY_INSTANTIATE_BOOLS("gemv", einsums::RuntimeTensorView<std::complex<float>>,                                            einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, std::complex<float>)
APIARY_INSTANTIATE_BOOLS("gemv", einsums::RuntimeTensorView<std::complex<float>>,                                            einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::RuntimeTensorView<std::complex<float>>,                                            std::complex<float>)
APIARY_INSTANTIATE_BOOLS("gemv", einsums::RuntimeTensorView<std::complex<float>>,                                            einsums::RuntimeTensorView<std::complex<float>>,                                            einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, std::complex<float>)
APIARY_INSTANTIATE_BOOLS("gemv", einsums::RuntimeTensorView<std::complex<float>>,                                            einsums::RuntimeTensorView<std::complex<float>>,                                            einsums::RuntimeTensorView<std::complex<float>>,                                            std::complex<float>)
// complex<double>
APIARY_INSTANTIATE_BOOLS("gemv", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, std::complex<double>)
APIARY_INSTANTIATE_BOOLS("gemv", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::RuntimeTensorView<std::complex<double>>,                                          std::complex<double>)
APIARY_INSTANTIATE_BOOLS("gemv", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::RuntimeTensorView<std::complex<double>>,                                          einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, std::complex<double>)
APIARY_INSTANTIATE_BOOLS("gemv", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::RuntimeTensorView<std::complex<double>>,                                          einsums::RuntimeTensorView<std::complex<double>>,                                          std::complex<double>)
APIARY_INSTANTIATE_BOOLS("gemv", einsums::RuntimeTensorView<std::complex<double>>,                                          einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, std::complex<double>)
APIARY_INSTANTIATE_BOOLS("gemv", einsums::RuntimeTensorView<std::complex<double>>,                                          einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::RuntimeTensorView<std::complex<double>>,                                          std::complex<double>)
APIARY_INSTANTIATE_BOOLS("gemv", einsums::RuntimeTensorView<std::complex<double>>,                                          einsums::RuntimeTensorView<std::complex<double>>,                                          einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, std::complex<double>)
APIARY_INSTANTIATE_BOOLS("gemv", einsums::RuntimeTensorView<std::complex<double>>,                                          einsums::RuntimeTensorView<std::complex<double>>,                                          einsums::RuntimeTensorView<std::complex<double>>,                                          std::complex<double>)
    // clang-format on
    void gemv(U const alpha, AType const &A, XType const &z, U const beta, YType *y) {
    if (A.rank() != 2 || z.rank() != 1 || y->rank() != 1) {
        EINSUMS_THROW_EXCEPTION(rank_error, "cg::gemv requires A rank-2 and x/y rank-1; got {}, {}, {}.", A.rank(), z.rank(), y->rank());
    }

    auto &ctx = CaptureContext::current();
    if (!ctx.is_capturing()) {
        LabeledSection("gemv eager");
        linear_algebra::gemv<TransA>(alpha, A, z, beta, y);
        return;
    }

    LabeledSection("gemv capture");
    auto [a_id, a_slot] = ctx.get_slot(A);
    auto [z_id, z_slot] = ctx.get_slot(z);
    auto [y_id, y_slot] = ctx.get_slot(*y);

    auto label    = fmt::format("gemv<{}>", TransA ? "T" : "N");
    auto executor = [alpha, a_slot, z_slot, beta, y_slot]() {
        LabeledSection("gemv execute");
        ProfileAnnotate("trans", TransA ? "T" : "N");
        ProfileAnnotate("m", static_cast<int64_t>(static_cast<AType const *>(a_slot->ptr)->dim(0)));
        ProfileAnnotate("n", static_cast<int64_t>(static_cast<AType const *>(a_slot->ptr)->dim(1)));
        linear_algebra::gemv<TransA>(alpha, *static_cast<AType const *>(a_slot->ptr), *static_cast<XType const *>(z_slot->ptr), beta,
                                     static_cast<YType *>(y_slot->ptr));
    };

    // beta != 0 → gemv reads y as well as writing it; list it as an input so
    // loop-invariance and scheduling see the read (see the gemm note above).
    std::vector<TensorId> inputs = {a_id, z_id};
    if (beta != U{}) {
        inputs.push_back(y_id);
    }
    ctx.record(OpKind::Gemv, std::move(label), std::move(inputs), {y_id}, std::move(executor));
}

// ─────────────────────────────────────────────────────────────────────────────
// ger: A += alpha * X * Y^T
// ─────────────────────────────────────────────────────────────────────────────

template <MatrixConcept AType, VectorConcept XType, VectorConcept YType>
    requires SameUnderlying<AType, XType, YType>
void ger(typename AType::ValueType alpha, XType const &X, YType const &Y, AType *A) {
    auto &ctx = CaptureContext::current();
    if (!ctx.is_capturing()) {
        LabeledSection("ger eager");
        linear_algebra::ger(alpha, X, Y, A);
        return;
    }

    LabeledSection("ger capture");
    auto [x_id, x_slot] = ctx.get_slot(X);
    auto [y_id, y_slot] = ctx.get_slot(Y);
    auto [a_id, a_slot] = ctx.get_slot(*A);

    auto executor = [alpha, x_slot, y_slot, a_slot]() {
        LabeledSection("ger execute");
        ProfileAnnotate("m", static_cast<int64_t>(static_cast<XType const *>(x_slot->ptr)->dim(0)));
        ProfileAnnotate("n", static_cast<int64_t>(static_cast<YType const *>(y_slot->ptr)->dim(0)));
        linear_algebra::ger(alpha, *static_cast<XType const *>(x_slot->ptr), *static_cast<YType const *>(y_slot->ptr),
                            static_cast<AType *>(a_slot->ptr));
    };

    // ger always accumulates (``A += α·X·Y^T``), so it reads A as well as
    // writing it, list A as an input so loop-invariance and scheduling see the
    // read-modify-write (see the gemm note above).
    ctx.record(OpKind::Ger, "ger", {x_id, y_id, a_id}, {a_id}, std::move(executor));
}

/// Graph-aware GER (rank-1 update): ``A += alpha * X * Y^T``.
///
/// Outer product of vectors X and Y added to matrix A. X and Y must be
/// rank 1; A must be rank 2.
template <RuntimeRankTensorConcept AType, RuntimeRankTensorConcept XType, RuntimeRankTensorConcept YType>
    requires SameUnderlying<AType, XType, YType>
// clang-format off
APIARY_EXPOSE
APIARY_MODULE("linalg")
// All 8 combinations of (A, X, Y) x (owning, view), per dtype. Same-dtype
// across operands is enforced by SameUnderlying above.
//
// float
APIARY_INSTANTIATE_AS("ger", einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::GeneralRuntimeTensor<float, std::allocator<float>>)
APIARY_INSTANTIATE_AS("ger", einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::RuntimeTensorView<float>)
APIARY_INSTANTIATE_AS("ger", einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::RuntimeTensorView<float>,                          einsums::GeneralRuntimeTensor<float, std::allocator<float>>)
APIARY_INSTANTIATE_AS("ger", einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::RuntimeTensorView<float>,                          einsums::RuntimeTensorView<float>)
APIARY_INSTANTIATE_AS("ger", einsums::RuntimeTensorView<float>,                          einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::GeneralRuntimeTensor<float, std::allocator<float>>)
APIARY_INSTANTIATE_AS("ger", einsums::RuntimeTensorView<float>,                          einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::RuntimeTensorView<float>)
APIARY_INSTANTIATE_AS("ger", einsums::RuntimeTensorView<float>,                          einsums::RuntimeTensorView<float>,                          einsums::GeneralRuntimeTensor<float, std::allocator<float>>)
APIARY_INSTANTIATE_AS("ger", einsums::RuntimeTensorView<float>,                          einsums::RuntimeTensorView<float>,                          einsums::RuntimeTensorView<float>)
// double
APIARY_INSTANTIATE_AS("ger", einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::GeneralRuntimeTensor<double, std::allocator<double>>)
APIARY_INSTANTIATE_AS("ger", einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::RuntimeTensorView<double>)
APIARY_INSTANTIATE_AS("ger", einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::RuntimeTensorView<double>,                          einsums::GeneralRuntimeTensor<double, std::allocator<double>>)
APIARY_INSTANTIATE_AS("ger", einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::RuntimeTensorView<double>,                          einsums::RuntimeTensorView<double>)
APIARY_INSTANTIATE_AS("ger", einsums::RuntimeTensorView<double>,                          einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::GeneralRuntimeTensor<double, std::allocator<double>>)
APIARY_INSTANTIATE_AS("ger", einsums::RuntimeTensorView<double>,                          einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::RuntimeTensorView<double>)
APIARY_INSTANTIATE_AS("ger", einsums::RuntimeTensorView<double>,                          einsums::RuntimeTensorView<double>,                          einsums::GeneralRuntimeTensor<double, std::allocator<double>>)
APIARY_INSTANTIATE_AS("ger", einsums::RuntimeTensorView<double>,                          einsums::RuntimeTensorView<double>,                          einsums::RuntimeTensorView<double>)
// complex<float>
APIARY_INSTANTIATE_AS("ger", einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>)
APIARY_INSTANTIATE_AS("ger", einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::RuntimeTensorView<std::complex<float>>)
APIARY_INSTANTIATE_AS("ger", einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::RuntimeTensorView<std::complex<float>>,                                            einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>)
APIARY_INSTANTIATE_AS("ger", einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::RuntimeTensorView<std::complex<float>>,                                            einsums::RuntimeTensorView<std::complex<float>>)
APIARY_INSTANTIATE_AS("ger", einsums::RuntimeTensorView<std::complex<float>>,                                            einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>)
APIARY_INSTANTIATE_AS("ger", einsums::RuntimeTensorView<std::complex<float>>,                                            einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::RuntimeTensorView<std::complex<float>>)
APIARY_INSTANTIATE_AS("ger", einsums::RuntimeTensorView<std::complex<float>>,                                            einsums::RuntimeTensorView<std::complex<float>>,                                            einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>)
APIARY_INSTANTIATE_AS("ger", einsums::RuntimeTensorView<std::complex<float>>,                                            einsums::RuntimeTensorView<std::complex<float>>,                                            einsums::RuntimeTensorView<std::complex<float>>)
// complex<double>
APIARY_INSTANTIATE_AS("ger", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
APIARY_INSTANTIATE_AS("ger", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::RuntimeTensorView<std::complex<double>>)
APIARY_INSTANTIATE_AS("ger", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::RuntimeTensorView<std::complex<double>>,                                          einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
APIARY_INSTANTIATE_AS("ger", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::RuntimeTensorView<std::complex<double>>,                                          einsums::RuntimeTensorView<std::complex<double>>)
APIARY_INSTANTIATE_AS("ger", einsums::RuntimeTensorView<std::complex<double>>,                                          einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
APIARY_INSTANTIATE_AS("ger", einsums::RuntimeTensorView<std::complex<double>>,                                          einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::RuntimeTensorView<std::complex<double>>)
APIARY_INSTANTIATE_AS("ger", einsums::RuntimeTensorView<std::complex<double>>,                                          einsums::RuntimeTensorView<std::complex<double>>,                                          einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
APIARY_INSTANTIATE_AS("ger", einsums::RuntimeTensorView<std::complex<double>>,                                          einsums::RuntimeTensorView<std::complex<double>>,                                          einsums::RuntimeTensorView<std::complex<double>>)
    // clang-format on
    void ger(typename AType::ValueType alpha, XType const &X, YType const &Y, AType *A) {
    if (X.rank() != 1 || Y.rank() != 1 || A->rank() != 2) {
        EINSUMS_THROW_EXCEPTION(rank_error, "cg::ger requires X/Y rank-1 and A rank-2; got {}, {}, {}.", X.rank(), Y.rank(), A->rank());
    }

    auto &ctx = CaptureContext::current();
    if (!ctx.is_capturing()) {
        LabeledSection("ger eager");
        linear_algebra::ger(alpha, X, Y, A);
        return;
    }

    LabeledSection("ger capture");
    auto [x_id, x_slot] = ctx.get_slot(X);
    auto [y_id, y_slot] = ctx.get_slot(Y);
    auto [a_id, a_slot] = ctx.get_slot(*A);

    auto executor = [alpha, x_slot, y_slot, a_slot]() {
        LabeledSection("ger execute");
        ProfileAnnotate("m", static_cast<int64_t>(static_cast<XType const *>(x_slot->ptr)->dim(0)));
        ProfileAnnotate("n", static_cast<int64_t>(static_cast<YType const *>(y_slot->ptr)->dim(0)));
        linear_algebra::ger(alpha, *static_cast<XType const *>(x_slot->ptr), *static_cast<YType const *>(y_slot->ptr),
                            static_cast<AType *>(a_slot->ptr));
    };

    // ger always accumulates (``A += α·X·Y^T``), so it reads A as well as
    // writing it, list A as an input so loop-invariance and scheduling see the
    // read-modify-write (see the gemm note above).
    ctx.record(OpKind::Ger, "ger", {x_id, y_id, a_id}, {a_id}, std::move(executor));
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
APIARY_EXPOSE
APIARY_MODULE("linalg")
APIARY_INSTANTIATE_AS("dot", einsums::GeneralRuntimeTensor<float,                std::allocator<float>>,                einsums::GeneralRuntimeTensor<float,                std::allocator<float>>)
APIARY_INSTANTIATE_AS("dot", einsums::GeneralRuntimeTensor<double,               std::allocator<double>>,               einsums::GeneralRuntimeTensor<double,               std::allocator<double>>)
APIARY_INSTANTIATE_AS("dot", einsums::GeneralRuntimeTensor<std::complex<float>,  std::allocator<std::complex<float>>>,  einsums::GeneralRuntimeTensor<std::complex<float>,  std::allocator<std::complex<float>>>)
APIARY_INSTANTIATE_AS("dot", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
// View operands: match the 3-arg dot(result, A, B) form, which already accepts
// non-contiguous views. Without these the scalar-returning dot(A, B) rejected a
// view argument (no matching overload) even though its template handles any
// TensorConcept.
APIARY_INSTANTIATE_AS("dot", einsums::GeneralRuntimeTensor<float,                std::allocator<float>>,                einsums::RuntimeTensorView<float>)
APIARY_INSTANTIATE_AS("dot", einsums::RuntimeTensorView<float>,                                                       einsums::GeneralRuntimeTensor<float,                std::allocator<float>>)
APIARY_INSTANTIATE_AS("dot", einsums::RuntimeTensorView<float>,                                                       einsums::RuntimeTensorView<float>)
APIARY_INSTANTIATE_AS("dot", einsums::GeneralRuntimeTensor<double,               std::allocator<double>>,               einsums::RuntimeTensorView<double>)
APIARY_INSTANTIATE_AS("dot", einsums::RuntimeTensorView<double>,                                                      einsums::GeneralRuntimeTensor<double,               std::allocator<double>>)
APIARY_INSTANTIATE_AS("dot", einsums::RuntimeTensorView<double>,                                                      einsums::RuntimeTensorView<double>)
APIARY_INSTANTIATE_AS("dot", einsums::GeneralRuntimeTensor<std::complex<float>,  std::allocator<std::complex<float>>>,  einsums::RuntimeTensorView<std::complex<float>>)
APIARY_INSTANTIATE_AS("dot", einsums::RuntimeTensorView<std::complex<float>>,                                          einsums::GeneralRuntimeTensor<std::complex<float>,  std::allocator<std::complex<float>>>)
APIARY_INSTANTIATE_AS("dot", einsums::RuntimeTensorView<std::complex<float>>,                                          einsums::RuntimeTensorView<std::complex<float>>)
APIARY_INSTANTIATE_AS("dot", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::RuntimeTensorView<std::complex<double>>)
APIARY_INSTANTIATE_AS("dot", einsums::RuntimeTensorView<std::complex<double>>,                                         einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
APIARY_INSTANTIATE_AS("dot", einsums::RuntimeTensorView<std::complex<double>>,                                         einsums::RuntimeTensorView<std::complex<double>>)
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
        LabeledSection("dot eager");
        *result = linear_algebra::dot(A, B);
        return;
    }

    LabeledSection("dot capture");
    auto [a_id, a_slot] = ctx.get_slot(A);
    auto [b_id, b_slot] = ctx.get_slot(B);
    TensorId r_id       = ctx.get_or_register_scalar(result, "dot_result");

    auto executor = [result, a_slot, b_slot]() {
        LabeledSection("dot execute");
        *result = linear_algebra::dot(*static_cast<AType const *>(a_slot->ptr), *static_cast<BType const *>(b_slot->ptr));
    };

    ctx.record(OpKind::Dot, "dot", {a_id, b_id}, {r_id}, std::move(executor));
}

/// Python-friendly graph-aware dot: writes the result into ``result->data()[0]``.
///
/// ``result`` is a pre-allocated rank-1 (or higher, but only element 0 is
/// touched) tensor that gives Python users a graph-native scalar handle, so
/// SCF energy patterns like ``e = ½ Σ D · (H+F)`` can be captured.
template <CoreBasicTensorConcept ResultType, typename AType, typename BType>
    requires requires {
        requires std::is_same_v<typename ResultType::ValueType, typename AType::ValueType>;
        requires std::is_same_v<typename AType::ValueType, typename BType::ValueType>;
        requires(CoreBasicTensorConcept<AType> || IsTiledTensorV<std::remove_cvref_t<AType>>);
        requires(CoreBasicTensorConcept<BType> || IsTiledTensorV<std::remove_cvref_t<BType>>);
    }
// clang-format off
APIARY_EXPOSE
APIARY_MODULE("linalg")
// All 8 combinations of (Result, A, B) x (owning, view), per dtype. Same-dtype
// across operands is enforced by the requires clause above. View arguments
// alias their parent so reads through them participate in the graph's
// dependency edges via TensorHandle::aliases.
//
// float: RRR/RRV/RVR/RVV/VRR/VRV/VVR/VVV
APIARY_INSTANTIATE_AS("dot", einsums::GeneralRuntimeTensor<float, std::allocator<float>>,                einsums::GeneralRuntimeTensor<float, std::allocator<float>>,                einsums::GeneralRuntimeTensor<float, std::allocator<float>>)
APIARY_INSTANTIATE_AS("dot", einsums::GeneralRuntimeTensor<float, std::allocator<float>>,                einsums::GeneralRuntimeTensor<float, std::allocator<float>>,                einsums::RuntimeTensorView<float>)
APIARY_INSTANTIATE_AS("dot", einsums::GeneralRuntimeTensor<float, std::allocator<float>>,                einsums::RuntimeTensorView<float>,                                                    einsums::GeneralRuntimeTensor<float, std::allocator<float>>)
APIARY_INSTANTIATE_AS("dot", einsums::GeneralRuntimeTensor<float, std::allocator<float>>,                einsums::RuntimeTensorView<float>,                                                    einsums::RuntimeTensorView<float>)
APIARY_INSTANTIATE_AS("dot", einsums::RuntimeTensorView<float>,                                          einsums::GeneralRuntimeTensor<float, std::allocator<float>>,                einsums::GeneralRuntimeTensor<float, std::allocator<float>>)
APIARY_INSTANTIATE_AS("dot", einsums::RuntimeTensorView<float>,                                          einsums::GeneralRuntimeTensor<float, std::allocator<float>>,                einsums::RuntimeTensorView<float>)
APIARY_INSTANTIATE_AS("dot", einsums::RuntimeTensorView<float>,                                          einsums::RuntimeTensorView<float>,                                                    einsums::GeneralRuntimeTensor<float, std::allocator<float>>)
APIARY_INSTANTIATE_AS("dot", einsums::RuntimeTensorView<float>,                                          einsums::RuntimeTensorView<float>,                                                    einsums::RuntimeTensorView<float>)
// double
APIARY_INSTANTIATE_AS("dot", einsums::GeneralRuntimeTensor<double, std::allocator<double>>,              einsums::GeneralRuntimeTensor<double, std::allocator<double>>,              einsums::GeneralRuntimeTensor<double, std::allocator<double>>)
APIARY_INSTANTIATE_AS("dot", einsums::GeneralRuntimeTensor<double, std::allocator<double>>,              einsums::GeneralRuntimeTensor<double, std::allocator<double>>,              einsums::RuntimeTensorView<double>)
APIARY_INSTANTIATE_AS("dot", einsums::GeneralRuntimeTensor<double, std::allocator<double>>,              einsums::RuntimeTensorView<double>,                                                  einsums::GeneralRuntimeTensor<double, std::allocator<double>>)
APIARY_INSTANTIATE_AS("dot", einsums::GeneralRuntimeTensor<double, std::allocator<double>>,              einsums::RuntimeTensorView<double>,                                                  einsums::RuntimeTensorView<double>)
APIARY_INSTANTIATE_AS("dot", einsums::RuntimeTensorView<double>,                                         einsums::GeneralRuntimeTensor<double, std::allocator<double>>,              einsums::GeneralRuntimeTensor<double, std::allocator<double>>)
APIARY_INSTANTIATE_AS("dot", einsums::RuntimeTensorView<double>,                                         einsums::GeneralRuntimeTensor<double, std::allocator<double>>,              einsums::RuntimeTensorView<double>)
APIARY_INSTANTIATE_AS("dot", einsums::RuntimeTensorView<double>,                                         einsums::RuntimeTensorView<double>,                                                  einsums::GeneralRuntimeTensor<double, std::allocator<double>>)
APIARY_INSTANTIATE_AS("dot", einsums::RuntimeTensorView<double>,                                         einsums::RuntimeTensorView<double>,                                                  einsums::RuntimeTensorView<double>)
// complex<float>
APIARY_INSTANTIATE_AS("dot", einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>,    einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>,    einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>)
APIARY_INSTANTIATE_AS("dot", einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>,    einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>,    einsums::RuntimeTensorView<std::complex<float>>)
APIARY_INSTANTIATE_AS("dot", einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>,    einsums::RuntimeTensorView<std::complex<float>>,                                            einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>)
APIARY_INSTANTIATE_AS("dot", einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>,    einsums::RuntimeTensorView<std::complex<float>>,                                            einsums::RuntimeTensorView<std::complex<float>>)
APIARY_INSTANTIATE_AS("dot", einsums::RuntimeTensorView<std::complex<float>>,                                            einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>,    einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>)
APIARY_INSTANTIATE_AS("dot", einsums::RuntimeTensorView<std::complex<float>>,                                            einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>,    einsums::RuntimeTensorView<std::complex<float>>)
APIARY_INSTANTIATE_AS("dot", einsums::RuntimeTensorView<std::complex<float>>,                                            einsums::RuntimeTensorView<std::complex<float>>,                                            einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>)
APIARY_INSTANTIATE_AS("dot", einsums::RuntimeTensorView<std::complex<float>>,                                            einsums::RuntimeTensorView<std::complex<float>>,                                            einsums::RuntimeTensorView<std::complex<float>>)
// complex<double>
APIARY_INSTANTIATE_AS("dot", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>,  einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>,  einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
APIARY_INSTANTIATE_AS("dot", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>,  einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>,  einsums::RuntimeTensorView<std::complex<double>>)
APIARY_INSTANTIATE_AS("dot", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>,  einsums::RuntimeTensorView<std::complex<double>>,                                          einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
APIARY_INSTANTIATE_AS("dot", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>,  einsums::RuntimeTensorView<std::complex<double>>,                                          einsums::RuntimeTensorView<std::complex<double>>)
APIARY_INSTANTIATE_AS("dot", einsums::RuntimeTensorView<std::complex<double>>,                                          einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>,  einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
APIARY_INSTANTIATE_AS("dot", einsums::RuntimeTensorView<std::complex<double>>,                                          einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>,  einsums::RuntimeTensorView<std::complex<double>>)
APIARY_INSTANTIATE_AS("dot", einsums::RuntimeTensorView<std::complex<double>>,                                          einsums::RuntimeTensorView<std::complex<double>>,                                          einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
APIARY_INSTANTIATE_AS("dot", einsums::RuntimeTensorView<std::complex<double>>,                                          einsums::RuntimeTensorView<std::complex<double>>,                                          einsums::RuntimeTensorView<std::complex<double>>)
// all-tiled operands, dense scalar result, per dtype
APIARY_INSTANTIATE_AS("dot", einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::TiledRuntimeTensor<float>, einsums::TiledRuntimeTensor<float>)
APIARY_INSTANTIATE_AS("dot", einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::TiledRuntimeTensor<double>, einsums::TiledRuntimeTensor<double>)
APIARY_INSTANTIATE_AS("dot", einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::TiledRuntimeTensor<std::complex<float>>, einsums::TiledRuntimeTensor<std::complex<float>>)
APIARY_INSTANTIATE_AS("dot", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::TiledRuntimeTensor<std::complex<double>>, einsums::TiledRuntimeTensor<std::complex<double>>)
    // clang-format on
    void dot_python(ResultType *result, AType const &A, BType const &B) {
    using T = typename AType::ValueType;
    if (result->size() < 1) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "cg::dot: result tensor must have at least one element");
    }

    // Tiled operands compose per-tile dots; dense delegate to linear_algebra.
    auto compute = [](AType const &a, BType const &b) -> T {
        if constexpr (IsTiledTensorV<std::remove_cvref_t<AType>>) {
            return detail::tiled_dot<T>(a, b);
        } else {
            return linear_algebra::dot(a, b);
        }
    };

    auto &ctx = CaptureContext::current();
    if (!ctx.is_capturing()) {
        LabeledSection("dot_python eager");
        result->data()[0] = compute(A, B);
        return;
    }

    LabeledSection("dot_python capture");
    // Register the result as a normal tensor slot (not a scalar handle) so
    // downstream tensor ops (scale, axpy, ...) on the same tensor see the
    // same slot id, get_or_register_scalar would key by data()[0] and
    // collide with get_slot(*result), giving rank-0 metadata to the scale.
    auto [a_id, a_slot] = ctx.get_slot(A);
    auto [b_id, b_slot] = ctx.get_slot(B);
    auto [r_id, r_slot] = ctx.get_slot(*result);

    auto executor = [a_slot, b_slot, r_slot, compute]() {
        LabeledSection("dot_python execute");
        auto *r_ptr      = static_cast<ResultType *>(r_slot->ptr);
        r_ptr->data()[0] = compute(*static_cast<AType const *>(a_slot->ptr), *static_cast<BType const *>(b_slot->ptr));
    };
    ctx.record(OpKind::Dot, "dot", {a_id, b_id}, {r_id}, std::move(executor));
}

/// Graph-aware Hermitian inner product: ``result := sum_i conj(A_i) * B_i``.
///
/// The conjugating counterpart of ``dot`` (which is the bilinear ``sum A_i B_i``).
/// For real dtypes this coincides with ``dot``. Backed by ``true_dot`` (BLAS
/// dotc on the contiguous complex path).
template <CoreBasicTensorConcept ResultType, CoreBasicTensorConcept AType, CoreBasicTensorConcept BType>
    requires requires {
        requires std::is_same_v<typename ResultType::ValueType, typename AType::ValueType>;
        requires std::is_same_v<typename AType::ValueType, typename BType::ValueType>;
    }
// clang-format off
APIARY_EXPOSE
APIARY_MODULE("linalg")
// float — RRR/RRV/RVR/RVV/VRR/VRV/VVR/VVV
APIARY_INSTANTIATE_AS("dotc", einsums::GeneralRuntimeTensor<float, std::allocator<float>>,                einsums::GeneralRuntimeTensor<float, std::allocator<float>>,                einsums::GeneralRuntimeTensor<float, std::allocator<float>>)
APIARY_INSTANTIATE_AS("dotc", einsums::GeneralRuntimeTensor<float, std::allocator<float>>,                einsums::GeneralRuntimeTensor<float, std::allocator<float>>,                einsums::RuntimeTensorView<float>)
APIARY_INSTANTIATE_AS("dotc", einsums::GeneralRuntimeTensor<float, std::allocator<float>>,                einsums::RuntimeTensorView<float>,                                                    einsums::GeneralRuntimeTensor<float, std::allocator<float>>)
APIARY_INSTANTIATE_AS("dotc", einsums::GeneralRuntimeTensor<float, std::allocator<float>>,                einsums::RuntimeTensorView<float>,                                                    einsums::RuntimeTensorView<float>)
APIARY_INSTANTIATE_AS("dotc", einsums::RuntimeTensorView<float>,                                          einsums::GeneralRuntimeTensor<float, std::allocator<float>>,                einsums::GeneralRuntimeTensor<float, std::allocator<float>>)
APIARY_INSTANTIATE_AS("dotc", einsums::RuntimeTensorView<float>,                                          einsums::GeneralRuntimeTensor<float, std::allocator<float>>,                einsums::RuntimeTensorView<float>)
APIARY_INSTANTIATE_AS("dotc", einsums::RuntimeTensorView<float>,                                          einsums::RuntimeTensorView<float>,                                                    einsums::GeneralRuntimeTensor<float, std::allocator<float>>)
APIARY_INSTANTIATE_AS("dotc", einsums::RuntimeTensorView<float>,                                          einsums::RuntimeTensorView<float>,                                                    einsums::RuntimeTensorView<float>)
// double
APIARY_INSTANTIATE_AS("dotc", einsums::GeneralRuntimeTensor<double, std::allocator<double>>,              einsums::GeneralRuntimeTensor<double, std::allocator<double>>,              einsums::GeneralRuntimeTensor<double, std::allocator<double>>)
APIARY_INSTANTIATE_AS("dotc", einsums::GeneralRuntimeTensor<double, std::allocator<double>>,              einsums::GeneralRuntimeTensor<double, std::allocator<double>>,              einsums::RuntimeTensorView<double>)
APIARY_INSTANTIATE_AS("dotc", einsums::GeneralRuntimeTensor<double, std::allocator<double>>,              einsums::RuntimeTensorView<double>,                                                  einsums::GeneralRuntimeTensor<double, std::allocator<double>>)
APIARY_INSTANTIATE_AS("dotc", einsums::GeneralRuntimeTensor<double, std::allocator<double>>,              einsums::RuntimeTensorView<double>,                                                  einsums::RuntimeTensorView<double>)
APIARY_INSTANTIATE_AS("dotc", einsums::RuntimeTensorView<double>,                                         einsums::GeneralRuntimeTensor<double, std::allocator<double>>,              einsums::GeneralRuntimeTensor<double, std::allocator<double>>)
APIARY_INSTANTIATE_AS("dotc", einsums::RuntimeTensorView<double>,                                         einsums::GeneralRuntimeTensor<double, std::allocator<double>>,              einsums::RuntimeTensorView<double>)
APIARY_INSTANTIATE_AS("dotc", einsums::RuntimeTensorView<double>,                                         einsums::RuntimeTensorView<double>,                                                  einsums::GeneralRuntimeTensor<double, std::allocator<double>>)
APIARY_INSTANTIATE_AS("dotc", einsums::RuntimeTensorView<double>,                                         einsums::RuntimeTensorView<double>,                                                  einsums::RuntimeTensorView<double>)
// complex<float>
APIARY_INSTANTIATE_AS("dotc", einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>,    einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>,    einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>)
APIARY_INSTANTIATE_AS("dotc", einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>,    einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>,    einsums::RuntimeTensorView<std::complex<float>>)
APIARY_INSTANTIATE_AS("dotc", einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>,    einsums::RuntimeTensorView<std::complex<float>>,                                            einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>)
APIARY_INSTANTIATE_AS("dotc", einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>,    einsums::RuntimeTensorView<std::complex<float>>,                                            einsums::RuntimeTensorView<std::complex<float>>)
APIARY_INSTANTIATE_AS("dotc", einsums::RuntimeTensorView<std::complex<float>>,                                            einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>,    einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>)
APIARY_INSTANTIATE_AS("dotc", einsums::RuntimeTensorView<std::complex<float>>,                                            einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>,    einsums::RuntimeTensorView<std::complex<float>>)
APIARY_INSTANTIATE_AS("dotc", einsums::RuntimeTensorView<std::complex<float>>,                                            einsums::RuntimeTensorView<std::complex<float>>,                                            einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>)
APIARY_INSTANTIATE_AS("dotc", einsums::RuntimeTensorView<std::complex<float>>,                                            einsums::RuntimeTensorView<std::complex<float>>,                                            einsums::RuntimeTensorView<std::complex<float>>)
// complex<double>
APIARY_INSTANTIATE_AS("dotc", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>,  einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>,  einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
APIARY_INSTANTIATE_AS("dotc", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>,  einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>,  einsums::RuntimeTensorView<std::complex<double>>)
APIARY_INSTANTIATE_AS("dotc", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>,  einsums::RuntimeTensorView<std::complex<double>>,                                          einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
APIARY_INSTANTIATE_AS("dotc", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>,  einsums::RuntimeTensorView<std::complex<double>>,                                          einsums::RuntimeTensorView<std::complex<double>>)
APIARY_INSTANTIATE_AS("dotc", einsums::RuntimeTensorView<std::complex<double>>,                                          einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>,  einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
APIARY_INSTANTIATE_AS("dotc", einsums::RuntimeTensorView<std::complex<double>>,                                          einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>,  einsums::RuntimeTensorView<std::complex<double>>)
APIARY_INSTANTIATE_AS("dotc", einsums::RuntimeTensorView<std::complex<double>>,                                          einsums::RuntimeTensorView<std::complex<double>>,                                          einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
APIARY_INSTANTIATE_AS("dotc", einsums::RuntimeTensorView<std::complex<double>>,                                          einsums::RuntimeTensorView<std::complex<double>>,                                          einsums::RuntimeTensorView<std::complex<double>>)
    // clang-format on
    void dotc_python(ResultType *result, AType const &A, BType const &B) {
    using T = typename AType::ValueType;
    if (result->size() < 1) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "cg::dotc: result tensor must have at least one element");
    }
    auto compute = [](AType const &a, BType const &b) -> T { return linear_algebra::true_dot(a, b); };

    auto &ctx = CaptureContext::current();
    if (!ctx.is_capturing()) {
        LabeledSection("dotc_python eager");
        result->data()[0] = compute(A, B);
        return;
    }

    LabeledSection("dotc_python capture");
    auto [a_id, a_slot] = ctx.get_slot(A);
    auto [b_id, b_slot] = ctx.get_slot(B);
    auto [r_id, r_slot] = ctx.get_slot(*result);

    auto executor = [a_slot, b_slot, r_slot, compute]() {
        LabeledSection("dotc_python execute");
        auto *r_ptr      = static_cast<ResultType *>(r_slot->ptr);
        r_ptr->data()[0] = compute(*static_cast<AType const *>(a_slot->ptr), *static_cast<BType const *>(b_slot->ptr));
    };
    ctx.record(OpKind::Dot, "dotc", {a_id, b_id}, {r_id}, std::move(executor));
}

// ─────────────────────────────────────────────────────────────────────────────
// reductions: sum / max  (write a scalar into result->data()[0])
// ─────────────────────────────────────────────────────────────────────────────

namespace detail {
/// Stride-correct fold over every element of a dense tensor or view.
///
/// Walks the logical index space with an odometer (multi-index -> offset via
/// strides), so non-contiguous views (slices, transposes) reduce correctly,
/// not just contiguous storage. O(size * rank).
template <typename TensorType, typename Acc, typename Op>
Acc reduce_elements(TensorType const &A, Acc init, Op op) {
    using T           = typename TensorType::ValueType;
    size_t const rank = A.rank();
    size_t const n    = A.size();
    T const     *base = A.data();
    if (base == nullptr || n == 0)
        return init;
    std::vector<size_t> dims(rank), strides(rank), idx(rank, 0);
    for (size_t a = 0; a < rank; ++a) {
        dims[a]    = A.dim(a);
        strides[a] = A.stride(a);
    }
    Acc acc = init;
    for (size_t k = 0; k < n; ++k) {
        size_t off = 0;
        for (size_t a = 0; a < rank; ++a)
            off += idx[a] * strides[a];
        acc = op(acc, base[off]);
        for (size_t a = rank; a-- > 0;) { // increment the odometer
            if (++idx[a] < dims[a])
                break;
            idx[a] = 0;
        }
    }
    return acc;
}
} // namespace detail

/// Graph-aware sum of every element, written into ``result->data()[0]``.
///
/// Mirrors dot_python's scalar-into-[1]-tensor convention: eager when not
/// capturing, an opaque @ref OpKind::Custom node otherwise. Stride-correct
/// (works on slice/transpose views). Backs the numpy-style ``A.sum()`` /
/// ``A.mean()``.
template <CoreBasicTensorConcept ResultType, CoreBasicTensorConcept AType>
    requires(std::is_same_v<typename ResultType::ValueType, typename AType::ValueType>)
// clang-format off
APIARY_EXPOSE
APIARY_MODULE("linalg")
APIARY_INSTANTIATE_AS("sum", einsums::GeneralRuntimeTensor<float, std::allocator<float>>,                              einsums::GeneralRuntimeTensor<float, std::allocator<float>>)
APIARY_INSTANTIATE_AS("sum", einsums::GeneralRuntimeTensor<float, std::allocator<float>>,                              einsums::RuntimeTensorView<float>)
APIARY_INSTANTIATE_AS("sum", einsums::GeneralRuntimeTensor<double, std::allocator<double>>,                            einsums::GeneralRuntimeTensor<double, std::allocator<double>>)
APIARY_INSTANTIATE_AS("sum", einsums::GeneralRuntimeTensor<double, std::allocator<double>>,                            einsums::RuntimeTensorView<double>)
APIARY_INSTANTIATE_AS("sum", einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>,  einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>)
APIARY_INSTANTIATE_AS("sum", einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>,  einsums::RuntimeTensorView<std::complex<float>>)
APIARY_INSTANTIATE_AS("sum", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
APIARY_INSTANTIATE_AS("sum", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::RuntimeTensorView<std::complex<double>>)
    // clang-format on
    void sum_python(ResultType *result, AType const &A) {
    using T = typename AType::ValueType;
    if (result->size() < 1)
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "cg::sum: result tensor must have at least one element");

    auto compute = [](AType const &a) -> T { return detail::reduce_elements(a, T{0}, [](T acc, T x) { return acc + x; }); };

    auto &ctx = CaptureContext::current();
    if (!ctx.is_capturing()) {
        LabeledSection("sum_python eager");
        result->data()[0] = compute(A);
        return;
    }
    LabeledSection("sum_python capture");
    auto [a_id, a_slot] = ctx.get_slot(A);
    auto [r_id, r_slot] = ctx.get_slot(*result);
    auto executor       = [a_slot, r_slot, compute]() {
        LabeledSection("sum_python execute");
        static_cast<ResultType *>(r_slot->ptr)->data()[0] = compute(*static_cast<AType const *>(a_slot->ptr));
    };
    ctx.record(OpKind::Custom, "sum", {a_id}, {r_id}, std::move(executor));
}

/// Graph-aware maximum element (real dtypes), written into ``result->data()[0]``.
/// Backs the numpy-style ``A.max()``. Real dtypes only, since complex ordering
/// is not meaningful; use ``norm(MAXABS)`` for the largest magnitude.
template <CoreBasicTensorConcept ResultType, CoreBasicTensorConcept AType>
    requires(std::is_same_v<typename ResultType::ValueType, typename AType::ValueType>)
// clang-format off
APIARY_EXPOSE
APIARY_MODULE("linalg")
APIARY_INSTANTIATE_AS("max", einsums::GeneralRuntimeTensor<float, std::allocator<float>>,   einsums::GeneralRuntimeTensor<float, std::allocator<float>>)
APIARY_INSTANTIATE_AS("max", einsums::GeneralRuntimeTensor<float, std::allocator<float>>,   einsums::RuntimeTensorView<float>)
APIARY_INSTANTIATE_AS("max", einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::GeneralRuntimeTensor<double, std::allocator<double>>)
APIARY_INSTANTIATE_AS("max", einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::RuntimeTensorView<double>)
    // clang-format on
    void max_python(ResultType *result, AType const &A) {
    using T = typename AType::ValueType;
    if (result->size() < 1)
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "cg::max: result tensor must have at least one element");
    if (A.size() == 0)
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "cg::max: cannot reduce an empty tensor");

    auto compute = [](AType const &a) -> T {
        // Propagate NaN like numpy.max: a plain ``x > acc`` comparison is false for
        // a NaN x, so NaN would be silently dropped, and an all-NaN reduction would
        // leak the ``lowest()`` seed. Test ``isnan(x)`` so a NaN poisons the
        // accumulator (and ``acc`` stays NaN thereafter, since ``x > NaN`` is false).
        return detail::reduce_elements(a, std::numeric_limits<T>::lowest(),
                                       [](T acc, T x) { return (std::isnan(x) || x > acc) ? x : acc; });
    };

    auto &ctx = CaptureContext::current();
    if (!ctx.is_capturing()) {
        LabeledSection("max_python eager");
        result->data()[0] = compute(A);
        return;
    }
    LabeledSection("max_python capture");
    auto [a_id, a_slot] = ctx.get_slot(A);
    auto [r_id, r_slot] = ctx.get_slot(*result);
    auto executor       = [a_slot, r_slot, compute]() {
        LabeledSection("max_python execute");
        static_cast<ResultType *>(r_slot->ptr)->data()[0] = compute(*static_cast<AType const *>(a_slot->ptr));
    };
    ctx.record(OpKind::Custom, "max", {a_id}, {r_id}, std::move(executor));
}

// ─────────────────────────────────────────────────────────────────────────────
// direct_product: C = alpha * (A ⊙ B) + beta * C
// ─────────────────────────────────────────────────────────────────────────────

template <typename T, TensorConcept AType, TensorConcept BType, TensorConcept CType>
// clang-format off
APIARY_EXPOSE
APIARY_MODULE("linalg")
// All 8 combinations of (A, B, C) x (owning, view), per dtype. The first
// template argument is the scalar dtype for alpha/beta.
//
// float
APIARY_INSTANTIATE_AS("direct_product", float, einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::GeneralRuntimeTensor<float, std::allocator<float>>)
APIARY_INSTANTIATE_AS("direct_product", float, einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::RuntimeTensorView<float>)
APIARY_INSTANTIATE_AS("direct_product", float, einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::RuntimeTensorView<float>,                          einsums::GeneralRuntimeTensor<float, std::allocator<float>>)
APIARY_INSTANTIATE_AS("direct_product", float, einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::RuntimeTensorView<float>,                          einsums::RuntimeTensorView<float>)
APIARY_INSTANTIATE_AS("direct_product", float, einsums::RuntimeTensorView<float>,                          einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::GeneralRuntimeTensor<float, std::allocator<float>>)
APIARY_INSTANTIATE_AS("direct_product", float, einsums::RuntimeTensorView<float>,                          einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::RuntimeTensorView<float>)
APIARY_INSTANTIATE_AS("direct_product", float, einsums::RuntimeTensorView<float>,                          einsums::RuntimeTensorView<float>,                          einsums::GeneralRuntimeTensor<float, std::allocator<float>>)
APIARY_INSTANTIATE_AS("direct_product", float, einsums::RuntimeTensorView<float>,                          einsums::RuntimeTensorView<float>,                          einsums::RuntimeTensorView<float>)
// double
APIARY_INSTANTIATE_AS("direct_product", double, einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::GeneralRuntimeTensor<double, std::allocator<double>>)
APIARY_INSTANTIATE_AS("direct_product", double, einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::RuntimeTensorView<double>)
APIARY_INSTANTIATE_AS("direct_product", double, einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::RuntimeTensorView<double>,                          einsums::GeneralRuntimeTensor<double, std::allocator<double>>)
APIARY_INSTANTIATE_AS("direct_product", double, einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::RuntimeTensorView<double>,                          einsums::RuntimeTensorView<double>)
APIARY_INSTANTIATE_AS("direct_product", double, einsums::RuntimeTensorView<double>,                          einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::GeneralRuntimeTensor<double, std::allocator<double>>)
APIARY_INSTANTIATE_AS("direct_product", double, einsums::RuntimeTensorView<double>,                          einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::RuntimeTensorView<double>)
APIARY_INSTANTIATE_AS("direct_product", double, einsums::RuntimeTensorView<double>,                          einsums::RuntimeTensorView<double>,                          einsums::GeneralRuntimeTensor<double, std::allocator<double>>)
APIARY_INSTANTIATE_AS("direct_product", double, einsums::RuntimeTensorView<double>,                          einsums::RuntimeTensorView<double>,                          einsums::RuntimeTensorView<double>)
// complex<float>
APIARY_INSTANTIATE_AS("direct_product", std::complex<float>, einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>)
APIARY_INSTANTIATE_AS("direct_product", std::complex<float>, einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::RuntimeTensorView<std::complex<float>>)
APIARY_INSTANTIATE_AS("direct_product", std::complex<float>, einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::RuntimeTensorView<std::complex<float>>,                                            einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>)
APIARY_INSTANTIATE_AS("direct_product", std::complex<float>, einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::RuntimeTensorView<std::complex<float>>,                                            einsums::RuntimeTensorView<std::complex<float>>)
APIARY_INSTANTIATE_AS("direct_product", std::complex<float>, einsums::RuntimeTensorView<std::complex<float>>,                                            einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>)
APIARY_INSTANTIATE_AS("direct_product", std::complex<float>, einsums::RuntimeTensorView<std::complex<float>>,                                            einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::RuntimeTensorView<std::complex<float>>)
APIARY_INSTANTIATE_AS("direct_product", std::complex<float>, einsums::RuntimeTensorView<std::complex<float>>,                                            einsums::RuntimeTensorView<std::complex<float>>,                                            einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>)
APIARY_INSTANTIATE_AS("direct_product", std::complex<float>, einsums::RuntimeTensorView<std::complex<float>>,                                            einsums::RuntimeTensorView<std::complex<float>>,                                            einsums::RuntimeTensorView<std::complex<float>>)
// complex<double>
APIARY_INSTANTIATE_AS("direct_product", std::complex<double>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
APIARY_INSTANTIATE_AS("direct_product", std::complex<double>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::RuntimeTensorView<std::complex<double>>)
APIARY_INSTANTIATE_AS("direct_product", std::complex<double>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::RuntimeTensorView<std::complex<double>>,                                          einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
APIARY_INSTANTIATE_AS("direct_product", std::complex<double>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::RuntimeTensorView<std::complex<double>>,                                          einsums::RuntimeTensorView<std::complex<double>>)
APIARY_INSTANTIATE_AS("direct_product", std::complex<double>, einsums::RuntimeTensorView<std::complex<double>>,                                          einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
APIARY_INSTANTIATE_AS("direct_product", std::complex<double>, einsums::RuntimeTensorView<std::complex<double>>,                                          einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::RuntimeTensorView<std::complex<double>>)
APIARY_INSTANTIATE_AS("direct_product", std::complex<double>, einsums::RuntimeTensorView<std::complex<double>>,                                          einsums::RuntimeTensorView<std::complex<double>>,                                          einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
APIARY_INSTANTIATE_AS("direct_product", std::complex<double>, einsums::RuntimeTensorView<std::complex<double>>,                                          einsums::RuntimeTensorView<std::complex<double>>,                                          einsums::RuntimeTensorView<std::complex<double>>)
    // clang-format on
    void direct_product(T alpha, AType const &A, BType const &B, T beta, CType *C) {
    auto &ctx = CaptureContext::current();
    if (!ctx.is_capturing()) {
        LabeledSection("direct_product eager");
        linear_algebra::direct_product(alpha, A, B, beta, C);
        return;
    }

    LabeledSection("direct_product capture");
    auto [a_id, a_slot] = ctx.get_slot(A);
    auto [b_id, b_slot] = ctx.get_slot(B);
    auto [c_id, c_slot] = ctx.get_slot(*C);

    auto executor = [alpha, a_slot, b_slot, beta, c_slot]() {
        LabeledSection("direct_product execute");
        ProfileAnnotate("size", static_cast<int64_t>(static_cast<CType *>(c_slot->ptr)->size()));
        linear_algebra::direct_product(alpha, *static_cast<AType const *>(a_slot->ptr), *static_cast<BType const *>(b_slot->ptr), beta,
                                       static_cast<CType *>(c_slot->ptr));
    };

    // When beta != 0 the op reads its destination (C = alpha*A*B + beta*C), so C
    // is an input as well as the output. List it -- otherwise dependency-based
    // passes (LoopInvariantHoisting, Reorder, ...) don't see the read and may
    // hoist the accumulation out of a loop or reorder it past another writer of C.
    // (gemm already does this; matches the out-tensor-as-input convention.)
    std::vector<TensorId> dp_inputs = (beta != T{0}) ? std::vector<TensorId>{a_id, b_id, c_id} : std::vector<TensorId>{a_id, b_id};
    ctx.record(OpKind::DirectProduct, "direct_product", std::move(dp_inputs), {c_id}, std::move(executor));
}

// ─────────────────────────────────────────────────────────────────────────────
// direct_division: C = alpha * (A ⊘ B) + beta * C
// ─────────────────────────────────────────────────────────────────────────────

template <typename T, TensorConcept AType, TensorConcept BType, TensorConcept CType>
// clang-format off
APIARY_EXPOSE
APIARY_MODULE("linalg")
// All 8 combinations of (A, B, C) x (owning, view), per dtype. The first
// template argument is the scalar dtype for alpha/beta.
//
// float
APIARY_INSTANTIATE_AS("direct_division", float, einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::GeneralRuntimeTensor<float, std::allocator<float>>)
APIARY_INSTANTIATE_AS("direct_division", float, einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::RuntimeTensorView<float>)
APIARY_INSTANTIATE_AS("direct_division", float, einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::RuntimeTensorView<float>,                          einsums::GeneralRuntimeTensor<float, std::allocator<float>>)
APIARY_INSTANTIATE_AS("direct_division", float, einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::RuntimeTensorView<float>,                          einsums::RuntimeTensorView<float>)
APIARY_INSTANTIATE_AS("direct_division", float, einsums::RuntimeTensorView<float>,                          einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::GeneralRuntimeTensor<float, std::allocator<float>>)
APIARY_INSTANTIATE_AS("direct_division", float, einsums::RuntimeTensorView<float>,                          einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::RuntimeTensorView<float>)
APIARY_INSTANTIATE_AS("direct_division", float, einsums::RuntimeTensorView<float>,                          einsums::RuntimeTensorView<float>,                          einsums::GeneralRuntimeTensor<float, std::allocator<float>>)
APIARY_INSTANTIATE_AS("direct_division", float, einsums::RuntimeTensorView<float>,                          einsums::RuntimeTensorView<float>,                          einsums::RuntimeTensorView<float>)
// double
APIARY_INSTANTIATE_AS("direct_division", double, einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::GeneralRuntimeTensor<double, std::allocator<double>>)
APIARY_INSTANTIATE_AS("direct_division", double, einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::RuntimeTensorView<double>)
APIARY_INSTANTIATE_AS("direct_division", double, einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::RuntimeTensorView<double>,                          einsums::GeneralRuntimeTensor<double, std::allocator<double>>)
APIARY_INSTANTIATE_AS("direct_division", double, einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::RuntimeTensorView<double>,                          einsums::RuntimeTensorView<double>)
APIARY_INSTANTIATE_AS("direct_division", double, einsums::RuntimeTensorView<double>,                          einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::GeneralRuntimeTensor<double, std::allocator<double>>)
APIARY_INSTANTIATE_AS("direct_division", double, einsums::RuntimeTensorView<double>,                          einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::RuntimeTensorView<double>)
APIARY_INSTANTIATE_AS("direct_division", double, einsums::RuntimeTensorView<double>,                          einsums::RuntimeTensorView<double>,                          einsums::GeneralRuntimeTensor<double, std::allocator<double>>)
APIARY_INSTANTIATE_AS("direct_division", double, einsums::RuntimeTensorView<double>,                          einsums::RuntimeTensorView<double>,                          einsums::RuntimeTensorView<double>)
// complex<float>
APIARY_INSTANTIATE_AS("direct_division", std::complex<float>, einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>)
APIARY_INSTANTIATE_AS("direct_division", std::complex<float>, einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::RuntimeTensorView<std::complex<float>>)
APIARY_INSTANTIATE_AS("direct_division", std::complex<float>, einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::RuntimeTensorView<std::complex<float>>,                                            einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>)
APIARY_INSTANTIATE_AS("direct_division", std::complex<float>, einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::RuntimeTensorView<std::complex<float>>,                                            einsums::RuntimeTensorView<std::complex<float>>)
APIARY_INSTANTIATE_AS("direct_division", std::complex<float>, einsums::RuntimeTensorView<std::complex<float>>,                                            einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>)
APIARY_INSTANTIATE_AS("direct_division", std::complex<float>, einsums::RuntimeTensorView<std::complex<float>>,                                            einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::RuntimeTensorView<std::complex<float>>)
APIARY_INSTANTIATE_AS("direct_division", std::complex<float>, einsums::RuntimeTensorView<std::complex<float>>,                                            einsums::RuntimeTensorView<std::complex<float>>,                                            einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>)
APIARY_INSTANTIATE_AS("direct_division", std::complex<float>, einsums::RuntimeTensorView<std::complex<float>>,                                            einsums::RuntimeTensorView<std::complex<float>>,                                            einsums::RuntimeTensorView<std::complex<float>>)
// complex<double>
APIARY_INSTANTIATE_AS("direct_division", std::complex<double>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
APIARY_INSTANTIATE_AS("direct_division", std::complex<double>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::RuntimeTensorView<std::complex<double>>)
APIARY_INSTANTIATE_AS("direct_division", std::complex<double>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::RuntimeTensorView<std::complex<double>>,                                          einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
APIARY_INSTANTIATE_AS("direct_division", std::complex<double>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::RuntimeTensorView<std::complex<double>>,                                          einsums::RuntimeTensorView<std::complex<double>>)
APIARY_INSTANTIATE_AS("direct_division", std::complex<double>, einsums::RuntimeTensorView<std::complex<double>>,                                          einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
APIARY_INSTANTIATE_AS("direct_division", std::complex<double>, einsums::RuntimeTensorView<std::complex<double>>,                                          einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::RuntimeTensorView<std::complex<double>>)
APIARY_INSTANTIATE_AS("direct_division", std::complex<double>, einsums::RuntimeTensorView<std::complex<double>>,                                          einsums::RuntimeTensorView<std::complex<double>>,                                          einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
APIARY_INSTANTIATE_AS("direct_division", std::complex<double>, einsums::RuntimeTensorView<std::complex<double>>,                                          einsums::RuntimeTensorView<std::complex<double>>,                                          einsums::RuntimeTensorView<std::complex<double>>)
    // clang-format on
    void direct_division(T alpha, AType const &A, BType const &B, T beta, CType *C) {
    auto &ctx = CaptureContext::current();
    if (!ctx.is_capturing()) {
        LabeledSection("direct_division eager");
        linear_algebra::direct_division(alpha, A, B, beta, C);
        return;
    }

    LabeledSection("direct_division capture");
    auto [a_id, a_slot] = ctx.get_slot(A);
    auto [b_id, b_slot] = ctx.get_slot(B);
    auto [c_id, c_slot] = ctx.get_slot(*C);

    auto executor = [alpha, a_slot, b_slot, beta, c_slot]() {
        LabeledSection("direct_division execute");
        ProfileAnnotate("size", static_cast<int64_t>(static_cast<CType *>(c_slot->ptr)->size()));
        linear_algebra::direct_division(alpha, *static_cast<AType const *>(a_slot->ptr), *static_cast<BType const *>(b_slot->ptr), beta,
                                        static_cast<CType *>(c_slot->ptr));
    };

    // beta != 0 reads the destination (C = alpha*A/B + beta*C) -- list C as an
    // input so dependency-based passes see the read (see direct_product).
    std::vector<TensorId> dd_inputs = (beta != T{0}) ? std::vector<TensorId>{a_id, b_id, c_id} : std::vector<TensorId>{a_id, b_id};
    ctx.record(OpKind::DirectDivision, "direct_division", std::move(dd_inputs), {c_id}, std::move(executor));
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
APIARY_EXPOSE
APIARY_MODULE("linalg")
// All 4 combinations of (Result, Vectors) x (owning, view), per dtype. The
// vectors list is homogeneous, every element must be the same C++ type,
// since VectorType is a single template parameter shared across the list.
// For the canonical MP2 denominator use case (all four eps vectors are
// owning tensors) this is fine. If you really need a mix of owning and
// view vectors in one call, materialize the view side into an owning
// tensor first via block_copy.
//
// float
APIARY_INSTANTIATE_AS("outer_sum", einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::GeneralRuntimeTensor<float, std::allocator<float>>)
APIARY_INSTANTIATE_AS("outer_sum", einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::RuntimeTensorView<float>)
APIARY_INSTANTIATE_AS("outer_sum", einsums::RuntimeTensorView<float>,                          einsums::GeneralRuntimeTensor<float, std::allocator<float>>)
APIARY_INSTANTIATE_AS("outer_sum", einsums::RuntimeTensorView<float>,                          einsums::RuntimeTensorView<float>)
// double
APIARY_INSTANTIATE_AS("outer_sum", einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::GeneralRuntimeTensor<double, std::allocator<double>>)
APIARY_INSTANTIATE_AS("outer_sum", einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::RuntimeTensorView<double>)
APIARY_INSTANTIATE_AS("outer_sum", einsums::RuntimeTensorView<double>,                          einsums::GeneralRuntimeTensor<double, std::allocator<double>>)
APIARY_INSTANTIATE_AS("outer_sum", einsums::RuntimeTensorView<double>,                          einsums::RuntimeTensorView<double>)
// complex<float>
APIARY_INSTANTIATE_AS("outer_sum", einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>)
APIARY_INSTANTIATE_AS("outer_sum", einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::RuntimeTensorView<std::complex<float>>)
APIARY_INSTANTIATE_AS("outer_sum", einsums::RuntimeTensorView<std::complex<float>>,                                            einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>)
APIARY_INSTANTIATE_AS("outer_sum", einsums::RuntimeTensorView<std::complex<float>>,                                            einsums::RuntimeTensorView<std::complex<float>>)
// complex<double>
APIARY_INSTANTIATE_AS("outer_sum", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
APIARY_INSTANTIATE_AS("outer_sum", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::RuntimeTensorView<std::complex<double>>)
APIARY_INSTANTIATE_AS("outer_sum", einsums::RuntimeTensorView<std::complex<double>>,                                          einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
APIARY_INSTANTIATE_AS("outer_sum", einsums::RuntimeTensorView<std::complex<double>>,                                          einsums::RuntimeTensorView<std::complex<double>>)
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
    // Capture-time checks: only rank and null. Dim matching is deferred to
    // execute time because views report their parent's dims at capture time,
    // the slice dims aren't resolved until the View executor runs.
    for (size_t k = 0; k < N; ++k) {
        if (vectors[k] == nullptr) {
            EINSUMS_THROW_EXCEPTION(std::invalid_argument, "cg::outer_sum: vector[{}] is null", k);
        }
        if (vectors[k]->rank() != 1) {
            EINSUMS_THROW_EXCEPTION(rank_error, "cg::outer_sum: vector[{}] must be rank-1; got rank {}", k, vectors[k]->rank());
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
        // Dim check (deferred from capture time so view operands can resolve).
        for (size_t k = 0; k < N; ++k) {
            if (vectors[k]->dim(0) != r->dim(k)) {
                EINSUMS_THROW_EXCEPTION(std::invalid_argument, "cg::outer_sum: vector[{}] length ({}) doesn't match result dim {} ({})", k,
                                        vectors[k]->dim(0), k, r->dim(k));
            }
        }
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
            // Increment multi-index (axis 0 fastest, direction is irrelevant for correctness).
            for (size_t k = 0; k < N; ++k) {
                if (++idx[k] < dims[k])
                    break;
                idx[k] = 0;
            }
        }
    };

    auto &ctx = CaptureContext::current();
    if (!ctx.is_capturing()) {
        LabeledSection("outer_sum eager");
        apply(result);
        return;
    }

    LabeledSection("outer_sum capture");
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
        LabeledSection("outer_sum execute");
        auto                           *r_ptr = static_cast<ResultType *>(r_slot->ptr);
        std::vector<VectorType const *> rebound(N);
        for (size_t k = 0; k < N; ++k)
            rebound[k] = static_cast<VectorType const *>(v_slots[k]->ptr);

        // Dim check (deferred from capture time so view operands can resolve).
        for (size_t k = 0; k < N; ++k) {
            if (rebound[k]->dim(0) != r_ptr->dim(k)) {
                EINSUMS_THROW_EXCEPTION(std::invalid_argument, "cg::outer_sum: vector[{}] length ({}) doesn't match result dim {} ({})", k,
                                        rebound[k]->dim(0), k, r_ptr->dim(k));
            }
        }
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
APIARY_EXPOSE
APIARY_MODULE("linalg")
APIARY_INSTANTIATE_AS("norm", einsums::GeneralRuntimeTensor<float,                std::allocator<float>>)
APIARY_INSTANTIATE_AS("norm", einsums::GeneralRuntimeTensor<double,               std::allocator<double>>)
APIARY_INSTANTIATE_AS("norm", einsums::GeneralRuntimeTensor<std::complex<float>,  std::allocator<std::complex<float>>>)
APIARY_INSTANTIATE_AS("norm", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
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
        LabeledSection("norm eager");
        *result = linear_algebra::norm(norm_type, A);
        return;
    }

    LabeledSection("norm capture");
    auto [a_id, a_slot] = ctx.get_slot(A);
    TensorId r_id       = ctx.get_or_register_scalar(result, "norm_result");

    auto executor = [result, norm_type, a_slot]() {
        LabeledSection("norm execute");
        *result = linear_algebra::norm(norm_type, *static_cast<AType const *>(a_slot->ptr));
    };

    ctx.record(OpKind::Norm, "norm", {a_id}, {r_id}, std::move(executor));
}

/// Python-friendly graph-aware norm: writes the result into ``result->data()[0]``.
///
/// For complex inputs the result is real-valued (e.g. complex<double> input
/// requires a ``double`` result tensor). Use ``Norm::ONE``, ``Norm::TWO``,
/// ``Norm::INFINITY_``, ``Norm::FROBENIUS``, etc.
template <CoreBasicTensorConcept ResultType, typename AType>
    requires requires {
        requires std::is_same_v<typename ResultType::ValueType, RemoveComplexT<typename AType::ValueType>>;
        requires(CoreBasicTensorConcept<AType> || IsTiledTensorV<std::remove_cvref_t<AType>>);
    }
// clang-format off
APIARY_EXPOSE
APIARY_MODULE("linalg")
// All 4 combinations of (Result, A) x (owning, view), per dtype mapping. Norm
// returns a real value, so for complex inputs the result must be the
// corresponding real dtype (float for complex<float>, double for complex<double>).
//
// float input -> float result
APIARY_INSTANTIATE_AS("norm", einsums::GeneralRuntimeTensor<float,  std::allocator<float>>,  einsums::GeneralRuntimeTensor<float,                std::allocator<float>>)
APIARY_INSTANTIATE_AS("norm", einsums::GeneralRuntimeTensor<float,  std::allocator<float>>,  einsums::RuntimeTensorView<float>)
APIARY_INSTANTIATE_AS("norm", einsums::RuntimeTensorView<float>,                             einsums::GeneralRuntimeTensor<float,                std::allocator<float>>)
APIARY_INSTANTIATE_AS("norm", einsums::RuntimeTensorView<float>,                             einsums::RuntimeTensorView<float>)
// double input -> double result
APIARY_INSTANTIATE_AS("norm", einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::GeneralRuntimeTensor<double,               std::allocator<double>>)
APIARY_INSTANTIATE_AS("norm", einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::RuntimeTensorView<double>)
APIARY_INSTANTIATE_AS("norm", einsums::RuntimeTensorView<double>,                            einsums::GeneralRuntimeTensor<double,               std::allocator<double>>)
APIARY_INSTANTIATE_AS("norm", einsums::RuntimeTensorView<double>,                            einsums::RuntimeTensorView<double>)
// complex<float> input -> float result
APIARY_INSTANTIATE_AS("norm", einsums::GeneralRuntimeTensor<float,  std::allocator<float>>,  einsums::GeneralRuntimeTensor<std::complex<float>,  std::allocator<std::complex<float>>>)
APIARY_INSTANTIATE_AS("norm", einsums::GeneralRuntimeTensor<float,  std::allocator<float>>,  einsums::RuntimeTensorView<std::complex<float>>)
APIARY_INSTANTIATE_AS("norm", einsums::RuntimeTensorView<float>,                             einsums::GeneralRuntimeTensor<std::complex<float>,  std::allocator<std::complex<float>>>)
APIARY_INSTANTIATE_AS("norm", einsums::RuntimeTensorView<float>,                             einsums::RuntimeTensorView<std::complex<float>>)
// complex<double> input -> double result
APIARY_INSTANTIATE_AS("norm", einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
APIARY_INSTANTIATE_AS("norm", einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::RuntimeTensorView<std::complex<double>>)
APIARY_INSTANTIATE_AS("norm", einsums::RuntimeTensorView<double>,                            einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
APIARY_INSTANTIATE_AS("norm", einsums::RuntimeTensorView<double>,                            einsums::RuntimeTensorView<std::complex<double>>)
// tiled operand, real scalar result (RemoveComplexT of operand's type)
APIARY_INSTANTIATE_AS("norm", einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::TiledRuntimeTensor<float>)
APIARY_INSTANTIATE_AS("norm", einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::TiledRuntimeTensor<double>)
APIARY_INSTANTIATE_AS("norm", einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::TiledRuntimeTensor<std::complex<float>>)
APIARY_INSTANTIATE_AS("norm", einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::TiledRuntimeTensor<std::complex<double>>)
    // clang-format on
    void norm_python(ResultType *result, linear_algebra::Norm norm_type, AType const &A) {
    using R = RemoveComplexT<typename AType::ValueType>;
    if (result->size() < 1) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "cg::norm: result tensor must have at least one element");
    }

    auto compute = [](linear_algebra::Norm nt, AType const &a) -> R {
        if constexpr (IsTiledTensorV<std::remove_cvref_t<AType>>) {
            return detail::tiled_norm<typename AType::ValueType>(nt, a);
        } else {
            return linear_algebra::norm(nt, a);
        }
    };

    auto &ctx = CaptureContext::current();
    if (!ctx.is_capturing()) {
        LabeledSection("norm_python eager");
        result->data()[0] = compute(norm_type, A);
        return;
    }

    LabeledSection("norm_python capture");
    auto [a_id, a_slot] = ctx.get_slot(A);
    auto [r_id, r_slot] = ctx.get_slot(*result);

    auto executor = [norm_type, a_slot, r_slot, compute]() {
        LabeledSection("norm_python execute");
        auto *r_ptr      = static_cast<ResultType *>(r_slot->ptr);
        r_ptr->data()[0] = compute(norm_type, *static_cast<AType const *>(a_slot->ptr));
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
//   - eager:    ``auto t = cg::trace(A);``, executes immediately. Throws if
//               called during graph capture (capture has no scalar return).
//   - recorded: ``cg::trace(&t, A);``, records into the active graph. ``t``
//               is read at execute time and is the destination scalar.
//
// Trace doesn't need a dedicated OpKind, it lowers to a one-liner inside an
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
APIARY_EXPOSE
APIARY_MODULE("linalg")
APIARY_INSTANTIATE_AS("trace", einsums::GeneralRuntimeTensor<float,                std::allocator<float>>)
APIARY_INSTANTIATE_AS("trace", einsums::GeneralRuntimeTensor<double,               std::allocator<double>>)
APIARY_INSTANTIATE_AS("trace", einsums::GeneralRuntimeTensor<std::complex<float>,  std::allocator<std::complex<float>>>)
APIARY_INSTANTIATE_AS("trace", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
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
        LabeledSection("trace eager");
        if (A.dim(0) != A.dim(1))
            EINSUMS_THROW_EXCEPTION(std::invalid_argument, "cg::trace: input must be square");
        T sum = T{};
        for (size_t i = 0; i < A.dim(0); ++i)
            sum += A(i, i);
        *result = sum;
        return;
    }

    LabeledSection("trace capture");
    auto [a_id, a_slot] = ctx.get_slot(A);
    TensorId r_id       = ctx.get_or_register_scalar(result, "trace_result");

    auto executor = [result, a_slot]() {
        LabeledSection("trace execute");
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
template <CoreBasicTensorConcept ResultType, typename AType>
    requires requires {
        requires std::is_same_v<typename ResultType::ValueType, typename AType::ValueType>;
        requires(CoreBasicTensorConcept<AType> || IsTiledTensorV<std::remove_cvref_t<AType>>);
    }
// clang-format off
APIARY_EXPOSE
APIARY_MODULE("linalg")
// All 4 combinations of (Result, A) x (owning, view), per dtype. Same-dtype
// across operands is enforced by the requires clause above.
//
// float
APIARY_INSTANTIATE_AS("trace", einsums::GeneralRuntimeTensor<float, std::allocator<float>>,                              einsums::GeneralRuntimeTensor<float, std::allocator<float>>)
APIARY_INSTANTIATE_AS("trace", einsums::GeneralRuntimeTensor<float, std::allocator<float>>,                              einsums::RuntimeTensorView<float>)
APIARY_INSTANTIATE_AS("trace", einsums::RuntimeTensorView<float>,                                                        einsums::GeneralRuntimeTensor<float, std::allocator<float>>)
APIARY_INSTANTIATE_AS("trace", einsums::RuntimeTensorView<float>,                                                        einsums::RuntimeTensorView<float>)
// double
APIARY_INSTANTIATE_AS("trace", einsums::GeneralRuntimeTensor<double, std::allocator<double>>,                            einsums::GeneralRuntimeTensor<double, std::allocator<double>>)
APIARY_INSTANTIATE_AS("trace", einsums::GeneralRuntimeTensor<double, std::allocator<double>>,                            einsums::RuntimeTensorView<double>)
APIARY_INSTANTIATE_AS("trace", einsums::RuntimeTensorView<double>,                                                       einsums::GeneralRuntimeTensor<double, std::allocator<double>>)
APIARY_INSTANTIATE_AS("trace", einsums::RuntimeTensorView<double>,                                                       einsums::RuntimeTensorView<double>)
// complex<float>
APIARY_INSTANTIATE_AS("trace", einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>,  einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>)
APIARY_INSTANTIATE_AS("trace", einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>,  einsums::RuntimeTensorView<std::complex<float>>)
APIARY_INSTANTIATE_AS("trace", einsums::RuntimeTensorView<std::complex<float>>,                                          einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>)
APIARY_INSTANTIATE_AS("trace", einsums::RuntimeTensorView<std::complex<float>>,                                          einsums::RuntimeTensorView<std::complex<float>>)
// complex<double>
APIARY_INSTANTIATE_AS("trace", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>,einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
APIARY_INSTANTIATE_AS("trace", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>,einsums::RuntimeTensorView<std::complex<double>>)
APIARY_INSTANTIATE_AS("trace", einsums::RuntimeTensorView<std::complex<double>>,                                         einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
APIARY_INSTANTIATE_AS("trace", einsums::RuntimeTensorView<std::complex<double>>,                                         einsums::RuntimeTensorView<std::complex<double>>)
// tiled operand, dense scalar result, per dtype
APIARY_INSTANTIATE_AS("trace", einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::TiledRuntimeTensor<float>)
APIARY_INSTANTIATE_AS("trace", einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::TiledRuntimeTensor<double>)
APIARY_INSTANTIATE_AS("trace", einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::TiledRuntimeTensor<std::complex<float>>)
APIARY_INSTANTIATE_AS("trace", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::TiledRuntimeTensor<std::complex<double>>)
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
        if constexpr (IsTiledTensorV<std::remove_cvref_t<AType>>) {
            return detail::tiled_trace<T>(a);
        } else {
            T sum = T{};
            for (size_t i = 0; i < a.dim(0); ++i)
                sum += a(i, i);
            return sum;
        }
    };

    auto &ctx = CaptureContext::current();
    if (!ctx.is_capturing()) {
        LabeledSection("trace_python eager");
        result->data()[0] = compute(A);
        return;
    }

    LabeledSection("trace_python capture");
    auto [a_id, a_slot] = ctx.get_slot(A);
    auto [r_id, r_slot] = ctx.get_slot(*result);

    auto executor = [a_slot, r_slot, compute]() {
        LabeledSection("trace_python execute");
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
APIARY_EXPOSE
APIARY_MODULE("linalg")
APIARY_TEMPLATE_KWARGS("trans_a", "trans_b")
// All 8 combinations of (A, B, C) x (owning, view), per dtype, per
// (TransA, TransB) bool pair. INSTANTIATE_BOOLS expands each line to 4
// entries.
//
// float
APIARY_INSTANTIATE_BOOLS("symm_gemm", einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::GeneralRuntimeTensor<float, std::allocator<float>>)
APIARY_INSTANTIATE_BOOLS("symm_gemm", einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::RuntimeTensorView<float>)
APIARY_INSTANTIATE_BOOLS("symm_gemm", einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::RuntimeTensorView<float>,                          einsums::GeneralRuntimeTensor<float, std::allocator<float>>)
APIARY_INSTANTIATE_BOOLS("symm_gemm", einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::RuntimeTensorView<float>,                          einsums::RuntimeTensorView<float>)
APIARY_INSTANTIATE_BOOLS("symm_gemm", einsums::RuntimeTensorView<float>,                          einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::GeneralRuntimeTensor<float, std::allocator<float>>)
APIARY_INSTANTIATE_BOOLS("symm_gemm", einsums::RuntimeTensorView<float>,                          einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::RuntimeTensorView<float>)
APIARY_INSTANTIATE_BOOLS("symm_gemm", einsums::RuntimeTensorView<float>,                          einsums::RuntimeTensorView<float>,                          einsums::GeneralRuntimeTensor<float, std::allocator<float>>)
APIARY_INSTANTIATE_BOOLS("symm_gemm", einsums::RuntimeTensorView<float>,                          einsums::RuntimeTensorView<float>,                          einsums::RuntimeTensorView<float>)
// double
APIARY_INSTANTIATE_BOOLS("symm_gemm", einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::GeneralRuntimeTensor<double, std::allocator<double>>)
APIARY_INSTANTIATE_BOOLS("symm_gemm", einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::RuntimeTensorView<double>)
APIARY_INSTANTIATE_BOOLS("symm_gemm", einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::RuntimeTensorView<double>,                          einsums::GeneralRuntimeTensor<double, std::allocator<double>>)
APIARY_INSTANTIATE_BOOLS("symm_gemm", einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::RuntimeTensorView<double>,                          einsums::RuntimeTensorView<double>)
APIARY_INSTANTIATE_BOOLS("symm_gemm", einsums::RuntimeTensorView<double>,                          einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::GeneralRuntimeTensor<double, std::allocator<double>>)
APIARY_INSTANTIATE_BOOLS("symm_gemm", einsums::RuntimeTensorView<double>,                          einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::RuntimeTensorView<double>)
APIARY_INSTANTIATE_BOOLS("symm_gemm", einsums::RuntimeTensorView<double>,                          einsums::RuntimeTensorView<double>,                          einsums::GeneralRuntimeTensor<double, std::allocator<double>>)
APIARY_INSTANTIATE_BOOLS("symm_gemm", einsums::RuntimeTensorView<double>,                          einsums::RuntimeTensorView<double>,                          einsums::RuntimeTensorView<double>)
// complex<float>
APIARY_INSTANTIATE_BOOLS("symm_gemm", einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>)
APIARY_INSTANTIATE_BOOLS("symm_gemm", einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::RuntimeTensorView<std::complex<float>>)
APIARY_INSTANTIATE_BOOLS("symm_gemm", einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::RuntimeTensorView<std::complex<float>>,                                            einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>)
APIARY_INSTANTIATE_BOOLS("symm_gemm", einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::RuntimeTensorView<std::complex<float>>,                                            einsums::RuntimeTensorView<std::complex<float>>)
APIARY_INSTANTIATE_BOOLS("symm_gemm", einsums::RuntimeTensorView<std::complex<float>>,                                            einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>)
APIARY_INSTANTIATE_BOOLS("symm_gemm", einsums::RuntimeTensorView<std::complex<float>>,                                            einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::RuntimeTensorView<std::complex<float>>)
APIARY_INSTANTIATE_BOOLS("symm_gemm", einsums::RuntimeTensorView<std::complex<float>>,                                            einsums::RuntimeTensorView<std::complex<float>>,                                            einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>)
APIARY_INSTANTIATE_BOOLS("symm_gemm", einsums::RuntimeTensorView<std::complex<float>>,                                            einsums::RuntimeTensorView<std::complex<float>>,                                            einsums::RuntimeTensorView<std::complex<float>>)
// complex<double>
APIARY_INSTANTIATE_BOOLS("symm_gemm", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
APIARY_INSTANTIATE_BOOLS("symm_gemm", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::RuntimeTensorView<std::complex<double>>)
APIARY_INSTANTIATE_BOOLS("symm_gemm", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::RuntimeTensorView<std::complex<double>>,                                          einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
APIARY_INSTANTIATE_BOOLS("symm_gemm", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::RuntimeTensorView<std::complex<double>>,                                          einsums::RuntimeTensorView<std::complex<double>>)
APIARY_INSTANTIATE_BOOLS("symm_gemm", einsums::RuntimeTensorView<std::complex<double>>,                                          einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
APIARY_INSTANTIATE_BOOLS("symm_gemm", einsums::RuntimeTensorView<std::complex<double>>,                                          einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::RuntimeTensorView<std::complex<double>>)
APIARY_INSTANTIATE_BOOLS("symm_gemm", einsums::RuntimeTensorView<std::complex<double>>,                                          einsums::RuntimeTensorView<std::complex<double>>,                                          einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
APIARY_INSTANTIATE_BOOLS("symm_gemm", einsums::RuntimeTensorView<std::complex<double>>,                                          einsums::RuntimeTensorView<std::complex<double>>,                                          einsums::RuntimeTensorView<std::complex<double>>)
    // clang-format on
    void symm_gemm(AType const &A, BType const &B, CType *C) {
    if (A.rank() != 2 || B.rank() != 2 || C->rank() != 2) {
        EINSUMS_THROW_EXCEPTION(rank_error, "cg::symm_gemm requires rank-2 tensors; got ranks {}, {}, {}.", A.rank(), B.rank(), C->rank());
    }
    auto &ctx = CaptureContext::current();
    if (!ctx.is_capturing()) {
        LabeledSection("symm_gemm eager");
        linear_algebra::symm_gemm<TransA, TransB>(A, B, C);
        return;
    }
    LabeledSection("symm_gemm capture");
    auto [a_id, a_slot] = ctx.get_slot(A);
    auto [b_id, b_slot] = ctx.get_slot(B);
    auto [c_id, c_slot] = ctx.get_slot(*C);
    auto executor       = [a_slot, b_slot, c_slot]() {
        LabeledSection("symm_gemm execute");
        ProfileAnnotate("a_n", static_cast<int64_t>(static_cast<AType const *>(a_slot->ptr)->dim(0)));
        ProfileAnnotate("b_m", static_cast<int64_t>(static_cast<BType const *>(b_slot->ptr)->dim(0)));
        ProfileAnnotate("b_n", static_cast<int64_t>(static_cast<BType const *>(b_slot->ptr)->dim(1)));
        linear_algebra::symm_gemm<TransA, TransB>(*static_cast<AType const *>(a_slot->ptr), *static_cast<BType const *>(b_slot->ptr),
                                                  static_cast<CType *>(c_slot->ptr));
    };
    ctx.record(OpKind::SymmGemm, "symm_gemm", {a_id, b_id}, {c_id}, std::move(executor));
}

// Original compile-time-rank symm_gemm, kept under a different signature
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
        LabeledSection("symm_gemm eager");
        linear_algebra::symm_gemm<TransA, TransB>(A, B, C);
        return;
    }

    LabeledSection("symm_gemm capture");
    auto [a_id, a_slot] = ctx.get_slot(A);
    auto [b_id, b_slot] = ctx.get_slot(B);
    auto [c_id, c_slot] = ctx.get_slot(*C);

    auto executor = [a_slot, b_slot, c_slot]() {
        LabeledSection("symm_gemm execute");
        ProfileAnnotate("a_n", static_cast<int64_t>(static_cast<AType const *>(a_slot->ptr)->dim(0)));
        ProfileAnnotate("b_m", static_cast<int64_t>(static_cast<BType const *>(b_slot->ptr)->dim(0)));
        ProfileAnnotate("b_n", static_cast<int64_t>(static_cast<BType const *>(b_slot->ptr)->dim(1)));
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
        LabeledSection("syev eager");
        linear_algebra::syev<ComputeEigenvectors>(A, W);
        return;
    }

    LabeledSection("syev capture");
    auto [a_id, a_slot] = ctx.get_slot(*A);
    auto [w_id, w_slot] = ctx.get_slot(*W);

    auto executor = [a_slot, w_slot]() {
        LabeledSection("syev execute");
        ProfileAnnotate("n", static_cast<int64_t>(static_cast<AType *>(a_slot->ptr)->dim(0)));
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
APIARY_EXPOSE
APIARY_MODULE("linalg")
APIARY_TEMPLATE_KWARGS("compute_eigenvectors")
APIARY_INSTANTIATE_BOOLS("syev", einsums::GeneralRuntimeTensor<float,  std::allocator<float>>,  einsums::GeneralRuntimeTensor<float,  std::allocator<float>>)
APIARY_INSTANTIATE_BOOLS("syev", einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::GeneralRuntimeTensor<double, std::allocator<double>>)
    // clang-format on
    void syev(AType *A, WType *W) {
    if (A->rank() != 2 || W->rank() != 1) {
        EINSUMS_THROW_EXCEPTION(rank_error, "cg::syev requires A rank-2 and W rank-1; got {}, {}.", A->rank(), W->rank());
    }

    auto &ctx = CaptureContext::current();
    if (!ctx.is_capturing()) {
        LabeledSection("syev eager");
        linear_algebra::syev<ComputeEigenvectors>(A, W);
        return;
    }

    LabeledSection("syev capture");
    auto [a_id, a_slot] = ctx.get_slot(*A);
    auto [w_id, w_slot] = ctx.get_slot(*W);

    auto executor = [a_slot, w_slot]() {
        LabeledSection("syev execute");
        ProfileAnnotate("n", static_cast<int64_t>(static_cast<AType *>(a_slot->ptr)->dim(0)));
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
APIARY_EXPOSE
APIARY_MODULE("linalg")
APIARY_TEMPLATE_KWARGS("compute_eigenvectors")
APIARY_INSTANTIATE_BOOLS("syev_eig", einsums::GeneralRuntimeTensor<float,  std::allocator<float>>)
APIARY_INSTANTIATE_BOOLS("syev_eig", einsums::GeneralRuntimeTensor<double, std::allocator<double>>)
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
        LabeledSection("heev eager");
        linear_algebra::heev<ComputeEigenvectors>(A, W);
        return;
    }

    LabeledSection("heev capture");
    auto [a_id, a_slot] = ctx.get_slot(*A);
    auto [w_id, w_slot] = ctx.get_slot(*W);

    auto executor = [a_slot, w_slot]() {
        LabeledSection("heev execute");
        ProfileAnnotate("n", static_cast<int64_t>(static_cast<AType *>(a_slot->ptr)->dim(0)));
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
APIARY_EXPOSE
APIARY_MODULE("linalg")
APIARY_TEMPLATE_KWARGS("compute_eigenvectors")
APIARY_INSTANTIATE_BOOLS("heev", einsums::GeneralRuntimeTensor<std::complex<float>,  std::allocator<std::complex<float>>>,  einsums::GeneralRuntimeTensor<float,  std::allocator<float>>)
APIARY_INSTANTIATE_BOOLS("heev", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::GeneralRuntimeTensor<double, std::allocator<double>>)
    // clang-format on
    void heev(AType *A, WType *W) {
    if (A->rank() != 2 || W->rank() != 1) {
        EINSUMS_THROW_EXCEPTION(rank_error, "cg::heev requires A rank-2 and W rank-1; got {}, {}.", A->rank(), W->rank());
    }

    auto &ctx = CaptureContext::current();
    if (!ctx.is_capturing()) {
        LabeledSection("heev eager");
        linear_algebra::heev<ComputeEigenvectors>(A, W);
        return;
    }

    LabeledSection("heev capture");
    auto [a_id, a_slot] = ctx.get_slot(*A);
    auto [w_id, w_slot] = ctx.get_slot(*W);

    auto executor = [a_slot, w_slot]() {
        LabeledSection("heev execute");
        ProfileAnnotate("n", static_cast<int64_t>(static_cast<AType *>(a_slot->ptr)->dim(0)));
        linear_algebra::heev<ComputeEigenvectors>(static_cast<AType *>(a_slot->ptr), static_cast<WType *>(w_slot->ptr));
    };
    ctx.record(OpKind::Heev, "heev", {a_id}, {a_id, w_id}, std::move(executor));
}

/// Tiled real-symmetric eigendecomposition: independently diagonalize each
/// diagonal block of a block-diagonal tiled matrix. A is overwritten in place
/// with the per-block eigenvectors; W (tiled rank-1, partitioned like A's rows)
/// receives each block's eigenvalues. Off-diagonal tiles are rejected (the op is
/// only meaningful for a block-diagonal operator).
template <bool ComputeEigenvectors = true, TiledTensorConcept AType, TiledTensorConcept WType>
    requires(std::is_same_v<typename AType::ValueType, typename WType::ValueType> && !IsComplexV<typename AType::ValueType>)
void syev(AType *A, WType *W) {
    using T   = typename AType::ValueType;
    auto &ctx = CaptureContext::current();
    if (!ctx.is_capturing()) {
        LabeledSection("syev eager");
        detail::tiled_syev<ComputeEigenvectors, T>(A, W);
        return;
    }
    LabeledSection("syev capture");
    auto [a_id, a_slot] = ctx.get_slot(*A);
    auto [w_id, w_slot] = ctx.get_slot(*W);
    auto executor       = [a_slot, w_slot]() {
        LabeledSection("syev execute");
        detail::tiled_syev<ComputeEigenvectors, T>(static_cast<AType *>(a_slot->ptr), static_cast<WType *>(w_slot->ptr));
    };
    ctx.record(OpKind::Syev, "syev", {a_id}, {a_id, w_id}, std::move(executor));
}

/// Tiled Hermitian eigendecomposition (complex analogue of the tiled syev). W
/// holds the real eigenvalues.
template <bool ComputeEigenvectors = true, TiledTensorConcept AType, TiledTensorConcept WType>
    requires(IsComplexV<typename AType::ValueType> && std::is_same_v<typename WType::ValueType, RemoveComplexT<typename AType::ValueType>>)
void heev(AType *A, WType *W) {
    using T   = typename AType::ValueType;
    auto &ctx = CaptureContext::current();
    if (!ctx.is_capturing()) {
        LabeledSection("heev eager");
        detail::tiled_heev<ComputeEigenvectors, T>(A, W);
        return;
    }
    LabeledSection("heev capture");
    auto [a_id, a_slot] = ctx.get_slot(*A);
    auto [w_id, w_slot] = ctx.get_slot(*W);
    auto executor       = [a_slot, w_slot]() {
        LabeledSection("heev execute");
        detail::tiled_heev<ComputeEigenvectors, T>(static_cast<AType *>(a_slot->ptr), static_cast<WType *>(w_slot->ptr));
    };
    ctx.record(OpKind::Heev, "heev", {a_id}, {a_id, w_id}, std::move(executor));
}

/// Python-facing syev: real-symmetric eigendecomposition (in place; A receives
/// eigenvectors, W the eigenvalues). A wrapper because the templated syev has a
/// leading non-type ``bool ComputeEigenvectors`` parameter the pybind codegen
/// can't pin via INSTANTIATE_AS; this fixes it to true and presents a clean,
/// type-only signature. Accepts dense (RuntimeTensor) or tiled operands; the
/// inner syev<true>() dispatches accordingly.
template <typename AType, typename WType>
    requires(std::is_same_v<typename AType::ValueType, typename WType::ValueType> && !IsComplexV<typename AType::ValueType> &&
             (CoreBasicTensorConcept<AType> || IsTiledTensorV<std::remove_cvref_t<AType>>) &&
             (CoreBasicTensorConcept<WType> || IsTiledTensorV<std::remove_cvref_t<WType>>))
// clang-format off
APIARY_EXPOSE
APIARY_MODULE("linalg")
APIARY_INSTANTIATE_AS("syev", einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::GeneralRuntimeTensor<float, std::allocator<float>>)
APIARY_INSTANTIATE_AS("syev", einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::GeneralRuntimeTensor<double, std::allocator<double>>)
APIARY_INSTANTIATE_AS("syev", einsums::TiledRuntimeTensor<float>, einsums::TiledRuntimeTensor<float>)
APIARY_INSTANTIATE_AS("syev", einsums::TiledRuntimeTensor<double>, einsums::TiledRuntimeTensor<double>)
    // clang-format on
    void syev_python(AType *A, WType *W) {
    syev<true>(A, W);
}

/// Python-facing heev: Hermitian eigendecomposition (complex A, real W). See
/// syev_python for why this wrapper exists.
template <typename AType, typename WType>
    requires(IsComplexV<typename AType::ValueType> &&
             std::is_same_v<typename WType::ValueType, RemoveComplexT<typename AType::ValueType>> &&
             (CoreBasicTensorConcept<AType> || IsTiledTensorV<std::remove_cvref_t<AType>>) &&
             (CoreBasicTensorConcept<WType> || IsTiledTensorV<std::remove_cvref_t<WType>>))
// clang-format off
APIARY_EXPOSE
APIARY_MODULE("linalg")
APIARY_INSTANTIATE_AS("heev", einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::GeneralRuntimeTensor<float, std::allocator<float>>)
APIARY_INSTANTIATE_AS("heev", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::GeneralRuntimeTensor<double, std::allocator<double>>)
APIARY_INSTANTIATE_AS("heev", einsums::TiledRuntimeTensor<std::complex<float>>, einsums::TiledRuntimeTensor<float>)
APIARY_INSTANTIATE_AS("heev", einsums::TiledRuntimeTensor<std::complex<double>>, einsums::TiledRuntimeTensor<double>)
    // clang-format on
    void heev_python(AType *A, WType *W) {
    heev<true>(A, W);
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
        LabeledSection("gesv eager");
        return linear_algebra::gesv(A, B);
    }

    LabeledSection("gesv capture");
    auto [a_id, a_slot] = ctx.get_slot(*A);
    auto [b_id, b_slot] = ctx.get_slot(*B);

    auto executor = [a_slot, b_slot]() {
        LabeledSection("gesv execute");
        ProfileAnnotate("n", static_cast<int64_t>(static_cast<AType *>(a_slot->ptr)->dim(0)));
        ProfileAnnotate("nrhs", static_cast<int64_t>(static_cast<BType *>(b_slot->ptr)->dim(1)));
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
APIARY_EXPOSE
APIARY_MODULE("linalg")
APIARY_INSTANTIATE_AS("gesv", einsums::GeneralRuntimeTensor<float,                std::allocator<float>>,                einsums::GeneralRuntimeTensor<float,                std::allocator<float>>)
APIARY_INSTANTIATE_AS("gesv", einsums::GeneralRuntimeTensor<double,               std::allocator<double>>,               einsums::GeneralRuntimeTensor<double,               std::allocator<double>>)
APIARY_INSTANTIATE_AS("gesv", einsums::GeneralRuntimeTensor<std::complex<float>,  std::allocator<std::complex<float>>>,  einsums::GeneralRuntimeTensor<std::complex<float>,  std::allocator<std::complex<float>>>)
APIARY_INSTANTIATE_AS("gesv", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
    // clang-format on
    auto gesv(AType *A, BType *B) -> int {
    if (A->rank() != 2 || (B->rank() != 1 && B->rank() != 2)) {
        EINSUMS_THROW_EXCEPTION(rank_error, "cg::gesv requires A rank-2 and B rank-1 or rank-2; got {}, {}.", A->rank(), B->rank());
    }

    auto &ctx = CaptureContext::current();
    if (!ctx.is_capturing()) {
        LabeledSection("gesv eager");
        return linear_algebra::gesv(A, B);
    }

    LabeledSection("gesv capture");
    auto [a_id, a_slot] = ctx.get_slot(*A);
    auto [b_id, b_slot] = ctx.get_slot(*B);

    auto executor = [a_slot, b_slot]() {
        LabeledSection("gesv execute");
        ProfileAnnotate("n", static_cast<int64_t>(static_cast<AType *>(a_slot->ptr)->dim(0)));
        ProfileAnnotate("nrhs", static_cast<int64_t>(static_cast<BType *>(b_slot->ptr)->dim(1)));
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
        LabeledSection("invert eager");
        linear_algebra::invert(A);
        return;
    }

    LabeledSection("invert capture");
    auto [a_id, a_slot] = ctx.get_slot(*A);

    auto executor = [a_slot]() {
        LabeledSection("invert execute");
        ProfileAnnotate("n", static_cast<int64_t>(static_cast<AType *>(a_slot->ptr)->dim(0)));
        linear_algebra::invert(static_cast<AType *>(a_slot->ptr));
    };
    ctx.record(OpKind::Invert, "invert", {a_id}, {a_id}, std::move(executor));
}

/// In-place matrix inverse: ``A := A^-1``.
///
/// ``A`` must be rank 2 and square. Internally calls ``getrf`` followed
/// by ``getri``; raises ``rank_error`` if the input rank is wrong and
/// the LAPACK kernel raises if ``A`` is singular.
template <RuntimeRankTensorConcept AType>
// clang-format off
APIARY_EXPOSE
APIARY_MODULE("linalg")
APIARY_INSTANTIATE_AS("invert", einsums::GeneralRuntimeTensor<float,                std::allocator<float>>)
APIARY_INSTANTIATE_AS("invert", einsums::GeneralRuntimeTensor<double,               std::allocator<double>>)
APIARY_INSTANTIATE_AS("invert", einsums::GeneralRuntimeTensor<std::complex<float>,  std::allocator<std::complex<float>>>)
APIARY_INSTANTIATE_AS("invert", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
    // clang-format on
    void invert(AType *A) {
    if (A->rank() != 2) {
        EINSUMS_THROW_EXCEPTION(rank_error, "cg::invert requires rank-2 tensor; got rank {}.", A->rank());
    }

    auto &ctx = CaptureContext::current();
    if (!ctx.is_capturing()) {
        LabeledSection("invert eager");
        linear_algebra::invert(A);
        return;
    }

    LabeledSection("invert capture");
    auto [a_id, a_slot] = ctx.get_slot(*A);

    auto executor = [a_slot]() {
        LabeledSection("invert execute");
        ProfileAnnotate("n", static_cast<int64_t>(static_cast<AType *>(a_slot->ptr)->dim(0)));
        linear_algebra::invert(static_cast<AType *>(a_slot->ptr));
    };
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
/// inputs. ``U`` and ``Vt`` are always present, and the user can post-
/// filter for the "no vectors" case if needed. Cannot be used during
/// graph capture (returns by value); ``A`` is left unmodified.
template <RuntimeRankTensorConcept AType>
// clang-format off
APIARY_EXPOSE
APIARY_MODULE("linalg")
APIARY_INSTANTIATE_AS("svd", einsums::GeneralRuntimeTensor<float,                std::allocator<float>>)
APIARY_INSTANTIATE_AS("svd", einsums::GeneralRuntimeTensor<double,               std::allocator<double>>)
APIARY_INSTANTIATE_AS("svd", einsums::GeneralRuntimeTensor<std::complex<float>,  std::allocator<std::complex<float>>>)
APIARY_INSTANTIATE_AS("svd", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
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
APIARY_EXPOSE
APIARY_MODULE("linalg")
APIARY_INSTANTIATE_AS("svd_dd", einsums::GeneralRuntimeTensor<float,                std::allocator<float>>)
APIARY_INSTANTIATE_AS("svd_dd", einsums::GeneralRuntimeTensor<double,               std::allocator<double>>)
APIARY_INSTANTIATE_AS("svd_dd", einsums::GeneralRuntimeTensor<std::complex<float>,  std::allocator<std::complex<float>>>)
APIARY_INSTANTIATE_AS("svd_dd", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
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
/// Randomized algorithm with over-sampling factor 5, which requires
/// ``A.dim(0) >= k + 5``. Smaller inputs raise ``IndexError`` from the
/// projection step. Results are approximate; expect small drift versus a
/// full ``svd``.
///
/// Cannot be used during graph capture (returns by value); ``A`` is left
/// unmodified.
template <RuntimeRankTensorConcept AType>
// clang-format off
APIARY_EXPOSE
APIARY_MODULE("linalg")
APIARY_INSTANTIATE_AS("truncated_svd", einsums::GeneralRuntimeTensor<float,                std::allocator<float>>)
APIARY_INSTANTIATE_AS("truncated_svd", einsums::GeneralRuntimeTensor<double,               std::allocator<double>>)
APIARY_INSTANTIATE_AS("truncated_svd", einsums::GeneralRuntimeTensor<std::complex<float>,  std::allocator<std::complex<float>>>)
APIARY_INSTANTIATE_AS("truncated_svd", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
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
/// Randomized algorithm with over-sampling factor 5, which requires
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
APIARY_EXPOSE
APIARY_MODULE("linalg")
APIARY_INSTANTIATE_AS("truncated_syev", einsums::GeneralRuntimeTensor<float,  std::allocator<float>>)
APIARY_INSTANTIATE_AS("truncated_syev", einsums::GeneralRuntimeTensor<double, std::allocator<double>>)
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
APIARY_EXPOSE
APIARY_MODULE("linalg")
APIARY_INSTANTIATE_AS("qr", einsums::GeneralRuntimeTensor<float,                std::allocator<float>>)
APIARY_INSTANTIATE_AS("qr", einsums::GeneralRuntimeTensor<double,               std::allocator<double>>)
APIARY_INSTANTIATE_AS("qr", einsums::GeneralRuntimeTensor<std::complex<float>,  std::allocator<std::complex<float>>>)
APIARY_INSTANTIATE_AS("qr", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
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
APIARY_EXPOSE
APIARY_MODULE("linalg")
APIARY_INSTANTIATE_AS("pow", einsums::GeneralRuntimeTensor<float,  std::allocator<float>>)
APIARY_INSTANTIATE_AS("pow", einsums::GeneralRuntimeTensor<double, std::allocator<double>>)
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
APIARY_EXPOSE
APIARY_MODULE("linalg")
APIARY_INSTANTIATE_AS("det", einsums::GeneralRuntimeTensor<float,                std::allocator<float>>)
APIARY_INSTANTIATE_AS("det", einsums::GeneralRuntimeTensor<double,               std::allocator<double>>)
APIARY_INSTANTIATE_AS("det", einsums::GeneralRuntimeTensor<std::complex<float>,  std::allocator<std::complex<float>>>)
APIARY_INSTANTIATE_AS("det", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
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
        if (std::cmp_not_equal(pivots[i], i + 1)) {
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
        requires !detail::any_tiled_v<AType, BType, CType>;
    }
void einsum(EinsumFormatString spec, typename AType::ValueType c_pf, CType *C, typename AType::ValueType ab_pf, AType const &A,
            BType const &B) {
    using T = typename AType::ValueType;

    // Operand rank ↔ spec consistency check. When the spec is a literal,
    // ``spec.counts`` is populated at consteval time and folds to compile-
    // time constants here; for typed tensors with a static ::Rank the whole
    // condition is a constant comparison and the throw-branch is dead-code-
    // eliminated. For runtime-rank tensors (RuntimeTensor) the check fires
    // against ``tensor.rank()``. Spec strings built at runtime, ``Python``
    // bindings, user input, leave ``counts.known == false`` and skip the
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
        LabeledSection("einsum eager");
        dispatch::string_einsum(parsed, c_pf, C, ab_pf, A, B);
        return;
    }

    LabeledSection("einsum capture");
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
    // them: avoids recomputing link indices on every execute().
    auto indices = ctx.graph()->create_indices(parsed.a_indices, parsed.b_indices, parsed.c_indices, desc.spec.link_indices);
    desc.indices = indices;

    // BLAS-level batching hint. Only populated when the contraction is
    // a pure 2D GEMM pattern (rank-2 inputs/output, one link index),
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
            // (handles graph.rebind(), the slot's ptr points at the
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
    // If the einsum expresses a batched matrix multiply, a 3D tensor
    // where one index appears in A, B, AND C (the batch) and the other
    // three indices form a standard 2D GEMM pattern (target-A, link,
    // target-B), we can collapse N per-batch 2D gemms into a single
    // `blas::gemm_batch` call at execute time. This is the same layout
    // `cublasDgemmStridedBatched` expects on GPU, so the descriptor
    // carries enough info to dispatch there too once the GPU backend is
    // wired up.
    //
    // Both conventions are supported so users don't have to transpose
    // their data to match some arbitrary choice:
    //   - Row-major tensors with batch(es) at the FIRST axes
    //     (e.g. "bij;bjk->bik" shape (B, M, K); or "abij;abjk->abik"
    //     shape (A, B, M, K)), the ML/CUDA convention
    //   - Column-major tensors with batch(es) at the LAST axes
    //     (e.g. "ijb;jkb->ikb" shape (M, K, B); or "ijab;jkab->ikab"
    //     shape (M, K, A, B)), Einsums's default layout
    //
    // Multiple batch indices (rank 4+) are flattened: N batch dims with
    // sizes (d1, d2, ..., dN) become a single effective batch of size
    // prod(di) with uniform stride equal to the product of the per-slice
    // 2D dims. This works as long as all batch indices appear in the
    // outermost contiguous region in memory, in the same relative order
    // across A, B, C, typical of how tensors carry "free" axes like
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
            // operands: otherwise flattening the batch doesn't produce
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

                // The descriptor below requires C's two non-batch slice axes in
                // canonical (M, N) order -- M (shared with A) first, N (shared with
                // B) second. Both modes assume this: col_mode maps it to BLAS m/n
                // directly; row_mode emits the transposed product (so it swaps m/n
                // and trans_a/trans_b) to honor row-major storage, but still on a
                // canonical (M, N) output. A transposed output -- e.g.
                // "kji <- jli ; lki", whose slice is (N, M) -- would mis-map m/n
                // against the operands and gemm_batch would silently miscompute
                // (often to zero), so detect it and fall through to the generic
                // einsum (string_einsum) below.
                std::vector<std::string> const c_rest =
                    row_mode ? std::vector<std::string>{parsed.c_indices[Rank - 2], parsed.c_indices[Rank - 1]}
                             : std::vector<std::string>{parsed.c_indices[0], parsed.c_indices[1]};
                std::string const m_index      = (a_rest[0] == link) ? a_rest[1] : a_rest[0];
                std::string const n_index      = (b_rest[0] == link) ? b_rest[1] : b_rest[0];
                bool const        canonical_mn = (c_rest[0] == m_index && c_rest[1] == n_index);
                if (canonical_mn) {

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

                    d.alpha          = as<std::complex<double>>(params->ab_pf);
                    d.beta           = as<std::complex<double>>(params->c_pf);
                    d.batch_count    = static_cast<int>(flat_batch);
                    d.strided        = true;
                    d.batch_stride_a = static_cast<std::int64_t>(a_slice_dim(0)) * static_cast<std::int64_t>(a_slice_dim(1));
                    d.batch_stride_b = static_cast<std::int64_t>(b_slice_dim(0)) * static_cast<std::int64_t>(b_slice_dim(1));
                    d.batch_stride_c = static_cast<std::int64_t>(c_slice_dim(0)) * static_cast<std::int64_t>(c_slice_dim(1));

                    bool const swap_ab  = row_mode;
                    auto       executor = [d, swap_ab, a_slot, b_slot, c_slot]() {
                        LabeledSection("einsum batched execute");
                        ProfileAnnotate("m", static_cast<int64_t>(d.m));
                        ProfileAnnotate("n", static_cast<int64_t>(d.n));
                        ProfileAnnotate("k", static_cast<int64_t>(d.k));
                        ProfileAnnotate("batch", static_cast<int64_t>(d.batch_count));
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
                            using R = typename T::value_type;
                            T alpha{static_cast<R>(d.alpha.real()), static_cast<R>(d.alpha.imag())};
                            T beta{static_cast<R>(d.beta.real()), static_cast<R>(d.beta.imag())};
                            blas::gemm_batch<T>(d.trans_a, d.trans_b, d.m, d.n, d.k, alpha, blas_a, d.lda, blas_b, d.ldb, beta,
                                                c_arr.data(), d.ldc, d.batch_count);
                        } else {
                            blas::gemm_batch<T>(d.trans_a, d.trans_b, d.m, d.n, d.k, static_cast<T>(d.alpha.real()), blas_a, d.lda, blas_b,
                                                d.ldb, static_cast<T>(d.beta.real()), c_arr.data(), d.ldc, d.batch_count);
                        }
                    };

                    auto label = fmt::format("gemm_batch_strided x{} ({}-major, batch={}, M={}, K={}, N={})", d.batch_count,
                                             col_mode ? "col" : "row", fmt::join(batch_names, ","), d.m, d.k, d.n);
                    ctx.record(OpKind::BatchedGemm, std::move(label), {a_id, b_id}, {c_id}, std::move(executor), std::move(d));
                    return;
                } // canonical_mn, otherwise fall through to the generic einsum
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
        LabeledSection("einsum execute");
        ProfileAnnotate("a_size", static_cast<int64_t>(static_cast<AType const *>(a_slot->ptr)->size()));
        ProfileAnnotate("b_size", static_cast<int64_t>(static_cast<BType const *>(b_slot->ptr)->size()));
        ProfileAnnotate("c_size", static_cast<int64_t>(static_cast<CType *>(c_slot->ptr)->size()));
        ParsedEinsumSpec parsed_live{indices->c_indices, indices->a_indices, indices->b_indices, /*raw*/ std::string{}};
        dispatch::string_einsum(parsed_live, as<T>(params->c_pf), static_cast<CType *>(c_slot->ptr), as<T>(params->ab_pf),
                                *static_cast<AType const *>(a_slot->ptr), *static_cast<BType const *>(b_slot->ptr));
    };

    ctx.record(OpKind::Einsum, std::move(label), {a_id, b_id}, {c_id}, std::move(executor), std::move(desc));
}

/**
 * @brief Tiled einsum (Tier B1): einsum over TiledRuntimeTensor operands.
 *
 * Selected when any operand is tiled. Walks the tile grid and composes dense
 * per-tile contractions (see detail::tiled_runtime_einsum). All three operands
 * must be tiled; the contraction is recorded as a single opaque ``Custom`` node
 * so the einsum-rewriting passes don't try to synthesize dense intermediates
 * for it.
 */
template <TiledTensorConcept AType, TiledTensorConcept BType, TiledTensorConcept CType>
    requires requires {
        requires std::is_same_v<typename AType::ValueType, typename BType::ValueType>;
        requires std::is_same_v<typename AType::ValueType, typename CType::ValueType>;
    }
void einsum(EinsumFormatString spec, typename AType::ValueType c_pf, CType *C, typename AType::ValueType ab_pf, AType const &A,
            BType const &B) {
    using T = typename AType::ValueType;
    static_assert(IsTiledTensorV<std::remove_cvref_t<AType>> && IsTiledTensorV<std::remove_cvref_t<BType>> &&
                      IsTiledTensorV<std::remove_cvref_t<CType>>,
                  "cg::einsum with a tiled operand currently requires all of A, B, C to be TiledRuntimeTensor "
                  "(mixed tiled/dense is not supported yet)");

    auto parse_result = parse_einsum_spec(static_cast<std::string_view>(spec));
    if (!parse_result) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "{}", parse_result.error().message);
    }
    auto &parsed = parse_result.value();

    auto &ctx = CaptureContext::current();
    if (!ctx.is_capturing()) {
        LabeledSection("einsum eager");
        detail::tiled_runtime_einsum<T>(parsed, c_pf, C, ab_pf, A, B);
        return;
    }

    LabeledSection("einsum capture");
    auto [a_id, a_slot] = ctx.get_slot(A);
    auto [b_id, b_slot] = ctx.get_slot(B);
    auto [c_id, c_slot] = ctx.get_slot(*C);

    auto params  = ctx.graph()->create_params(c_pf, ab_pf);
    auto indices = ctx.graph()->create_indices(parsed.a_indices, parsed.b_indices, parsed.c_indices, std::vector<std::string>{});

    auto label = fmt::format("tiled einsum: C[{}] = A[{}] * B[{}]", fmt::join(parsed.c_indices, ","), fmt::join(parsed.a_indices, ","),
                             fmt::join(parsed.b_indices, ","));

    auto executor = [indices, params, a_slot, b_slot, c_slot]() {
        LabeledSection("einsum execute");
        ParsedEinsumSpec live{.c_indices   = indices->c_indices,
                              .a_indices   = indices->a_indices,
                              .b_indices   = indices->b_indices,
                              /*raw*/ .raw = std::string{}};
        detail::tiled_runtime_einsum<T>(live, as<T>(params->c_pf), static_cast<CType *>(c_slot->ptr), as<T>(params->ab_pf),
                                        *static_cast<AType const *>(a_slot->ptr), *static_cast<BType const *>(b_slot->ptr));
    };

    // Opaque Custom node: participates in dependency ordering + lifecycle but is
    // invisible to the einsum-rewriting passes (chain parenthesization /
    // contraction planning), which would otherwise synthesize *dense*
    // intermediates. Tier B1 = no cross-tile graph optimization.
    ctx.record(OpKind::Custom, std::move(label), {a_id, b_id}, {c_id}, std::move(executor));
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

/// Default-prefactor tiled einsum (delegates to the tiled prefactor overload).
template <TiledTensorConcept AType, TiledTensorConcept BType, TiledTensorConcept CType>
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
/// default to 0 and 1, giving ``C = A op B``, but can be set to
/// accumulate (``c_pf=1``) or scale.
///
/// Complex dtypes (``RuntimeTensor<complex<T>>``) accept only the
/// 4-argument form (no explicit prefactors); Graph::create_params still
/// stores prefactors as doubles internally, which would narrow the
/// imaginary part out of a complex ``c_pf``/``ab_pf``.
template <TensorConcept AType, TensorConcept BType, TensorConcept CType>
    requires(std::is_same_v<typename AType::ValueType, typename BType::ValueType> &&
             std::is_same_v<typename AType::ValueType, typename CType::ValueType>)
// clang-format off
APIARY_EXPOSE
// All 8 combinations of (C, A, B) x (owning, view), per dtype. Same-dtype
// across operands is enforced by the requires clause above. Plus the all-tiled
// case (TiledRuntimeTensor), TensorConcept (not BasicTensorConcept) so a tiled
// operand is admitted; the body's einsum() dispatches to the tiled tile-walk.
//
// float: OOO/OOV/OVO/OVV/VOO/VOV/VVO/VVV in (C, A, B) order
APIARY_INSTANTIATE_AS("einsum", einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::GeneralRuntimeTensor<float, std::allocator<float>>)
APIARY_INSTANTIATE_AS("einsum", einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::RuntimeTensorView<float>)
APIARY_INSTANTIATE_AS("einsum", einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::RuntimeTensorView<float>,                          einsums::GeneralRuntimeTensor<float, std::allocator<float>>)
APIARY_INSTANTIATE_AS("einsum", einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::RuntimeTensorView<float>,                          einsums::RuntimeTensorView<float>)
APIARY_INSTANTIATE_AS("einsum", einsums::RuntimeTensorView<float>,                          einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::GeneralRuntimeTensor<float, std::allocator<float>>)
APIARY_INSTANTIATE_AS("einsum", einsums::RuntimeTensorView<float>,                          einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::RuntimeTensorView<float>)
APIARY_INSTANTIATE_AS("einsum", einsums::RuntimeTensorView<float>,                          einsums::RuntimeTensorView<float>,                          einsums::GeneralRuntimeTensor<float, std::allocator<float>>)
APIARY_INSTANTIATE_AS("einsum", einsums::RuntimeTensorView<float>,                          einsums::RuntimeTensorView<float>,                          einsums::RuntimeTensorView<float>)
// double
APIARY_INSTANTIATE_AS("einsum", einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::GeneralRuntimeTensor<double, std::allocator<double>>)
APIARY_INSTANTIATE_AS("einsum", einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::RuntimeTensorView<double>)
APIARY_INSTANTIATE_AS("einsum", einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::RuntimeTensorView<double>,                          einsums::GeneralRuntimeTensor<double, std::allocator<double>>)
APIARY_INSTANTIATE_AS("einsum", einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::RuntimeTensorView<double>,                          einsums::RuntimeTensorView<double>)
APIARY_INSTANTIATE_AS("einsum", einsums::RuntimeTensorView<double>,                          einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::GeneralRuntimeTensor<double, std::allocator<double>>)
APIARY_INSTANTIATE_AS("einsum", einsums::RuntimeTensorView<double>,                          einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::RuntimeTensorView<double>)
APIARY_INSTANTIATE_AS("einsum", einsums::RuntimeTensorView<double>,                          einsums::RuntimeTensorView<double>,                          einsums::GeneralRuntimeTensor<double, std::allocator<double>>)
APIARY_INSTANTIATE_AS("einsum", einsums::RuntimeTensorView<double>,                          einsums::RuntimeTensorView<double>,                          einsums::RuntimeTensorView<double>)
// complex<float>
APIARY_INSTANTIATE_AS("einsum", einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>)
APIARY_INSTANTIATE_AS("einsum", einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::RuntimeTensorView<std::complex<float>>)
APIARY_INSTANTIATE_AS("einsum", einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::RuntimeTensorView<std::complex<float>>,                                            einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>)
APIARY_INSTANTIATE_AS("einsum", einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::RuntimeTensorView<std::complex<float>>,                                            einsums::RuntimeTensorView<std::complex<float>>)
APIARY_INSTANTIATE_AS("einsum", einsums::RuntimeTensorView<std::complex<float>>,                                            einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>)
APIARY_INSTANTIATE_AS("einsum", einsums::RuntimeTensorView<std::complex<float>>,                                            einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::RuntimeTensorView<std::complex<float>>)
APIARY_INSTANTIATE_AS("einsum", einsums::RuntimeTensorView<std::complex<float>>,                                            einsums::RuntimeTensorView<std::complex<float>>,                                            einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>)
APIARY_INSTANTIATE_AS("einsum", einsums::RuntimeTensorView<std::complex<float>>,                                            einsums::RuntimeTensorView<std::complex<float>>,                                            einsums::RuntimeTensorView<std::complex<float>>)
// complex<double>
APIARY_INSTANTIATE_AS("einsum", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
APIARY_INSTANTIATE_AS("einsum", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::RuntimeTensorView<std::complex<double>>)
APIARY_INSTANTIATE_AS("einsum", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::RuntimeTensorView<std::complex<double>>,                                          einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
APIARY_INSTANTIATE_AS("einsum", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::RuntimeTensorView<std::complex<double>>,                                          einsums::RuntimeTensorView<std::complex<double>>)
APIARY_INSTANTIATE_AS("einsum", einsums::RuntimeTensorView<std::complex<double>>,                                          einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
APIARY_INSTANTIATE_AS("einsum", einsums::RuntimeTensorView<std::complex<double>>,                                          einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::RuntimeTensorView<std::complex<double>>)
APIARY_INSTANTIATE_AS("einsum", einsums::RuntimeTensorView<std::complex<double>>,                                          einsums::RuntimeTensorView<std::complex<double>>,                                          einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
APIARY_INSTANTIATE_AS("einsum", einsums::RuntimeTensorView<std::complex<double>>,                                          einsums::RuntimeTensorView<std::complex<double>>,                                          einsums::RuntimeTensorView<std::complex<double>>)
// all-tiled (TiledRuntimeTensor), per dtype
APIARY_INSTANTIATE_AS("einsum", einsums::TiledRuntimeTensor<float>, einsums::TiledRuntimeTensor<float>, einsums::TiledRuntimeTensor<float>)
APIARY_INSTANTIATE_AS("einsum", einsums::TiledRuntimeTensor<double>, einsums::TiledRuntimeTensor<double>, einsums::TiledRuntimeTensor<double>)
APIARY_INSTANTIATE_AS("einsum", einsums::TiledRuntimeTensor<std::complex<float>>, einsums::TiledRuntimeTensor<std::complex<float>>, einsums::TiledRuntimeTensor<std::complex<float>>)
APIARY_INSTANTIATE_AS("einsum", einsums::TiledRuntimeTensor<std::complex<double>>, einsums::TiledRuntimeTensor<std::complex<double>>, einsums::TiledRuntimeTensor<std::complex<double>>)
    // clang-format on
    void einsum_python(std::string const &spec, CType *C, AType const &A, BType const &B,
                       typename CType::ValueType c_pf  = typename CType::ValueType{0},
                       typename AType::ValueType ab_pf = typename AType::ValueType{1}) {
    einsum(EinsumFormatString(std::string_view{spec}), c_pf, C, ab_pf, A, B);
}

// ─────────────────────────────────────────────────────────────────────────────
// parallel_for: graph-capturable data-parallel loop via TaskPool
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
        LabeledSection("parallel_for eager");
        task_pool::TaskPool::get_singleton().parallel_for(name, begin, end, std::forward<F>(body));
        return;
    }

    LabeledSection("parallel_for capture");
    // Collect input tensor IDs
    std::vector<TensorId> input_ids;
    std::apply([&](auto *...ptrs) { (input_ids.push_back(ctx.get_or_register(*ptrs)), ...); }, reads);

    // Collect output tensor IDs
    std::vector<TensorId> output_ids;
    std::apply([&](auto *...ptrs) { (output_ids.push_back(ctx.get_or_register(*ptrs)), ...); }, writes);

    auto executor = [name, begin, end, body = std::forward<F>(body)]() mutable {
        LabeledSection("parallel_for execute");
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
        LabeledSection("parallel_for eager");
        task_pool::TaskPool::get_singleton().parallel_for(name, begin, end, std::forward<F>(body));
        return;
    }

    LabeledSection("parallel_for capture");
    // All listed tensors are both inputs and outputs
    std::vector<TensorId> tensor_ids;
    (tensor_ids.push_back(ctx.get_or_register(*tensors)), ...);

    auto executor = [name, begin, end, body = std::forward<F>(body)]() mutable {
        LabeledSection("parallel_for execute");
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
        LabeledSection("parallel_reduce eager");
        *result = task_pool::TaskPool::get_singleton().parallel_reduce<Acc>(name, begin, end, std::forward<InitFactory>(init),
                                                                            std::forward<Body>(body), std::forward<Combiner>(combine));
        return;
    }

    LabeledSection("parallel_reduce capture");
    // Input tensors
    std::vector<TensorId> tensor_ids;
    (tensor_ids.push_back(ctx.get_or_register(*tensors)), ...);

    auto executor = [name, begin, end, result, init = std::forward<InitFactory>(init), body = std::forward<Body>(body),
                     combine = std::forward<Combiner>(combine)]() mutable {
        LabeledSection("parallel_reduce execute");
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
    // This overload takes raw pointers, prefer the typed overload below.
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
 * @brief No-dependency custom node, Python-friendly overload.
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
 * @brief No-dependency custom node, Python-friendly overload.
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
APIARY_EXPOSE APIARY_MODULE("graph") inline void custom(std::string label, std::function<void()> executor) {
    auto &ctx = CaptureContext::current();
    if (!ctx.is_capturing()) {
        executor();
        return;
    }
    ctx.record(OpKind::Custom, std::move(label), {}, {}, std::move(executor));
}

/**
 * @brief Single-tensor read-modify-write custom node, Python-friendly.
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
APIARY_EXPOSE
APIARY_MODULE("graph")
APIARY_INSTANTIATE_AS("custom", einsums::GeneralRuntimeTensor<float, std::allocator<float>>)
APIARY_INSTANTIATE_AS("custom", einsums::GeneralRuntimeTensor<double, std::allocator<double>>)
APIARY_INSTANTIATE_AS("custom", einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>)
APIARY_INSTANTIATE_AS("custom", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
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
