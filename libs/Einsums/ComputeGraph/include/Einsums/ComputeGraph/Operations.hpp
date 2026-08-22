//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/BLAS.hpp>
#include <Einsums/BLAS/ThreadControl.hpp>
#include <Einsums/ComputeGraph/CaptureContext.hpp>
#include <Einsums/ComputeGraph/Detail/BatchedGemm.hpp>
#include <Einsums/ComputeGraph/Detail/GroupedBatchedGemm.hpp>
#include <Einsums/ComputeGraph/Detail/TiledRuntimeEinsum.hpp>
#include <Einsums/ComputeGraph/Detail/TiledRuntimeElementwise.hpp>
#include <Einsums/ComputeGraph/Diis.hpp>
#include <Einsums/ComputeGraph/EinsumSpec.hpp>
#include <Einsums/ComputeGraph/LuPivots.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/StringDispatch.hpp>
#include <Einsums/ComputeGraph/TensorRank.hpp>
#include <Einsums/Concepts/TensorConcepts.hpp>
#include <Einsums/Config/Namespace.hpp>
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

EINSUMS_NAMESPACE_BEGIN(compute_graph)

namespace detail {
/// Guard for the returning-form ops (dot/svd/syev/qr/det/...), which build and
/// return a fresh result tensor and therefore cannot run inside graph capture.
/// Throws std::logic_error carrying @p message when a capture is active; a no-op
/// otherwise. Centralizes the identical `if (is_capturing()) throw ...` guard
/// that each returning-form overload would otherwise open-code.
inline void reject_if_capturing(char const *message) {
    if (CaptureContext::current().is_capturing()) {
        EINSUMS_THROW_EXCEPTION(std::logic_error, "{}", message);
    }
}
} // namespace detail

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
        auto params         = std::make_shared<TiledElementwiseParams>();
        params->alpha       = PrefactorScalar{factor};
        auto executor       = [params, a_slot]() {
            LabeledSection("scale execute");
            detail::tiled_scale<T>(as<T>(params->alpha), static_cast<AType *>(a_slot->ptr));
        };
        // Carries a descriptor so TiledExpansion can lower it per tile; the
        // executor reads the scalar back out of the same params the descriptor
        // exposes, so a pass that rewrites it is actually obeyed.
        TiledElementwiseDescriptor edesc;
        edesc.op     = TiledElementwiseOp::Scale;
        edesc.params = params;
        ctx.record(OpKind::Custom, std::move(label), {a_id}, {a_id}, std::move(executor), std::move(edesc));
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
// conj / real / imag / abs: complex-conjugation primitives
// ─────────────────────────────────────────────────────────────────────────────

/// Graph-aware in-place complex conjugate: ``A := conj(A)``. A no-op for real
/// dtypes, matching ``numpy.conj`` on a real array. Captured as an opaque Custom
/// node, since the einsum-rewriting passes do not reason about conjugation.
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
APIARY_INSTANTIATE_AS("conj", einsums::TiledRuntimeTensor<float>)
APIARY_INSTANTIATE_AS("conj", einsums::TiledRuntimeTensor<double>)
APIARY_INSTANTIATE_AS("conj", einsums::TiledRuntimeTensor<std::complex<float>>)
APIARY_INSTANTIATE_AS("conj", einsums::TiledRuntimeTensor<std::complex<double>>)
    // clang-format on
    void conj(AType *A) {
    if constexpr (IsTiledTensorV<std::remove_cvref_t<AType>>) {
        using T   = typename AType::ValueType;
        auto &ctx = CaptureContext::current();
        if (!ctx.is_capturing()) {
            LabeledSection("conj eager");
            detail::tiled_conj<T>(A);
            return;
        }
        LabeledSection("conj capture");
        auto [a_id, a_slot] = ctx.get_slot(*A);
        auto label          = fmt::format("tiled conj({})", A->name());
        auto executor       = [a_slot]() {
            LabeledSection("conj execute");
            detail::tiled_conj<T>(static_cast<AType *>(a_slot->ptr));
        };
        ctx.record(OpKind::Custom, std::move(label), {a_id}, {a_id}, std::move(executor));
    } else {
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
}

/// Graph-aware real part: ``out := Re(A)``. Complex ``A`` produces real ``out``.
/// For real ``A`` it is a copy, since Re(x) == x, matching numpy ``.real``. Dense
/// or tiled.
template <typename ResultType, typename AType>
    requires requires {
        requires(CoreBasicTensorConcept<ResultType> || IsTiledTensorV<std::remove_cvref_t<ResultType>>);
        requires(CoreBasicTensorConcept<AType> || IsTiledTensorV<std::remove_cvref_t<AType>>);
    }
// clang-format off
APIARY_EXPOSE
APIARY_MODULE("linalg")
APIARY_INSTANTIATE_AS("real", einsums::GeneralRuntimeTensor<float, std::allocator<float>>,   einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>)
APIARY_INSTANTIATE_AS("real", einsums::GeneralRuntimeTensor<float, std::allocator<float>>,   einsums::RuntimeTensorView<std::complex<float>>)
APIARY_INSTANTIATE_AS("real", einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
APIARY_INSTANTIATE_AS("real", einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::RuntimeTensorView<std::complex<double>>)
APIARY_INSTANTIATE_AS("real", einsums::GeneralRuntimeTensor<float, std::allocator<float>>,   einsums::GeneralRuntimeTensor<float, std::allocator<float>>)
APIARY_INSTANTIATE_AS("real", einsums::GeneralRuntimeTensor<float, std::allocator<float>>,   einsums::RuntimeTensorView<float>)
APIARY_INSTANTIATE_AS("real", einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::GeneralRuntimeTensor<double, std::allocator<double>>)
APIARY_INSTANTIATE_AS("real", einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::RuntimeTensorView<double>)
APIARY_INSTANTIATE_AS("real", einsums::TiledRuntimeTensor<float>,  einsums::TiledRuntimeTensor<std::complex<float>>)
APIARY_INSTANTIATE_AS("real", einsums::TiledRuntimeTensor<double>, einsums::TiledRuntimeTensor<std::complex<double>>)
    // clang-format on
    void real(ResultType *out, AType const &A) {
    auto &ctx = CaptureContext::current();
    if constexpr (IsTiledTensorV<std::remove_cvref_t<AType>>) {
        if (!ctx.is_capturing()) {
            LabeledSection("real eager");
            detail::tiled_real(A, out);
            return;
        }
        LabeledSection("real capture");
        auto [a_id, a_slot] = ctx.get_slot(A);
        auto [r_id, r_slot] = ctx.get_slot(*out);
        auto executor       = [a_slot, r_slot]() {
            detail::tiled_real(*static_cast<AType const *>(a_slot->ptr), static_cast<ResultType *>(r_slot->ptr));
        };
        ctx.record(OpKind::Custom, "real", {a_id}, {r_id}, std::move(executor));
    } else {
        auto compute = [](AType const &a, ResultType *o) {
            if constexpr (IsComplexV<typename AType::ValueType>) {
                einsums::detail::impl_real(a.impl(), o->impl());
            } else {
                einsums::detail::impl_copy(a.impl(), o->impl()); // Re(x) == x for real x
            }
        };
        if (!ctx.is_capturing()) {
            LabeledSection("real eager");
            compute(A, out);
            return;
        }
        LabeledSection("real capture");
        auto [a_id, a_slot] = ctx.get_slot(A);
        auto [r_id, r_slot] = ctx.get_slot(*out);
        auto executor       = [a_slot, r_slot, compute]() {
            LabeledSection("real execute");
            compute(*static_cast<AType const *>(a_slot->ptr), static_cast<ResultType *>(r_slot->ptr));
        };
        ctx.record(OpKind::Custom, "real", {a_id}, {r_id}, std::move(executor));
    }
}

/// Graph-aware imaginary part: ``out := Im(A)``. Complex ``A`` produces real
/// ``out``. For real ``A`` it is zeros, since Im(x) == 0, matching numpy
/// ``.imag``. Dense or tiled.
template <typename ResultType, typename AType>
    requires requires {
        requires(CoreBasicTensorConcept<ResultType> || IsTiledTensorV<std::remove_cvref_t<ResultType>>);
        requires(CoreBasicTensorConcept<AType> || IsTiledTensorV<std::remove_cvref_t<AType>>);
    }
// clang-format off
APIARY_EXPOSE
APIARY_MODULE("linalg")
APIARY_INSTANTIATE_AS("imag", einsums::GeneralRuntimeTensor<float, std::allocator<float>>,   einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>)
APIARY_INSTANTIATE_AS("imag", einsums::GeneralRuntimeTensor<float, std::allocator<float>>,   einsums::RuntimeTensorView<std::complex<float>>)
APIARY_INSTANTIATE_AS("imag", einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
APIARY_INSTANTIATE_AS("imag", einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::RuntimeTensorView<std::complex<double>>)
APIARY_INSTANTIATE_AS("imag", einsums::GeneralRuntimeTensor<float, std::allocator<float>>,   einsums::GeneralRuntimeTensor<float, std::allocator<float>>)
APIARY_INSTANTIATE_AS("imag", einsums::GeneralRuntimeTensor<float, std::allocator<float>>,   einsums::RuntimeTensorView<float>)
APIARY_INSTANTIATE_AS("imag", einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::GeneralRuntimeTensor<double, std::allocator<double>>)
APIARY_INSTANTIATE_AS("imag", einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::RuntimeTensorView<double>)
APIARY_INSTANTIATE_AS("imag", einsums::TiledRuntimeTensor<float>,  einsums::TiledRuntimeTensor<std::complex<float>>)
APIARY_INSTANTIATE_AS("imag", einsums::TiledRuntimeTensor<double>, einsums::TiledRuntimeTensor<std::complex<double>>)
    // clang-format on
    void imag(ResultType *out, AType const &A) {
    auto &ctx = CaptureContext::current();
    if constexpr (IsTiledTensorV<std::remove_cvref_t<AType>>) {
        if (!ctx.is_capturing()) {
            LabeledSection("imag eager");
            detail::tiled_imag(A, out);
            return;
        }
        LabeledSection("imag capture");
        auto [a_id, a_slot] = ctx.get_slot(A);
        auto [r_id, r_slot] = ctx.get_slot(*out);
        auto executor       = [a_slot, r_slot]() {
            detail::tiled_imag(*static_cast<AType const *>(a_slot->ptr), static_cast<ResultType *>(r_slot->ptr));
        };
        ctx.record(OpKind::Custom, "imag", {a_id}, {r_id}, std::move(executor));
    } else {
        auto compute = [](AType const &a, ResultType *o) {
            if constexpr (IsComplexV<typename AType::ValueType>) {
                einsums::detail::impl_imag(a.impl(), o->impl());
            } else {
                // Im(x) == 0 for real x: copy then scale by zero (avoids reading
                // uninitialized output the way a bare scal(0) would).
                einsums::detail::impl_copy(a.impl(), o->impl());
                einsums::detail::impl_scal(typename ResultType::ValueType{0}, o->impl());
            }
        };
        if (!ctx.is_capturing()) {
            LabeledSection("imag eager");
            compute(A, out);
            return;
        }
        LabeledSection("imag capture");
        auto [a_id, a_slot] = ctx.get_slot(A);
        auto [r_id, r_slot] = ctx.get_slot(*out);
        auto executor       = [a_slot, r_slot, compute]() {
            LabeledSection("imag execute");
            compute(*static_cast<AType const *>(a_slot->ptr), static_cast<ResultType *>(r_slot->ptr));
        };
        ctx.record(OpKind::Custom, "imag", {a_id}, {r_id}, std::move(executor));
    }
}

/// Graph-aware magnitude: ``out := |A|``. Real or complex ``A`` produces real
/// ``out``. Dense or tiled.
template <typename ResultType, typename AType>
    requires requires {
        requires(CoreBasicTensorConcept<ResultType> || IsTiledTensorV<std::remove_cvref_t<ResultType>>);
        requires(CoreBasicTensorConcept<AType> || IsTiledTensorV<std::remove_cvref_t<AType>>);
    }
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
APIARY_INSTANTIATE_AS("abs", einsums::TiledRuntimeTensor<float>,  einsums::TiledRuntimeTensor<float>)
APIARY_INSTANTIATE_AS("abs", einsums::TiledRuntimeTensor<double>, einsums::TiledRuntimeTensor<double>)
APIARY_INSTANTIATE_AS("abs", einsums::TiledRuntimeTensor<float>,  einsums::TiledRuntimeTensor<std::complex<float>>)
APIARY_INSTANTIATE_AS("abs", einsums::TiledRuntimeTensor<double>, einsums::TiledRuntimeTensor<std::complex<double>>)
    // clang-format on
    void abs(ResultType *out, AType const &A) {
    auto &ctx = CaptureContext::current();
    if constexpr (IsTiledTensorV<std::remove_cvref_t<AType>>) {
        if (!ctx.is_capturing()) {
            LabeledSection("abs eager");
            detail::tiled_abs(A, out);
            return;
        }
        LabeledSection("abs capture");
        auto [a_id, a_slot] = ctx.get_slot(A);
        auto [r_id, r_slot] = ctx.get_slot(*out);
        auto executor       = [a_slot, r_slot]() {
            detail::tiled_abs(*static_cast<AType const *>(a_slot->ptr), static_cast<ResultType *>(r_slot->ptr));
        };
        ctx.record(OpKind::Custom, "abs", {a_id}, {r_id}, std::move(executor));
    } else {
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
template <typename AType, typename CType>
    requires requires {
        requires std::is_same_v<typename AType::ValueType, typename CType::ValueType>;
        requires(BasicTensorConcept<AType> || IsTiledTensorV<std::remove_cvref_t<AType>>);
        requires(BasicTensorConcept<CType> || IsTiledTensorV<std::remove_cvref_t<CType>>);
    }
void permute(PermuteFormatString spec, typename CType::ValueType beta, CType *C, typename AType::ValueType alpha, AType const &A) {
    using T = typename AType::ValueType;

    auto parse_result = parse_permute_spec(static_cast<std::string_view>(spec));
    if (!parse_result) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "{}", parse_result.error().message);
    }
    auto &parsed = parse_result.value();

    if constexpr (IsTiledTensorV<std::remove_cvref_t<AType>> || IsTiledTensorV<std::remove_cvref_t<CType>>) {
        static_assert(IsTiledTensorV<std::remove_cvref_t<AType>> && IsTiledTensorV<std::remove_cvref_t<CType>>,
                      "cg::permute with a tiled operand requires both A and C to be TiledRuntimeTensor");
        auto &tctx = CaptureContext::current();
        if (!tctx.is_capturing()) {
            LabeledSection("permute eager");
            detail::tiled_permute(parsed, beta, C, alpha, A);
            return;
        }
        LabeledSection("permute capture");
        auto [a_id, a_slot] = tctx.get_slot(A);
        auto [c_id, c_slot] = tctx.get_slot(*C);
        auto label    = fmt::format("tiled permute: C[{}] = A[{}]", fmt::join(parsed.c_indices, ","), fmt::join(parsed.a_indices, ","));
        auto executor = [parsed, beta, alpha, a_slot, c_slot]() {
            LabeledSection("permute execute");
            detail::tiled_permute(parsed, static_cast<T>(beta), static_cast<CType *>(c_slot->ptr), static_cast<T>(alpha),
                                  *static_cast<AType const *>(a_slot->ptr));
        };
        // Recorded as Custom with a TiledPermuteDescriptor - NOT the dense
        // PermuteDescriptor, which passes probe without checking the node kind
        // and then reason about a single dense buffer. The tiled descriptor is
        // what lets TiledExpansion lower this node into per-tile dense
        // permutes instead of stranding every tensor it touches out of
        // expansion. RMW convention: beta != 0 reads C.
        TiledPermuteDescriptor tdesc;
        tdesc.c_indices                      = parsed.c_indices;
        tdesc.a_indices                      = parsed.a_indices;
        tdesc.alpha                          = PrefactorScalar{alpha};
        tdesc.beta                           = PrefactorScalar{beta};
        std::vector<TensorId> permute_inputs = (beta != T{0}) ? std::vector<TensorId>{a_id, c_id} : std::vector<TensorId>{a_id};
        tctx.record(OpKind::Custom, std::move(label), std::move(permute_inputs), {c_id}, std::move(executor), std::move(tdesc));
        return;
    } else {
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
            desc.alpha = std::complex<double>{static_cast<double>(alpha.real()), static_cast<double>(alpha.imag())};
            desc.beta  = std::complex<double>{static_cast<double>(beta.real()), static_cast<double>(beta.imag())};
        } else {
            desc.alpha = std::complex<double>{static_cast<double>(alpha), 0.0};
            desc.beta  = std::complex<double>{static_cast<double>(beta), 0.0};
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
}

/// String-based permute with default prefactors (beta=0, alpha=1): C = permute(A).
template <typename AType, typename CType>
    requires requires {
        requires std::is_same_v<typename AType::ValueType, typename CType::ValueType>;
        requires(BasicTensorConcept<AType> || IsTiledTensorV<std::remove_cvref_t<AType>>);
        requires(BasicTensorConcept<CType> || IsTiledTensorV<std::remove_cvref_t<CType>>);
    }
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
template <typename AType, typename CType>
    requires requires {
        requires std::is_same_v<typename AType::ValueType, typename CType::ValueType>;
        requires(BasicTensorConcept<AType> || IsTiledTensorV<std::remove_cvref_t<AType>>);
        requires(BasicTensorConcept<CType> || IsTiledTensorV<std::remove_cvref_t<CType>>);
    }
// clang-format off
APIARY_EXPOSE
APIARY_INSTANTIATE_AS("permute", einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::GeneralRuntimeTensor<float, std::allocator<float>>)
APIARY_INSTANTIATE_AS("permute", einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::GeneralRuntimeTensor<double, std::allocator<double>>)
APIARY_INSTANTIATE_AS("permute", einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>)
APIARY_INSTANTIATE_AS("permute", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
APIARY_INSTANTIATE_AS("permute", einsums::TiledRuntimeTensor<float>, einsums::TiledRuntimeTensor<float>)
APIARY_INSTANTIATE_AS("permute", einsums::TiledRuntimeTensor<double>, einsums::TiledRuntimeTensor<double>)
APIARY_INSTANTIATE_AS("permute", einsums::TiledRuntimeTensor<std::complex<float>>, einsums::TiledRuntimeTensor<std::complex<float>>)
APIARY_INSTANTIATE_AS("permute", einsums::TiledRuntimeTensor<std::complex<double>>, einsums::TiledRuntimeTensor<std::complex<double>>)
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
// gather: index-list extraction along any subset of axes
// ─────────────────────────────────────────────────────────────────────────────

/// Copy an arbitrary index selection out of @p src into @p dst.
///
/// For each axis k, @p indices[k] lists the positions of src to take along that
/// axis:
///
///   dst[i0, i1, ...] = src[indices[0][i0], indices[1][i1], ...]
///
/// This is the outer-product selection numpy spells `A[np.ix_(rows, cols)]`,
/// not the zipped selection `A[rows, cols]`.
///
/// Every axis needs an explicit list; there is deliberately no "whole axis"
/// wildcard. Spelling it as the empty list would be the natural choice and is
/// the wrong one: the callers are domain-restricted methods, an empty domain is
/// a legitimate (if degenerate) input there, and having it silently expand to
/// the entire axis would turn a screened-out domain into a full-rank one with
/// no error. An empty list therefore selects nothing, and a whole axis is
/// `std::iota` / `range(n)` at the call site.
///
/// @ref block_copy covers the contiguous case; this covers the case where the
/// wanted elements are scattered, which is what domain-restricted methods do
/// constantly: a local-correlation pair domain is a sorted list of orbital
/// indices, and every operand is `A[domain, domain]`. Doing that on the host
/// forces the extraction out of the graph, which is what this exists to avoid.
///
/// Both tensors must have the same rank and dtype, and dst's extent on each
/// axis must equal the number of indices selected there. Capture-aware: outside
/// capture it runs immediately; inside, it records a node with src as an input
/// and dst as an output, so a whole per-pair setup can be one graph.
///
/// @p axes optionally reorders the axes on the way out: source axis k lands on
/// destination axis `axes[k]`, so
///
///   dst[..., i_{axes[k]} = ik, ...] = src[..., indices[k][ik], ...]
///
/// Empty (the default) is the identity. This exists because the alternative is
/// a gather followed by a permute, and both are full passes over the result:
/// selecting and reordering together is one pass, and the traversal already
/// walks the destination through per-axis strides, so permuting is only a
/// different set of strides. Callers that need to reinterpret rather than move
/// axes want @ref RuntimeTensor::reshape_view, which does not copy at all.
template <CoreBasicTensorConcept DstType, CoreBasicTensorConcept SrcType>
    requires std::is_same_v<typename DstType::ValueType, typename SrcType::ValueType>
// clang-format off
APIARY_EXPOSE
APIARY_MODULE("linalg")
APIARY_INSTANTIATE_AS("gather", einsums::GeneralRuntimeTensor<float,                std::allocator<float>>,                einsums::GeneralRuntimeTensor<float,                std::allocator<float>>)
APIARY_INSTANTIATE_AS("gather", einsums::GeneralRuntimeTensor<double,               std::allocator<double>>,               einsums::GeneralRuntimeTensor<double,               std::allocator<double>>)
APIARY_INSTANTIATE_AS("gather", einsums::GeneralRuntimeTensor<std::complex<float>,  std::allocator<std::complex<float>>>,  einsums::GeneralRuntimeTensor<std::complex<float>,  std::allocator<std::complex<float>>>)
APIARY_INSTANTIATE_AS("gather", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
APIARY_INSTANTIATE_AS("gather", einsums::RuntimeTensorView<float>,                                                   einsums::GeneralRuntimeTensor<float,                std::allocator<float>>)
APIARY_INSTANTIATE_AS("gather", einsums::RuntimeTensorView<double>,                                                  einsums::GeneralRuntimeTensor<double,               std::allocator<double>>)
APIARY_INSTANTIATE_AS("gather", einsums::RuntimeTensorView<std::complex<float>>,                                     einsums::GeneralRuntimeTensor<std::complex<float>,  std::allocator<std::complex<float>>>)
APIARY_INSTANTIATE_AS("gather", einsums::RuntimeTensorView<std::complex<double>>,                                    einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
APIARY_INSTANTIATE_AS("gather", einsums::GeneralRuntimeTensor<float,                std::allocator<float>>,                einsums::RuntimeTensorView<float>)
APIARY_INSTANTIATE_AS("gather", einsums::GeneralRuntimeTensor<double,               std::allocator<double>>,               einsums::RuntimeTensorView<double>)
APIARY_INSTANTIATE_AS("gather", einsums::GeneralRuntimeTensor<std::complex<float>,  std::allocator<std::complex<float>>>,  einsums::RuntimeTensorView<std::complex<float>>)
APIARY_INSTANTIATE_AS("gather", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::RuntimeTensorView<std::complex<double>>)
APIARY_INSTANTIATE_AS("gather", einsums::RuntimeTensorView<float>,                                                   einsums::RuntimeTensorView<float>)
APIARY_INSTANTIATE_AS("gather", einsums::RuntimeTensorView<double>,                                                  einsums::RuntimeTensorView<double>)
APIARY_INSTANTIATE_AS("gather", einsums::RuntimeTensorView<std::complex<float>>,                                     einsums::RuntimeTensorView<std::complex<float>>)
APIARY_INSTANTIATE_AS("gather", einsums::RuntimeTensorView<std::complex<double>>,                                    einsums::RuntimeTensorView<std::complex<double>>)
    // clang-format on
    // The default is spelled out rather than `{}` because the binding generator
    // copies the token through to py::arg, and pybind cannot deduce a type from
    // an empty braced list.
    void gather(DstType *dst, SrcType const &src, std::vector<std::vector<size_t>> const &indices,
                std::vector<size_t> const &axes = std::vector<size_t>{}) {
    size_t const N = indices.size();
    if (N == 0) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "cg::gather: indices must be non-empty");
    }
    // tensor_rank rather than .rank(): compile-time Tensor<T, N> has no rank()
    // member, and gather is useful on those too.
    size_t const dst_rank = detail::tensor_rank(*dst);
    size_t const src_rank = detail::tensor_rank(src);
    if (dst_rank != N || src_rank != N) {
        EINSUMS_THROW_EXCEPTION(rank_error, "cg::gather: rank mismatch - dst rank={}, src rank={}, indices.size()={}", dst_rank, src_rank,
                                N);
    }

    // axes[k] is the DESTINATION axis that source axis k lands on, so a gather
    // can reorder axes on its way out instead of needing a separate permute
    // pass over the whole result. Empty means the identity.
    std::vector<size_t> dst_axis(N);
    if (axes.empty()) {
        std::iota(dst_axis.begin(), dst_axis.end(), size_t{0});
    } else {
        if (axes.size() != N) {
            EINSUMS_THROW_EXCEPTION(std::invalid_argument, "cg::gather: axes has {} entries but there are {} axes", axes.size(), N);
        }
        std::vector<bool> seen(N, false);
        for (size_t k = 0; k < N; ++k) {
            if (axes[k] >= N || seen[axes[k]]) {
                EINSUMS_THROW_EXCEPTION(std::invalid_argument, "cg::gather: axes must be a permutation of [0, {}), got {}", N, axes);
            }
            seen[axes[k]] = true;
        }
        dst_axis = axes;
    }

    std::vector<size_t> extents(N);
    for (size_t k = 0; k < N; ++k) {
        extents[k] = indices[k].size();
        if (dst->dim(dst_axis[k]) != extents[k]) {
            EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                    "cg::gather: dst axis {} has extent {}, but {} indices were given for source "
                                    "axis {}",
                                    dst_axis[k], dst->dim(dst_axis[k]), extents[k], k);
        }
        for (size_t p : indices[k]) {
            if (p >= src.dim(k)) {
                EINSUMS_THROW_EXCEPTION(std::out_of_range, "cg::gather: index {} on axis {} is out of range for src dim {}", p, k,
                                        src.dim(k));
            }
        }
    }

    auto apply = [indices, extents, dst_axis, N](DstType *d, SrcType const *s) {
        using T = typename DstType::ValueType;
        std::vector<size_t> d_str(N), s_str(N);
        for (size_t k = 0; k < N; ++k) {
            // Indexed by SOURCE axis, so the traversal below is unchanged: it
            // walks the destination linearly through whatever strides it is
            // handed, and a permutation is just a different set of strides. The
            // contiguous-run fast path keys on d_str[0] == 1, so it switches
            // itself off exactly when the permutation breaks that.
            d_str[k] = d->stride(dst_axis[k]);
            s_str[k] = s->stride(k);
        }
        T       *d_data = d->data();
        T const *s_data = s->data();

        // The source is the indexed side; the destination is walked linearly -
        // which is what makes the parallel opt-in sound here: every op call
        // writes a disjoint destination run no matter what the index lists
        // hold. scatter and scatter_add stay serial; see the walker's contract.
        detail::for_each_selection_run(
            indices, extents, s_str, d_str, [&](size_t s_off, size_t d_off, size_t n) { std::copy_n(s_data + s_off, n, d_data + d_off); },
            /*parallel=*/true);
    };

    auto &ctx = CaptureContext::current();
    if (!ctx.is_capturing()) {
        LabeledSection("gather eager");
        apply(dst, &src);
        return;
    }

    LabeledSection("gather capture");
    auto [s_id, s_slot] = ctx.get_slot(src);
    auto [d_id, d_slot] = ctx.get_slot(*dst);

    auto executor = [s_slot, d_slot, apply]() {
        LabeledSection("gather execute");
        apply(static_cast<DstType *>(d_slot->ptr), static_cast<SrcType const *>(s_slot->ptr));
    };
    ctx.record(OpKind::Custom, "gather", {s_id}, {d_id}, std::move(executor));
}

// ─────────────────────────────────────────────────────────────────────────────
// scatter: index-list placement, the inverse of gather
// ─────────────────────────────────────────────────────────────────────────────

/// Write @p src into an arbitrary index selection of @p dst.
///
/// The inverse of @ref gather, with the index lists naming positions in the
/// DESTINATION rather than the source:
///
///   dst[indices[0][i0], indices[1][i1], ...] = src[i0, i1, ...]
///
/// which is numpy's `A[np.ix_(rows, cols)] = B`. Elements of dst outside the
/// selection are left alone, so this is a placement, not an assignment of the
/// whole tensor. That is what a domain-restricted result needs: a pair's block
/// is computed in its own small domain basis and then dropped into its slot in
/// the full-length container.
///
/// As with gather, an empty index list selects nothing rather than acting as a
/// whole-axis wildcard, and a whole axis is an explicit range at the call site.
///
/// **Repeated indices are rejected.** In a gather a repeat is harmless - the
/// same element is read twice - but in a scatter it means two writes to one
/// destination element, and which one survives depends on iteration order. The
/// callers here build index lists from orbital domains, where a repeat is a
/// bug rather than an intent, so it is diagnosed instead of silently resolved.
/// An accumulating scatter would be a different operation, and is not this one.
template <CoreBasicTensorConcept DstType, CoreBasicTensorConcept SrcType>
    requires std::is_same_v<typename DstType::ValueType, typename SrcType::ValueType>
// clang-format off
APIARY_EXPOSE
APIARY_MODULE("linalg")
APIARY_INSTANTIATE_AS("scatter", einsums::GeneralRuntimeTensor<float,                std::allocator<float>>,                einsums::GeneralRuntimeTensor<float,                std::allocator<float>>)
APIARY_INSTANTIATE_AS("scatter", einsums::GeneralRuntimeTensor<double,               std::allocator<double>>,               einsums::GeneralRuntimeTensor<double,               std::allocator<double>>)
APIARY_INSTANTIATE_AS("scatter", einsums::GeneralRuntimeTensor<std::complex<float>,  std::allocator<std::complex<float>>>,  einsums::GeneralRuntimeTensor<std::complex<float>,  std::allocator<std::complex<float>>>)
APIARY_INSTANTIATE_AS("scatter", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
APIARY_INSTANTIATE_AS("scatter", einsums::RuntimeTensorView<float>,                                                   einsums::GeneralRuntimeTensor<float,                std::allocator<float>>)
APIARY_INSTANTIATE_AS("scatter", einsums::RuntimeTensorView<double>,                                                  einsums::GeneralRuntimeTensor<double,               std::allocator<double>>)
APIARY_INSTANTIATE_AS("scatter", einsums::RuntimeTensorView<std::complex<float>>,                                     einsums::GeneralRuntimeTensor<std::complex<float>,  std::allocator<std::complex<float>>>)
APIARY_INSTANTIATE_AS("scatter", einsums::RuntimeTensorView<std::complex<double>>,                                    einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
APIARY_INSTANTIATE_AS("scatter", einsums::GeneralRuntimeTensor<float,                std::allocator<float>>,                einsums::RuntimeTensorView<float>)
APIARY_INSTANTIATE_AS("scatter", einsums::GeneralRuntimeTensor<double,               std::allocator<double>>,               einsums::RuntimeTensorView<double>)
APIARY_INSTANTIATE_AS("scatter", einsums::GeneralRuntimeTensor<std::complex<float>,  std::allocator<std::complex<float>>>,  einsums::RuntimeTensorView<std::complex<float>>)
APIARY_INSTANTIATE_AS("scatter", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::RuntimeTensorView<std::complex<double>>)
APIARY_INSTANTIATE_AS("scatter", einsums::RuntimeTensorView<float>,                                                   einsums::RuntimeTensorView<float>)
APIARY_INSTANTIATE_AS("scatter", einsums::RuntimeTensorView<double>,                                                  einsums::RuntimeTensorView<double>)
APIARY_INSTANTIATE_AS("scatter", einsums::RuntimeTensorView<std::complex<float>>,                                     einsums::RuntimeTensorView<std::complex<float>>)
APIARY_INSTANTIATE_AS("scatter", einsums::RuntimeTensorView<std::complex<double>>,                                    einsums::RuntimeTensorView<std::complex<double>>)
    // clang-format on
    void scatter(DstType *dst, SrcType const &src, std::vector<std::vector<size_t>> const &indices) {
    size_t const N = indices.size();
    if (N == 0) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "cg::scatter: indices must be non-empty");
    }
    size_t const dst_rank = detail::tensor_rank(*dst);
    size_t const src_rank = detail::tensor_rank(src);
    if (dst_rank != N || src_rank != N) {
        EINSUMS_THROW_EXCEPTION(rank_error, "cg::scatter: rank mismatch - dst rank={}, src rank={}, indices.size()={}", dst_rank, src_rank,
                                N);
    }

    std::vector<size_t> extents(N);
    for (size_t k = 0; k < N; ++k) {
        extents[k] = indices[k].size();
        if (src.dim(k) != extents[k]) {
            EINSUMS_THROW_EXCEPTION(std::invalid_argument, "cg::scatter: src axis {} has extent {}, but {} indices were given", k,
                                    src.dim(k), extents[k]);
        }
        for (size_t p : indices[k]) {
            if (p >= dst->dim(k)) {
                EINSUMS_THROW_EXCEPTION(std::out_of_range, "cg::scatter: index {} on axis {} is out of range for dst dim {}", p, k,
                                        dst->dim(k));
            }
        }
        // O(n log n) once at record time, not per element.
        std::vector<size_t> sorted(indices[k]);
        std::sort(sorted.begin(), sorted.end());
        auto dup = std::adjacent_find(sorted.begin(), sorted.end());
        if (dup != sorted.end()) {
            EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                    "cg::scatter: index {} is repeated on axis {}; two writes would target the same element and the "
                                    "result would depend on iteration order",
                                    *dup, k);
        }
    }

    auto apply = [indices, extents, N](DstType *d, SrcType const *s) {
        using T = typename DstType::ValueType;
        std::vector<size_t> d_str(N), s_str(N);
        for (size_t k = 0; k < N; ++k) {
            d_str[k] = d->stride(k);
            s_str[k] = s->stride(k);
        }
        T       *d_data = d->data();
        T const *s_data = s->data();

        // The destination is the indexed side here, which is what makes this
        // the inverse of gather; the source is walked linearly.
        detail::for_each_selection_run(indices, extents, d_str, s_str,
                                       [&](size_t d_off, size_t s_off, size_t n) { std::copy_n(s_data + s_off, n, d_data + d_off); });
    };

    auto &ctx = CaptureContext::current();
    if (!ctx.is_capturing()) {
        LabeledSection("scatter eager");
        apply(dst, &src);
        return;
    }

    LabeledSection("scatter capture");
    auto [s_id, s_slot] = ctx.get_slot(src);
    auto [d_id, d_slot] = ctx.get_slot(*dst);

    // dst is BOTH an input and an output: a scatter leaves everything outside
    // the selection untouched, so whatever wrote those elements has to be
    // ordered before this node.
    auto executor = [s_slot, d_slot, apply]() {
        LabeledSection("scatter execute");
        apply(static_cast<DstType *>(d_slot->ptr), static_cast<SrcType const *>(s_slot->ptr));
    };
    ctx.record(OpKind::Custom, "scatter", {s_id, d_id}, {d_id}, std::move(executor));
}

// ─────────────────────────────────────────────────────────────────────────────
// sqrt: element-wise square root
// ─────────────────────────────────────────────────────────────────────────────

/// Graph-aware element-wise square root: ``out := sqrt(A)``.
///
/// Distinct from @ref pow, which is a MATRIX power computed by
/// eigendecomposition (that is what its cutoff argument discards) and is a
/// returning form that rejects capture. This is the element-wise partner to
/// @ref abs, and the two together are what a numerical `sqrt(abs(X))` needs
/// without dropping to the host between them.
///
/// Real dtypes only: the square root of a negative real is where a caller
/// wants to decide, not have the library pick a branch. Negative inputs throw.
template <CoreBasicTensorConcept ResultType, CoreBasicTensorConcept AType>
    requires std::is_same_v<typename ResultType::ValueType, typename AType::ValueType>
// clang-format off
APIARY_EXPOSE
APIARY_MODULE("linalg")
APIARY_INSTANTIATE_AS("sqrt", einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::GeneralRuntimeTensor<float, std::allocator<float>>)
APIARY_INSTANTIATE_AS("sqrt", einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::RuntimeTensorView<float>)
APIARY_INSTANTIATE_AS("sqrt", einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::GeneralRuntimeTensor<double, std::allocator<double>>)
APIARY_INSTANTIATE_AS("sqrt", einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::RuntimeTensorView<double>)
    // clang-format on
    void sqrt(ResultType *out, AType const &A) {
    using T = typename ResultType::ValueType;
    static_assert(!IsComplexV<T>, "cg::sqrt is real-only; complex needs a branch choice the caller must make");
    if (detail::tensor_rank(*out) != detail::tensor_rank(A)) {
        EINSUMS_THROW_EXCEPTION(rank_error, "cg::sqrt: rank mismatch - out rank={}, A rank={}", detail::tensor_rank(*out),
                                detail::tensor_rank(A));
    }
    size_t const N = detail::tensor_rank(A);
    for (size_t k = 0; k < N; ++k) {
        if (out->dim(k) != A.dim(k)) {
            EINSUMS_THROW_EXCEPTION(dimension_error, "cg::sqrt: axis {} - out dim {} does not match A dim {}", k, out->dim(k), A.dim(k));
        }
    }

    auto apply = [N](ResultType *o, AType const *a) {
        size_t total = 1;
        for (size_t k = 0; k < N; ++k)
            total *= a->dim(k);
        if (total == 0)
            return;
        std::vector<size_t> idx(N, 0), o_str(N), a_str(N), dims(N);
        for (size_t k = 0; k < N; ++k) {
            o_str[k] = o->stride(k);
            a_str[k] = a->stride(k);
            dims[k]  = a->dim(k);
        }
        T       *o_data = o->data();
        T const *a_data = a->data();
        for (size_t count = 0; count < total; ++count) {
            size_t o_off = 0, a_off = 0;
            for (size_t k = 0; k < N; ++k) {
                o_off += idx[k] * o_str[k];
                a_off += idx[k] * a_str[k];
            }
            T const v = a_data[a_off];
            if (v < T{0}) {
                EINSUMS_THROW_EXCEPTION(std::domain_error, "cg::sqrt: negative input {}; use abs() first if that is intended",
                                        static_cast<double>(v));
            }
            o_data[o_off] = std::sqrt(v);
            for (size_t k = 0; k < N; ++k) {
                if (++idx[k] < dims[k])
                    break;
                idx[k] = 0;
            }
        }
    };

    auto &ctx = CaptureContext::current();
    if (!ctx.is_capturing()) {
        LabeledSection("sqrt eager");
        apply(out, &A);
        return;
    }
    LabeledSection("sqrt capture");
    auto [a_id, a_slot] = ctx.get_slot(A);
    auto [r_id, r_slot] = ctx.get_slot(*out);
    auto executor       = [a_slot, r_slot, apply]() {
        LabeledSection("sqrt execute");
        apply(static_cast<ResultType *>(r_slot->ptr), static_cast<AType const *>(a_slot->ptr));
    };
    ctx.record(OpKind::Custom, "sqrt", {a_id}, {r_id}, std::move(executor));
}

// ─────────────────────────────────────────────────────────────────────────────
// sum_axes: reduction over a subset of axes
// ─────────────────────────────────────────────────────────────────────────────

/// Graph-aware reduction: sum @p A over @p axes, writing the surviving axes to
/// @p out in their original order.
///
/// @ref sum already reduces a whole tensor to a scalar; this is the axis-wise
/// form, numpy's ``A.sum(axis=...)``. ``out`` must have rank
/// ``rank(A) - axes.size()`` with the extents of the axes not being summed.
/// ``out`` is zeroed first, so this assigns rather than accumulates.
template <CoreBasicTensorConcept ResultType, CoreBasicTensorConcept AType>
    requires std::is_same_v<typename ResultType::ValueType, typename AType::ValueType>
// clang-format off
APIARY_EXPOSE
APIARY_MODULE("linalg")
APIARY_INSTANTIATE_AS("sum_axes", einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::GeneralRuntimeTensor<float, std::allocator<float>>)
APIARY_INSTANTIATE_AS("sum_axes", einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::RuntimeTensorView<float>)
APIARY_INSTANTIATE_AS("sum_axes", einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::GeneralRuntimeTensor<double, std::allocator<double>>)
APIARY_INSTANTIATE_AS("sum_axes", einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::RuntimeTensorView<double>)
APIARY_INSTANTIATE_AS("sum_axes", einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>)
APIARY_INSTANTIATE_AS("sum_axes", einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::RuntimeTensorView<std::complex<float>>)
APIARY_INSTANTIATE_AS("sum_axes", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
APIARY_INSTANTIATE_AS("sum_axes", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::RuntimeTensorView<std::complex<double>>)
    // clang-format on
    void sum_axes(ResultType *out, AType const &A, std::vector<size_t> axes) {
    using T        = typename ResultType::ValueType;
    size_t const N = detail::tensor_rank(A);
    std::sort(axes.begin(), axes.end());
    if (std::adjacent_find(axes.begin(), axes.end()) != axes.end()) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "cg::sum_axes: an axis is listed twice");
    }
    for (size_t ax : axes) {
        if (ax >= N) {
            EINSUMS_THROW_EXCEPTION(std::out_of_range, "cg::sum_axes: axis {} is out of range for a rank-{} tensor", ax, N);
        }
    }
    std::vector<bool> reduced(N, false);
    for (size_t const ax : axes)
        reduced[ax] = true;
    std::vector<size_t> kept;
    for (size_t k = 0; k < N; ++k)
        if (!reduced[k])
            kept.push_back(k);
    if (detail::tensor_rank(*out) != kept.size()) {
        EINSUMS_THROW_EXCEPTION(rank_error, "cg::sum_axes: out rank {} should be {} after summing {} of {} axes", detail::tensor_rank(*out),
                                kept.size(), axes.size(), N);
    }
    for (size_t k = 0; k < kept.size(); ++k) {
        if (out->dim(k) != A.dim(kept[k])) {
            EINSUMS_THROW_EXCEPTION(dimension_error, "cg::sum_axes: out axis {} has extent {}, expected {}", k, out->dim(k),
                                    A.dim(kept[k]));
        }
    }

    auto apply = [N, kept](ResultType *o, AType const *a) {
        size_t              total = 1;
        std::vector<size_t> dims(N), a_str(N);
        for (size_t k = 0; k < N; ++k) {
            dims[k]  = a->dim(k);
            a_str[k] = a->stride(k);
            total *= dims[k];
        }
        size_t              out_total = 1;
        std::vector<size_t> o_str(kept.size());
        for (size_t k = 0; k < kept.size(); ++k) {
            o_str[k] = o->stride(k);
            out_total *= o->dim(k);
        }
        T *o_data = o->data();
        // Assign, not accumulate: zero first so a replay does not add to the
        // previous execution's result.
        for (size_t k = 0; k < out_total; ++k) {
            size_t off = 0, rem = k;
            for (size_t d = 0; d < kept.size(); ++d) {
                off += (rem % o->dim(d)) * o_str[d];
                rem /= o->dim(d);
            }
            o_data[off] = T{0};
        }
        if (total == 0)
            return;
        T const            *a_data = a->data();
        std::vector<size_t> idx(N, 0);
        for (size_t count = 0; count < total; ++count) {
            size_t a_off = 0, o_off = 0;
            for (size_t k = 0; k < N; ++k)
                a_off += idx[k] * a_str[k];
            for (size_t k = 0; k < kept.size(); ++k)
                o_off += idx[kept[k]] * o_str[k];
            o_data[o_off] += a_data[a_off];
            for (size_t k = 0; k < N; ++k) {
                if (++idx[k] < dims[k])
                    break;
                idx[k] = 0;
            }
        }
    };

    auto &ctx = CaptureContext::current();
    if (!ctx.is_capturing()) {
        LabeledSection("sum_axes eager");
        apply(out, &A);
        return;
    }
    LabeledSection("sum_axes capture");
    auto [a_id, a_slot] = ctx.get_slot(A);
    auto [r_id, r_slot] = ctx.get_slot(*out);
    auto executor       = [a_slot, r_slot, apply]() {
        LabeledSection("sum_axes execute");
        apply(static_cast<ResultType *>(r_slot->ptr), static_cast<AType const *>(a_slot->ptr));
    };
    ctx.record(OpKind::Custom, "sum_axes", {a_id}, {r_id}, std::move(executor));
}

// ─────────────────────────────────────────────────────────────────────────────
// reshape: same elements, different shape
// ─────────────────────────────────────────────────────────────────────────────

/// Graph-aware reshape: copy @p A into @p out, which has the same number of
/// elements in a different shape.
///
/// @p row_major picks which linear order the elements are walked in, and it is
/// NOT a formality. Einsums tensors are column major, so the natural order here
/// is column major (``row_major = false``); numpy's ``reshape`` defaults to row
/// major, and code ported from it means the other one. Getting it wrong
/// silently transposes blocks rather than failing, so the argument is required
/// and has no default.
///
/// This copies. It is not a view: a reshape that could alias would have to
/// reason about strides, and the callers here are assembling operands for BLAS
/// which wants a fresh contiguous buffer anyway.
template <CoreBasicTensorConcept ResultType, CoreBasicTensorConcept AType>
    requires std::is_same_v<typename ResultType::ValueType, typename AType::ValueType>
// clang-format off
APIARY_EXPOSE
APIARY_MODULE("linalg")
APIARY_INSTANTIATE_AS("reshape", einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::GeneralRuntimeTensor<float, std::allocator<float>>)
APIARY_INSTANTIATE_AS("reshape", einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::RuntimeTensorView<float>)
APIARY_INSTANTIATE_AS("reshape", einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::GeneralRuntimeTensor<double, std::allocator<double>>)
APIARY_INSTANTIATE_AS("reshape", einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::RuntimeTensorView<double>)
APIARY_INSTANTIATE_AS("reshape", einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>)
APIARY_INSTANTIATE_AS("reshape", einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::RuntimeTensorView<std::complex<float>>)
APIARY_INSTANTIATE_AS("reshape", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
APIARY_INSTANTIATE_AS("reshape", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::RuntimeTensorView<std::complex<double>>)
    // clang-format on
    void reshape(ResultType *out, AType const &A, bool row_major) {
    using T              = typename ResultType::ValueType;
    size_t const a_rank  = detail::tensor_rank(A);
    size_t const o_rank  = detail::tensor_rank(*out);
    size_t       a_total = 1, o_total = 1;
    for (size_t k = 0; k < a_rank; ++k)
        a_total *= A.dim(k);
    for (size_t k = 0; k < o_rank; ++k)
        o_total *= out->dim(k);
    if (a_total != o_total) {
        EINSUMS_THROW_EXCEPTION(dimension_error, "cg::reshape: {} elements cannot be reshaped into {}", a_total, o_total);
    }

    auto apply = [a_rank, o_rank, a_total, row_major](ResultType *o, AType const *a) {
        if (a_total == 0)
            return;
        std::vector<size_t> a_dims(a_rank), a_str(a_rank), o_dims(o_rank), o_str(o_rank);
        for (size_t k = 0; k < a_rank; ++k) {
            a_dims[k] = a->dim(k);
            a_str[k]  = a->stride(k);
        }
        for (size_t k = 0; k < o_rank; ++k) {
            o_dims[k] = o->dim(k);
            o_str[k]  = o->stride(k);
        }
        T       *o_data = o->data();
        T const *a_data = a->data();
        // Walk the shared linear index and decompose it into each shape.
        for (size_t lin = 0; lin < a_total; ++lin) {
            size_t a_off = 0, o_off = 0, rem = lin;
            if (row_major) {
                for (size_t k = a_rank; k-- > 0;) {
                    a_off += (rem % a_dims[k]) * a_str[k];
                    rem /= a_dims[k];
                }
                rem = lin;
                for (size_t k = o_rank; k-- > 0;) {
                    o_off += (rem % o_dims[k]) * o_str[k];
                    rem /= o_dims[k];
                }
            } else {
                for (size_t k = 0; k < a_rank; ++k) {
                    a_off += (rem % a_dims[k]) * a_str[k];
                    rem /= a_dims[k];
                }
                rem = lin;
                for (size_t k = 0; k < o_rank; ++k) {
                    o_off += (rem % o_dims[k]) * o_str[k];
                    rem /= o_dims[k];
                }
            }
            o_data[o_off] = a_data[a_off];
        }
    };

    auto &ctx = CaptureContext::current();
    if (!ctx.is_capturing()) {
        LabeledSection("reshape eager");
        apply(out, &A);
        return;
    }
    LabeledSection("reshape capture");
    auto [a_id, a_slot] = ctx.get_slot(A);
    auto [r_id, r_slot] = ctx.get_slot(*out);
    auto executor       = [a_slot, r_slot, apply]() {
        LabeledSection("reshape execute");
        apply(static_cast<ResultType *>(r_slot->ptr), static_cast<AType const *>(a_slot->ptr));
    };
    ctx.record(OpKind::Custom, "reshape", {a_id}, {r_id}, std::move(executor));
}

// ─────────────────────────────────────────────────────────────────────────────
// diagonal: extract the diagonal of a rank-2 tensor
// ─────────────────────────────────────────────────────────────────────────────

/// Graph-aware diagonal extraction: ``out[i] := A[i, i]``.
///
/// @p out is rank-1 with extent ``min(A.dim(0), A.dim(1))``, matching numpy's
/// ``np.diag`` on a matrix. A rectangular @p A is fine; the shorter axis wins.
template <CoreBasicTensorConcept ResultType, CoreBasicTensorConcept AType>
    requires std::is_same_v<typename ResultType::ValueType, typename AType::ValueType>
// clang-format off
APIARY_EXPOSE
APIARY_MODULE("linalg")
APIARY_INSTANTIATE_AS("diagonal", einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::GeneralRuntimeTensor<float, std::allocator<float>>)
APIARY_INSTANTIATE_AS("diagonal", einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::RuntimeTensorView<float>)
APIARY_INSTANTIATE_AS("diagonal", einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::GeneralRuntimeTensor<double, std::allocator<double>>)
APIARY_INSTANTIATE_AS("diagonal", einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::RuntimeTensorView<double>)
APIARY_INSTANTIATE_AS("diagonal", einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>)
APIARY_INSTANTIATE_AS("diagonal", einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::RuntimeTensorView<std::complex<float>>)
APIARY_INSTANTIATE_AS("diagonal", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
APIARY_INSTANTIATE_AS("diagonal", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::RuntimeTensorView<std::complex<double>>)
    // clang-format on
    void diagonal(ResultType *out, AType const &A) {
    if (detail::tensor_rank(A) != 2) {
        EINSUMS_THROW_EXCEPTION(rank_error, "cg::diagonal: A must be rank 2, got rank {}", detail::tensor_rank(A));
    }
    if (detail::tensor_rank(*out) != 1) {
        EINSUMS_THROW_EXCEPTION(rank_error, "cg::diagonal: out must be rank 1, got rank {}", detail::tensor_rank(*out));
    }
    size_t const n = std::min(A.dim(0), A.dim(1));
    if (out->dim(0) != n) {
        EINSUMS_THROW_EXCEPTION(dimension_error, "cg::diagonal: out has extent {}, expected {}", out->dim(0), n);
    }

    auto apply = [n](ResultType *o, AType const *a) {
        using T             = typename ResultType::ValueType;
        T           *o_data = o->data();
        T const     *a_data = a->data();
        size_t const s0 = a->stride(0), s1 = a->stride(1), os = o->stride(0);
        for (size_t i = 0; i < n; ++i)
            o_data[i * os] = a_data[i * s0 + i * s1];
    };

    auto &ctx = CaptureContext::current();
    if (!ctx.is_capturing()) {
        LabeledSection("diagonal eager");
        apply(out, &A);
        return;
    }
    LabeledSection("diagonal capture");
    auto [a_id, a_slot] = ctx.get_slot(A);
    auto [r_id, r_slot] = ctx.get_slot(*out);
    auto executor       = [a_slot, r_slot, apply]() {
        LabeledSection("diagonal execute");
        apply(static_cast<ResultType *>(r_slot->ptr), static_cast<AType const *>(a_slot->ptr));
    };
    ctx.record(OpKind::Custom, "diagonal", {a_id}, {r_id}, std::move(executor));
}

// ─────────────────────────────────────────────────────────────────────────────
// scatter_add: accumulating placement
// ─────────────────────────────────────────────────────────────────────────────

/// Graph-aware accumulating scatter: ``dst[indices...] += src[...]``.
///
/// numpy's ``np.add.at``. The companion to @ref scatter rather than a
/// relaxation of it: scatter REJECTS repeated indices, because a plain write
/// twice to one element leaves the result depending on loop order. Under an
/// accumulation that ambiguity disappears - addition is commutative - so
/// repeats are meaningful here and are allowed.
template <CoreBasicTensorConcept DstType, CoreBasicTensorConcept SrcType>
    requires std::is_same_v<typename DstType::ValueType, typename SrcType::ValueType>
// clang-format off
APIARY_EXPOSE
APIARY_MODULE("linalg")
APIARY_INSTANTIATE_AS("scatter_add", einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::GeneralRuntimeTensor<float, std::allocator<float>>)
APIARY_INSTANTIATE_AS("scatter_add", einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::RuntimeTensorView<float>)
APIARY_INSTANTIATE_AS("scatter_add", einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::GeneralRuntimeTensor<double, std::allocator<double>>)
APIARY_INSTANTIATE_AS("scatter_add", einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::RuntimeTensorView<double>)
APIARY_INSTANTIATE_AS("scatter_add", einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>)
APIARY_INSTANTIATE_AS("scatter_add", einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::RuntimeTensorView<std::complex<float>>)
APIARY_INSTANTIATE_AS("scatter_add", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
APIARY_INSTANTIATE_AS("scatter_add", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::RuntimeTensorView<std::complex<double>>)
    // clang-format on
    void scatter_add(DstType *dst, SrcType const &src, std::vector<std::vector<size_t>> const &indices) {
    size_t const N = indices.size();
    if (N == 0) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "cg::scatter_add: indices must be non-empty");
    }
    if (detail::tensor_rank(*dst) != N || detail::tensor_rank(src) != N) {
        EINSUMS_THROW_EXCEPTION(rank_error, "cg::scatter_add: rank mismatch - dst rank={}, src rank={}, indices.size()={}",
                                detail::tensor_rank(*dst), detail::tensor_rank(src), N);
    }
    std::vector<size_t> extents(N);
    for (size_t k = 0; k < N; ++k) {
        extents[k] = indices[k].size();
        if (src.dim(k) != extents[k]) {
            EINSUMS_THROW_EXCEPTION(std::invalid_argument, "cg::scatter_add: src axis {} has extent {}, but {} indices were given", k,
                                    src.dim(k), extents[k]);
        }
        for (size_t p : indices[k]) {
            if (p >= dst->dim(k)) {
                EINSUMS_THROW_EXCEPTION(std::out_of_range, "cg::scatter_add: index {} on axis {} is out of range for dst dim {}", p, k,
                                        dst->dim(k));
            }
        }
    }

    auto apply = [indices, extents, N](DstType *d, SrcType const *s) {
        using T = typename DstType::ValueType;
        std::vector<size_t> d_str(N), s_str(N);
        for (size_t k = 0; k < N; ++k) {
            d_str[k] = d->stride(k);
            s_str[k] = s->stride(k);
        }
        T       *d_data = d->data();
        T const *s_data = s->data();

        // As scatter, but accumulating. A repeated index only collapses into a
        // run when it repeats the PREVIOUS index plus one, which it cannot, so
        // the run path never merges two writes to the same element.
        detail::for_each_selection_run(indices, extents, d_str, s_str, [&](size_t d_off, size_t s_off, size_t n) {
            for (size_t i = 0; i < n; ++i) {
                d_data[d_off + i] += s_data[s_off + i];
            }
        });
    };

    auto &ctx = CaptureContext::current();
    if (!ctx.is_capturing()) {
        LabeledSection("scatter_add eager");
        apply(dst, &src);
        return;
    }
    LabeledSection("scatter_add capture");
    auto [s_id, s_slot] = ctx.get_slot(src);
    auto [d_id, d_slot] = ctx.get_slot(*dst);
    auto executor       = [s_slot, d_slot, apply]() {
        LabeledSection("scatter_add execute");
        apply(static_cast<DstType *>(d_slot->ptr), static_cast<SrcType const *>(s_slot->ptr));
    };
    // dst is read as well as written: this accumulates onto what is there.
    ctx.record(OpKind::Custom, "scatter_add", {s_id, d_id}, {d_id}, std::move(executor));
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
        auto params         = std::make_shared<TiledElementwiseParams>();
        params->alpha       = PrefactorScalar{alpha};
        auto executor       = [params, x_slot, y_slot]() {
            LabeledSection("axpy execute");
            detail::tiled_axpy<T>(as<T>(params->alpha), *static_cast<XType const *>(x_slot->ptr), static_cast<YType *>(y_slot->ptr));
        };
        TiledElementwiseDescriptor edesc;
        edesc.op     = TiledElementwiseOp::Axpy;
        edesc.params = params;
        // Y is listed as an INPUT as well as an output: tiled_axpy computes
        // Y += alpha*X, so it reads its destination. Omitting it hides the
        // accumulation from the scheduler and the liveness passes -- Reorder could
        // move this past another writer of Y, and DeadNodeElimination could treat
        // the value being accumulated onto as dead. This is the convention the
        // dense axpy already uses (bug-1009); the tiled overload was written later
        // and missed it.
        ctx.record(OpKind::Custom, std::move(label), {x_id, y_id}, {y_id}, std::move(executor), std::move(edesc));
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

        using T = typename XType::ValueType;

        // axpy IS an axpby with beta == 1, so it records as one. It used to be
        // its own OpKind carrying no op_data at all, which made the op opaque:
        // a pass could see "something accumulates into Y" but not by how much,
        // so every scalar-aware rewrite (ScaleAbsorption, CSE, ElementWiseFusion,
        // SymmetrizedAccumulation, ...) gated on OpKind::Axpby and skipped it.
        // Since `Y += X` is how this operation is spelled in every other
        // library, the most natural spelling was the one the optimizer could not
        // see.
        //
        // Recording the same kind rather than a parallel one is what makes those
        // passes work on it, with no per-pass special-casing. The kernel choice
        // is the executor's, not the kind's: it still calls the BLAS axpy fast
        // path whenever beta == 1, which is every capture from here.
        auto params   = std::make_shared<AxpbyParams>();
        params->alpha = PrefactorScalar{alpha};
        params->beta  = PrefactorScalar{T{1}};

        auto label = fmt::format("axpy(alpha={}, {}, {})", alpha, X.name(), Y->name());
        // Reads the scalars through the shared params, exactly as axpby does, so
        // the descriptor is the single source of truth rather than a snapshot the
        // executor can silently disagree with. A pass that rewrites beta away
        // from 1 turns this into a genuine axpby, so honor that rather than
        // ignoring the write - a baked beta would compute the wrong thing.
        auto executor = [params, x_slot, y_slot]() {
            LabeledSection("axpy execute");
            auto const a = as<T>(params->alpha);
            auto const b = as<T>(params->beta);
            if (b == T{1}) {
                linear_algebra::axpy(a, *static_cast<XType const *>(x_slot->ptr), static_cast<YType *>(y_slot->ptr));
            } else {
                linear_algebra::axpby(a, *static_cast<XType const *>(x_slot->ptr), b, static_cast<YType *>(y_slot->ptr));
            }
        };

        AxpbyDescriptor desc;
        desc.alpha  = params->alpha;
        desc.beta   = params->beta;
        desc.params = params;

        // Y += alpha*X reads its destination unconditionally (beta == 1); list it
        // as an input so dependency passes see the read (matches gemm's and
        // direct_product's out-tensor-as-input convention - without it,
        // LoopInvariantHoisting's reads-its-output guard is blind to the
        // accumulation and Reorder misses the WAR hazard on Y's old value).
        ctx.record(OpKind::Axpby, std::move(label), {x_id, y_id}, {y_id}, std::move(executor), std::move(desc));
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
APIARY_INSTANTIATE_AS("axpby", einsums::TiledRuntimeTensor<float>, einsums::TiledRuntimeTensor<float>)
APIARY_INSTANTIATE_AS("axpby", einsums::TiledRuntimeTensor<double>, einsums::TiledRuntimeTensor<double>)
APIARY_INSTANTIATE_AS("axpby", einsums::TiledRuntimeTensor<std::complex<float>>, einsums::TiledRuntimeTensor<std::complex<float>>)
APIARY_INSTANTIATE_AS("axpby", einsums::TiledRuntimeTensor<std::complex<double>>, einsums::TiledRuntimeTensor<std::complex<double>>)
    // clang-format on
    void axpby(typename XType::ValueType alpha, XType const &X, typename XType::ValueType beta, YType *Y) {
    if constexpr (IsTiledTensorV<std::remove_cvref_t<XType>> || IsTiledTensorV<std::remove_cvref_t<YType>>) {
        static_assert(IsTiledTensorV<std::remove_cvref_t<XType>> && IsTiledTensorV<std::remove_cvref_t<YType>>,
                      "cg::axpby with a tiled operand requires both X and Y to be TiledRuntimeTensor");
        // Y = alpha*X + beta*Y decomposes into the two tiled primitives the
        // graph already executes, captures, and lowers (TiledExpansion):
        // scale the destination, then accumulate. Grid agreement and
        // absent-tile semantics are those ops' own rules: a tile absent from
        // Y stays absent under the scale (it is a rigorous zero), and a tile
        // present only in X is created zeroed by the accumulate.
        scale(beta, Y);
        axpy(alpha, X, Y);
        return;
    } else {
        auto &ctx = CaptureContext::current();
        if (!ctx.is_capturing()) {
            LabeledSection("axpby eager");
            linear_algebra::axpby(alpha, X, beta, Y);
            return;
        }

        LabeledSection("axpby capture");
        auto [x_id, x_slot] = ctx.get_slot(X);
        auto [y_id, y_slot] = ctx.get_slot(*Y);

        using T = typename XType::ValueType;

        // Live-mutable scalars shared with the executor (single source of truth:
        // a pass that folds a scale into this axpby writes beta through params and
        // the executor honors it on replay). The descriptor keeps the at-capture
        // snapshot for analysis passes.
        auto params   = std::make_shared<AxpbyParams>();
        params->alpha = PrefactorScalar{alpha};
        params->beta  = PrefactorScalar{beta};

        auto label    = fmt::format("axpby(alpha={}, beta={})", alpha, beta);
        auto executor = [params, x_slot, y_slot]() {
            LabeledSection("axpby execute");
            linear_algebra::axpby(as<T>(params->alpha), *static_cast<XType const *>(x_slot->ptr), as<T>(params->beta),
                                  static_cast<YType *>(y_slot->ptr));
        };

        AxpbyDescriptor desc;
        desc.alpha  = params->alpha;
        desc.beta   = params->beta;
        desc.params = params;

        // Y = alpha*X + beta*Y reads its destination when beta != 0; same
        // out-tensor-as-input convention as gemm/direct_product (see axpy).
        std::vector<TensorId> axpby_inputs =
            (beta != typename XType::ValueType{0}) ? std::vector<TensorId>{x_id, y_id} : std::vector<TensorId>{x_id};
        ctx.record(OpKind::Axpby, std::move(label), std::move(axpby_inputs), {y_id}, std::move(executor), std::move(desc));
    }
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
// float: AAA/AAV/AVA/AVV/VAA/VAV/VVA/VVV, where A = owning and V = view
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

/// Graph-aware GEMV with a runtime ``Transpose`` op flag (N / T / C).
///
/// The companion of the bool ``trans_a`` overload above; adds conjugate-transpose
/// (``Transpose::C``) for complex A via the runtime-char ``linear_algebra::gemv``.
/// The bool overload still resolves ``trans_a=True/False``; pass
/// ``trans_a=Transpose.C`` to reach this one. Default ``Transpose::N``.
template <RuntimeRankTensorConcept AType, RuntimeRankTensorConcept XType, RuntimeRankTensorConcept YType, typename U>
    requires(SameUnderlying<AType, XType, YType> && std::convertible_to<U, typename AType::ValueType>)
// clang-format off
APIARY_EXPOSE
APIARY_MODULE("linalg")
// float
APIARY_INSTANTIATE_AS("gemv", einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::GeneralRuntimeTensor<float, std::allocator<float>>, float)
APIARY_INSTANTIATE_AS("gemv", einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::RuntimeTensorView<float>,                          float)
APIARY_INSTANTIATE_AS("gemv", einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::RuntimeTensorView<float>,                          einsums::GeneralRuntimeTensor<float, std::allocator<float>>, float)
APIARY_INSTANTIATE_AS("gemv", einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::RuntimeTensorView<float>,                          einsums::RuntimeTensorView<float>,                          float)
APIARY_INSTANTIATE_AS("gemv", einsums::RuntimeTensorView<float>,                          einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::GeneralRuntimeTensor<float, std::allocator<float>>, float)
APIARY_INSTANTIATE_AS("gemv", einsums::RuntimeTensorView<float>,                          einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::RuntimeTensorView<float>,                          float)
APIARY_INSTANTIATE_AS("gemv", einsums::RuntimeTensorView<float>,                          einsums::RuntimeTensorView<float>,                          einsums::GeneralRuntimeTensor<float, std::allocator<float>>, float)
APIARY_INSTANTIATE_AS("gemv", einsums::RuntimeTensorView<float>,                          einsums::RuntimeTensorView<float>,                          einsums::RuntimeTensorView<float>,                          float)
// double
APIARY_INSTANTIATE_AS("gemv", einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::GeneralRuntimeTensor<double, std::allocator<double>>, double)
APIARY_INSTANTIATE_AS("gemv", einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::RuntimeTensorView<double>,                          double)
APIARY_INSTANTIATE_AS("gemv", einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::RuntimeTensorView<double>,                          einsums::GeneralRuntimeTensor<double, std::allocator<double>>, double)
APIARY_INSTANTIATE_AS("gemv", einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::RuntimeTensorView<double>,                          einsums::RuntimeTensorView<double>,                          double)
APIARY_INSTANTIATE_AS("gemv", einsums::RuntimeTensorView<double>,                          einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::GeneralRuntimeTensor<double, std::allocator<double>>, double)
APIARY_INSTANTIATE_AS("gemv", einsums::RuntimeTensorView<double>,                          einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::RuntimeTensorView<double>,                          double)
APIARY_INSTANTIATE_AS("gemv", einsums::RuntimeTensorView<double>,                          einsums::RuntimeTensorView<double>,                          einsums::GeneralRuntimeTensor<double, std::allocator<double>>, double)
APIARY_INSTANTIATE_AS("gemv", einsums::RuntimeTensorView<double>,                          einsums::RuntimeTensorView<double>,                          einsums::RuntimeTensorView<double>,                          double)
// complex<float>
APIARY_INSTANTIATE_AS("gemv", einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, std::complex<float>)
APIARY_INSTANTIATE_AS("gemv", einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::RuntimeTensorView<std::complex<float>>,                                            std::complex<float>)
APIARY_INSTANTIATE_AS("gemv", einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::RuntimeTensorView<std::complex<float>>,                                            einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, std::complex<float>)
APIARY_INSTANTIATE_AS("gemv", einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::RuntimeTensorView<std::complex<float>>,                                            einsums::RuntimeTensorView<std::complex<float>>,                                            std::complex<float>)
APIARY_INSTANTIATE_AS("gemv", einsums::RuntimeTensorView<std::complex<float>>,                                            einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, std::complex<float>)
APIARY_INSTANTIATE_AS("gemv", einsums::RuntimeTensorView<std::complex<float>>,                                            einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::RuntimeTensorView<std::complex<float>>,                                            std::complex<float>)
APIARY_INSTANTIATE_AS("gemv", einsums::RuntimeTensorView<std::complex<float>>,                                            einsums::RuntimeTensorView<std::complex<float>>,                                            einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, std::complex<float>)
APIARY_INSTANTIATE_AS("gemv", einsums::RuntimeTensorView<std::complex<float>>,                                            einsums::RuntimeTensorView<std::complex<float>>,                                            einsums::RuntimeTensorView<std::complex<float>>,                                            std::complex<float>)
// complex<double>
APIARY_INSTANTIATE_AS("gemv", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, std::complex<double>)
APIARY_INSTANTIATE_AS("gemv", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::RuntimeTensorView<std::complex<double>>,                                          std::complex<double>)
APIARY_INSTANTIATE_AS("gemv", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::RuntimeTensorView<std::complex<double>>,                                          einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, std::complex<double>)
APIARY_INSTANTIATE_AS("gemv", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::RuntimeTensorView<std::complex<double>>,                                          einsums::RuntimeTensorView<std::complex<double>>,                                          std::complex<double>)
APIARY_INSTANTIATE_AS("gemv", einsums::RuntimeTensorView<std::complex<double>>,                                          einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, std::complex<double>)
APIARY_INSTANTIATE_AS("gemv", einsums::RuntimeTensorView<std::complex<double>>,                                          einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::RuntimeTensorView<std::complex<double>>,                                          std::complex<double>)
APIARY_INSTANTIATE_AS("gemv", einsums::RuntimeTensorView<std::complex<double>>,                                          einsums::RuntimeTensorView<std::complex<double>>,                                          einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, std::complex<double>)
APIARY_INSTANTIATE_AS("gemv", einsums::RuntimeTensorView<std::complex<double>>,                                          einsums::RuntimeTensorView<std::complex<double>>,                                          einsums::RuntimeTensorView<std::complex<double>>,                                          std::complex<double>)
    // clang-format on
    void gemv(U const alpha, AType const &A, XType const &z, U const beta, YType *y,
              linear_algebra::Transpose trans_a = linear_algebra::Transpose::N) {
    if (A.rank() != 2 || z.rank() != 1 || y->rank() != 1) {
        EINSUMS_THROW_EXCEPTION(rank_error, "cg::gemv requires A rank-2 and x/y rank-1; got {}, {}, {}.", A.rank(), z.rank(), y->rank());
    }
    char const ta = static_cast<char>(trans_a);

    auto &ctx = CaptureContext::current();
    if (!ctx.is_capturing()) {
        LabeledSection("gemv eager");
        linear_algebra::gemv(ta, alpha, A, z, beta, y);
        return;
    }

    LabeledSection("gemv capture");
    auto [a_id, a_slot] = ctx.get_slot(A);
    auto [z_id, z_slot] = ctx.get_slot(z);
    auto [y_id, y_slot] = ctx.get_slot(*y);

    auto label    = fmt::format("gemv({})", static_cast<char>(trans_a));
    auto executor = [alpha, a_slot, z_slot, beta, y_slot, ta]() {
        LabeledSection("gemv execute");
        linear_algebra::gemv(ta, alpha, *static_cast<AType const *>(a_slot->ptr), *static_cast<XType const *>(z_slot->ptr), beta,
                             static_cast<YType *>(y_slot->ptr));
    };

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

/// Graph-aware conjugating rank-1 update (GERC): ``A += alpha * X * Y^H``.
///
/// The Hermitian counterpart of ``ger``, which computes ``A += alpha * X * Y^T``:
/// here the second vector is conjugated. Complex operands only; for real operands
/// use ``ger``. Backed by linear_algebra::gerc (BLAS cgerc/zgerc).
template <RuntimeRankTensorConcept AType, RuntimeRankTensorConcept XType, RuntimeRankTensorConcept YType>
    requires SameUnderlying<AType, XType, YType>
// clang-format off
APIARY_EXPOSE
APIARY_MODULE("linalg")
// complex<float>
APIARY_INSTANTIATE_AS("gerc", einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>)
APIARY_INSTANTIATE_AS("gerc", einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::RuntimeTensorView<std::complex<float>>)
APIARY_INSTANTIATE_AS("gerc", einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::RuntimeTensorView<std::complex<float>>,                                            einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>)
APIARY_INSTANTIATE_AS("gerc", einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::RuntimeTensorView<std::complex<float>>,                                            einsums::RuntimeTensorView<std::complex<float>>)
APIARY_INSTANTIATE_AS("gerc", einsums::RuntimeTensorView<std::complex<float>>,                                            einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>)
APIARY_INSTANTIATE_AS("gerc", einsums::RuntimeTensorView<std::complex<float>>,                                            einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::RuntimeTensorView<std::complex<float>>)
APIARY_INSTANTIATE_AS("gerc", einsums::RuntimeTensorView<std::complex<float>>,                                            einsums::RuntimeTensorView<std::complex<float>>,                                            einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>)
APIARY_INSTANTIATE_AS("gerc", einsums::RuntimeTensorView<std::complex<float>>,                                            einsums::RuntimeTensorView<std::complex<float>>,                                            einsums::RuntimeTensorView<std::complex<float>>)
// complex<double>
APIARY_INSTANTIATE_AS("gerc", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
APIARY_INSTANTIATE_AS("gerc", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::RuntimeTensorView<std::complex<double>>)
APIARY_INSTANTIATE_AS("gerc", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::RuntimeTensorView<std::complex<double>>,                                          einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
APIARY_INSTANTIATE_AS("gerc", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::RuntimeTensorView<std::complex<double>>,                                          einsums::RuntimeTensorView<std::complex<double>>)
APIARY_INSTANTIATE_AS("gerc", einsums::RuntimeTensorView<std::complex<double>>,                                          einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
APIARY_INSTANTIATE_AS("gerc", einsums::RuntimeTensorView<std::complex<double>>,                                          einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::RuntimeTensorView<std::complex<double>>)
APIARY_INSTANTIATE_AS("gerc", einsums::RuntimeTensorView<std::complex<double>>,                                          einsums::RuntimeTensorView<std::complex<double>>,                                          einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
APIARY_INSTANTIATE_AS("gerc", einsums::RuntimeTensorView<std::complex<double>>,                                          einsums::RuntimeTensorView<std::complex<double>>,                                          einsums::RuntimeTensorView<std::complex<double>>)
    // clang-format on
    void gerc(typename AType::ValueType alpha, XType const &X, YType const &Y, AType *A) {
    if (X.rank() != 1 || Y.rank() != 1 || A->rank() != 2) {
        EINSUMS_THROW_EXCEPTION(rank_error, "cg::gerc requires X/Y rank-1 and A rank-2; got {}, {}, {}.", X.rank(), Y.rank(), A->rank());
    }

    auto &ctx = CaptureContext::current();
    if (!ctx.is_capturing()) {
        LabeledSection("gerc eager");
        linear_algebra::gerc(alpha, X, Y, A);
        return;
    }

    LabeledSection("gerc capture");
    auto [x_id, x_slot] = ctx.get_slot(X);
    auto [y_id, y_slot] = ctx.get_slot(Y);
    auto [a_id, a_slot] = ctx.get_slot(*A);

    auto executor = [alpha, x_slot, y_slot, a_slot]() {
        LabeledSection("gerc execute");
        linear_algebra::gerc(alpha, *static_cast<XType const *>(x_slot->ptr), *static_cast<YType const *>(y_slot->ptr),
                             static_cast<AType *>(a_slot->ptr));
    };

    // gerc accumulates into A (``A += α·X·Y^H``); list A as an input for the RMW.
    ctx.record(OpKind::Ger, "gerc", {x_id, y_id, a_id}, {a_id}, std::move(executor));
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
// Tiled operands. A tiled dot reduces over the grid, which is the shape a
// per-iteration amplitude container has, and it was reachable from C++ only -
// so the loop that reduces over it could not be measured from Python at all,
// which is how it went unnoticed that it opened a parallel region per call
// whatever the grid held. See BenchmarkBlockTileReduction.
APIARY_INSTANTIATE_AS("dot", einsums::TiledRuntimeTensor<float>,                einsums::TiledRuntimeTensor<float>)
APIARY_INSTANTIATE_AS("dot", einsums::TiledRuntimeTensor<double>,               einsums::TiledRuntimeTensor<double>)
APIARY_INSTANTIATE_AS("dot", einsums::TiledRuntimeTensor<std::complex<float>>,  einsums::TiledRuntimeTensor<std::complex<float>>)
APIARY_INSTANTIATE_AS("dot", einsums::TiledRuntimeTensor<std::complex<double>>, einsums::TiledRuntimeTensor<std::complex<double>>)
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
    detail::reject_if_capturing("cg::dot(A, B) returning scalar cannot be used during graph capture. "
                                "Use cg::einsum(\" <- i ; i\", &result, A, B) instead.");
    // A reduction's summation order is its thread count's, so an unfenced dot is a function of the machine as well as of
    // the operands. See @ref blas::SerialVendorScope.
    blas::SerialVendorScope const serial;
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
        // A reduction's summation order is its thread count's, so the fence is what makes this a function of the operands alone.
        blas::SerialVendorScope const serial;
        *result = linear_algebra::dot(A, B);
        return;
    }

    LabeledSection("dot capture");
    auto [a_id, a_slot] = ctx.get_slot(A);
    auto [b_id, b_slot] = ctx.get_slot(B);
    TensorId r_id       = ctx.get_or_register_scalar(result, "dot_result");

    auto executor = [result, a_slot, b_slot]() {
        LabeledSection("dot execute");
        blas::SerialVendorScope const serial;
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
        // A reduction's summation order is its thread count's, so the fence is what makes this a function of the operands alone.
        blas::SerialVendorScope const serial;
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
    if constexpr (IsTiledTensorV<std::remove_cvref_t<AType>>) {
        // The descriptor lets TiledExpansion lower this node onto per-tile
        // ids instead of stranding its whole-tensor tiled operands.
        TiledDotDescriptor td;
        td.conjugated = false;
        ctx.record(OpKind::Dot, "dot", {a_id, b_id}, {r_id}, std::move(executor), std::move(td));
    } else {
        ctx.record(OpKind::Dot, "dot", {a_id, b_id}, {r_id}, std::move(executor));
    }
}

/// Graph-aware Hermitian inner product: ``result := sum_i conj(A_i) * B_i``.
///
/// The conjugating counterpart of ``dot``, the bilinear ``sum A_i B_i``. For real
/// dtypes this coincides with ``dot``. Backed by ``true_dot``, which uses BLAS
/// dotc on the contiguous complex path.
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
// float: RRR/RRV/RVR/RVV/VRR/VRV/VVR/VVV
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
APIARY_INSTANTIATE_AS("dotc", einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::TiledRuntimeTensor<float>, einsums::TiledRuntimeTensor<float>)
APIARY_INSTANTIATE_AS("dotc", einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::TiledRuntimeTensor<double>, einsums::TiledRuntimeTensor<double>)
APIARY_INSTANTIATE_AS("dotc", einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::TiledRuntimeTensor<std::complex<float>>, einsums::TiledRuntimeTensor<std::complex<float>>)
APIARY_INSTANTIATE_AS("dotc", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::TiledRuntimeTensor<std::complex<double>>, einsums::TiledRuntimeTensor<std::complex<double>>)
    // clang-format on
    void dotc_python(ResultType *result, AType const &A, BType const &B) {
    using T = typename AType::ValueType;
    if (result->size() < 1) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "cg::dotc: result tensor must have at least one element");
    }
    auto compute = [](AType const &a, BType const &b) -> T {
        // A reduction's summation order is its thread count's, so the fence is what makes this a function of the operands alone.
        blas::SerialVendorScope const serial;
        if constexpr (IsTiledTensorV<std::remove_cvref_t<AType>>) {
            return detail::tiled_dotc<T>(a, b);
        } else {
            return linear_algebra::true_dot(a, b);
        }
    };

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
    if constexpr (IsTiledTensorV<std::remove_cvref_t<AType>>) {
        // Same expansion hook as dot_python's, with the conjugation recorded.
        TiledDotDescriptor td;
        td.conjugated = true;
        ctx.record(OpKind::Dot, "dotc", {a_id, b_id}, {r_id}, std::move(executor), std::move(td));
    } else {
        ctx.record(OpKind::Dot, "dotc", {a_id, b_id}, {r_id}, std::move(executor));
    }
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
// tiled: all three operands tiled (mixed tiled/dense is rejected by static_assert)
APIARY_INSTANTIATE_AS("direct_division", float, einsums::TiledRuntimeTensor<float>, einsums::TiledRuntimeTensor<float>, einsums::TiledRuntimeTensor<float>)
APIARY_INSTANTIATE_AS("direct_division", double, einsums::TiledRuntimeTensor<double>, einsums::TiledRuntimeTensor<double>, einsums::TiledRuntimeTensor<double>)
APIARY_INSTANTIATE_AS("direct_division", std::complex<float>, einsums::TiledRuntimeTensor<std::complex<float>>, einsums::TiledRuntimeTensor<std::complex<float>>, einsums::TiledRuntimeTensor<std::complex<float>>)
APIARY_INSTANTIATE_AS("direct_division", std::complex<double>, einsums::TiledRuntimeTensor<std::complex<double>>, einsums::TiledRuntimeTensor<std::complex<double>>, einsums::TiledRuntimeTensor<std::complex<double>>)
    // clang-format on
    void direct_division(T alpha, AType const &A, BType const &B, T beta, CType *C) {
    if constexpr (IsTiledTensorV<std::remove_cvref_t<AType>> || IsTiledTensorV<std::remove_cvref_t<BType>> ||
                  IsTiledTensorV<std::remove_cvref_t<CType>>) {
        static_assert(IsTiledTensorV<std::remove_cvref_t<AType>> && IsTiledTensorV<std::remove_cvref_t<BType>> &&
                          IsTiledTensorV<std::remove_cvref_t<CType>>,
                      "cg::direct_division with a tiled operand requires all of A, B, C to be TiledRuntimeTensor");
        auto &ctx = CaptureContext::current();
        if (!ctx.is_capturing()) {
            LabeledSection("direct_division eager");
            detail::tiled_direct_division<T>(alpha, A, B, beta, C);
            return;
        }
        LabeledSection("direct_division capture");
        auto [a_id, a_slot] = ctx.get_slot(A);
        auto [b_id, b_slot] = ctx.get_slot(B);
        auto [c_id, c_slot] = ctx.get_slot(*C);
        auto label          = fmt::format("tiled direct_division({}, {})", A.name(), C->name());
        auto params         = std::make_shared<TiledElementwiseParams>();
        params->alpha       = PrefactorScalar{alpha};
        params->beta        = PrefactorScalar{beta};
        auto executor       = [params, a_slot, b_slot, c_slot]() {
            LabeledSection("direct_division execute");
            detail::tiled_direct_division<T>(as<T>(params->alpha), *static_cast<AType const *>(a_slot->ptr),
                                             *static_cast<BType const *>(b_slot->ptr), as<T>(params->beta),
                                             static_cast<CType *>(c_slot->ptr));
        };
        TiledElementwiseDescriptor edesc;
        edesc.op     = TiledElementwiseOp::Divide;
        edesc.params = params;
        // beta != 0 reads the destination, same RMW convention as the dense path.
        std::vector<TensorId> ids = (beta != T{0}) ? std::vector<TensorId>{a_id, b_id, c_id} : std::vector<TensorId>{a_id, b_id};
        ctx.record(OpKind::DirectDivision, std::move(label), std::move(ids), {c_id}, std::move(executor), std::move(edesc));
        return;
    } else {
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
}

// ─────────────────────────────────────────────────────────────────────────────
// outer_sum: rank-N result(i_0,...,i_{N-1}) = Σ_k c_k * v_k(i_k)
// ─────────────────────────────────────────────────────────────────────────────

/// Outer sum of N rank-1 vectors with per-axis coefficients.
///
/// Fills ``result`` with
///
/// @code result(i_0, i_1, ..., i_{N-1}) = Σ_k coefficients[k] * vectors[k](i_k)
/// @endcode
///
/// Canonical use case is the MP2/CC energy denominator:
///
/// @code Δ(i,j,a,b) = ε_i + ε_j − ε_a − ε_b
///   ↪ outer_sum(&Δ, {ε_occ, ε_occ, ε_virt, ε_virt}, {+1, +1, -1, -1})
/// @endcode
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
// batched_gemm: many independent GEMMs as ONE node
// ─────────────────────────────────────────────────────────────────────────────

/// @brief Emit one `blas::gemm_batch` over a batch of independent GEMMs:
/// ``C_i = alpha * op(A_i) op(B_i) + beta * C_i`` for every i.
///
/// The GEMMBatching pass already fuses independent 2D einsums into a single
/// BatchedGemm node, so the fused form was reachable without this. What it was
/// not reachable *cheaply*: the caller had to emit one node per GEMM and let
/// the pass collapse them, and capture costs on the order of tens of
/// microseconds a node. A DLPNO-MP2 residual emitted 8112 nodes for the passes
/// to fuse into 33, spending more time building the graph than replaying it.
/// This lets a caller who already knows the batch say so directly.
///
/// Every member must share m, n, k, the transpose flags, the element type and
/// the leading dimensions, which is what `gemm_batch` takes as scalars. The
/// prefactors are shared too, so a per-member prefactor has to be folded into
/// the operands beforehand. Mismatches throw here rather than at execute time.
///
/// Outside capture this executes immediately, so the same call works eagerly.
///
/// @param alpha   Prefactor on op(A_i) op(B_i), shared by the batch.
/// @param a_list  Left operands. Must all be rank 2 with identical dims.
/// @param b_list  Right operands, same length as @p a_list.
/// @param beta    Prefactor on C_i. Non-zero means every C_i is read as well as
///                written, which the node records as a dependency.
/// @param c_list  Destinations, same length as @p a_list. Must be distinct
///                tensors: `gemm_batch` gives no ordering between members, so
///                two members sharing a destination race.
/// @param trans_a Transpose each A_i.
/// @param trans_b Transpose each B_i.
template <CoreBasicTensorConcept AType, CoreBasicTensorConcept BType, CoreBasicTensorConcept CType>
    requires(std::is_same_v<typename AType::ValueType, typename BType::ValueType> &&
             std::is_same_v<typename AType::ValueType, typename CType::ValueType>)
// clang-format off
APIARY_EXPOSE
APIARY_MODULE("graph")
// All 8 owning/view combinations of (A, B, C) per dtype. Unlike outer_sum,
// where a homogeneous list is the natural shape, a batched GEMM routinely
// mixes them: scratch destinations are owning tensors while the operands are
// slices of a larger store. Each LIST is still homogeneous, since one template
// parameter covers it.
//
// float
APIARY_INSTANTIATE_AS("batched_gemm", einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::GeneralRuntimeTensor<float, std::allocator<float>>)
APIARY_INSTANTIATE_AS("batched_gemm", einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::RuntimeTensorView<float>)
APIARY_INSTANTIATE_AS("batched_gemm", einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::RuntimeTensorView<float>, einsums::GeneralRuntimeTensor<float, std::allocator<float>>)
APIARY_INSTANTIATE_AS("batched_gemm", einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::RuntimeTensorView<float>, einsums::RuntimeTensorView<float>)
APIARY_INSTANTIATE_AS("batched_gemm", einsums::RuntimeTensorView<float>, einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::GeneralRuntimeTensor<float, std::allocator<float>>)
APIARY_INSTANTIATE_AS("batched_gemm", einsums::RuntimeTensorView<float>, einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::RuntimeTensorView<float>)
APIARY_INSTANTIATE_AS("batched_gemm", einsums::RuntimeTensorView<float>, einsums::RuntimeTensorView<float>, einsums::GeneralRuntimeTensor<float, std::allocator<float>>)
APIARY_INSTANTIATE_AS("batched_gemm", einsums::RuntimeTensorView<float>, einsums::RuntimeTensorView<float>, einsums::RuntimeTensorView<float>)
//
// double
APIARY_INSTANTIATE_AS("batched_gemm", einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::GeneralRuntimeTensor<double, std::allocator<double>>)
APIARY_INSTANTIATE_AS("batched_gemm", einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::RuntimeTensorView<double>)
APIARY_INSTANTIATE_AS("batched_gemm", einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::RuntimeTensorView<double>, einsums::GeneralRuntimeTensor<double, std::allocator<double>>)
APIARY_INSTANTIATE_AS("batched_gemm", einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::RuntimeTensorView<double>, einsums::RuntimeTensorView<double>)
APIARY_INSTANTIATE_AS("batched_gemm", einsums::RuntimeTensorView<double>, einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::GeneralRuntimeTensor<double, std::allocator<double>>)
APIARY_INSTANTIATE_AS("batched_gemm", einsums::RuntimeTensorView<double>, einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::RuntimeTensorView<double>)
APIARY_INSTANTIATE_AS("batched_gemm", einsums::RuntimeTensorView<double>, einsums::RuntimeTensorView<double>, einsums::GeneralRuntimeTensor<double, std::allocator<double>>)
APIARY_INSTANTIATE_AS("batched_gemm", einsums::RuntimeTensorView<double>, einsums::RuntimeTensorView<double>, einsums::RuntimeTensorView<double>)
//
// std::complex<float>
APIARY_INSTANTIATE_AS("batched_gemm", einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>)
APIARY_INSTANTIATE_AS("batched_gemm", einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::RuntimeTensorView<std::complex<float>>)
APIARY_INSTANTIATE_AS("batched_gemm", einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::RuntimeTensorView<std::complex<float>>, einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>)
APIARY_INSTANTIATE_AS("batched_gemm", einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::RuntimeTensorView<std::complex<float>>, einsums::RuntimeTensorView<std::complex<float>>)
APIARY_INSTANTIATE_AS("batched_gemm", einsums::RuntimeTensorView<std::complex<float>>, einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>)
APIARY_INSTANTIATE_AS("batched_gemm", einsums::RuntimeTensorView<std::complex<float>>, einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::RuntimeTensorView<std::complex<float>>)
APIARY_INSTANTIATE_AS("batched_gemm", einsums::RuntimeTensorView<std::complex<float>>, einsums::RuntimeTensorView<std::complex<float>>, einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>)
APIARY_INSTANTIATE_AS("batched_gemm", einsums::RuntimeTensorView<std::complex<float>>, einsums::RuntimeTensorView<std::complex<float>>, einsums::RuntimeTensorView<std::complex<float>>)
//
// std::complex<double>
APIARY_INSTANTIATE_AS("batched_gemm", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
APIARY_INSTANTIATE_AS("batched_gemm", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::RuntimeTensorView<std::complex<double>>)
APIARY_INSTANTIATE_AS("batched_gemm", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::RuntimeTensorView<std::complex<double>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
APIARY_INSTANTIATE_AS("batched_gemm", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::RuntimeTensorView<std::complex<double>>, einsums::RuntimeTensorView<std::complex<double>>)
APIARY_INSTANTIATE_AS("batched_gemm", einsums::RuntimeTensorView<std::complex<double>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
APIARY_INSTANTIATE_AS("batched_gemm", einsums::RuntimeTensorView<std::complex<double>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::RuntimeTensorView<std::complex<double>>)
APIARY_INSTANTIATE_AS("batched_gemm", einsums::RuntimeTensorView<std::complex<double>>, einsums::RuntimeTensorView<std::complex<double>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
APIARY_INSTANTIATE_AS("batched_gemm", einsums::RuntimeTensorView<std::complex<double>>, einsums::RuntimeTensorView<std::complex<double>>, einsums::RuntimeTensorView<std::complex<double>>)
    // clang-format on
    void batched_gemm(double alpha, std::vector<AType const *> a_list, std::vector<BType const *> b_list, double beta,
                      std::vector<CType *> c_list, bool trans_a = false, bool trans_b = false) {
    using T = typename AType::ValueType;

    size_t const count = a_list.size();
    if (count == 0) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "cg::batched_gemm: batch is empty");
    }
    if (b_list.size() != count || c_list.size() != count) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "cg::batched_gemm: A, B and C lists must be the same length; got {}, {}, {}", count,
                                b_list.size(), c_list.size());
    }
    for (size_t i = 0; i < count; ++i) {
        if (a_list[i] == nullptr || b_list[i] == nullptr || c_list[i] == nullptr) {
            EINSUMS_THROW_EXCEPTION(std::invalid_argument, "cg::batched_gemm: member {} has a null operand", i);
        }
        // tensor_rank, not rank(): statically-ranked tensors carry Rank as a
        // constant and have no rank() member.
        if (detail::tensor_rank(*a_list[i]) != 2 || detail::tensor_rank(*b_list[i]) != 2 || detail::tensor_rank(*c_list[i]) != 2) {
            EINSUMS_THROW_EXCEPTION(rank_error, "cg::batched_gemm: member {} is not rank 2 (got {}, {}, {})", i,
                                    detail::tensor_rank(*a_list[i]), detail::tensor_rank(*b_list[i]), detail::tensor_rank(*c_list[i]));
        }
    }

    BatchedGemmDescriptor d;
    d.trans_a     = trans_a ? 'T' : 'N';
    d.trans_b     = trans_b ? 'T' : 'N';
    d.alpha       = std::complex<double>{alpha, 0.0};
    d.beta        = std::complex<double>{beta, 0.0};
    d.batch_count = static_cast<int>(count);
    d.m           = static_cast<int>(c_list[0]->dim(0));
    d.n           = static_cast<int>(c_list[0]->dim(1));
    d.k           = static_cast<int>(trans_a ? a_list[0]->dim(0) : a_list[0]->dim(1));
    d.lda         = static_cast<int>(a_list[0]->impl().get_lda());
    d.ldb         = static_cast<int>(b_list[0]->impl().get_lda());
    d.ldc         = static_cast<int>(c_list[0]->impl().get_lda());
    if constexpr (std::is_same_v<T, float>) {
        d.scalar = BlasScalar::Float;
    } else if constexpr (std::is_same_v<T, double>) {
        d.scalar = BlasScalar::Double;
    } else if constexpr (std::is_same_v<T, std::complex<float>>) {
        d.scalar = BlasScalar::ComplexFloat;
    } else {
        d.scalar = BlasScalar::ComplexDouble;
    }

    // gemm_batch takes ONE lda/ldb/ldc and one m/n/k for the whole batch, so a
    // member that differs is not expressible. Caught here, where the caller can
    // see which member and why, rather than as corruption at execute time.
    auto const require = [&](bool ok, size_t i, char const *what) {
        if (!ok) {
            EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                    "cg::batched_gemm: member {} disagrees on {}; every member must share m/n/k and leading dimensions "
                                    "because gemm_batch takes them once for the whole batch",
                                    i, what);
        }
    };
    for (size_t i = 1; i < count; ++i) {
        require(static_cast<int>(c_list[i]->dim(0)) == d.m, i, "m");
        require(static_cast<int>(c_list[i]->dim(1)) == d.n, i, "n");
        require(static_cast<int>(trans_a ? a_list[i]->dim(0) : a_list[i]->dim(1)) == d.k, i, "k");
        require(static_cast<int>(a_list[i]->impl().get_lda()) == d.lda, i, "lda");
        require(static_cast<int>(b_list[i]->impl().get_lda()) == d.ldb, i, "ldb");
        require(static_cast<int>(c_list[i]->impl().get_lda()) == d.ldc, i, "ldc");
    }
    // The contraction dimension has to agree between A and B as well.
    for (size_t i = 0; i < count; ++i) {
        require(static_cast<int>(trans_b ? b_list[i]->dim(1) : b_list[i]->dim(0)) == d.k, i, "the link dimension shared with A");
        require(static_cast<int>(trans_b ? b_list[i]->dim(0) : b_list[i]->dim(1)) == d.n, i, "n against B");
        require(static_cast<int>(trans_a ? a_list[i]->dim(1) : a_list[i]->dim(0)) == d.m, i, "m against A");
    }

    auto &ctx = CaptureContext::current();
    if (!ctx.is_capturing()) {
        LabeledSection("batched_gemm eager");
        std::vector<void const *> a_vs(count), b_vs(count);
        std::vector<void *>       c_vs(count);
        for (size_t i = 0; i < count; ++i) {
            a_vs[i] = static_cast<void const *>(a_list[i]->data());
            b_vs[i] = static_cast<void const *>(b_list[i]->data());
            c_vs[i] = static_cast<void *>(c_list[i]->data());
        }
        if constexpr (IsComplexV<T>) {
            detail::run_batched_gemm_complex<T>(d, a_vs, b_vs, c_vs);
        } else {
            detail::run_batched_gemm<T>(d, a_vs, b_vs, c_vs);
        }
        return;
    }

    LabeledSection("batched_gemm capture");
    detail::BatchedGemmExtractors       a_exs, b_exs;
    detail::BatchedGemmOutputExtractors c_exs;
    a_exs.reserve(count);
    b_exs.reserve(count);
    c_exs.reserve(count);
    // Node I/O keeps the pass's convention: inputs interleaved A_0, B_0, A_1,
    // B_1, ... and outputs C_0, C_1, ... in batch order.
    std::vector<TensorId> inputs;
    std::vector<TensorId> outputs;
    inputs.reserve(2 * count);
    outputs.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        auto [a_id, a_slot] = ctx.get_slot(*a_list[i]);
        auto [b_id, b_slot] = ctx.get_slot(*b_list[i]);
        auto [c_id, c_slot] = ctx.get_slot(*c_list[i]);
        inputs.push_back(a_id);
        inputs.push_back(b_id);
        outputs.push_back(c_id);
        // Read through the slot, not the captured pointer: rebind() and the
        // MemoryPlanning arena can both move a tensor's storage.
        a_exs.emplace_back([a_slot]() -> std::pair<void const *, int> {
            auto const *t = static_cast<AType const *>(a_slot->ptr);
            return {static_cast<void const *>(t->data()), static_cast<int>(t->impl().get_lda())};
        });
        b_exs.emplace_back([b_slot]() -> std::pair<void const *, int> {
            auto const *t = static_cast<BType const *>(b_slot->ptr);
            return {static_cast<void const *>(t->data()), static_cast<int>(t->impl().get_lda())};
        });
        c_exs.emplace_back([c_slot]() -> std::pair<void *, int> {
            auto *t = static_cast<CType *>(c_slot->ptr);
            return {static_cast<void *>(t->data()), static_cast<int>(t->impl().get_lda())};
        });
    }
    // beta != 0 means gemm_batch reads every destination before writing it, so
    // the RAW edge from whoever produced each C must survive (bug-1009).
    if (beta != 0.0) {
        inputs.insert(inputs.end(), outputs.begin(), outputs.end());
    }

    auto executor = detail::make_batched_gemm_executor(d, std::move(a_exs), std::move(b_exs), std::move(c_exs));
    ctx.record(OpKind::BatchedGemm,
               fmt::format("gemm_batch x{} ({}x{}x{}, trans={}{})", d.batch_count, d.m, d.k, d.n, d.trans_a, d.trans_b), std::move(inputs),
               std::move(outputs), std::move(executor), d);
}

// ─────────────────────────────────────────────────────────────────────────────
// batched_gemm_blocked: a batch whose C operands are blocks of ONE tensor
// ─────────────────────────────────────────────────────────────────────────────

/// @ref batched_gemm where the destinations are uniform-shaped blocks of a
/// single tensor, described by their offsets rather than enumerated as views.
///
/// The list form needs one tensor object per member, and building those is not
/// free: a caller whose destinations are column ranges of one store has to
/// materialize a view apiece, which in a local-correlation method is tens of
/// thousands of them. On the DLPNO port's overlap build that is 32948 view
/// constructions on the Python side and 32948 more slot registrations here,
/// together about half the phase, and none of it is arithmetic.
///
/// @p c_offsets gives each block's first element as an offset into @p c_base,
/// in elements. Every block is @p c_rows by @p c_cols and inherits the base's
/// leading dimension, which is what makes them expressible this way: a column
/// range of a column-major tensor has the parent's ``lda``.
///
/// A and B stay lists, because the callers that want this have destinations
/// that repeat one tensor and sources that do not. Passing a list of existing
/// tensors is cheap; it is *constructing* the views that is not.
///
/// One further difference from the list form, and it is a correctness one: the
/// node declares a single output, @p c_base itself, so a later read of the whole
/// base is ordered against this write. The list form declares the views, and
/// the graph does not know a view write touches its parent, so that ordering
/// has to be forced with a graph boundary.
template <TensorConcept AType, TensorConcept BType, TensorConcept CType>
    requires(std::is_same_v<typename AType::ValueType, typename BType::ValueType> &&
             std::is_same_v<typename AType::ValueType, typename CType::ValueType>)
// clang-format off
APIARY_EXPOSE
APIARY_MODULE("graph")
APIARY_INSTANTIATE_AS("batched_gemm_blocked", einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::GeneralRuntimeTensor<float, std::allocator<float>>)
APIARY_INSTANTIATE_AS("batched_gemm_blocked", einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::RuntimeTensorView<float>, einsums::GeneralRuntimeTensor<float, std::allocator<float>>)
APIARY_INSTANTIATE_AS("batched_gemm_blocked", einsums::RuntimeTensorView<float>, einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::GeneralRuntimeTensor<float, std::allocator<float>>)
APIARY_INSTANTIATE_AS("batched_gemm_blocked", einsums::RuntimeTensorView<float>, einsums::RuntimeTensorView<float>, einsums::GeneralRuntimeTensor<float, std::allocator<float>>)
APIARY_INSTANTIATE_AS("batched_gemm_blocked", einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::GeneralRuntimeTensor<double, std::allocator<double>>)
APIARY_INSTANTIATE_AS("batched_gemm_blocked", einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::RuntimeTensorView<double>, einsums::GeneralRuntimeTensor<double, std::allocator<double>>)
APIARY_INSTANTIATE_AS("batched_gemm_blocked", einsums::RuntimeTensorView<double>, einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::GeneralRuntimeTensor<double, std::allocator<double>>)
APIARY_INSTANTIATE_AS("batched_gemm_blocked", einsums::RuntimeTensorView<double>, einsums::RuntimeTensorView<double>, einsums::GeneralRuntimeTensor<double, std::allocator<double>>)
APIARY_INSTANTIATE_AS("batched_gemm_blocked", einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>)
APIARY_INSTANTIATE_AS("batched_gemm_blocked", einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::RuntimeTensorView<std::complex<float>>, einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>)
APIARY_INSTANTIATE_AS("batched_gemm_blocked", einsums::RuntimeTensorView<std::complex<float>>, einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>)
APIARY_INSTANTIATE_AS("batched_gemm_blocked", einsums::RuntimeTensorView<std::complex<float>>, einsums::RuntimeTensorView<std::complex<float>>, einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>)
APIARY_INSTANTIATE_AS("batched_gemm_blocked", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
APIARY_INSTANTIATE_AS("batched_gemm_blocked", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::RuntimeTensorView<std::complex<double>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
APIARY_INSTANTIATE_AS("batched_gemm_blocked", einsums::RuntimeTensorView<std::complex<double>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
APIARY_INSTANTIATE_AS("batched_gemm_blocked", einsums::RuntimeTensorView<std::complex<double>>, einsums::RuntimeTensorView<std::complex<double>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
    // clang-format on
    void batched_gemm_blocked(double alpha, std::vector<AType const *> a_list, std::vector<BType const *> b_list, double beta,
                              CType *c_base, std::vector<size_t> const &c_offsets, size_t c_rows, size_t c_cols, bool trans_a = false,
                              bool trans_b = false) {
    using T = typename AType::ValueType;

    size_t const count = a_list.size();
    if (count == 0) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "cg::batched_gemm_blocked: batch is empty");
    }
    if (b_list.size() != count || c_offsets.size() != count) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                "cg::batched_gemm_blocked: A, B and the offset list must be the same length; got {}, {}, {}", count,
                                b_list.size(), c_offsets.size());
    }
    if (c_base == nullptr) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "cg::batched_gemm_blocked: c_base is null");
    }
    if (detail::tensor_rank(*c_base) != 2) {
        EINSUMS_THROW_EXCEPTION(rank_error, "cg::batched_gemm_blocked: c_base is not rank 2 (got {})", detail::tensor_rank(*c_base));
    }
    if (c_rows == 0 || c_cols == 0) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "cg::batched_gemm_blocked: block shape is {}x{}; neither may be zero", c_rows,
                                c_cols);
    }

    size_t const ldc_s = c_base->impl().get_lda();
    size_t const span  = c_base->size();
    // A block starting at `off` reaches off + (c_cols-1)*ldc + c_rows - 1. Checked
    // here because the executor only ever sees a raw pointer, where an offset past
    // the end is silent corruption rather than a diagnosable error.
    if (c_rows > ldc_s) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                "cg::batched_gemm_blocked: block has {} rows but c_base's leading dimension is {}; a block taller than the "
                                "parent's column stride would overlap the next column",
                                c_rows, ldc_s);
    }
    for (size_t i = 0; i < count; ++i) {
        size_t const last = c_offsets[i] + (c_cols - 1) * ldc_s + c_rows;
        if (last > span) {
            EINSUMS_THROW_EXCEPTION(std::out_of_range,
                                    "cg::batched_gemm_blocked: block {} at offset {} spans {} elements of a {}-element c_base", i,
                                    c_offsets[i], last - c_offsets[i], span);
        }
        if (a_list[i] == nullptr || b_list[i] == nullptr) {
            EINSUMS_THROW_EXCEPTION(std::invalid_argument, "cg::batched_gemm_blocked: member {} has a null operand", i);
        }
        if (detail::tensor_rank(*a_list[i]) != 2 || detail::tensor_rank(*b_list[i]) != 2) {
            EINSUMS_THROW_EXCEPTION(rank_error, "cg::batched_gemm_blocked: member {} is not rank 2 (got {}, {})", i,
                                    detail::tensor_rank(*a_list[i]), detail::tensor_rank(*b_list[i]));
        }
    }

    BatchedGemmDescriptor d;
    d.trans_a     = trans_a ? 'T' : 'N';
    d.trans_b     = trans_b ? 'T' : 'N';
    d.alpha       = std::complex<double>{alpha, 0.0};
    d.beta        = std::complex<double>{beta, 0.0};
    d.batch_count = static_cast<int>(count);
    d.m           = static_cast<int>(c_rows);
    d.n           = static_cast<int>(c_cols);
    d.k           = static_cast<int>(trans_a ? a_list[0]->dim(0) : a_list[0]->dim(1));
    d.lda         = static_cast<int>(a_list[0]->impl().get_lda());
    d.ldb         = static_cast<int>(b_list[0]->impl().get_lda());
    d.ldc         = static_cast<int>(ldc_s);
    if constexpr (std::is_same_v<T, float>) {
        d.scalar = BlasScalar::Float;
    } else if constexpr (std::is_same_v<T, double>) {
        d.scalar = BlasScalar::Double;
    } else if constexpr (std::is_same_v<T, std::complex<float>>) {
        d.scalar = BlasScalar::ComplexFloat;
    } else {
        d.scalar = BlasScalar::ComplexDouble;
    }

    auto const require = [&](bool ok, size_t i, char const *what) {
        if (!ok) {
            EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                    "cg::batched_gemm_blocked: member {} disagrees on {}; every member must share m/n/k and leading "
                                    "dimensions because gemm_batch takes them once for the whole batch",
                                    i, what);
        }
    };
    for (size_t i = 0; i < count; ++i) {
        require(static_cast<int>(trans_a ? a_list[i]->dim(1) : a_list[i]->dim(0)) == d.m, i, "m against A");
        require(static_cast<int>(trans_a ? a_list[i]->dim(0) : a_list[i]->dim(1)) == d.k, i, "k");
        require(static_cast<int>(trans_b ? b_list[i]->dim(1) : b_list[i]->dim(0)) == d.k, i, "the link dimension shared with A");
        require(static_cast<int>(trans_b ? b_list[i]->dim(0) : b_list[i]->dim(1)) == d.n, i, "n against B");
        require(static_cast<int>(a_list[i]->impl().get_lda()) == d.lda, i, "lda");
        require(static_cast<int>(b_list[i]->impl().get_lda()) == d.ldb, i, "ldb");
    }

    auto &ctx = CaptureContext::current();
    if (!ctx.is_capturing()) {
        LabeledSection("batched_gemm_blocked eager");
        std::vector<void const *> a_vs(count), b_vs(count);
        std::vector<void *>       c_vs(count);
        T                        *base = c_base->data();
        for (size_t i = 0; i < count; ++i) {
            a_vs[i] = static_cast<void const *>(a_list[i]->data());
            b_vs[i] = static_cast<void const *>(b_list[i]->data());
            c_vs[i] = static_cast<void *>(base + c_offsets[i]);
        }
        if constexpr (IsComplexV<T>) {
            detail::run_batched_gemm_complex<T>(d, a_vs, b_vs, c_vs);
        } else {
            detail::run_batched_gemm<T>(d, a_vs, b_vs, c_vs);
        }
        return;
    }

    LabeledSection("batched_gemm_blocked capture");
    detail::BatchedGemmExtractors       a_exs, b_exs;
    detail::BatchedGemmOutputExtractors c_exs;
    a_exs.reserve(count);
    b_exs.reserve(count);
    c_exs.reserve(count);
    std::vector<TensorId> inputs;
    inputs.reserve(2 * count);

    // One slot for the whole destination, against one per member in the list
    // form. That is the point of this overload.
    auto [c_id, c_slot] = ctx.get_slot(*c_base);

    for (size_t i = 0; i < count; ++i) {
        auto [a_id, a_slot] = ctx.get_slot(*a_list[i]);
        auto [b_id, b_slot] = ctx.get_slot(*b_list[i]);
        inputs.push_back(a_id);
        inputs.push_back(b_id);
        a_exs.emplace_back([a_slot]() -> std::pair<void const *, int> {
            auto const *t = static_cast<AType const *>(a_slot->ptr);
            return {static_cast<void const *>(t->data()), static_cast<int>(t->impl().get_lda())};
        });
        b_exs.emplace_back([b_slot]() -> std::pair<void const *, int> {
            auto const *t = static_cast<BType const *>(b_slot->ptr);
            return {static_cast<void const *>(t->data()), static_cast<int>(t->impl().get_lda())};
        });
        // Read the base through its slot and offset at execute time: rebind()
        // and the MemoryPlanning arena can both move the storage after capture.
        size_t const off = c_offsets[i];
        c_exs.emplace_back([c_slot, off]() -> std::pair<void *, int> {
            auto *t = static_cast<CType *>(c_slot->ptr);
            return {static_cast<void *>(t->data() + off), static_cast<int>(t->impl().get_lda())};
        });
    }

    std::vector<TensorId> outputs{c_id};
    // beta != 0 means gemm_batch reads every destination before writing it, so
    // the RAW edge from whoever produced c_base must survive (bug-1009).
    if (beta != 0.0) {
        inputs.push_back(c_id);
    }

    auto executor = detail::make_batched_gemm_executor(d, std::move(a_exs), std::move(b_exs), std::move(c_exs));
    ctx.record(OpKind::BatchedGemm,
               fmt::format("gemm_batch x{} into blocks ({}x{}x{}, trans={}{})", d.batch_count, d.m, d.k, d.n, d.trans_a, d.trans_b),
               std::move(inputs), std::move(outputs), std::move(executor), d);
}

// ─────────────────────────────────────────────────────────────────────────────
// grouped_batched_gemm: independent GEMMs of DIFFERING shape as ONE node
// ─────────────────────────────────────────────────────────────────────────────

namespace detail {

/// What makes two members of a grouped batch belong to the same uniform group.
struct GemmShapeKey {
    int m, n, k, lda, ldb, ldc;

    auto operator<=>(GemmShapeKey const &) const = default;
};

} // namespace detail

/// @brief Emit one `blas::gemm_batch_grouped` over independent GEMMs whose
/// shapes need NOT agree: ``C_i = alpha * op(A_i) op(B_i) + beta * C_i``.
///
/// The same signature as @ref batched_gemm, minus its central restriction.
/// Members are sorted into shape groups here, at capture, and the whole set
/// then runs under ONE OpenMP region instead of one per shape.
///
/// That is the entire point, and it is not a small one. Entering a region costs
/// tens of microseconds on a wide team, which is comparable to a GEMM of the
/// size local-correlation methods produce. Measured on a DLPNO-MP2 iteration
/// whose batched GEMMs want 16 ms of arithmetic, issuing them as 754 uniform
/// calls took 45 ms and issuing them as one grouped call took 16.7. The same
/// batch also stopped caring how finely it was split: the effective rate held
/// flat at ~230 GFLOPS across 120, 448 and 754 shape classes, where the looped
/// form fell 146 to 80.
///
/// Grouping is by (m, n, k, lda, ldb, ldc); the prefactors and transpose flags
/// are shared by the whole call, as they are for @ref batched_gemm. Groups are
/// numbered in order of first appearance and members keep their relative order
/// inside a group, but members of DIFFERENT shapes are reordered against each
/// other. That is within contract for the same reason the uniform form gives no
/// ordering between members: they must be independent.
///
/// Outside capture this executes immediately, so the same call works eagerly.
///
/// @param alpha   Prefactor on op(A_i) op(B_i), shared by the whole call.
/// @param a_list  Left operands, all rank 2. Shapes may differ between members.
/// @param b_list  Right operands, same length as @p a_list.
/// @param beta    Prefactor on C_i. Non-zero means every C_i is read as well as
///                written, which the node records as a dependency.
/// @param c_list  Destinations, same length as @p a_list. Must be distinct
///                tensors: the call gives no ordering between members, so two
///                members sharing a destination race.
/// @param trans_a Transpose each A_i.
/// @param trans_b Transpose each B_i.
template <CoreBasicTensorConcept AType, CoreBasicTensorConcept BType, CoreBasicTensorConcept CType>
    requires(std::is_same_v<typename AType::ValueType, typename BType::ValueType> &&
             std::is_same_v<typename AType::ValueType, typename CType::ValueType>)
// clang-format off
APIARY_EXPOSE
APIARY_MODULE("graph")
// The same 8 owning/view combinations per dtype that batched_gemm carries, and
// for the same reason: scratch destinations are owning tensors while operands
// are slices of a larger store.
//
// float
APIARY_INSTANTIATE_AS("grouped_batched_gemm", einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::GeneralRuntimeTensor<float, std::allocator<float>>)
APIARY_INSTANTIATE_AS("grouped_batched_gemm", einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::RuntimeTensorView<float>)
APIARY_INSTANTIATE_AS("grouped_batched_gemm", einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::RuntimeTensorView<float>, einsums::GeneralRuntimeTensor<float, std::allocator<float>>)
APIARY_INSTANTIATE_AS("grouped_batched_gemm", einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::RuntimeTensorView<float>, einsums::RuntimeTensorView<float>)
APIARY_INSTANTIATE_AS("grouped_batched_gemm", einsums::RuntimeTensorView<float>, einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::GeneralRuntimeTensor<float, std::allocator<float>>)
APIARY_INSTANTIATE_AS("grouped_batched_gemm", einsums::RuntimeTensorView<float>, einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::RuntimeTensorView<float>)
APIARY_INSTANTIATE_AS("grouped_batched_gemm", einsums::RuntimeTensorView<float>, einsums::RuntimeTensorView<float>, einsums::GeneralRuntimeTensor<float, std::allocator<float>>)
APIARY_INSTANTIATE_AS("grouped_batched_gemm", einsums::RuntimeTensorView<float>, einsums::RuntimeTensorView<float>, einsums::RuntimeTensorView<float>)
//
// double
APIARY_INSTANTIATE_AS("grouped_batched_gemm", einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::GeneralRuntimeTensor<double, std::allocator<double>>)
APIARY_INSTANTIATE_AS("grouped_batched_gemm", einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::RuntimeTensorView<double>)
APIARY_INSTANTIATE_AS("grouped_batched_gemm", einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::RuntimeTensorView<double>, einsums::GeneralRuntimeTensor<double, std::allocator<double>>)
APIARY_INSTANTIATE_AS("grouped_batched_gemm", einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::RuntimeTensorView<double>, einsums::RuntimeTensorView<double>)
APIARY_INSTANTIATE_AS("grouped_batched_gemm", einsums::RuntimeTensorView<double>, einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::GeneralRuntimeTensor<double, std::allocator<double>>)
APIARY_INSTANTIATE_AS("grouped_batched_gemm", einsums::RuntimeTensorView<double>, einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::RuntimeTensorView<double>)
APIARY_INSTANTIATE_AS("grouped_batched_gemm", einsums::RuntimeTensorView<double>, einsums::RuntimeTensorView<double>, einsums::GeneralRuntimeTensor<double, std::allocator<double>>)
APIARY_INSTANTIATE_AS("grouped_batched_gemm", einsums::RuntimeTensorView<double>, einsums::RuntimeTensorView<double>, einsums::RuntimeTensorView<double>)
//
// std::complex<float>
APIARY_INSTANTIATE_AS("grouped_batched_gemm", einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>)
APIARY_INSTANTIATE_AS("grouped_batched_gemm", einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::RuntimeTensorView<std::complex<float>>)
APIARY_INSTANTIATE_AS("grouped_batched_gemm", einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::RuntimeTensorView<std::complex<float>>, einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>)
APIARY_INSTANTIATE_AS("grouped_batched_gemm", einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::RuntimeTensorView<std::complex<float>>, einsums::RuntimeTensorView<std::complex<float>>)
APIARY_INSTANTIATE_AS("grouped_batched_gemm", einsums::RuntimeTensorView<std::complex<float>>, einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>)
APIARY_INSTANTIATE_AS("grouped_batched_gemm", einsums::RuntimeTensorView<std::complex<float>>, einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::RuntimeTensorView<std::complex<float>>)
APIARY_INSTANTIATE_AS("grouped_batched_gemm", einsums::RuntimeTensorView<std::complex<float>>, einsums::RuntimeTensorView<std::complex<float>>, einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>)
APIARY_INSTANTIATE_AS("grouped_batched_gemm", einsums::RuntimeTensorView<std::complex<float>>, einsums::RuntimeTensorView<std::complex<float>>, einsums::RuntimeTensorView<std::complex<float>>)
//
// std::complex<double>
APIARY_INSTANTIATE_AS("grouped_batched_gemm", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
APIARY_INSTANTIATE_AS("grouped_batched_gemm", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::RuntimeTensorView<std::complex<double>>)
APIARY_INSTANTIATE_AS("grouped_batched_gemm", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::RuntimeTensorView<std::complex<double>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
APIARY_INSTANTIATE_AS("grouped_batched_gemm", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::RuntimeTensorView<std::complex<double>>, einsums::RuntimeTensorView<std::complex<double>>)
APIARY_INSTANTIATE_AS("grouped_batched_gemm", einsums::RuntimeTensorView<std::complex<double>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
APIARY_INSTANTIATE_AS("grouped_batched_gemm", einsums::RuntimeTensorView<std::complex<double>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::RuntimeTensorView<std::complex<double>>)
APIARY_INSTANTIATE_AS("grouped_batched_gemm", einsums::RuntimeTensorView<std::complex<double>>, einsums::RuntimeTensorView<std::complex<double>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
APIARY_INSTANTIATE_AS("grouped_batched_gemm", einsums::RuntimeTensorView<std::complex<double>>, einsums::RuntimeTensorView<std::complex<double>>, einsums::RuntimeTensorView<std::complex<double>>)
    // clang-format on
    void grouped_batched_gemm(double alpha, std::vector<AType const *> a_list, std::vector<BType const *> b_list, double beta,
                              std::vector<CType *> c_list, bool trans_a = false, bool trans_b = false) {
    using T = typename AType::ValueType;

    size_t const count = a_list.size();
    if (count == 0) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "cg::grouped_batched_gemm: batch is empty");
    }
    if (b_list.size() != count || c_list.size() != count) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "cg::grouped_batched_gemm: A, B and C lists must be the same length; got {}, {}, {}",
                                count, b_list.size(), c_list.size());
    }

    // Per member: check it is a well-formed GEMM on its own terms, then read
    // its shape key. Unlike the uniform form there is nothing to compare
    // against across members, so every consistency check is internal to one.
    std::vector<detail::GemmShapeKey> keys(count);
    for (size_t i = 0; i < count; ++i) {
        if (a_list[i] == nullptr || b_list[i] == nullptr || c_list[i] == nullptr) {
            EINSUMS_THROW_EXCEPTION(std::invalid_argument, "cg::grouped_batched_gemm: member {} has a null operand", i);
        }
        // tensor_rank, not rank(): statically-ranked tensors carry Rank as a
        // constant and have no rank() member.
        if (detail::tensor_rank(*a_list[i]) != 2 || detail::tensor_rank(*b_list[i]) != 2 || detail::tensor_rank(*c_list[i]) != 2) {
            EINSUMS_THROW_EXCEPTION(rank_error, "cg::grouped_batched_gemm: member {} is not rank 2 (got {}, {}, {})", i,
                                    detail::tensor_rank(*a_list[i]), detail::tensor_rank(*b_list[i]), detail::tensor_rank(*c_list[i]));
        }

        auto const m = static_cast<int>(c_list[i]->dim(0));
        auto const n = static_cast<int>(c_list[i]->dim(1));
        auto const k = static_cast<int>(trans_a ? a_list[i]->dim(0) : a_list[i]->dim(1));

        auto const require = [&](bool ok, char const *what) {
            if (!ok) {
                EINSUMS_THROW_EXCEPTION(std::invalid_argument, "cg::grouped_batched_gemm: member {} disagrees with itself on {}", i, what);
            }
        };
        require(static_cast<int>(trans_a ? a_list[i]->dim(1) : a_list[i]->dim(0)) == m, "m between A and C");
        require(static_cast<int>(trans_b ? b_list[i]->dim(1) : b_list[i]->dim(0)) == k, "the link dimension between A and B");
        require(static_cast<int>(trans_b ? b_list[i]->dim(0) : b_list[i]->dim(1)) == n, "n between B and C");

        keys[i] = {.m   = m,
                   .n   = n,
                   .k   = k,
                   .lda = static_cast<int>(a_list[i]->impl().get_lda()),
                   .ldb = static_cast<int>(b_list[i]->impl().get_lda()),
                   .ldc = static_cast<int>(c_list[i]->impl().get_lda())};
    }

    // Group by shape, first appearance first. Members of one shape keep their
    // relative order, which is what makes a single-shape call identical to
    // batched_gemm rather than merely equivalent to it.
    std::map<detail::GemmShapeKey, size_t> index_of;
    std::vector<detail::GemmShapeKey>      order;
    std::vector<std::vector<size_t>>       members;
    for (size_t i = 0; i < count; ++i) {
        auto const [it, inserted] = index_of.try_emplace(keys[i], order.size());
        if (inserted) {
            order.push_back(keys[i]);
            members.emplace_back();
        }
        members[it->second].push_back(i);
    }

    GroupedBatchedGemmDescriptor d;
    d.total = static_cast<int>(count);
    if constexpr (std::is_same_v<T, float>) {
        d.scalar = BlasScalar::Float;
    } else if constexpr (std::is_same_v<T, double>) {
        d.scalar = BlasScalar::Double;
    } else if constexpr (std::is_same_v<T, std::complex<float>>) {
        d.scalar = BlasScalar::ComplexFloat;
    } else {
        d.scalar = BlasScalar::ComplexDouble;
    }
    d.groups.reserve(order.size());
    d.labels.reserve(order.size());

    int first = 0;
    for (size_t g = 0; g < order.size(); ++g) {
        auto const &key = order[g];
        d.groups.push_back(GemmGroup{.m       = key.m,
                                     .n       = key.n,
                                     .k       = key.k,
                                     .lda     = key.lda,
                                     .ldb     = key.ldb,
                                     .ldc     = key.ldc,
                                     .trans_a = static_cast<char>(trans_a ? 'T' : 'N'),
                                     .trans_b = static_cast<char>(trans_b ? 'T' : 'N'),
                                     .alpha   = std::complex<double>{alpha, 0.0},
                                     .beta    = std::complex<double>{beta, 0.0},
                                     .count   = static_cast<int>(members[g].size()),
                                     .first   = first});
        d.labels.push_back(
            fmt::format("gemm {}x{}x{} trans={}{} x{}", key.m, key.k, key.n, trans_a ? 'T' : 'N', trans_b ? 'T' : 'N', members[g].size()));
        first += static_cast<int>(members[g].size());
    }

    // The flattened member order the descriptor's offsets index.
    std::vector<size_t> flat;
    flat.reserve(count);
    for (auto const &m : members) {
        flat.insert(flat.end(), m.begin(), m.end());
    }

    // Two members sharing a destination race, because the call gives no
    // ordering between them. @ref batched_gemm carries the same contract and
    // leaves it to the caller, but this entry point exists to MERGE calls that
    // used to be separate, and merging two that accumulate into one C is
    // exactly the mistake it makes newly reachable. Cheap to catch here, and
    // silently wrong if it is not.
    {
        std::vector<CType const *> seen(c_list.begin(), c_list.end());
        std::sort(seen.begin(), seen.end());
        auto const dup = std::adjacent_find(seen.begin(), seen.end());
        if (dup != seen.end()) {
            EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                    "cg::grouped_batched_gemm: two members share a destination tensor. The batch gives no ordering "
                                    "between members, so they would race; split them into separate calls");
        }
    }

    auto &ctx = CaptureContext::current();
    if (!ctx.is_capturing()) {
        LabeledSection("grouped_batched_gemm eager");
        std::vector<void const *> a_vs(count), b_vs(count);
        std::vector<void *>       c_vs(count);
        for (size_t i = 0; i < count; ++i) {
            a_vs[i] = static_cast<void const *>(a_list[flat[i]]->data());
            b_vs[i] = static_cast<void const *>(b_list[flat[i]]->data());
            c_vs[i] = static_cast<void *>(c_list[flat[i]]->data());
        }
        detail::run_grouped_batched_gemm<T>(d, a_vs, b_vs, c_vs);
        return;
    }

    LabeledSection("grouped_batched_gemm capture");
    detail::BatchedGemmExtractors       a_exs, b_exs;
    detail::BatchedGemmOutputExtractors c_exs;
    a_exs.reserve(count);
    b_exs.reserve(count);
    c_exs.reserve(count);
    // Node I/O keeps the batched form's convention, inputs interleaved
    // A_0, B_0, A_1, B_1, ... and outputs C_0, C_1, ..., in the FLATTENED
    // order, so a group's offset indexes the extractors and the node lists
    // alike.
    std::vector<TensorId> inputs;
    std::vector<TensorId> outputs;
    inputs.reserve(2 * count);
    outputs.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        size_t const s      = flat[i];
        auto [a_id, a_slot] = ctx.get_slot(*a_list[s]);
        auto [b_id, b_slot] = ctx.get_slot(*b_list[s]);
        auto [c_id, c_slot] = ctx.get_slot(*c_list[s]);
        inputs.push_back(a_id);
        inputs.push_back(b_id);
        outputs.push_back(c_id);
        // Read through the slot, not the captured pointer: rebind() and the
        // MemoryPlanning arena can both move a tensor's storage.
        a_exs.emplace_back([a_slot]() -> std::pair<void const *, int> {
            auto const *t = static_cast<AType const *>(a_slot->ptr);
            return {static_cast<void const *>(t->data()), static_cast<int>(t->impl().get_lda())};
        });
        b_exs.emplace_back([b_slot]() -> std::pair<void const *, int> {
            auto const *t = static_cast<BType const *>(b_slot->ptr);
            return {static_cast<void const *>(t->data()), static_cast<int>(t->impl().get_lda())};
        });
        c_exs.emplace_back([c_slot]() -> std::pair<void *, int> {
            auto *t = static_cast<CType *>(c_slot->ptr);
            return {static_cast<void *>(t->data()), static_cast<int>(t->impl().get_lda())};
        });
    }
    // beta != 0 means every destination is read before it is written, so the
    // RAW edge from whoever produced each C must survive (bug-1009).
    if (beta != 0.0) {
        inputs.insert(inputs.end(), outputs.begin(), outputs.end());
    }

    auto executor = detail::make_grouped_batched_gemm_executor(d, std::move(a_exs), std::move(b_exs), std::move(c_exs));
    ctx.record(
        OpKind::GroupedBatchedGemm,
        fmt::format("gemm_batch_grouped x{} in {} shapes (trans={}{})", d.total, d.groups.size(), trans_a ? 'T' : 'N', trans_b ? 'T' : 'N'),
        std::move(inputs), std::move(outputs), std::move(executor), std::move(d));
}

/// @ref grouped_batched_gemm where the destinations are blocks of existing
/// tensors, described by a base and an offset rather than enumerated as views.
///
/// The blocked form of @ref batched_gemm exists because materializing one view
/// per destination is tens of thousands of pybind round trips in a local
/// correlation method. The grouped form exists because one OpenMP region per
/// shape class costs more than the arithmetic. A caller whose destinations are
/// blocks AND whose shapes differ between classes needs both, and the DLPNO
/// overlap build is exactly that caller: one destination tensor per shape
/// class, every coupling a column range of its class's tensor.
///
/// So @p c_bases is parallel to the members, not to the groups. Repeating a
/// base pointer across a class's members is a pointer copy; it is constructing
/// the views that is expensive, and none are constructed. Each block's shape is
/// derived from its own A and B rather than passed, and its leading dimension
/// from its base, which is what makes a column range expressible this way.
///
/// The node declares each DISTINCT base as an output, so a later read of a
/// whole base is ordered against this write. The list form declares the views,
/// and the graph does not know a view write touches its parent.
///
/// @param c_bases  Per member, the tensor its block lives in.
/// @param c_offsets Per member, the block's first element as an offset into its
///                  base, in elements.
template <TensorConcept AType, TensorConcept BType, TensorConcept CType>
    requires(std::is_same_v<typename AType::ValueType, typename BType::ValueType> &&
             std::is_same_v<typename AType::ValueType, typename CType::ValueType>)
// clang-format off
APIARY_EXPOSE
APIARY_MODULE("graph")
    // clang-format on
    void grouped_batched_gemm_blocked(double alpha, std::vector<AType const *> a_list, std::vector<BType const *> b_list, double beta,
                                      std::vector<CType *> const &c_bases, std::vector<size_t> const &c_offsets, bool trans_a = false,
                                      bool trans_b = false) {
    using T = typename AType::ValueType;

    size_t const count = a_list.size();
    if (count == 0) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "cg::grouped_batched_gemm_blocked: batch is empty");
    }
    if (b_list.size() != count || c_bases.size() != count || c_offsets.size() != count) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                "cg::grouped_batched_gemm_blocked: A, B, the base list and the offset list must be the same length; got "
                                "{}, {}, {}, {}",
                                count, b_list.size(), c_bases.size(), c_offsets.size());
    }

    std::vector<detail::GemmShapeKey> keys(count);
    for (size_t i = 0; i < count; ++i) {
        if (a_list[i] == nullptr || b_list[i] == nullptr || c_bases[i] == nullptr) {
            EINSUMS_THROW_EXCEPTION(std::invalid_argument, "cg::grouped_batched_gemm_blocked: member {} has a null operand", i);
        }
        if (detail::tensor_rank(*a_list[i]) != 2 || detail::tensor_rank(*b_list[i]) != 2 || detail::tensor_rank(*c_bases[i]) != 2) {
            EINSUMS_THROW_EXCEPTION(rank_error, "cg::grouped_batched_gemm_blocked: member {} is not rank 2 (got {}, {}, {})", i,
                                    detail::tensor_rank(*a_list[i]), detail::tensor_rank(*b_list[i]), detail::tensor_rank(*c_bases[i]));
        }

        auto const m = static_cast<size_t>(trans_a ? a_list[i]->dim(1) : a_list[i]->dim(0));
        auto const n = static_cast<size_t>(trans_b ? b_list[i]->dim(0) : b_list[i]->dim(1));
        auto const k = static_cast<size_t>(trans_a ? a_list[i]->dim(0) : a_list[i]->dim(1));
        if (static_cast<size_t>(trans_b ? b_list[i]->dim(1) : b_list[i]->dim(0)) != k) {
            EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                    "cg::grouped_batched_gemm_blocked: member {} disagrees with itself on the link dimension between A "
                                    "and B",
                                    i);
        }

        size_t const ldc  = c_bases[i]->impl().get_lda();
        size_t const span = c_bases[i]->size();
        // The executor only ever sees a raw pointer, where an offset past the
        // end is silent corruption rather than a diagnosable error.
        if (m > ldc) {
            EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                    "cg::grouped_batched_gemm_blocked: block {} has {} rows but its base's leading dimension is {}; a "
                                    "block taller than the parent's column stride would overlap the next column",
                                    i, m, ldc);
        }
        size_t const last = c_offsets[i] + (n - 1) * ldc + m;
        if (last > span) {
            EINSUMS_THROW_EXCEPTION(std::out_of_range,
                                    "cg::grouped_batched_gemm_blocked: block {} at offset {} spans {} elements of a {}-element base", i,
                                    c_offsets[i], last - c_offsets[i], span);
        }

        keys[i] = {.m   = static_cast<int>(m),
                   .n   = static_cast<int>(n),
                   .k   = static_cast<int>(k),
                   .lda = static_cast<int>(a_list[i]->impl().get_lda()),
                   .ldb = static_cast<int>(b_list[i]->impl().get_lda()),
                   .ldc = static_cast<int>(ldc)};
    }

    // Two blocks of one base that overlap race, the same hazard the list form
    // rejects by comparing destination tensors. Checked exactly for the
    // column-aligned case - an offset that is a whole number of columns, which
    // is what a column range of a column-major tensor is - and skipped
    // otherwise rather than approximated, because an interval test over a
    // strided block over-reports and would reject legitimate interleaving.
    {
        std::map<CType const *, std::vector<std::pair<size_t, size_t>>> columns;
        bool                                                            aligned = true;
        for (size_t i = 0; i < count && aligned; ++i) {
            size_t const ldc = static_cast<size_t>(keys[i].ldc);
            if (ldc == 0 || c_offsets[i] % ldc != 0) {
                aligned = false;
                break;
            }
            size_t const first = c_offsets[i] / ldc;
            columns[c_bases[i]].emplace_back(first, first + static_cast<size_t>(keys[i].n));
        }
        if (aligned) {
            for (auto &[base, spans] : columns) {
                std::sort(spans.begin(), spans.end());
                for (size_t i = 1; i < spans.size(); ++i) {
                    if (spans[i].first < spans[i - 1].second) {
                        EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                                "cg::grouped_batched_gemm_blocked: two blocks of one base overlap (columns [{}, {}) and "
                                                "[{}, {})). The batch gives no ordering between members, so they would race",
                                                spans[i - 1].first, spans[i - 1].second, spans[i].first, spans[i].second);
                    }
                }
            }
        }
    }

    std::map<detail::GemmShapeKey, size_t> index_of;
    std::vector<detail::GemmShapeKey>      order;
    std::vector<std::vector<size_t>>       members;
    for (size_t i = 0; i < count; ++i) {
        auto const [it, inserted] = index_of.try_emplace(keys[i], order.size());
        if (inserted) {
            order.push_back(keys[i]);
            members.emplace_back();
        }
        members[it->second].push_back(i);
    }

    GroupedBatchedGemmDescriptor d;
    d.total = static_cast<int>(count);
    if constexpr (std::is_same_v<T, float>) {
        d.scalar = BlasScalar::Float;
    } else if constexpr (std::is_same_v<T, double>) {
        d.scalar = BlasScalar::Double;
    } else if constexpr (std::is_same_v<T, std::complex<float>>) {
        d.scalar = BlasScalar::ComplexFloat;
    } else {
        d.scalar = BlasScalar::ComplexDouble;
    }
    d.groups.reserve(order.size());
    d.labels.reserve(order.size());

    int first = 0;
    for (size_t g = 0; g < order.size(); ++g) {
        auto const &key = order[g];
        d.groups.push_back(GemmGroup{.m       = key.m,
                                     .n       = key.n,
                                     .k       = key.k,
                                     .lda     = key.lda,
                                     .ldb     = key.ldb,
                                     .ldc     = key.ldc,
                                     .trans_a = static_cast<char>(trans_a ? 'T' : 'N'),
                                     .trans_b = static_cast<char>(trans_b ? 'T' : 'N'),
                                     .alpha   = std::complex<double>{alpha, 0.0},
                                     .beta    = std::complex<double>{beta, 0.0},
                                     .count   = static_cast<int>(members[g].size()),
                                     .first   = first});
        d.labels.push_back(fmt::format("gemm {}x{}x{} trans={}{} x{} into blocks", key.m, key.k, key.n, trans_a ? 'T' : 'N',
                                       trans_b ? 'T' : 'N', members[g].size()));
        first += static_cast<int>(members[g].size());
    }

    std::vector<size_t> flat;
    flat.reserve(count);
    for (auto const &m : members) {
        flat.insert(flat.end(), m.begin(), m.end());
    }

    auto &ctx = CaptureContext::current();
    if (!ctx.is_capturing()) {
        LabeledSection("grouped_batched_gemm_blocked eager");
        std::vector<void const *> a_vs(count), b_vs(count);
        std::vector<void *>       c_vs(count);
        for (size_t i = 0; i < count; ++i) {
            size_t const s = flat[i];
            a_vs[i]        = static_cast<void const *>(a_list[s]->data());
            b_vs[i]        = static_cast<void const *>(b_list[s]->data());
            c_vs[i]        = static_cast<void *>(c_bases[s]->data() + c_offsets[s]);
        }
        detail::run_grouped_batched_gemm<T>(d, a_vs, b_vs, c_vs);
        return;
    }

    LabeledSection("grouped_batched_gemm_blocked capture");
    detail::BatchedGemmExtractors       a_exs, b_exs;
    detail::BatchedGemmOutputExtractors c_exs;
    a_exs.reserve(count);
    b_exs.reserve(count);
    c_exs.reserve(count);
    std::vector<TensorId> inputs;
    inputs.reserve(2 * count);

    // One slot per DISTINCT base, against one per member in the list form.
    std::map<CType *, std::pair<TensorId, TensorSlot *>> base_slots;
    std::vector<TensorId>                                outputs;
    for (size_t i = 0; i < count; ++i) {
        size_t const s = flat[i];
        if (!base_slots.contains(c_bases[s])) {
            auto const entry = ctx.get_slot(*c_bases[s]);
            base_slots.emplace(c_bases[s], entry);
            outputs.push_back(entry.first);
        }
    }

    for (size_t i = 0; i < count; ++i) {
        size_t const s      = flat[i];
        auto [a_id, a_slot] = ctx.get_slot(*a_list[s]);
        auto [b_id, b_slot] = ctx.get_slot(*b_list[s]);
        inputs.push_back(a_id);
        inputs.push_back(b_id);
        a_exs.emplace_back([a_slot]() -> std::pair<void const *, int> {
            auto const *t = static_cast<AType const *>(a_slot->ptr);
            return {static_cast<void const *>(t->data()), static_cast<int>(t->impl().get_lda())};
        });
        b_exs.emplace_back([b_slot]() -> std::pair<void const *, int> {
            auto const *t = static_cast<BType const *>(b_slot->ptr);
            return {static_cast<void const *>(t->data()), static_cast<int>(t->impl().get_lda())};
        });
        // Read the base through its slot and offset at execute time: rebind()
        // and the MemoryPlanning arena can both move the storage after capture.
        auto const  *c_slot = base_slots.at(c_bases[s]).second;
        size_t const off    = c_offsets[s];
        c_exs.emplace_back([c_slot, off]() -> std::pair<void *, int> {
            auto *t = static_cast<CType *>(c_slot->ptr);
            return {static_cast<void *>(t->data() + off), static_cast<int>(t->impl().get_lda())};
        });
    }

    // beta != 0 means every destination is read before it is written, so the
    // RAW edge from whoever produced each base must survive (bug-1009).
    if (beta != 0.0) {
        inputs.insert(inputs.end(), outputs.begin(), outputs.end());
    }

    auto executor = detail::make_grouped_batched_gemm_executor(d, std::move(a_exs), std::move(b_exs), std::move(c_exs));
    ctx.record(OpKind::GroupedBatchedGemm,
               fmt::format("gemm_batch_grouped x{} in {} shapes into blocks of {} (trans={}{})", d.total, d.groups.size(), outputs.size(),
                           trans_a ? 'T' : 'N', trans_b ? 'T' : 'N'),
               std::move(inputs), std::move(outputs), std::move(executor), std::move(d));
}

// ─────────────────────────────────────────────────────────────────────────────
// grouped_dot / grouped_axpby: a run of scalar-sized operations as ONE node
// ─────────────────────────────────────────────────────────────────────────────

namespace detail {

/// Per-entry dims of a runtime- or statically-ranked tensor, for the shape
/// agreement checks the grouped forms make once at capture.
template <typename TensorType>
std::vector<size_t> tensor_dims(TensorType const &t) {
    size_t const        r = tensor_rank(t);
    std::vector<size_t> dims(r);
    for (size_t d = 0; d < r; d++) {
        dims[d] = t.dim(d);
    }
    return dims;
}

} // namespace detail

/// @brief One node holding many independent dot products:
/// ``results[i]->data()[0] = sum(A_i * B_i)`` for every entry.
///
/// The entries run SEQUENTIALLY inside the node, in the order they were passed,
/// each through the same `linear_algebra::dot` a single @ref dot_python call
/// makes. That is the point of the operation and not an implementation detail:
/// this exists for workloads whose gate is bit-identity, so the result has to be
/// the same bits as the loop of single calls it replaces, entry by entry. A
/// parallel-within-node form would reduce in a different order on nothing but
/// its own schedule, and it would have nothing to gain - a dot into a scalar has
/// no arithmetic to spread. What is saved is the per-node dispatch, which at the
/// block sizes local-correlation methods produce is the whole cost: a DLPNO-CCSD
/// iteration reached ~1,700 of these, one node each.
///
/// Each entry's own reduction runs at VENDOR WIDTH ONE, under
/// @ref blas::SerialVendorScope, and that is part of the contract rather than a
/// tuning choice. A threaded dot sums per-thread partials, so its last bits are
/// a function of how many threads the caller happened to present - and this node
/// is reached at whatever width its context has: eagerly from the main thread,
/// from an executor worker, from inside a node-scoped width. Pinning the width
/// makes every entry a function of its operands alone, so the same inputs give
/// the same bits whatever is running around them. The entries may be dispatched
/// from anywhere; it is each entry's INTERNAL reduction that is fixed, and the
/// entries themselves are already ordered by the caller's list.
///
/// Because the entries are sequential, repeating a destination is well defined
/// and means what the loop means - the last entry writing it wins. Unlike
/// @ref grouped_batched_gemm there is therefore nothing to reject.
///
/// Outside capture this executes immediately, so the same call works eagerly.
///
/// @param results One length-1-or-larger destination per entry; only element 0
///                is written, matching the single form.
/// @param a_list,b_list The operands, same length as @p results. Each entry's A
///                and B must agree on rank and shape, as `dot` requires; the
///                entries need not agree with each other.
template <CoreBasicTensorConcept ResultType, CoreBasicTensorConcept AType, CoreBasicTensorConcept BType>
    requires requires {
        requires std::is_same_v<typename ResultType::ValueType, typename AType::ValueType>;
        requires std::is_same_v<typename AType::ValueType, typename BType::ValueType>;
    }
// clang-format off
APIARY_EXPOSE
APIARY_MODULE("linalg")
// The same 8 owning/view combinations per dtype that the single `dot` carries,
// and for the same reason: a destination scalar is an owning tensor while the
// operands are slices of a larger store.
//
// float
APIARY_INSTANTIATE_AS("grouped_dot", einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::GeneralRuntimeTensor<float, std::allocator<float>>)
APIARY_INSTANTIATE_AS("grouped_dot", einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::RuntimeTensorView<float>)
APIARY_INSTANTIATE_AS("grouped_dot", einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::RuntimeTensorView<float>, einsums::GeneralRuntimeTensor<float, std::allocator<float>>)
APIARY_INSTANTIATE_AS("grouped_dot", einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::RuntimeTensorView<float>, einsums::RuntimeTensorView<float>)
APIARY_INSTANTIATE_AS("grouped_dot", einsums::RuntimeTensorView<float>, einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::GeneralRuntimeTensor<float, std::allocator<float>>)
APIARY_INSTANTIATE_AS("grouped_dot", einsums::RuntimeTensorView<float>, einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::RuntimeTensorView<float>)
APIARY_INSTANTIATE_AS("grouped_dot", einsums::RuntimeTensorView<float>, einsums::RuntimeTensorView<float>, einsums::GeneralRuntimeTensor<float, std::allocator<float>>)
APIARY_INSTANTIATE_AS("grouped_dot", einsums::RuntimeTensorView<float>, einsums::RuntimeTensorView<float>, einsums::RuntimeTensorView<float>)
//
// double
APIARY_INSTANTIATE_AS("grouped_dot", einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::GeneralRuntimeTensor<double, std::allocator<double>>)
APIARY_INSTANTIATE_AS("grouped_dot", einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::RuntimeTensorView<double>)
APIARY_INSTANTIATE_AS("grouped_dot", einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::RuntimeTensorView<double>, einsums::GeneralRuntimeTensor<double, std::allocator<double>>)
APIARY_INSTANTIATE_AS("grouped_dot", einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::RuntimeTensorView<double>, einsums::RuntimeTensorView<double>)
APIARY_INSTANTIATE_AS("grouped_dot", einsums::RuntimeTensorView<double>, einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::GeneralRuntimeTensor<double, std::allocator<double>>)
APIARY_INSTANTIATE_AS("grouped_dot", einsums::RuntimeTensorView<double>, einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::RuntimeTensorView<double>)
APIARY_INSTANTIATE_AS("grouped_dot", einsums::RuntimeTensorView<double>, einsums::RuntimeTensorView<double>, einsums::GeneralRuntimeTensor<double, std::allocator<double>>)
APIARY_INSTANTIATE_AS("grouped_dot", einsums::RuntimeTensorView<double>, einsums::RuntimeTensorView<double>, einsums::RuntimeTensorView<double>)
//
// std::complex<float>
APIARY_INSTANTIATE_AS("grouped_dot", einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>)
APIARY_INSTANTIATE_AS("grouped_dot", einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::RuntimeTensorView<std::complex<float>>)
APIARY_INSTANTIATE_AS("grouped_dot", einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::RuntimeTensorView<std::complex<float>>, einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>)
APIARY_INSTANTIATE_AS("grouped_dot", einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::RuntimeTensorView<std::complex<float>>, einsums::RuntimeTensorView<std::complex<float>>)
APIARY_INSTANTIATE_AS("grouped_dot", einsums::RuntimeTensorView<std::complex<float>>, einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>)
APIARY_INSTANTIATE_AS("grouped_dot", einsums::RuntimeTensorView<std::complex<float>>, einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::RuntimeTensorView<std::complex<float>>)
APIARY_INSTANTIATE_AS("grouped_dot", einsums::RuntimeTensorView<std::complex<float>>, einsums::RuntimeTensorView<std::complex<float>>, einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>)
APIARY_INSTANTIATE_AS("grouped_dot", einsums::RuntimeTensorView<std::complex<float>>, einsums::RuntimeTensorView<std::complex<float>>, einsums::RuntimeTensorView<std::complex<float>>)
//
// std::complex<double>
APIARY_INSTANTIATE_AS("grouped_dot", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
APIARY_INSTANTIATE_AS("grouped_dot", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::RuntimeTensorView<std::complex<double>>)
APIARY_INSTANTIATE_AS("grouped_dot", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::RuntimeTensorView<std::complex<double>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
APIARY_INSTANTIATE_AS("grouped_dot", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::RuntimeTensorView<std::complex<double>>, einsums::RuntimeTensorView<std::complex<double>>)
APIARY_INSTANTIATE_AS("grouped_dot", einsums::RuntimeTensorView<std::complex<double>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
APIARY_INSTANTIATE_AS("grouped_dot", einsums::RuntimeTensorView<std::complex<double>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::RuntimeTensorView<std::complex<double>>)
APIARY_INSTANTIATE_AS("grouped_dot", einsums::RuntimeTensorView<std::complex<double>>, einsums::RuntimeTensorView<std::complex<double>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
APIARY_INSTANTIATE_AS("grouped_dot", einsums::RuntimeTensorView<std::complex<double>>, einsums::RuntimeTensorView<std::complex<double>>, einsums::RuntimeTensorView<std::complex<double>>)
    // clang-format on
    void grouped_dot(std::vector<ResultType *> results, std::vector<AType const *> a_list, std::vector<BType const *> b_list) {
    size_t const count = results.size();
    if (count == 0) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "cg::grouped_dot: the run is empty");
    }
    if (a_list.size() != count || b_list.size() != count) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "cg::grouped_dot: the result, A and B lists must be the same length; got {}, {}, {}",
                                count, a_list.size(), b_list.size());
    }

    // Every check is internal to one entry: there is nothing to compare across
    // entries, which is exactly what the grouped form buys.
    for (size_t i = 0; i < count; i++) {
        if (results[i] == nullptr || a_list[i] == nullptr || b_list[i] == nullptr) {
            EINSUMS_THROW_EXCEPTION(std::invalid_argument, "cg::grouped_dot: entry {} has a null operand", i);
        }
        if (results[i]->size() < 1) {
            EINSUMS_THROW_EXCEPTION(std::invalid_argument, "cg::grouped_dot: entry {}'s result tensor must have at least one element", i);
        }
        if (detail::tensor_rank(*a_list[i]) != detail::tensor_rank(*b_list[i])) {
            EINSUMS_THROW_EXCEPTION(rank_error, "cg::grouped_dot: entry {}'s operands disagree on rank ({} and {})", i,
                                    detail::tensor_rank(*a_list[i]), detail::tensor_rank(*b_list[i]));
        }
        if (detail::tensor_dims(*a_list[i]) != detail::tensor_dims(*b_list[i])) {
            EINSUMS_THROW_EXCEPTION(dimension_error, "cg::grouped_dot: entry {}'s operands disagree on shape", i);
        }
    }

    auto &ctx = CaptureContext::current();
    if (!ctx.is_capturing()) {
        LabeledSection("grouped_dot eager");
        blas::SerialVendorScope const serial;
        for (size_t i = 0; i < count; i++) {
            results[i]->data()[0] = linear_algebra::dot(*a_list[i], *b_list[i]);
        }
        return;
    }

    LabeledSection("grouped_dot capture");
    // Inputs interleaved A_0, B_0, A_1, B_1, ... and outputs in entry order, so
    // entry i indexes both. The same convention the batched nodes use.
    std::vector<TensorId>     inputs, outputs;
    std::vector<TensorSlot *> r_slots, a_slots, b_slots;
    inputs.reserve(2 * count);
    outputs.reserve(count);
    r_slots.reserve(count);
    a_slots.reserve(count);
    b_slots.reserve(count);
    for (size_t i = 0; i < count; i++) {
        auto [a_id, a_slot] = ctx.get_slot(*a_list[i]);
        auto [b_id, b_slot] = ctx.get_slot(*b_list[i]);
        auto [r_id, r_slot] = ctx.get_slot(*results[i]);
        inputs.push_back(a_id);
        inputs.push_back(b_id);
        outputs.push_back(r_id);
        // Read through the slot, not the captured pointer: rebind() and the
        // MemoryPlanning arena can both move a tensor's storage.
        a_slots.push_back(a_slot);
        b_slots.push_back(b_slot);
        r_slots.push_back(r_slot);
    }

    auto executor = [r_slots = std::move(r_slots), a_slots = std::move(a_slots), b_slots = std::move(b_slots)]() {
        LabeledSection("grouped_dot execute");
        blas::SerialVendorScope const serial;
        for (size_t i = 0; i < r_slots.size(); i++) {
            static_cast<ResultType *>(r_slots[i]->ptr)->data()[0] =
                linear_algebra::dot(*static_cast<AType const *>(a_slots[i]->ptr), *static_cast<BType const *>(b_slots[i]->ptr));
        }
    };

    GroupedDotDescriptor d;
    d.total = static_cast<int>(count);
    ctx.record(OpKind::GroupedDot, fmt::format("dot x{}", count), std::move(inputs), std::move(outputs), std::move(executor), std::move(d));
}

/// @brief One node holding many independent AXPBYs:
/// ``Y_i = alphas[i] * X_i + betas[i] * Y_i`` for every entry.
///
/// The grouped counterpart of @ref axpby, and the accumulating half of the
/// pattern @ref grouped_dot reduces into: a local-correlation residual computes
/// a scalar per pair or per neighbour and then adds it, scaled, into one element
/// of a shared matrix, and both halves used to cost a node apiece.
///
/// Entries run SEQUENTIALLY inside the node, in the order they were passed. That
/// is load-bearing twice over. It makes each entry the same bits the single call
/// would have written, and it makes REPEATED DESTINATIONS legal: a family whose
/// entries all accumulate into one element - a sum over a pair's neighbours, say
/// - keeps its term order, because the entries are applied in the order the
/// caller wrote them and no two run at once. So unlike
/// @ref grouped_batched_gemm, which rejects a shared destination because its
/// members may run concurrently, this operation accepts one and defines it.
///
/// The prefactors are per entry, which is the difference from @ref axpby that
/// makes merging worthwhile: the accumulations a residual wants to merge
/// disagree on their coefficients.
///
/// Outside capture this executes immediately, so the same call works eagerly.
///
/// @param alphas Per-entry prefactor on X_i.
/// @param x_list Sources, same length as @p alphas.
/// @param betas  Per-entry prefactor on Y_i. A non-zero entry means that Y_i is
///               read as well as written, which the node records as a dependency.
/// @param y_list Destinations, same length as @p alphas. May repeat.
template <CoreBasicTensorConcept XType, CoreBasicTensorConcept YType>
    requires SameUnderlying<XType, YType>
// clang-format off
APIARY_EXPOSE
APIARY_MODULE("linalg")
// The same 4 owning/view combinations per dtype that the single `axpby` carries.
//
// float
APIARY_INSTANTIATE_AS("grouped_axpby", einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::GeneralRuntimeTensor<float, std::allocator<float>>)
APIARY_INSTANTIATE_AS("grouped_axpby", einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::RuntimeTensorView<float>)
APIARY_INSTANTIATE_AS("grouped_axpby", einsums::RuntimeTensorView<float>,                          einsums::GeneralRuntimeTensor<float, std::allocator<float>>)
APIARY_INSTANTIATE_AS("grouped_axpby", einsums::RuntimeTensorView<float>,                          einsums::RuntimeTensorView<float>)
// double
APIARY_INSTANTIATE_AS("grouped_axpby", einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::GeneralRuntimeTensor<double, std::allocator<double>>)
APIARY_INSTANTIATE_AS("grouped_axpby", einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::RuntimeTensorView<double>)
APIARY_INSTANTIATE_AS("grouped_axpby", einsums::RuntimeTensorView<double>,                          einsums::GeneralRuntimeTensor<double, std::allocator<double>>)
APIARY_INSTANTIATE_AS("grouped_axpby", einsums::RuntimeTensorView<double>,                          einsums::RuntimeTensorView<double>)
// complex<float>
APIARY_INSTANTIATE_AS("grouped_axpby", einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>)
APIARY_INSTANTIATE_AS("grouped_axpby", einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::RuntimeTensorView<std::complex<float>>)
APIARY_INSTANTIATE_AS("grouped_axpby", einsums::RuntimeTensorView<std::complex<float>>,                                        einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>)
APIARY_INSTANTIATE_AS("grouped_axpby", einsums::RuntimeTensorView<std::complex<float>>,                                        einsums::RuntimeTensorView<std::complex<float>>)
// complex<double>
APIARY_INSTANTIATE_AS("grouped_axpby", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
APIARY_INSTANTIATE_AS("grouped_axpby", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::RuntimeTensorView<std::complex<double>>)
APIARY_INSTANTIATE_AS("grouped_axpby", einsums::RuntimeTensorView<std::complex<double>>,                                         einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
APIARY_INSTANTIATE_AS("grouped_axpby", einsums::RuntimeTensorView<std::complex<double>>,                                         einsums::RuntimeTensorView<std::complex<double>>)
    // clang-format on
    void grouped_axpby(std::vector<double> alphas, std::vector<XType const *> x_list, std::vector<double> betas,
                       std::vector<YType *> y_list) {
    using T = typename XType::ValueType;

    size_t const count = alphas.size();
    if (count == 0) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "cg::grouped_axpby: the run is empty");
    }
    if (x_list.size() != count || betas.size() != count || y_list.size() != count) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                "cg::grouped_axpby: the alpha, X, beta and Y lists must be the same length; got {}, {}, {}, {}", count,
                                x_list.size(), betas.size(), y_list.size());
    }

    for (size_t i = 0; i < count; i++) {
        if (x_list[i] == nullptr || y_list[i] == nullptr) {
            EINSUMS_THROW_EXCEPTION(std::invalid_argument, "cg::grouped_axpby: entry {} has a null operand", i);
        }
        if (detail::tensor_rank(*x_list[i]) != detail::tensor_rank(*y_list[i])) {
            EINSUMS_THROW_EXCEPTION(rank_error, "cg::grouped_axpby: entry {}'s operands disagree on rank ({} and {})", i,
                                    detail::tensor_rank(*x_list[i]), detail::tensor_rank(*y_list[i]));
        }
        if (detail::tensor_dims(*x_list[i]) != detail::tensor_dims(*y_list[i])) {
            EINSUMS_THROW_EXCEPTION(dimension_error, "cg::grouped_axpby: entry {}'s operands disagree on shape", i);
        }
    }

    std::vector<T> a_typed(count), b_typed(count);
    for (size_t i = 0; i < count; i++) {
        a_typed[i] = static_cast<T>(alphas[i]);
        b_typed[i] = static_cast<T>(betas[i]);
    }

    auto &ctx = CaptureContext::current();
    if (!ctx.is_capturing()) {
        LabeledSection("grouped_axpby eager");
        for (size_t i = 0; i < count; i++) {
            linear_algebra::axpby(a_typed[i], *x_list[i], b_typed[i], y_list[i]);
        }
        return;
    }

    LabeledSection("grouped_axpby capture");
    std::vector<TensorId>     inputs, outputs;
    std::vector<TensorSlot *> x_slots, y_slots;
    inputs.reserve(2 * count);
    outputs.reserve(count);
    x_slots.reserve(count);
    y_slots.reserve(count);
    for (size_t i = 0; i < count; i++) {
        auto [x_id, x_slot] = ctx.get_slot(*x_list[i]);
        auto [y_id, y_slot] = ctx.get_slot(*y_list[i]);
        inputs.push_back(x_id);
        // beta != 0 means this entry reads its destination before writing it, so
        // the RAW edge from whoever produced Y must survive (bug-1009).
        if (b_typed[i] != T{0}) {
            inputs.push_back(y_id);
        }
        outputs.push_back(y_id);
        x_slots.push_back(x_slot);
        y_slots.push_back(y_slot);
    }

    auto executor = [a_typed, b_typed, x_slots = std::move(x_slots), y_slots = std::move(y_slots)]() {
        LabeledSection("grouped_axpby execute");
        for (size_t i = 0; i < x_slots.size(); i++) {
            linear_algebra::axpby(a_typed[i], *static_cast<XType const *>(x_slots[i]->ptr), b_typed[i],
                                  static_cast<YType *>(y_slots[i]->ptr));
        }
    };

    GroupedAxpbyDescriptor d;
    d.total = static_cast<int>(count);
    d.alphas.reserve(count);
    d.betas.reserve(count);
    for (size_t i = 0; i < count; i++) {
        d.alphas.emplace_back(a_typed[i]);
        d.betas.emplace_back(b_typed[i]);
    }
    ctx.record(OpKind::GroupedAxpby, fmt::format("axpby x{}", count), std::move(inputs), std::move(outputs), std::move(executor),
               std::move(d));
}

// ─────────────────────────────────────────────────────────────────────────────
// grouped_permute / grouped_direct_product / grouped_direct_division:
// many independent ELEMENT-WISE members as ONE node
// ─────────────────────────────────────────────────────────────────────────────

namespace detail {

/// Reject a run in which two members would write the same destination tensor.
///
/// These forms thread over their members, so a shared destination is a data
/// race with no ordering to fall back on - the contract
/// @ref grouped_batched_gemm states, for the same reason. Like that check this
/// compares the operand handles, so two DISTINCT views of one buffer pass it;
/// splitting a tensor across members is the caller's promise either way.
template <typename CType>
void require_distinct_destinations(std::vector<CType *> const &c_list, char const *who) {
    std::vector<CType const *> seen(c_list.begin(), c_list.end());
    std::sort(seen.begin(), seen.end());
    if (std::adjacent_find(seen.begin(), seen.end()) != seen.end()) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                "{}: two members share a destination tensor. Members run concurrently, so they would "
                                "race; split them into separate calls",
                                who);
    }
}

/// Run @p member over every index as one OpenMP team, carrying the first
/// exception out by hand: one may not cross a region boundary.
template <typename F>
void run_grouped_members(size_t count, F &&member) {
    // A run of one IS the call the grouped form replaces, and forking a team
    // for it costs more than the member does. Worth the branch because a gated
    // capture is full of them: a conditional over one entity still wants the
    // grouped spelling, so that the ungated capture beside it can be the same
    // emitter with a longer list.
    if (count == 1) {
        member(size_t{0});
        return;
    }
    std::exception_ptr first;
    EINSUMS_OMP_PRAGMA(parallel for schedule(dynamic))
    for (size_t i = 0; i < count; i++) {
        try {
            member(i);
        } catch (...) {
            EINSUMS_OMP_PRAGMA(critical(grouped_elementwise_failure))
            if (!first) {
                first = std::current_exception();
            }
        }
    }
    if (first) {
        std::rethrow_exception(first);
    }
}

} // namespace detail

/// @brief One node holding many independent permutes:
/// ``C_i = c_pfs[i] * C_i + a_pfs[i] * permute(A_i)`` for every member, all of
/// them under one spec.
///
/// The grouped counterpart of @ref string_permute, for the shape a
/// local-correlation phase produces: one small reorder per pair or per triplet,
/// hundreds of them, every one a node of its own. The arithmetic is unchanged
/// and the dispatch is paid once.
///
/// **Members run CONCURRENTLY and that costs nothing in reproducibility.** A
/// permute is element-wise - every output element is one input element scaled
/// and added to what was there - so no member's result depends on how the run
/// was divided, and each member writes only its own destination. The last bit
/// of every member is therefore the last bit the single call would have
/// written, whatever the schedule. That is why this one threads where
/// @ref grouped_dot and @ref grouped_axpby do not: theirs is a REDUCTION whose
/// order a schedule would decide.
///
/// One spec for the whole run, because a batch drawn at one emission site has
/// one. Per-member prefactors, because that is the difference between members
/// worth merging: a term-by-term accumulation disagrees on its coefficients and
/// on whether it is the first term (``c_pf`` zero) or a later one.
///
/// Destinations must be distinct, for the concurrency above.
///
/// Outside capture this executes immediately, so the same call works eagerly.
///
/// @param spec   The permutation, e.g. ``"abc <- acb"``, shared by every member.
/// @param c_list Destinations, one per member. Must be distinct.
/// @param a_list Sources, same length as @p c_list.
/// @param c_pfs  Per-member prefactor on the destination. A non-zero entry
///               means the member reads C as well as writing it, which the node
///               records as a dependency.
/// @param a_pfs  Per-member prefactor on the source.
template <CoreBasicTensorConcept AType, CoreBasicTensorConcept CType>
    requires std::is_same_v<typename AType::ValueType, typename CType::ValueType>
// clang-format off
APIARY_EXPOSE
APIARY_MODULE("linalg")
// The 4 owning/view combinations per dtype: a per-entity block is an owning
// store and a slice of a padded one is a view, and a run may mix them.
//
// float
APIARY_INSTANTIATE_AS("grouped_permute", einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::GeneralRuntimeTensor<float, std::allocator<float>>)
APIARY_INSTANTIATE_AS("grouped_permute", einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::RuntimeTensorView<float>)
APIARY_INSTANTIATE_AS("grouped_permute", einsums::RuntimeTensorView<float>,                          einsums::GeneralRuntimeTensor<float, std::allocator<float>>)
APIARY_INSTANTIATE_AS("grouped_permute", einsums::RuntimeTensorView<float>,                          einsums::RuntimeTensorView<float>)
// double
APIARY_INSTANTIATE_AS("grouped_permute", einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::GeneralRuntimeTensor<double, std::allocator<double>>)
APIARY_INSTANTIATE_AS("grouped_permute", einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::RuntimeTensorView<double>)
APIARY_INSTANTIATE_AS("grouped_permute", einsums::RuntimeTensorView<double>,                           einsums::GeneralRuntimeTensor<double, std::allocator<double>>)
APIARY_INSTANTIATE_AS("grouped_permute", einsums::RuntimeTensorView<double>,                           einsums::RuntimeTensorView<double>)
// complex<float>
APIARY_INSTANTIATE_AS("grouped_permute", einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>)
APIARY_INSTANTIATE_AS("grouped_permute", einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::RuntimeTensorView<std::complex<float>>)
APIARY_INSTANTIATE_AS("grouped_permute", einsums::RuntimeTensorView<std::complex<float>>,                                        einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>)
APIARY_INSTANTIATE_AS("grouped_permute", einsums::RuntimeTensorView<std::complex<float>>,                                        einsums::RuntimeTensorView<std::complex<float>>)
// complex<double>
APIARY_INSTANTIATE_AS("grouped_permute", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
APIARY_INSTANTIATE_AS("grouped_permute", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::RuntimeTensorView<std::complex<double>>)
APIARY_INSTANTIATE_AS("grouped_permute", einsums::RuntimeTensorView<std::complex<double>>,                                         einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
APIARY_INSTANTIATE_AS("grouped_permute", einsums::RuntimeTensorView<std::complex<double>>,                                         einsums::RuntimeTensorView<std::complex<double>>)
    // clang-format on
    void grouped_permute(std::string const &spec, std::vector<CType *> c_list, std::vector<AType const *> a_list, std::vector<double> c_pfs,
                         std::vector<double> a_pfs) {
    using T = typename CType::ValueType;

    size_t const count = c_list.size();
    if (count == 0) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "cg::grouped_permute: the run is empty");
    }
    if (a_list.size() != count || c_pfs.size() != count || a_pfs.size() != count) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                "cg::grouped_permute: the C, A, c_pf and a_pf lists must be the same length; got {}, {}, {}, {}", count,
                                a_list.size(), c_pfs.size(), a_pfs.size());
    }

    auto parse_result = parse_permute_spec(std::string_view{spec});
    if (!parse_result) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "{}", parse_result.error().message);
    }
    auto parsed = parse_result.value();

    for (size_t i = 0; i < count; i++) {
        if (c_list[i] == nullptr || a_list[i] == nullptr) {
            EINSUMS_THROW_EXCEPTION(std::invalid_argument, "cg::grouped_permute: member {} has a null operand", i);
        }
        if (detail::tensor_rank(*a_list[i]) != parsed.a_indices.size() || detail::tensor_rank(*c_list[i]) != parsed.c_indices.size()) {
            EINSUMS_THROW_EXCEPTION(rank_error, "cg::grouped_permute: member {} has ranks ({}, {}) where the spec wants ({}, {})", i,
                                    detail::tensor_rank(*c_list[i]), detail::tensor_rank(*a_list[i]), parsed.c_indices.size(),
                                    parsed.a_indices.size());
        }
    }
    detail::require_distinct_destinations(c_list, "cg::grouped_permute");

    std::vector<T> c_typed(count), a_typed(count);
    for (size_t i = 0; i < count; i++) {
        c_typed[i] = static_cast<T>(c_pfs[i]);
        a_typed[i] = static_cast<T>(a_pfs[i]);
    }

    auto &ctx = CaptureContext::current();
    if (!ctx.is_capturing()) {
        LabeledSection("grouped_permute eager");
        detail::run_grouped_members(
            count, [&](size_t i) { dispatch::string_permute<AType, CType>(parsed, c_typed[i], c_list[i], a_typed[i], *a_list[i]); });
        return;
    }

    LabeledSection("grouped_permute capture");
    std::vector<TensorId>     inputs, outputs;
    std::vector<TensorSlot *> c_slots, a_slots;
    inputs.reserve(2 * count);
    outputs.reserve(count);
    c_slots.reserve(count);
    a_slots.reserve(count);
    for (size_t i = 0; i < count; i++) {
        auto [a_id, a_slot] = ctx.get_slot(*a_list[i]);
        auto [c_id, c_slot] = ctx.get_slot(*c_list[i]);
        inputs.push_back(a_id);
        // A non-zero c_pf reads the destination before writing it, so the RAW
        // edge from whoever produced C must survive.
        if (c_typed[i] != T{0}) {
            inputs.push_back(c_id);
        }
        outputs.push_back(c_id);
        a_slots.push_back(a_slot);
        c_slots.push_back(c_slot);
    }

    auto executor = [parsed, c_typed, a_typed, c_slots = std::move(c_slots), a_slots = std::move(a_slots)]() {
        LabeledSection("grouped_permute execute");
        detail::run_grouped_members(c_slots.size(), [&](size_t i) {
            dispatch::string_permute<AType, CType>(parsed, c_typed[i], static_cast<CType *>(c_slots[i]->ptr), a_typed[i],
                                                   *static_cast<AType const *>(a_slots[i]->ptr));
        });
    };

    GroupedElementwiseDescriptor d;
    d.total = static_cast<int>(count);
    d.alphas.reserve(count);
    d.betas.reserve(count);
    for (size_t i = 0; i < count; i++) {
        d.alphas.emplace_back(a_typed[i]);
        d.betas.emplace_back(c_typed[i]);
    }
    ctx.record(OpKind::GroupedPermute,
               fmt::format("permute x{}: C[{}] = A[{}]", count, fmt::join(parsed.c_indices, ","), fmt::join(parsed.a_indices, ",")),
               std::move(inputs), std::move(outputs), std::move(executor), std::move(d));
}

namespace detail {

/// The body @ref grouped_direct_product and @ref grouped_direct_division share:
/// a run of independent ``C_i = alpha_i * (A_i op B_i) + beta_i * C_i``
/// members, threaded, recorded as one node of @p kind.
///
/// @p kernel is the single-tensor entry point the member calls, which is what
/// makes a member's bits the bits the ungrouped emission wrote: the grouped
/// form adds a loop and nothing else.
template <typename T, typename AType, typename BType, typename CType, typename Kernel>
void grouped_binary_elementwise(char const *who, OpKind kind, char const *label, Kernel kernel, std::vector<T> const &alphas,
                                std::vector<AType const *> const &a_list, std::vector<BType const *> const &b_list,
                                std::vector<T> const &betas, std::vector<CType *> const &c_list) {
    size_t const count = c_list.size();
    if (count == 0) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "{}: the run is empty", who);
    }
    if (alphas.size() != count || a_list.size() != count || b_list.size() != count || betas.size() != count) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                "{}: the alpha, A, B, beta and C lists must be the same length; got {}, {}, {}, {}, {}", who, alphas.size(),
                                a_list.size(), b_list.size(), betas.size(), count);
    }

    for (size_t i = 0; i < count; i++) {
        if (a_list[i] == nullptr || b_list[i] == nullptr || c_list[i] == nullptr) {
            EINSUMS_THROW_EXCEPTION(std::invalid_argument, "{}: member {} has a null operand", who, i);
        }
        if (tensor_rank(*a_list[i]) != tensor_rank(*b_list[i]) || tensor_rank(*a_list[i]) != tensor_rank(*c_list[i])) {
            EINSUMS_THROW_EXCEPTION(rank_error, "{}: member {}'s operands disagree on rank ({}, {} and {})", who, i,
                                    tensor_rank(*a_list[i]), tensor_rank(*b_list[i]), tensor_rank(*c_list[i]));
        }
        if (tensor_dims(*a_list[i]) != tensor_dims(*b_list[i]) || tensor_dims(*a_list[i]) != tensor_dims(*c_list[i])) {
            EINSUMS_THROW_EXCEPTION(dimension_error, "{}: member {}'s operands disagree on shape", who, i);
        }
    }
    require_distinct_destinations(c_list, who);

    auto &ctx = CaptureContext::current();
    if (!ctx.is_capturing()) {
        run_grouped_members(count, [&](size_t i) { kernel(alphas[i], a_list[i], b_list[i], betas[i], c_list[i]); });
        return;
    }

    std::vector<TensorId>     inputs, outputs;
    std::vector<TensorSlot *> a_slots, b_slots, c_slots;
    inputs.reserve(3 * count);
    outputs.reserve(count);
    a_slots.reserve(count);
    b_slots.reserve(count);
    c_slots.reserve(count);
    for (size_t i = 0; i < count; i++) {
        auto [a_id, a_slot] = ctx.get_slot(*a_list[i]);
        auto [b_id, b_slot] = ctx.get_slot(*b_list[i]);
        auto [c_id, c_slot] = ctx.get_slot(*c_list[i]);
        inputs.push_back(a_id);
        inputs.push_back(b_id);
        // A non-zero beta reads the destination before writing it.
        if (betas[i] != T{0}) {
            inputs.push_back(c_id);
        }
        outputs.push_back(c_id);
        a_slots.push_back(a_slot);
        b_slots.push_back(b_slot);
        c_slots.push_back(c_slot);
    }

    auto executor = [kernel, alphas, betas, a_slots = std::move(a_slots), b_slots = std::move(b_slots), c_slots = std::move(c_slots)]() {
        run_grouped_members(c_slots.size(), [&](size_t i) {
            kernel(alphas[i], static_cast<AType const *>(a_slots[i]->ptr), static_cast<BType const *>(b_slots[i]->ptr), betas[i],
                   static_cast<CType *>(c_slots[i]->ptr));
        });
    };

    GroupedElementwiseDescriptor d;
    d.total = static_cast<int>(count);
    d.alphas.reserve(count);
    d.betas.reserve(count);
    for (size_t i = 0; i < count; i++) {
        d.alphas.emplace_back(alphas[i]);
        d.betas.emplace_back(betas[i]);
    }
    ctx.record(kind, fmt::format("{} x{}", label, count), std::move(inputs), std::move(outputs), std::move(executor), std::move(d));
}

} // namespace detail

/// @brief One node holding many independent element-wise products:
/// ``C_i = alphas[i] * (A_i ⊙ B_i) + betas[i] * C_i`` for every member.
///
/// The grouped counterpart of @ref direct_product, and the shape a
/// local-correlation residual reaches it in: one Hadamard product per entity
/// against that entity's own denominator or weight block, hundreds of them.
///
/// Members run CONCURRENTLY, which an element-wise kernel can afford: no
/// member's result depends on how the run was divided, so every member writes
/// the bits the single call would have written whatever the schedule. Each
/// member calls the same @ref linear_algebra::direct_product a single
/// @ref direct_product node calls, so the grouped form adds a loop and nothing
/// else. Destinations must be distinct.
///
/// Outside capture this executes immediately, so the same call works eagerly.
template <typename T, CoreBasicTensorConcept AType, CoreBasicTensorConcept BType, CoreBasicTensorConcept CType>
    requires requires {
        requires std::is_same_v<typename AType::ValueType, T>;
        requires std::is_same_v<typename BType::ValueType, T>;
        requires std::is_same_v<typename CType::ValueType, T>;
    }
// clang-format off
APIARY_EXPOSE
APIARY_MODULE("linalg")
// The same 8 owning/view combinations per dtype the single form carries.
//
// float
APIARY_INSTANTIATE_AS("grouped_direct_product", float, einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::GeneralRuntimeTensor<float, std::allocator<float>>)
APIARY_INSTANTIATE_AS("grouped_direct_product", float, einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::RuntimeTensorView<float>)
APIARY_INSTANTIATE_AS("grouped_direct_product", float, einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::RuntimeTensorView<float>,                          einsums::GeneralRuntimeTensor<float, std::allocator<float>>)
APIARY_INSTANTIATE_AS("grouped_direct_product", float, einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::RuntimeTensorView<float>,                          einsums::RuntimeTensorView<float>)
APIARY_INSTANTIATE_AS("grouped_direct_product", float, einsums::RuntimeTensorView<float>,                          einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::GeneralRuntimeTensor<float, std::allocator<float>>)
APIARY_INSTANTIATE_AS("grouped_direct_product", float, einsums::RuntimeTensorView<float>,                          einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::RuntimeTensorView<float>)
APIARY_INSTANTIATE_AS("grouped_direct_product", float, einsums::RuntimeTensorView<float>,                          einsums::RuntimeTensorView<float>,                          einsums::GeneralRuntimeTensor<float, std::allocator<float>>)
APIARY_INSTANTIATE_AS("grouped_direct_product", float, einsums::RuntimeTensorView<float>,                          einsums::RuntimeTensorView<float>,                          einsums::RuntimeTensorView<float>)
// double
APIARY_INSTANTIATE_AS("grouped_direct_product", double, einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::GeneralRuntimeTensor<double, std::allocator<double>>)
APIARY_INSTANTIATE_AS("grouped_direct_product", double, einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::RuntimeTensorView<double>)
APIARY_INSTANTIATE_AS("grouped_direct_product", double, einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::RuntimeTensorView<double>,                           einsums::GeneralRuntimeTensor<double, std::allocator<double>>)
APIARY_INSTANTIATE_AS("grouped_direct_product", double, einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::RuntimeTensorView<double>,                           einsums::RuntimeTensorView<double>)
APIARY_INSTANTIATE_AS("grouped_direct_product", double, einsums::RuntimeTensorView<double>,                           einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::GeneralRuntimeTensor<double, std::allocator<double>>)
APIARY_INSTANTIATE_AS("grouped_direct_product", double, einsums::RuntimeTensorView<double>,                           einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::RuntimeTensorView<double>)
APIARY_INSTANTIATE_AS("grouped_direct_product", double, einsums::RuntimeTensorView<double>,                           einsums::RuntimeTensorView<double>,                           einsums::GeneralRuntimeTensor<double, std::allocator<double>>)
APIARY_INSTANTIATE_AS("grouped_direct_product", double, einsums::RuntimeTensorView<double>,                           einsums::RuntimeTensorView<double>,                           einsums::RuntimeTensorView<double>)
// complex<float>
APIARY_INSTANTIATE_AS("grouped_direct_product", std::complex<float>, einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>)
APIARY_INSTANTIATE_AS("grouped_direct_product", std::complex<float>, einsums::RuntimeTensorView<std::complex<float>>, einsums::RuntimeTensorView<std::complex<float>>, einsums::RuntimeTensorView<std::complex<float>>)
// complex<double>
APIARY_INSTANTIATE_AS("grouped_direct_product", std::complex<double>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
APIARY_INSTANTIATE_AS("grouped_direct_product", std::complex<double>, einsums::RuntimeTensorView<std::complex<double>>, einsums::RuntimeTensorView<std::complex<double>>, einsums::RuntimeTensorView<std::complex<double>>)
    // clang-format on
    void grouped_direct_product(std::vector<T> alphas, std::vector<AType const *> a_list, std::vector<BType const *> b_list,
                                std::vector<T> betas, std::vector<CType *> c_list) {
    LabeledSection("grouped_direct_product");
    detail::grouped_binary_elementwise<T, AType, BType, CType>(
        "cg::grouped_direct_product", OpKind::GroupedDirectProduct, "direct_product",
        [](T alpha, AType const *a, BType const *b, T beta, CType *c) { linear_algebra::direct_product(alpha, *a, *b, beta, c); }, alphas,
        a_list, b_list, betas, c_list);
}

/// @brief One node holding many independent element-wise quotients:
/// ``C_i = alphas[i] * (A_i ⊘ B_i) + betas[i] * C_i`` for every member.
///
/// The grouped counterpart of @ref direct_division, and the shape an amplitude
/// update reaches it in: one residual divided by its own denominator block per
/// entity. Everything @ref grouped_direct_product says about concurrency,
/// bit-identity and distinct destinations holds here unchanged.
///
/// Outside capture this executes immediately, so the same call works eagerly.
template <typename T, CoreBasicTensorConcept AType, CoreBasicTensorConcept BType, CoreBasicTensorConcept CType>
    requires requires {
        requires std::is_same_v<typename AType::ValueType, T>;
        requires std::is_same_v<typename BType::ValueType, T>;
        requires std::is_same_v<typename CType::ValueType, T>;
    }
// clang-format off
APIARY_EXPOSE
APIARY_MODULE("linalg")
// The same 8 owning/view combinations per dtype the single form carries.
//
// float
APIARY_INSTANTIATE_AS("grouped_direct_division", float, einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::GeneralRuntimeTensor<float, std::allocator<float>>)
APIARY_INSTANTIATE_AS("grouped_direct_division", float, einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::RuntimeTensorView<float>)
APIARY_INSTANTIATE_AS("grouped_direct_division", float, einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::RuntimeTensorView<float>,                          einsums::GeneralRuntimeTensor<float, std::allocator<float>>)
APIARY_INSTANTIATE_AS("grouped_direct_division", float, einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::RuntimeTensorView<float>,                          einsums::RuntimeTensorView<float>)
APIARY_INSTANTIATE_AS("grouped_direct_division", float, einsums::RuntimeTensorView<float>,                          einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::GeneralRuntimeTensor<float, std::allocator<float>>)
APIARY_INSTANTIATE_AS("grouped_direct_division", float, einsums::RuntimeTensorView<float>,                          einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::RuntimeTensorView<float>)
APIARY_INSTANTIATE_AS("grouped_direct_division", float, einsums::RuntimeTensorView<float>,                          einsums::RuntimeTensorView<float>,                          einsums::GeneralRuntimeTensor<float, std::allocator<float>>)
APIARY_INSTANTIATE_AS("grouped_direct_division", float, einsums::RuntimeTensorView<float>,                          einsums::RuntimeTensorView<float>,                          einsums::RuntimeTensorView<float>)
// double
APIARY_INSTANTIATE_AS("grouped_direct_division", double, einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::GeneralRuntimeTensor<double, std::allocator<double>>)
APIARY_INSTANTIATE_AS("grouped_direct_division", double, einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::RuntimeTensorView<double>)
APIARY_INSTANTIATE_AS("grouped_direct_division", double, einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::RuntimeTensorView<double>,                           einsums::GeneralRuntimeTensor<double, std::allocator<double>>)
APIARY_INSTANTIATE_AS("grouped_direct_division", double, einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::RuntimeTensorView<double>,                           einsums::RuntimeTensorView<double>)
APIARY_INSTANTIATE_AS("grouped_direct_division", double, einsums::RuntimeTensorView<double>,                           einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::GeneralRuntimeTensor<double, std::allocator<double>>)
APIARY_INSTANTIATE_AS("grouped_direct_division", double, einsums::RuntimeTensorView<double>,                           einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::RuntimeTensorView<double>)
APIARY_INSTANTIATE_AS("grouped_direct_division", double, einsums::RuntimeTensorView<double>,                           einsums::RuntimeTensorView<double>,                           einsums::GeneralRuntimeTensor<double, std::allocator<double>>)
APIARY_INSTANTIATE_AS("grouped_direct_division", double, einsums::RuntimeTensorView<double>,                           einsums::RuntimeTensorView<double>,                           einsums::RuntimeTensorView<double>)
// complex<float>
APIARY_INSTANTIATE_AS("grouped_direct_division", std::complex<float>, einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>)
APIARY_INSTANTIATE_AS("grouped_direct_division", std::complex<float>, einsums::RuntimeTensorView<std::complex<float>>, einsums::RuntimeTensorView<std::complex<float>>, einsums::RuntimeTensorView<std::complex<float>>)
// complex<double>
APIARY_INSTANTIATE_AS("grouped_direct_division", std::complex<double>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
APIARY_INSTANTIATE_AS("grouped_direct_division", std::complex<double>, einsums::RuntimeTensorView<std::complex<double>>, einsums::RuntimeTensorView<std::complex<double>>, einsums::RuntimeTensorView<std::complex<double>>)
    // clang-format on
    void grouped_direct_division(std::vector<T> alphas, std::vector<AType const *> a_list, std::vector<BType const *> b_list,
                                 std::vector<T> betas, std::vector<CType *> c_list) {
    LabeledSection("grouped_direct_division");
    detail::grouped_binary_elementwise<T, AType, BType, CType>(
        "cg::grouped_direct_division", OpKind::GroupedDirectDivision, "direct_division",
        [](T alpha, AType const *a, BType const *b, T beta, CType *c) { linear_algebra::direct_division(alpha, *a, *b, beta, c); }, alphas,
        a_list, b_list, betas, c_list);
}

// ─────────────────────────────────────────────────────────────────────────────
// grouped_sandwich: q-tiled dressed sandwich accumulations as ONE node
// ─────────────────────────────────────────────────────────────────────────────

namespace detail {

/// One member of a grouped sandwich: ``C += sum_q B_q S B_q^T`` with the
/// dressed slice ``B_q = A[q] - P^T M[q]`` built in cache, never in memory.
///
/// This is psi4's own shape for the Eq. 76/93 residual term (dlpno/ccsd.cc,
/// "the T1-dressing ... on the fly, as this intermediate is only used once"),
/// adapted to a q-fastest layout: slices of ``A`` along the auxiliary axis are
/// strided here, so a BLOCK of them is transposed into slice-major scratch
/// sized to stay cache-resident, and the dress plus both sandwich GEMMs run
/// out of that scratch. ``A`` streams from memory exactly once; the dressed
/// factor and the half product exist only as one cache-resident slice.
///
/// Deterministic by construction: the q blocks and the slices inside them run
/// in ascending order on one thread, so the accumulation order into ``C`` is a
/// function of the extents alone. What this DOES change, relative to the pair
/// of whole-q contractions it replaces, is that ``C`` accumulates per q slice
/// rather than in one GEMM reduction - the same operand values sum in a
/// different order, so results agree to accumulation roundoff, not bitwise.
template <typename T>
void sandwich_member(T const *A, std::size_t saq, std::size_t saa, std::size_t sab, T const *M, std::size_t smq, std::size_t smk,
                     std::size_t smb, T const *P, std::size_t ldp, T const *S, std::size_t lds, T *C, std::size_t ldc, std::size_t nq,
                     std::size_t nk, std::size_t na) {
    if (nq == 0 || na == 0) {
        return; // C += nothing: the accumulate form of the zero-extent contract
    }
    std::size_t const slice   = na * na;
    std::size_t const m_slice = nk * na;
    // Slices per block: enough to amortize the strided gather, small enough
    // that the staged block plus its M companion stay comfortably inside a
    // per-core L2 share.
    std::size_t tq = std::max<std::size_t>(4, (512UL * 1024) / (sizeof(T) * (slice + std::max<std::size_t>(1, m_slice))));
    tq             = std::min(tq, nq);

    // Plain heap scratch, deliberately not BufferVector: the metered buffer
    // pool is sized for contraction workspace (4 MiB by default), and a team
    // of these members would exhaust it. This scratch is bounded at a few
    // hundred kilobytes per running member by the block sizing above.
    std::vector<T> bblk(tq * slice);
    std::vector<T> mblk(std::max<std::size_t>(1, tq * m_slice));
    std::vector<T> w(slice);

    for (std::size_t q0 = 0; q0 < nq; q0 += tq) {
        std::size_t const tt = std::min(tq, nq - q0);

        // Stage the A block slice-major. The source walks are contiguous in q
        // (the fastest axis), so memory is read in order; the scattered writes
        // land in the cache-resident block.
        for (std::size_t b = 0; b < na; b++) {
            for (std::size_t a = 0; a < na; a++) {
                T const *from = A + q0 * saq + a * saa + b * sab;
                T       *to   = bblk.data() + a + na * b;
                for (std::size_t t = 0; t < tt; t++) {
                    to[t * slice] = from[t * saq];
                }
            }
        }
        if (nk != 0) {
            for (std::size_t b = 0; b < na; b++) {
                for (std::size_t k = 0; k < nk; k++) {
                    T const *from = M + q0 * smq + k * smk + b * smb;
                    T       *to   = mblk.data() + k + nk * b;
                    for (std::size_t t = 0; t < tt; t++) {
                        to[t * m_slice] = from[t * smq];
                    }
                }
            }
        }

        for (std::size_t t = 0; t < tt; t++) {
            T *Bs = bblk.data() + t * slice;
            // Dress: B_q -= P^T M_q. Skipped when there is nothing to dress
            // with, which is also what keeps a (1, na) placeholder P legal for
            // an nk == 0 member.
            if (nk != 0) {
                blas::gemm('T', 'N', static_cast<blas::int_t>(na), static_cast<blas::int_t>(na), static_cast<blas::int_t>(nk), T{-1}, P,
                           static_cast<blas::int_t>(ldp), mblk.data() + t * m_slice, static_cast<blas::int_t>(nk), T{1}, Bs,
                           static_cast<blas::int_t>(na));
            }
            // W = B_q S, then C += W B_q^T.
            blas::gemm('N', 'N', static_cast<blas::int_t>(na), static_cast<blas::int_t>(na), static_cast<blas::int_t>(na), T{1}, Bs,
                       static_cast<blas::int_t>(na), S, static_cast<blas::int_t>(lds), T{0}, w.data(), static_cast<blas::int_t>(na));
            blas::gemm('N', 'T', static_cast<blas::int_t>(na), static_cast<blas::int_t>(na), static_cast<blas::int_t>(na), T{1}, w.data(),
                       static_cast<blas::int_t>(na), Bs, static_cast<blas::int_t>(na), T{1}, C, static_cast<blas::int_t>(ldc));
        }
    }
}

} // namespace detail

/// @brief Emit one node holding many independent dressed sandwich
/// accumulations: ``C_i += sum_q B_q S_i B_q^T`` with
/// ``B_q = A_i[q] - P_i^T M_i[q]``, the whole run under one OpenMP region.
///
/// The shape this exists for is the DLPNO-CCSD Eq. 76 residual term with its
/// Eq. 93 T1-dressing, which the naive emission spells as a copy of ``A_i``,
/// an in-place dressing, a half product ``W`` the size of ``A_i``, and a
/// second contraction reading both - EIGHT full streams of the largest
/// per-pair data in the iteration where one suffices. Here every member
/// streams ``A_i`` once and keeps everything else in cache; see
/// @ref detail::sandwich_member for the tiling and for the accumulation-order
/// note (results match the naive emission to roundoff, not bitwise).
///
/// Every member is independent and members are threaded over, each one's
/// arithmetic serial inside its thread, so replays are deterministic whatever
/// the schedule.
///
/// Operand shapes per member, all column-major: ``A_i`` is ``(nq, na, na)``
/// with the auxiliary axis fastest, ``M_i`` is ``(nq, nk, na)``, ``P_i`` is
/// ``(nk, na)``, ``S_i`` is ``(na, na)``, and ``C_i`` is ``(na, na)``,
/// accumulated into (``beta = 1`` semantics, so the node reads its
/// destinations). ``P_i``, ``S_i`` and ``C_i`` must be column-contiguous;
/// ``A_i`` and ``M_i`` may be arbitrary views. Destinations must be distinct.
///
/// Outside capture this executes immediately, so the same call works eagerly.
template <CoreBasicTensorConcept CType, CoreBasicTensorConcept AType, CoreBasicTensorConcept MType, CoreBasicTensorConcept PType,
          CoreBasicTensorConcept SType>
    requires(std::is_same_v<typename CType::ValueType, typename AType::ValueType> &&
             std::is_same_v<typename CType::ValueType, typename MType::ValueType> &&
             std::is_same_v<typename CType::ValueType, typename PType::ValueType> &&
             std::is_same_v<typename CType::ValueType, typename SType::ValueType> && std::is_floating_point_v<typename CType::ValueType>)
// clang-format off
APIARY_EXPOSE
APIARY_MODULE("linalg")
// The combinations the DLPNO iteration actually emits (owning stores for the
// factors and accumulators, a view of the packed amplitudes for S), plus the
// all-owning and all-view forms for tests and for callers that stage
// differently. The mixed owning-C/owning-P with viewed A, M and S form is what
// a caller whose three-index factors are reshaped slabs rather than owning
// stores produces; only the double spelling exists because no float caller
// stages that way. Extend with more APIARY_INSTANTIATE_AS lines as needed.
APIARY_INSTANTIATE_AS("grouped_sandwich", einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::RuntimeTensorView<double>)
APIARY_INSTANTIATE_AS("grouped_sandwich", einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::GeneralRuntimeTensor<double, std::allocator<double>>)
APIARY_INSTANTIATE_AS("grouped_sandwich", einsums::RuntimeTensorView<double>, einsums::RuntimeTensorView<double>, einsums::RuntimeTensorView<double>, einsums::RuntimeTensorView<double>, einsums::RuntimeTensorView<double>)
APIARY_INSTANTIATE_AS("grouped_sandwich", einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::RuntimeTensorView<double>, einsums::RuntimeTensorView<double>, einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::RuntimeTensorView<double>)
APIARY_INSTANTIATE_AS("grouped_sandwich", einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::RuntimeTensorView<float>)
APIARY_INSTANTIATE_AS("grouped_sandwich", einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::GeneralRuntimeTensor<float, std::allocator<float>>)
APIARY_INSTANTIATE_AS("grouped_sandwich", einsums::RuntimeTensorView<float>, einsums::RuntimeTensorView<float>, einsums::RuntimeTensorView<float>, einsums::RuntimeTensorView<float>, einsums::RuntimeTensorView<float>)
    // clang-format on
    void grouped_sandwich(std::vector<CType *> c_list, std::vector<AType const *> a_list, std::vector<MType const *> m_list,
                          std::vector<PType const *> p_list, std::vector<SType const *> s_list) {
    using T            = typename CType::ValueType;
    size_t const count = c_list.size();
    if (count == 0) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "cg::grouped_sandwich: the run is empty");
    }
    if (a_list.size() != count || m_list.size() != count || p_list.size() != count || s_list.size() != count) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                "cg::grouped_sandwich: the C, A, M, P and S lists must be the same length; got {}, {}, {}, {}, {}", count,
                                a_list.size(), m_list.size(), p_list.size(), s_list.size());
    }

    for (size_t i = 0; i < count; i++) {
        if (c_list[i] == nullptr || a_list[i] == nullptr || m_list[i] == nullptr || p_list[i] == nullptr || s_list[i] == nullptr) {
            EINSUMS_THROW_EXCEPTION(std::invalid_argument, "cg::grouped_sandwich: entry {} has a null operand", i);
        }
        auto const &ai = a_list[i]->impl();
        auto const &mi = m_list[i]->impl();
        auto const &pi = p_list[i]->impl();
        auto const &si = s_list[i]->impl();
        auto const &ci = c_list[i]->impl();
        if (ai.rank() != 3 || mi.rank() != 3 || pi.rank() != 2 || si.rank() != 2 || ci.rank() != 2) {
            EINSUMS_THROW_EXCEPTION(rank_error,
                                    "cg::grouped_sandwich: entry {} wants ranks (A, M, P, S, C) = (3, 3, 2, 2, 2); got "
                                    "({}, {}, {}, {}, {})",
                                    i, ai.rank(), mi.rank(), pi.rank(), si.rank(), ci.rank());
        }
        size_t const nq = ai.dim(0), na = ai.dim(1), nk = mi.dim(1);
        if (ai.dim(2) != na || mi.dim(0) != nq || mi.dim(2) != na || si.dim(0) != na || si.dim(1) != na || ci.dim(0) != na ||
            ci.dim(1) != na || (nk != 0 && (pi.dim(0) != nk || pi.dim(1) != na))) {
            EINSUMS_THROW_EXCEPTION(dimension_error,
                                    "cg::grouped_sandwich: entry {}'s shapes disagree: A ({}, {}, {}), M ({}, {}, {}), P ({}, {}), "
                                    "S ({}, {}), C ({}, {})",
                                    i, ai.dim(0), ai.dim(1), ai.dim(2), mi.dim(0), mi.dim(1), mi.dim(2), pi.dim(0), pi.dim(1), si.dim(0),
                                    si.dim(1), ci.dim(0), ci.dim(1));
        }
        if ((nk != 0 && pi.stride(0) != 1) || si.stride(0) != 1 || ci.stride(0) != 1) {
            EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                    "cg::grouped_sandwich: entry {}'s P, S and C must be column-contiguous (stride 0 == 1)", i);
        }
    }

    auto run_member = [](CType *c, AType const *a, MType const *m, PType const *p, SType const *s) {
        auto const  &ai = a->impl();
        auto const  &mi = m->impl();
        auto const  &pi = p->impl();
        auto const  &si = s->impl();
        auto        &ci = c->impl();
        size_t const nk = mi.dim(1);
        detail::sandwich_member(ai.data(), ai.stride(0), ai.stride(1), ai.stride(2), mi.data(), mi.stride(0), mi.stride(1), mi.stride(2),
                                pi.data(), nk != 0 ? pi.stride(1) : 1, si.data(), si.stride(1), ci.data(), ci.stride(1), ai.dim(0), nk,
                                ai.dim(1));
    };

    auto &ctx = CaptureContext::current();
    if (!ctx.is_capturing()) {
        LabeledSection("grouped_sandwich eager");
        // Members are independent (distinct destinations, each accumulated
        // serially by one thread), and the whole run is one parallel region -
        // an OpenMP team, never a caller-created thread pool (trap 7). An
        // exception may not cross the region boundary (that terminates), so
        // the first one is carried out by hand.
        std::exception_ptr first;
        EINSUMS_OMP_PRAGMA(parallel for schedule(dynamic))
        for (size_t i = 0; i < count; i++) {
            try {
                run_member(c_list[i], a_list[i], m_list[i], p_list[i], s_list[i]);
            } catch (...) {
                EINSUMS_OMP_PRAGMA(critical(grouped_sandwich_failure))
                if (!first) {
                    first = std::current_exception();
                }
            }
        }
        if (first) {
            std::rethrow_exception(first);
        }
        return;
    }

    LabeledSection("grouped_sandwich capture");
    std::vector<TensorId>     inputs, outputs;
    std::vector<TensorSlot *> c_slots, a_slots, m_slots, p_slots, s_slots;
    inputs.reserve(5 * count);
    outputs.reserve(count);
    c_slots.reserve(count);
    a_slots.reserve(count);
    m_slots.reserve(count);
    p_slots.reserve(count);
    s_slots.reserve(count);
    for (size_t i = 0; i < count; i++) {
        auto [a_id, a_slot] = ctx.get_slot(*a_list[i]);
        auto [m_id, m_slot] = ctx.get_slot(*m_list[i]);
        auto [p_id, p_slot] = ctx.get_slot(*p_list[i]);
        auto [s_id, s_slot] = ctx.get_slot(*s_list[i]);
        auto [c_id, c_slot] = ctx.get_slot(*c_list[i]);
        inputs.push_back(a_id);
        inputs.push_back(m_id);
        inputs.push_back(p_id);
        inputs.push_back(s_id);
        // Accumulation reads the destination, so the RAW edge from whoever
        // produced C must survive - the same rule grouped_axpby records for a
        // non-zero beta.
        inputs.push_back(c_id);
        outputs.push_back(c_id);
        a_slots.push_back(a_slot);
        m_slots.push_back(m_slot);
        p_slots.push_back(p_slot);
        s_slots.push_back(s_slot);
        c_slots.push_back(c_slot);
    }

    auto executor = [run_member, c_slots = std::move(c_slots), a_slots = std::move(a_slots), m_slots = std::move(m_slots),
                     p_slots = std::move(p_slots), s_slots = std::move(s_slots)]() {
        LabeledSection("grouped_sandwich execute");
        size_t const       n = c_slots.size();
        std::exception_ptr first;
        EINSUMS_OMP_PRAGMA(parallel for schedule(dynamic))
        for (size_t i = 0; i < n; i++) {
            try {
                run_member(static_cast<CType *>(c_slots[i]->ptr), static_cast<AType const *>(a_slots[i]->ptr),
                           static_cast<MType const *>(m_slots[i]->ptr), static_cast<PType const *>(p_slots[i]->ptr),
                           static_cast<SType const *>(s_slots[i]->ptr));
            } catch (...) {
                EINSUMS_OMP_PRAGMA(critical(grouped_sandwich_failure))
                if (!first) {
                    first = std::current_exception();
                }
            }
        }
        if (first) {
            std::rethrow_exception(first);
        }
    };

    GroupedSandwichDescriptor d;
    d.total = static_cast<int>(count);
    d.nq.reserve(count);
    d.nk.reserve(count);
    d.na.reserve(count);
    for (size_t i = 0; i < count; i++) {
        d.nq.push_back(static_cast<std::int64_t>(a_list[i]->impl().dim(0)));
        d.nk.push_back(static_cast<std::int64_t>(m_list[i]->impl().dim(1)));
        d.na.push_back(static_cast<std::int64_t>(a_list[i]->impl().dim(1)));
    }
    ctx.record(OpKind::GroupedSandwich, fmt::format("sandwich x{}", count), std::move(inputs), std::move(outputs), std::move(executor),
               std::move(d));
}

// ─────────────────────────────────────────────────────────────────────────────
// grouped_gather_rotate: q-tiled gather plus two-sided rotation as ONE node
// ─────────────────────────────────────────────────────────────────────────────

namespace detail {

/// One member of a grouped gather-rotate:
/// ``C[q, a, b] = sum_uv src[Q[q], U[u], U[v]] X[u, a] X[v, b]``, with the
/// gathered ``(q, u, v)`` block built one cache-resident q tile at a time and
/// never materialized whole.
///
/// The emission this replaces is a gather of the whole ``(Q|u v)`` domain block
/// followed by two contractions through a half-transformed block of the same
/// leading extent, which is four full streams of the largest thing the phase
/// holds where one suffices. Here the source block streams once, and the half
/// product exists only for the tile in flight.
///
/// Constraints, all checked by the caller and relied on here:
/// - ``src`` is rank 3 and shared by every member; ``Q`` selects its axis 0,
///   ``U`` selects axes 1 and 2 SYMMETRICALLY (one list, because one ``X``
///   rotates both).
/// - ``X`` is ``(nu, nt)`` and column-contiguous, so it is a legal GEMM operand
///   at ``lda = ldx``.
/// - ``C`` is ``(nq, nt, nt)`` and ASSIGNED, not accumulated: every element is
///   written, so the destination may be uninitialized on entry.
/// - Offsets arrive premultiplied by the source strides, so the staging loop
///   adds rather than multiplies.
///
/// Bitwise reproducible, and independent of the tiling: q indexes no sum, so
/// each output element is one GEMM's reduction over the FULL ``u`` (then ``v``)
/// range whatever the tile size. That is a stronger contract than the grouped
/// sandwich's - there is no accumulation to reassociate - and it is why the
/// node needs no fixed-order argument. It is not bitwise agreement with the
/// gather-plus-two-einsums form it replaces, which blocks its reductions
/// differently; that agreement is to roundoff.
template <typename T>
void gather_rotate_member(T const *src, std::size_t sq, std::size_t const *qoff, std::size_t nq, std::size_t const *uoff,
                          std::size_t const *voff, std::size_t nu, T const *X, std::size_t ldx, std::size_t nt, T *C, std::size_t scq,
                          std::size_t sca, std::size_t scb) {
    if (nq == 0 || nt == 0) {
        return; // nothing to write
    }
    if (nu == 0) {
        // An empty sum, and the operation ASSIGNS, so the zeros are the answer
        // and have to be written rather than left as whatever was allocated.
        for (std::size_t b = 0; b < nt; b++) {
            for (std::size_t a = 0; a < nt; a++) {
                T *to = C + a * sca + b * scb;
                for (std::size_t q = 0; q < nq; q++) {
                    to[q * scq] = T{0};
                }
            }
        }
        return;
    }

    std::size_t const slice  = nu * nu;
    std::size_t const hslice = nt * nu;
    std::size_t const oslice = nt * nt;
    // Slices per tile: enough to amortize the gather and to give the first GEMM
    // a wide right-hand side, small enough that the staged block, the half
    // product and the result stay inside a per-core L2 share.
    std::size_t tq = std::max<std::size_t>(4, (512UL * 1024) / (sizeof(T) * (slice + hslice + oslice)));
    tq             = std::min(tq, nq);

    // Plain heap scratch, deliberately not BufferVector: the metered buffer pool
    // is sized for contraction workspace, and a team of these members running at
    // once would exhaust it - and a throw out of the pool inside this node's
    // OpenMP region is not recoverable.
    std::vector<T> blk(tq * slice);
    std::vector<T> half(tq * hslice);
    std::vector<T> out(tq * oslice);

    for (std::size_t q0 = 0; q0 < nq; q0 += tq) {
        std::size_t const tt = std::min(tq, nq - q0);

        // Whether this tile's auxiliary selection is one ascending run, which is
        // the common case: a domain is a sorted list of functions and the shells
        // behind it are contiguous. A run turns the gather into a strided walk
        // of the source, which is what the fastest axis wants.
        bool run = true;
        for (std::size_t t = 1; run && t < tt; t++) {
            run = qoff[q0 + t] == qoff[q0] + t * sq;
        }

        // Stage the tile slice-major, u fastest. The source walks are along the
        // FASTEST source axis, so memory is read in order; the strided writes
        // land inside the cache-resident tile. The layout is exactly the
        // ``(nu) x (nu * tt)`` matrix the first GEMM wants, which is why the
        // rotation below is one call and not one per slice.
        for (std::size_t b = 0; b < nu; b++) {
            for (std::size_t a = 0; a < nu; a++) {
                T const *from = src + uoff[a] + voff[b];
                T       *to   = blk.data() + a + nu * b;
                if (run) {
                    T const *base = from + qoff[q0];
                    for (std::size_t t = 0; t < tt; t++) {
                        to[t * slice] = base[t * sq];
                    }
                } else {
                    for (std::size_t t = 0; t < tt; t++) {
                        to[t * slice] = from[qoff[q0 + t]];
                    }
                }
            }
        }

        // half[a, v, t] = sum_u X[u, a] blk[u, v, t]: one GEMM for the whole
        // tile, because (v, t) is a single merged axis in this layout.
        blas::gemm('T', 'N', static_cast<blas::int_t>(nt), static_cast<blas::int_t>(nu * tt), static_cast<blas::int_t>(nu), T{1}, X,
                   static_cast<blas::int_t>(ldx), blk.data(), static_cast<blas::int_t>(nu), T{0}, half.data(),
                   static_cast<blas::int_t>(nt));
        // out[a, b, t] = sum_v half[a, v, t] X[v, b]: per slice, because the
        // contracted index is interior once the tile is laid out this way.
        for (std::size_t t = 0; t < tt; t++) {
            blas::gemm('N', 'N', static_cast<blas::int_t>(nt), static_cast<blas::int_t>(nt), static_cast<blas::int_t>(nu), T{1},
                       half.data() + t * hslice, static_cast<blas::int_t>(nt), X, static_cast<blas::int_t>(ldx), T{0},
                       out.data() + t * oslice, static_cast<blas::int_t>(nt));
        }

        // Scatter the tile out. The destination's auxiliary axis is the one
        // being walked, so a unit q stride - what an owning (nq, nt, nt) store
        // has - makes every one of these a contiguous run.
        for (std::size_t b = 0; b < nt; b++) {
            for (std::size_t a = 0; a < nt; a++) {
                T const *from = out.data() + a + nt * b;
                T       *to   = C + q0 * scq + a * sca + b * scb;
                if (scq == 1) {
                    for (std::size_t t = 0; t < tt; t++) {
                        to[t] = from[t * oslice];
                    }
                } else {
                    for (std::size_t t = 0; t < tt; t++) {
                        to[t * scq] = from[t * oslice];
                    }
                }
            }
        }
    }
}

} // namespace detail

/// @brief Emit one node holding many independent gather-and-rotate blocks:
/// ``C_i[q, a, b] = sum_uv src[Q_i[q], U_i[u], U_i[v]] X_i[u, a] X_i[v, b]``,
/// the whole run under one OpenMP region.
///
/// The shape this exists for is the DLPNO-(T0) three-external block, which the
/// naive emission spells as a @ref gather of the whole ``(Q|u v)`` domain block
/// followed by two contractions - the gathered block and a half-transformed one
/// of the same leading extent, both of them the largest tensors the phase ever
/// holds, streamed four times over where one pass suffices. Here every member
/// streams its selection of @p src once and keeps the rotation in cache; see
/// @ref detail::gather_rotate_member for the tiling.
///
/// One source for the whole run, because that is what makes the streaming claim
/// true: the members are domain-restricted selections of ONE parent, and giving
/// each its own would only let a caller take the traffic back.
///
/// Both source axes are selected by the same index list and rotated by the same
/// transform. The operation is the symmetric two-sided rotation
/// ``X^T (Q|u v) X``, which is the only form the callers have; an asymmetric one
/// would need a second list and a second transform and is not this operation.
///
/// Every member is independent - destinations are distinct and each is ASSIGNED
/// by one thread - so replays are deterministic whatever the schedule, and the
/// result does not depend on the tiling at all (the tiled axis indexes no sum).
/// Agreement with the gather-plus-two-contractions form it replaces is to
/// roundoff rather than bitwise, because the two block their reductions
/// differently.
///
/// Operand shapes per member, all column-major: @p src is ``(NQ, NU, NU)`` with
/// the auxiliary axis fastest, ``X_i`` is ``(nu_i, nt_i)`` and column-contiguous,
/// and ``C_i`` is ``(nq_i, nt_i, nt_i)`` where ``nq_i`` and ``nu_i`` are the
/// lengths of the member's index lists. A member with any extent zero is a quick
/// return, except that an empty ``u`` selection writes its destination's zeros.
///
/// Outside capture this executes immediately, so the same call works eagerly.
template <CoreBasicTensorConcept CType, CoreBasicTensorConcept SrcType, CoreBasicTensorConcept XType>
    requires(std::is_same_v<typename CType::ValueType, typename SrcType::ValueType> &&
             std::is_same_v<typename CType::ValueType, typename XType::ValueType> && std::is_floating_point_v<typename CType::ValueType>)
// clang-format off
APIARY_EXPOSE
APIARY_MODULE("linalg")
// The combination the DLPNO emission uses is all-owning; the view forms are
// there for callers that stage their destinations inside a larger store.
// Extend with more APIARY_INSTANTIATE_AS lines as needed.
APIARY_INSTANTIATE_AS("grouped_gather_rotate", einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::GeneralRuntimeTensor<double, std::allocator<double>>)
APIARY_INSTANTIATE_AS("grouped_gather_rotate", einsums::RuntimeTensorView<double>, einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::GeneralRuntimeTensor<double, std::allocator<double>>)
APIARY_INSTANTIATE_AS("grouped_gather_rotate", einsums::RuntimeTensorView<double>, einsums::RuntimeTensorView<double>, einsums::RuntimeTensorView<double>)
APIARY_INSTANTIATE_AS("grouped_gather_rotate", einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::GeneralRuntimeTensor<float, std::allocator<float>>)
APIARY_INSTANTIATE_AS("grouped_gather_rotate", einsums::RuntimeTensorView<float>, einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::GeneralRuntimeTensor<float, std::allocator<float>>)
APIARY_INSTANTIATE_AS("grouped_gather_rotate", einsums::RuntimeTensorView<float>, einsums::RuntimeTensorView<float>, einsums::RuntimeTensorView<float>)
    // clang-format on
    void grouped_gather_rotate(std::vector<CType *> c_list, SrcType const &src, std::vector<std::vector<size_t>> const &q_list,
                               std::vector<std::vector<size_t>> const &u_list, std::vector<XType const *> x_list) {
    using T            = typename CType::ValueType;
    size_t const count = c_list.size();
    if (count == 0) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "cg::grouped_gather_rotate: the run is empty");
    }
    if (q_list.size() != count || u_list.size() != count || x_list.size() != count) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                "cg::grouped_gather_rotate: the C, Q, U and X lists must be the same length; got {}, {}, {}, {}", count,
                                q_list.size(), u_list.size(), x_list.size());
    }
    if (detail::tensor_rank(src) != 3) {
        EINSUMS_THROW_EXCEPTION(rank_error, "cg::grouped_gather_rotate: the source must be rank 3, got {}", detail::tensor_rank(src));
    }
    size_t const NQ = src.dim(0), NU1 = src.dim(1), NU2 = src.dim(2);

    for (size_t i = 0; i < count; i++) {
        if (c_list[i] == nullptr || x_list[i] == nullptr) {
            EINSUMS_THROW_EXCEPTION(std::invalid_argument, "cg::grouped_gather_rotate: entry {} has a null operand", i);
        }
        auto const  &ci = c_list[i]->impl();
        auto const  &xi = x_list[i]->impl();
        size_t const nq = q_list[i].size(), nu = u_list[i].size();
        if (ci.rank() != 3 || xi.rank() != 2) {
            EINSUMS_THROW_EXCEPTION(rank_error, "cg::grouped_gather_rotate: entry {} wants ranks (C, X) = (3, 2); got ({}, {})", i,
                                    ci.rank(), xi.rank());
        }
        size_t const nt = xi.dim(1);
        if (xi.dim(0) != nu || ci.dim(0) != nq || ci.dim(1) != nt || ci.dim(2) != nt) {
            EINSUMS_THROW_EXCEPTION(dimension_error,
                                    "cg::grouped_gather_rotate: entry {}'s shapes disagree: {} q indices, {} u indices, X ({}, {}), "
                                    "C ({}, {}, {})",
                                    i, nq, nu, xi.dim(0), xi.dim(1), ci.dim(0), ci.dim(1), ci.dim(2));
        }
        if (nu != 0 && nt != 0 && xi.stride(0) != 1) {
            EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                    "cg::grouped_gather_rotate: entry {}'s X must be column-contiguous (stride 0 == 1)", i);
        }
        for (size_t p : q_list[i]) {
            if (p >= NQ) {
                EINSUMS_THROW_EXCEPTION(std::out_of_range,
                                        "cg::grouped_gather_rotate: entry {} selects auxiliary index {} from a source of extent {}", i, p,
                                        NQ);
            }
        }
        for (size_t p : u_list[i]) {
            if (p >= NU1 || p >= NU2) {
                EINSUMS_THROW_EXCEPTION(std::out_of_range,
                                        "cg::grouped_gather_rotate: entry {} selects index {} from source axes of extent ({}, {})", i, p,
                                        NU1, NU2);
            }
        }
    }

    // Two members sharing a destination race, because the run gives no ordering
    // between them - the same contract @ref grouped_batched_gemm carries, and
    // the same reason: this entry point exists to MERGE calls that used to be
    // separate.
    {
        std::vector<CType const *> seen(c_list.begin(), c_list.end());
        std::sort(seen.begin(), seen.end());
        if (std::adjacent_find(seen.begin(), seen.end()) != seen.end()) {
            EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                    "cg::grouped_gather_rotate: two members share a destination tensor. The run gives no ordering "
                                    "between members, so they would race; split them into separate calls");
        }
    }

    auto run_member = [](CType *c, SrcType const *s, XType const *x, std::vector<size_t> const &qs, std::vector<size_t> const &us) {
        auto const  &si = s->impl();
        auto const  &xi = x->impl();
        auto        &ci = c->impl();
        size_t const nq = qs.size(), nu = us.size(), nt = xi.dim(1);

        // Strides are read here rather than baked at capture: a slot may point
        // at a different tensor object on a later replay.
        size_t const        sq = si.stride(0);
        std::vector<size_t> qoff(nq), uoff(nu), voff(nu);
        for (size_t t = 0; t < nq; t++) {
            qoff[t] = qs[t] * sq;
        }
        for (size_t a = 0; a < nu; a++) {
            uoff[a] = us[a] * si.stride(1);
            voff[a] = us[a] * si.stride(2);
        }
        detail::gather_rotate_member(si.data(), sq, qoff.data(), nq, uoff.data(), voff.data(), nu, xi.data(), nu != 0 ? xi.stride(1) : 1,
                                     nt, ci.data(), ci.stride(0), ci.stride(1), ci.stride(2));
    };

    auto &ctx = CaptureContext::current();
    if (!ctx.is_capturing()) {
        LabeledSection("grouped_gather_rotate eager");
        // Members are independent (distinct destinations, each assigned by one
        // thread), and the whole run is one parallel region - an OpenMP team,
        // never a caller-created thread pool. An exception may not cross the
        // region boundary (that terminates, and takes a libomp worker with it),
        // so the first one is carried out by hand.
        std::exception_ptr first;
        EINSUMS_OMP_PRAGMA(parallel for schedule(dynamic))
        for (size_t i = 0; i < count; i++) {
            try {
                run_member(c_list[i], &src, x_list[i], q_list[i], u_list[i]);
            } catch (...) {
                EINSUMS_OMP_PRAGMA(critical(grouped_gather_rotate_failure))
                if (!first) {
                    first = std::current_exception();
                }
            }
        }
        if (first) {
            std::rethrow_exception(first);
        }
        return;
    }

    LabeledSection("grouped_gather_rotate capture");
    std::vector<TensorId>     inputs, outputs;
    std::vector<TensorSlot *> c_slots, x_slots;
    inputs.reserve(count + 1);
    outputs.reserve(count);
    c_slots.reserve(count);
    x_slots.reserve(count);
    auto [s_id, s_binding] = ctx.get_slot(src);
    // Copied out of the structured binding: a lambda that opens an OpenMP
    // region may not capture one.
    TensorSlot *const s_slot = s_binding;
    inputs.push_back(s_id);
    for (size_t i = 0; i < count; i++) {
        auto [x_id, x_slot] = ctx.get_slot(*x_list[i]);
        auto [c_id, c_slot] = ctx.get_slot(*c_list[i]);
        inputs.push_back(x_id);
        // The destination is ASSIGNED, not accumulated, so it is an output only:
        // unlike the grouped sandwich there is no read of C to keep an edge for.
        outputs.push_back(c_id);
        x_slots.push_back(x_slot);
        c_slots.push_back(c_slot);
    }

    // The index lists are copied into the executor rather than referenced: a
    // captured node outlives the call, and every replay reads them again.
    auto executor = [run_member, s_slot, c_slots = std::move(c_slots), x_slots = std::move(x_slots), q_list, u_list]() {
        LabeledSection("grouped_gather_rotate execute");
        size_t const       n = c_slots.size();
        std::exception_ptr first;
        EINSUMS_OMP_PRAGMA(parallel for schedule(dynamic))
        for (size_t i = 0; i < n; i++) {
            try {
                run_member(static_cast<CType *>(c_slots[i]->ptr), static_cast<SrcType const *>(s_slot->ptr),
                           static_cast<XType const *>(x_slots[i]->ptr), q_list[i], u_list[i]);
            } catch (...) {
                EINSUMS_OMP_PRAGMA(critical(grouped_gather_rotate_failure))
                if (!first) {
                    first = std::current_exception();
                }
            }
        }
        if (first) {
            std::rethrow_exception(first);
        }
    };

    GroupedGatherRotateDescriptor d;
    d.total      = static_cast<int>(count);
    d.elem_bytes = static_cast<std::int64_t>(sizeof(T));
    d.nq.reserve(count);
    d.nu.reserve(count);
    d.nt.reserve(count);
    for (size_t i = 0; i < count; i++) {
        d.nq.push_back(static_cast<std::int64_t>(q_list[i].size()));
        d.nu.push_back(static_cast<std::int64_t>(u_list[i].size()));
        d.nt.push_back(static_cast<std::int64_t>(x_list[i]->impl().dim(1)));
    }
    ctx.record(OpKind::GroupedGatherRotate, fmt::format("gather_rotate x{}", count), std::move(inputs), std::move(outputs),
               std::move(executor), std::move(d));
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
    detail::reject_if_capturing("cg::norm() returning scalar cannot be used during graph capture.");
    // A reduction's summation order is its thread count's, so an unfenced norm is a function of the machine as well as of
    // the operands. See @ref blas::SerialVendorScope.
    blas::SerialVendorScope const serial;
    return linear_algebra::norm(norm_type, A);
}

/// Graph-aware norm writing result to a pre-allocated scalar.
template <TensorConcept AType>
void norm(RemoveComplexT<typename AType::ValueType> *result, linear_algebra::Norm norm_type, AType const &A) {
    auto &ctx = CaptureContext::current();
    if (!ctx.is_capturing()) {
        LabeledSection("norm eager");
        // A reduction's summation order is its thread count's, so the fence is what makes this a function of the operands alone.
        blas::SerialVendorScope const serial;
        *result = linear_algebra::norm(norm_type, A);
        return;
    }

    LabeledSection("norm capture");
    auto [a_id, a_slot] = ctx.get_slot(A);
    TensorId r_id       = ctx.get_or_register_scalar(result, "norm_result");

    auto executor = [result, norm_type, a_slot]() {
        LabeledSection("norm execute");
        blas::SerialVendorScope const serial;
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
        // A reduction's summation order is its thread count's, so the fence is what makes this a function of the operands alone.
        blas::SerialVendorScope const serial;
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
    detail::reject_if_capturing("cg::trace(A) returning scalar cannot be used during graph capture. "
                                "Use cg::trace(&result, A) instead.");
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
    detail::reject_if_capturing("cg::trace(A) returning scalar cannot be used during graph capture. "
                                "Use cg::trace(&result, A) instead.");
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
    void symm_gemm(AType const &A, BType const &B, CType *C, bool conjugate = false) {
    if (A.rank() != 2 || B.rank() != 2 || C->rank() != 2) {
        EINSUMS_THROW_EXCEPTION(rank_error, "cg::symm_gemm requires rank-2 tensors; got ranks {}, {}, {}.", A.rank(), B.rank(), C->rank());
    }
    // conjugate=true computes the Hermitian congruence op(B)^H op(A) op(B); the
    // default is the bilinear op(B)^T op(A) op(B). For real dtypes they coincide.
    auto run = [conjugate](AType const &a, BType const &b, CType *c) {
        if (conjugate) {
            linear_algebra::hermitian_symm_gemm<TransA, TransB>(a, b, c);
        } else {
            linear_algebra::symm_gemm<TransA, TransB>(a, b, c);
        }
    };
    auto &ctx = CaptureContext::current();
    if (!ctx.is_capturing()) {
        LabeledSection("symm_gemm eager");
        run(A, B, C);
        return;
    }
    LabeledSection("symm_gemm capture");
    auto [a_id, a_slot] = ctx.get_slot(A);
    auto [b_id, b_slot] = ctx.get_slot(B);
    auto [c_id, c_slot] = ctx.get_slot(*C);
    auto executor       = [a_slot, b_slot, c_slot, run]() {
        LabeledSection("symm_gemm execute");
        run(*static_cast<AType const *>(a_slot->ptr), *static_cast<BType const *>(b_slot->ptr), static_cast<CType *>(c_slot->ptr));
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
APIARY_INSTANTIATE_BOOLS("syev", einsums::RuntimeTensorView<float>,  einsums::GeneralRuntimeTensor<float,  std::allocator<float>>)
APIARY_INSTANTIATE_BOOLS("syev", einsums::RuntimeTensorView<double>, einsums::GeneralRuntimeTensor<double, std::allocator<double>>)
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
    detail::reject_if_capturing("cg::syev(A) returning form cannot be used during graph capture. "
                                "Use the in-place form cg::syev(&A, &W) with pre-allocated tensors instead.");
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
    detail::reject_if_capturing("cg::syev(A) returning form cannot be used during graph capture. "
                                "Use the in-place form cg::syev(&A, &W) instead.");
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
APIARY_INSTANTIATE_BOOLS("heev", einsums::RuntimeTensorView<std::complex<float>>,  einsums::GeneralRuntimeTensor<float,  std::allocator<float>>)
APIARY_INSTANTIATE_BOOLS("heev", einsums::RuntimeTensorView<std::complex<double>>, einsums::GeneralRuntimeTensor<double, std::allocator<double>>)
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
/// eigenvectors, W the eigenvalues).
///
/// A wrapper because the templated syev has a leading non-type
/// ``bool ComputeEigenvectors`` parameter the pybind codegen can't pin via
/// INSTANTIATE_AS; this fixes it to true and presents a clean, type-only signature. Accepts dense (RuntimeTensor) or tiled operands; the
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
// getrf / getrs: factor once, solve many
//
// The pivots are not a tensor - a BLAS integer array is none of the dtypes an
// operand may carry - so they ride outside the dataflow in a LuPivots handle
// whose buffer both executors bake in at capture time. What orders a
// factorization against its solves is therefore the factorization TENSOR:
// getrf writes it, getrs reads it, and the hazard scan serializes that pair on
// its own. A getrs whose handle was filled by a getrf against a DIFFERENT
// tensor has no edge holding the two together.
// ─────────────────────────────────────────────────────────────────────────────

template <MatrixConcept AType>
auto getrf(AType *A, LuPivots *pivots) -> int {
    auto       &ctx    = CaptureContext::current();
    auto const &buffer = pivots->buffer();

    if (!ctx.is_capturing()) {
        LabeledSection("getrf eager");
        return linear_algebra::getrf(A, buffer.get());
    }

    LabeledSection("getrf capture");
    auto [a_id, a_slot] = ctx.get_slot(*A);

    auto executor = [a_slot, buffer]() {
        LabeledSection("getrf execute");
        ProfileAnnotate("n", static_cast<int64_t>(static_cast<AType *>(a_slot->ptr)->dim(0)));
        std::ignore = linear_algebra::getrf(static_cast<AType *>(a_slot->ptr), buffer.get());
    };
    ctx.record(OpKind::Getrf, "getrf", {a_id}, {a_id}, std::move(executor));

    return 0; // Return value not meaningful during capture
}

/// LU-factorize ``A`` in place, recording the row interchanges in ``pivots``.
///
/// On return ``A`` holds ``L`` and ``U`` in the LAPACK packing (the unit
/// diagonal of ``L`` is not stored) and ``pivots`` holds the interchanges,
/// sized to the order of ``A``. Unlike ``gesv``, the factorization survives
/// every solve made against it, so one ``getrf`` feeds any number of
/// ``getrs`` calls. Returns the LAPACK info code: 0 on success, positive
/// ``i`` if ``U(i,i)`` is exactly zero - the factorization is still complete,
/// but a solve against it is meaningless.
///
/// The same ``pivots`` object must be handed to the matching ``getrs``, and
/// ordering rides on ``A`` rather than on the pivots.
template <RuntimeRankTensorConcept AType>
// clang-format off
APIARY_EXPOSE
APIARY_MODULE("linalg")
APIARY_INSTANTIATE_AS("getrf", einsums::GeneralRuntimeTensor<float,                std::allocator<float>>)
APIARY_INSTANTIATE_AS("getrf", einsums::GeneralRuntimeTensor<double,               std::allocator<double>>)
APIARY_INSTANTIATE_AS("getrf", einsums::GeneralRuntimeTensor<std::complex<float>,  std::allocator<std::complex<float>>>)
APIARY_INSTANTIATE_AS("getrf", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
APIARY_INSTANTIATE_AS("getrf", einsums::RuntimeTensorView<float>)
APIARY_INSTANTIATE_AS("getrf", einsums::RuntimeTensorView<double>)
APIARY_INSTANTIATE_AS("getrf", einsums::RuntimeTensorView<std::complex<float>>)
APIARY_INSTANTIATE_AS("getrf", einsums::RuntimeTensorView<std::complex<double>>)
    // clang-format on
    auto getrf(AType *A, LuPivots *pivots) -> int {
    if (A->rank() != 2) {
        EINSUMS_THROW_EXCEPTION(rank_error, "cg::getrf requires a rank-2 tensor; got rank {}.", A->rank());
    }

    auto       &ctx    = CaptureContext::current();
    auto const &buffer = pivots->buffer();

    if (!ctx.is_capturing()) {
        LabeledSection("getrf eager");
        return linear_algebra::getrf(A, buffer.get());
    }

    LabeledSection("getrf capture");
    auto [a_id, a_slot] = ctx.get_slot(*A);

    auto executor = [a_slot, buffer]() {
        LabeledSection("getrf execute");
        ProfileAnnotate("n", static_cast<int64_t>(static_cast<AType *>(a_slot->ptr)->dim(0)));
        std::ignore = linear_algebra::getrf(static_cast<AType *>(a_slot->ptr), buffer.get());
    };
    ctx.record(OpKind::Getrf, "getrf", {a_id}, {a_id}, std::move(executor));

    return 0;
}

template <MatrixConcept AType, TensorConcept BType>
    requires requires {
        requires SameUnderlying<AType, BType>;
        requires MatrixConcept<BType> || VectorConcept<BType>;
    }
auto getrs(AType const &A, LuPivots const &pivots, BType *B) -> int {
    auto       &ctx    = CaptureContext::current();
    auto const &buffer = pivots.buffer();

    if (!ctx.is_capturing()) {
        LabeledSection("getrs eager");
        return linear_algebra::getrs(A, *buffer, B);
    }

    LabeledSection("getrs capture");
    auto [a_id, a_slot] = ctx.get_slot(A);
    auto [b_id, b_slot] = ctx.get_slot(*B);

    auto executor = [a_slot, b_slot, buffer]() {
        LabeledSection("getrs execute");
        ProfileAnnotate("n", static_cast<int64_t>(static_cast<AType const *>(a_slot->ptr)->dim(0)));
        std::ignore = linear_algebra::getrs(*static_cast<AType const *>(a_slot->ptr), *buffer, static_cast<BType *>(b_slot->ptr));
    };
    ctx.record(OpKind::Getrs, "getrs", {a_id, b_id}, {b_id}, std::move(executor));

    return 0; // Return value not meaningful during capture
}

/// Solve ``A * X = B`` in place against a factorization ``getrf`` produced.
///
/// ``A`` is the LU factorization and ``pivots`` the interchanges the matching
/// ``getrf`` left; both are READ, so the pair serves any number of
/// right-hand sides and any number of replays. ``B`` may be rank 1 (a single
/// right-hand side) or rank 2 (one per column), and on return holds ``X``.
/// Returns the LAPACK info code, which is 0 unless an argument was invalid.
template <RuntimeRankTensorConcept AType, RuntimeRankTensorConcept BType>
    requires SameUnderlying<AType, BType>
// clang-format off
APIARY_EXPOSE
APIARY_MODULE("linalg")
APIARY_INSTANTIATE_AS("getrs", einsums::GeneralRuntimeTensor<float,                std::allocator<float>>,                einsums::GeneralRuntimeTensor<float,                std::allocator<float>>)
APIARY_INSTANTIATE_AS("getrs", einsums::GeneralRuntimeTensor<double,               std::allocator<double>>,               einsums::GeneralRuntimeTensor<double,               std::allocator<double>>)
APIARY_INSTANTIATE_AS("getrs", einsums::GeneralRuntimeTensor<std::complex<float>,  std::allocator<std::complex<float>>>,  einsums::GeneralRuntimeTensor<std::complex<float>,  std::allocator<std::complex<float>>>)
APIARY_INSTANTIATE_AS("getrs", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
APIARY_INSTANTIATE_AS("getrs", einsums::GeneralRuntimeTensor<float,                std::allocator<float>>,                einsums::RuntimeTensorView<float>)
APIARY_INSTANTIATE_AS("getrs", einsums::GeneralRuntimeTensor<double,               std::allocator<double>>,               einsums::RuntimeTensorView<double>)
APIARY_INSTANTIATE_AS("getrs", einsums::GeneralRuntimeTensor<std::complex<float>,  std::allocator<std::complex<float>>>,  einsums::RuntimeTensorView<std::complex<float>>)
APIARY_INSTANTIATE_AS("getrs", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::RuntimeTensorView<std::complex<double>>)
    // clang-format on
    auto getrs(AType const &A, LuPivots const &pivots, BType *B) -> int {
    if (A.rank() != 2 || (B->rank() != 1 && B->rank() != 2)) {
        EINSUMS_THROW_EXCEPTION(rank_error, "cg::getrs requires A rank-2 and B rank-1 or rank-2; got {}, {}.", A.rank(), B->rank());
    }

    auto       &ctx    = CaptureContext::current();
    auto const &buffer = pivots.buffer();

    if (!ctx.is_capturing()) {
        LabeledSection("getrs eager");
        return linear_algebra::getrs(A, *buffer, B);
    }

    LabeledSection("getrs capture");
    auto [a_id, a_slot] = ctx.get_slot(A);
    auto [b_id, b_slot] = ctx.get_slot(*B);

    auto executor = [a_slot, b_slot, buffer]() {
        LabeledSection("getrs execute");
        ProfileAnnotate("n", static_cast<int64_t>(static_cast<AType const *>(a_slot->ptr)->dim(0)));
        std::ignore = linear_algebra::getrs(*static_cast<AType const *>(a_slot->ptr), *buffer, static_cast<BType *>(b_slot->ptr));
    };
    ctx.record(OpKind::Getrs, "getrs", {a_id, b_id}, {b_id}, std::move(executor));

    return 0;
}

// ─────────────────────────────────────────────────────────────────────────────
// diis_add_pair / diis_step: Pulay extrapolation as ONE node
//
// The accelerator's history is not tensor dataflow - it is a private list of
// snapshots the object owns - so like the LU pivots it rides outside the graph,
// in a DiisAccelerator whose shared_ptr the executor bakes in at capture time.
// The object therefore survives however long the graph does, and every replay
// steps the same history.
//
// What orders the node is the PAIR TENSORS. diis_step records every amplitude
// and every step tensor as an input and every amplitude as an output, which is
// exactly what the step does to them, so the hazard scan puts it after the body
// that computed the step and before whatever reads the extrapolated
// amplitudes. The operands must be the ones the accelerator was given: an
// accelerator holding tensors no captured op touches has nothing tying its node
// to the iteration and is a bug in the caller.
// ─────────────────────────────────────────────────────────────────────────────

/// Register one ``(amplitude, step)`` pair with a DIIS accelerator.
///
/// ``amplitude`` is the tensor the iteration updates and ``step`` is the update
/// the body computed, which DIIS also uses as the error vector. Both are bound
/// by storage, so they must outlive the accelerator, and every pair has to be
/// added before the first step - the snapshots are shaped for the pair list.
///
/// This is bookkeeping, not an operation: it records nothing into an active
/// capture and works the same inside and outside one.
template <RuntimeRankTensorConcept AmpType, RuntimeRankTensorConcept StepType>
    requires SameUnderlying<AmpType, StepType>
// clang-format off
APIARY_EXPOSE
APIARY_MODULE("graph")
// The 4 owning/view combinations per dtype: a DLPNO amplitude store is an
// owning tensor, a padded block of one is a view, and the pair may mix them.
//
// float
APIARY_INSTANTIATE_AS("diis_add_pair", einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::GeneralRuntimeTensor<float, std::allocator<float>>)
APIARY_INSTANTIATE_AS("diis_add_pair", einsums::GeneralRuntimeTensor<float, std::allocator<float>>, einsums::RuntimeTensorView<float>)
APIARY_INSTANTIATE_AS("diis_add_pair", einsums::RuntimeTensorView<float>,                          einsums::GeneralRuntimeTensor<float, std::allocator<float>>)
APIARY_INSTANTIATE_AS("diis_add_pair", einsums::RuntimeTensorView<float>,                          einsums::RuntimeTensorView<float>)
// double
APIARY_INSTANTIATE_AS("diis_add_pair", einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::GeneralRuntimeTensor<double, std::allocator<double>>)
APIARY_INSTANTIATE_AS("diis_add_pair", einsums::GeneralRuntimeTensor<double, std::allocator<double>>, einsums::RuntimeTensorView<double>)
APIARY_INSTANTIATE_AS("diis_add_pair", einsums::RuntimeTensorView<double>,                           einsums::GeneralRuntimeTensor<double, std::allocator<double>>)
APIARY_INSTANTIATE_AS("diis_add_pair", einsums::RuntimeTensorView<double>,                           einsums::RuntimeTensorView<double>)
// complex<float>
APIARY_INSTANTIATE_AS("diis_add_pair", einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>)
APIARY_INSTANTIATE_AS("diis_add_pair", einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>, einsums::RuntimeTensorView<std::complex<float>>)
APIARY_INSTANTIATE_AS("diis_add_pair", einsums::RuntimeTensorView<std::complex<float>>,                                        einsums::GeneralRuntimeTensor<std::complex<float>, std::allocator<std::complex<float>>>)
APIARY_INSTANTIATE_AS("diis_add_pair", einsums::RuntimeTensorView<std::complex<float>>,                                        einsums::RuntimeTensorView<std::complex<float>>)
// complex<double>
APIARY_INSTANTIATE_AS("diis_add_pair", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
APIARY_INSTANTIATE_AS("diis_add_pair", einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>, einsums::RuntimeTensorView<std::complex<double>>)
APIARY_INSTANTIATE_AS("diis_add_pair", einsums::RuntimeTensorView<std::complex<double>>,                                         einsums::GeneralRuntimeTensor<std::complex<double>, std::allocator<std::complex<double>>>)
APIARY_INSTANTIATE_AS("diis_add_pair", einsums::RuntimeTensorView<std::complex<double>>,                                         einsums::RuntimeTensorView<std::complex<double>>)
    // clang-format on
    void diis_add_pair(DiisAccelerator<typename AmpType::ValueType> *accelerator, AmpType *amplitude, StepType *step) {
    using T = typename AmpType::ValueType;
    if (accelerator == nullptr || amplitude == nullptr || step == nullptr) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "cg::diis_add_pair: null accelerator or operand");
    }

    // The registrars run at capture time, where the operand's static type is
    // needed to reach get_slot; that type is only available here.
    accelerator->add_pair(
        RuntimeTensorView<T>(*amplitude), RuntimeTensorView<T>(*step),
        [amplitude]() { return CaptureContext::current().get_slot(*amplitude).first; },
        [step]() { return CaptureContext::current().get_slot(*step).first; });
}

/// Take one DIIS step over an accelerator's registered pairs.
///
/// Outside capture this extrapolates immediately, which is how a host-side
/// convergence loop uses it. Inside capture it records one node whose executor
/// holds the accelerator, so a replay of the graph steps the same history; the
/// node reads every amplitude and step tensor and writes every amplitude, which
/// is what orders it against the body around it.
template <typename T>
// clang-format off
APIARY_EXPOSE
APIARY_MODULE("graph")
APIARY_INSTANTIATE_AS("diis_step", float)
APIARY_INSTANTIATE_AS("diis_step", double)
APIARY_INSTANTIATE_AS("diis_step", std::complex<float>)
APIARY_INSTANTIATE_AS("diis_step", std::complex<double>)
    // clang-format on
    void diis_step(std::shared_ptr<DiisAccelerator<T>> const &accelerator) {
    if (!accelerator) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "cg::diis_step: null accelerator");
    }
    if (accelerator->num_pairs() == 0) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "cg::diis_step: the accelerator has no (amplitude, step) pairs");
    }

    auto &ctx = CaptureContext::current();
    if (!ctx.is_capturing()) {
        LabeledSection("diis_step eager");
        accelerator->step();
        return;
    }

    LabeledSection("diis_step capture");
    std::vector<TensorId> inputs, outputs;
    inputs.reserve(2 * accelerator->num_pairs());
    outputs.reserve(accelerator->num_pairs());
    for (auto const &registrar : accelerator->amplitude_ids()) {
        TensorId const id = registrar();
        inputs.push_back(id);
        outputs.push_back(id);
    }
    for (auto const &registrar : accelerator->step_ids()) {
        inputs.push_back(registrar());
    }

    auto executor = [accelerator]() {
        LabeledSection("diis_step execute");
        accelerator->step();
    };
    ctx.record(OpKind::DiisStep, fmt::format("diis x{}", accelerator->num_pairs()), std::move(inputs), std::move(outputs),
               std::move(executor));
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
    detail::reject_if_capturing("cg::svd(A) returning form cannot be used during graph capture.");
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
    detail::reject_if_capturing("cg::svd(A) returning form cannot be used during graph capture.");
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
    detail::reject_if_capturing("cg::svd_dd(A) returning form cannot be used during graph capture.");
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
    detail::reject_if_capturing("cg::svd_dd(A) returning form cannot be used during graph capture.");
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
    detail::reject_if_capturing("cg::truncated_svd(A, k) returning form cannot be used during graph capture.");
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
    detail::reject_if_capturing("cg::truncated_svd(A, k) returning form cannot be used during graph capture.");
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
    detail::reject_if_capturing("cg::truncated_syev(A, k) returning form cannot be used during graph capture.");
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
    detail::reject_if_capturing("cg::truncated_syev(A, k) returning form cannot be used during graph capture.");
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
    detail::reject_if_capturing("cg::qr(A) returning form cannot be used during graph capture.");
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
    detail::reject_if_capturing("cg::qr(A) returning form cannot be used during graph capture.");
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
    detail::reject_if_capturing("cg::pow(A, alpha) returning form cannot be used during graph capture.");
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
    detail::reject_if_capturing("cg::pow(A, alpha) returning form cannot be used during graph capture.");
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
    detail::reject_if_capturing("cg::det(A) returning scalar cannot be used during graph capture.");
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
    detail::reject_if_capturing("cg::det(A) returning scalar cannot be used during graph capture.");
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
/// Every index name must denote ONE size wherever it appears. Checking at
/// capture (and on the eager path) puts the failure in the user's own stack
/// frame with names attached, instead of a garbage-dimension BLAS error - or
/// silent corruption - at execute time.
template <typename AType, typename BType, typename CType>
void validate_einsum_dims(ParsedEinsumSpec const &parsed, AType const &A, BType const &B, CType const &C) {
    std::unordered_map<std::string_view, std::pair<size_t, char>> sizes;

    auto scan = [&](std::vector<std::string> const &idx, auto const &t, char who) {
        size_t const rank = detail::tensor_rank(t);
        for (size_t d = 0; d < idx.size() && d < rank; d++) {
            size_t const dim      = t.dim(static_cast<int>(d));
            auto [it, first_seen] = sizes.try_emplace(std::string_view{idx[d]}, std::pair{dim, who});
            if (!first_seen && it->second.first != dim) {
                EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                        "cg::einsum: index '{}' spans {} elements in operand {} but {} in operand {} - the operand "
                                        "shapes disagree for spec '{}'",
                                        idx[d], it->second.first, it->second.second, dim, who, parsed.raw);
            }
        }
    };
    scan(parsed.a_indices, A, 'A');
    scan(parsed.b_indices, B, 'B');
    scan(parsed.c_indices, C, 'C');
}

// build_einsum_descriptor now lives in Node.hpp so Graph::make_einsum_node can
// share it; see einsums::compute_graph::detail there.

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
            BType const &B, bool conj_a = false, bool conj_b = false) {
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
    // A ``conj(...)`` wrapper in the spec ORs with the conj_a / conj_b kwargs.
    conj_a = conj_a || parsed.conj_a;
    conj_b = conj_b || parsed.conj_b;

    detail::validate_einsum_dims(parsed, A, B, *C);

    auto &ctx = CaptureContext::current();
    if (!ctx.is_capturing()) {
        LabeledSection("einsum eager");
        dispatch::string_einsum(parsed, c_pf, C, ab_pf, A, B, conj_a, conj_b);
        return;
    }

    LabeledSection("einsum capture");
    // Capture mode with slots
    auto [a_id, a_slot] = ctx.get_slot(A);
    auto [b_id, b_slot] = ctx.get_slot(B);
    auto [c_id, c_slot] = ctx.get_slot(*C);

    auto params    = ctx.graph()->create_params(c_pf, ab_pf);
    params->conj_a = conj_a;
    params->conj_b = conj_b;
    auto desc      = detail::build_einsum_descriptor(parsed, params->c_pf, params->ab_pf, params->conj_a, params->conj_b);

    // Runtime-mutable index state. Created once per einsum capture and
    // shared between the descriptor (for pass introspection / rewrite)
    // and the executor lambda. Seeded with the parsed indices and the
    // link set that `detail::build_einsum_descriptor` computes from
    // them: avoids recomputing link indices on every execute().
    auto indices = ctx.graph()->create_indices(parsed.a_indices, parsed.b_indices, parsed.c_indices, desc.spec.link_indices);
    desc.indices = indices;
    desc.params  = params;

    // BLAS-level batching hint. Only populated when the contraction is
    // a pure 2D GEMM pattern (rank-2 inputs/output, one link index),
    // that's the shape `blas::gemm_batch` accepts. For other shapes the
    // hint stays null and GEMMBatching falls through. `trans_a`/`trans_b`
    // follow string_gemm's convention: 'T' when the link is the first
    // index of A (resp. last index of B), 'N' otherwise.
    // A permute_view keeps the storage-order FLAG of its parent but presents
    // reordered strides, so is_row_major()/is_column_major() alone cannot
    // prove the canonical layout the fast paths below assume (found by the
    // large-rank differential fuzzer: a view with the slice axes swapped
    // passed the flag gates and produced wrong results). Verify the strides
    // are actually monotone in the flag's direction; size-1 axes are never
    // traversed, so their (possibly inflated) strides are ignored.
    auto layout_matches_flag = [](auto const &impl) {
        bool const   rm    = impl.is_row_major();
        size_t const rank  = impl.rank();
        size_t       prev  = 0;
        bool         first = true;
        for (size_t n = 0; n < rank; ++n) {
            size_t const d = rm ? rank - 1 - n : n;
            if (impl.dim(d) <= 1)
                continue;
            size_t const st = impl.stride(d);
            if (!first && st < prev)
                return false;
            prev  = st;
            first = false;
        }
        return true;
    };

    if (detail::tensor_rank(A) == 2 && detail::tensor_rank(B) == 2 && detail::tensor_rank(*C) == 2) {
        // C = op(A) * op(B) requires C's FIRST index to be the one A contributes
        // and its SECOND the one B contributes. "ia <- ma ; mi" has them swapped,
        // and the m/n/k below would describe a different matrix product. The
        // generic kernel never reads the hint, so a wrong one only surfaces once
        // GEMMBatching batches the node. See Graph::make_einsum_node, same gate.
        auto const roles_match = [&]() {
            if (parsed.a_indices.size() != 2 || parsed.b_indices.size() != 2 || parsed.c_indices.size() != 2 ||
                desc.spec.link_indices.size() != 1) {
                return false;
            }
            auto const &lnk  = desc.spec.link_indices[0];
            auto const  free = [&lnk](std::vector<std::string> const &idx) { return idx[0] == lnk ? idx[1] : idx[0]; };
            return parsed.c_indices[0] == free(parsed.a_indices) && parsed.c_indices[1] == free(parsed.b_indices);
        };
        if (parsed.a_indices.size() == 2 && parsed.b_indices.size() == 2 && parsed.c_indices.size() == 2 &&
            desc.spec.link_indices.size() == 1 && roles_match() && layout_matches_flag(A.impl()) && layout_matches_flag(B.impl()) &&
            layout_matches_flag(C->impl())) {
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
            bool const all_contig = A.impl().is_contiguous() && B.impl().is_contiguous() && C->impl().is_contiguous() &&
                                    layout_matches_flag(A.impl()) && layout_matches_flag(B.impl()) && layout_matches_flag(C->impl());
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

            // Conjugated batched einsums skip this gemm_batch fast path (it only
            // emits 'N'/'T' trans, never conjugation) and fall through to the
            // conj-aware generic string_einsum executor below.
            if (shape_ok && positions_match && all_contig && (row_mode || col_mode) && !params->conj_a && !params->conj_b) {
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

                    bool const swap_ab = row_mode;

                    // The per-slice pointer tables are pure functions of the
                    // three base pointers (base + i*stride); rebuild them only
                    // when a rebind moves a base, not on every replay. One
                    // cache per captured node; a node never runs concurrently
                    // with itself, so no synchronization is needed.
                    struct BatchPtrTables {
                        T const               *base_a{nullptr};
                        T const               *base_b{nullptr};
                        T                     *base_c{nullptr};
                        std::vector<T const *> a_arr;
                        std::vector<T const *> b_arr;
                        std::vector<T *>       c_arr;
                    };
                    auto tables = std::make_shared<BatchPtrTables>();

                    auto executor = [d, swap_ab, a_slot, b_slot, c_slot, tables]() {
                        LabeledSection("einsum batched execute");
                        ProfileAnnotate("m", static_cast<int64_t>(d.m));
                        ProfileAnnotate("n", static_cast<int64_t>(d.n));
                        ProfileAnnotate("k", static_cast<int64_t>(d.k));
                        ProfileAnnotate("batch", static_cast<int64_t>(d.batch_count));
                        auto const *base_a = static_cast<T const *>(static_cast<AType const *>(a_slot->ptr)->data());
                        auto const *base_b = static_cast<T const *>(static_cast<BType const *>(b_slot->ptr)->data());
                        auto       *base_c = static_cast<T *>(static_cast<CType *>(c_slot->ptr)->data());

                        if (base_a != tables->base_a || base_b != tables->base_b || base_c != tables->base_c) {
                            tables->a_arr.resize(d.batch_count);
                            tables->b_arr.resize(d.batch_count);
                            tables->c_arr.resize(d.batch_count);
                            for (int i = 0; i < d.batch_count; ++i) {
                                tables->a_arr[i] = base_a + i * d.batch_stride_a;
                                tables->b_arr[i] = base_b + i * d.batch_stride_b;
                                tables->c_arr[i] = base_c + i * d.batch_stride_c;
                            }
                            tables->base_a = base_a;
                            tables->base_b = base_b;
                            tables->base_c = base_c;
                        }

                        T const **blas_a = swap_ab ? tables->b_arr.data() : tables->a_arr.data();
                        T const **blas_b = swap_ab ? tables->a_arr.data() : tables->b_arr.data();

                        if constexpr (std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>) {
                            using R = typename T::value_type;
                            T alpha{static_cast<R>(d.alpha.real()), static_cast<R>(d.alpha.imag())};
                            T beta{static_cast<R>(d.beta.real()), static_cast<R>(d.beta.imag())};
                            blas::gemm_batch<T>(d.trans_a, d.trans_b, d.m, d.n, d.k, alpha, blas_a, d.lda, blas_b, d.ldb, beta,
                                                tables->c_arr.data(), d.ldc, d.batch_count);
                        } else {
                            blas::gemm_batch<T>(d.trans_a, d.trans_b, d.m, d.n, d.k, static_cast<T>(d.alpha.real()), blas_a, d.lda, blas_b,
                                                d.ldb, static_cast<T>(d.beta.real()), tables->c_arr.data(), d.ldc, d.batch_count);
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

    // This node's packed-GEMM memo, so a replay skips assembling the
    // contraction spec, the plan-cache key and its stride vectors, and the
    // lookup itself. Re-validated against the live indices and operand layout
    // on every call, so a pass rewriting either is honored (see
    // packed_gemm::ContractionSite).
    auto pg_site = std::make_shared<packed_gemm::ContractionSite>();

    // Capture the shared indices + params by shared_ptr and hand the dispatch
    // the LIVE spec by reference, so a pass that rewrites indices is honored
    // without copying three vector<string> per call.
    auto executor = [indices, params, a_slot, b_slot, c_slot, pg_site]() {
        LabeledSection("einsum execute");
        ProfileAnnotate("a_size", static_cast<int64_t>(static_cast<AType const *>(a_slot->ptr)->size()));
        ProfileAnnotate("b_size", static_cast<int64_t>(static_cast<BType const *>(b_slot->ptr)->size()));
        ProfileAnnotate("c_size", static_cast<int64_t>(static_cast<CType *>(c_slot->ptr)->size()));
        // link_indices was computed once at capture (sorted, same order the
        // dispatch derives); passing it spares every replay three set builds.
        dispatch::string_einsum(indices->spec, as<T>(params->c_pf), static_cast<CType *>(c_slot->ptr), as<T>(params->ab_pf),
                                *static_cast<AType const *>(a_slot->ptr), *static_cast<BType const *>(b_slot->ptr), params->conj_a,
                                params->conj_b, &indices->link_indices, pg_site.get());
    };

    // The same site the executor lambda holds, so a plan-time pass can pin this
    // node's kernel route where the dispatch will read it.
    desc.site = pg_site;

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
            BType const &B, bool conj_a = false, bool conj_b = false) {
    using T = typename AType::ValueType;
    if (conj_a || conj_b) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "cg::einsum: conjugation (conj_a/conj_b) is not yet supported for tiled operands");
    }
    static_assert(IsTiledTensorV<std::remove_cvref_t<AType>> && IsTiledTensorV<std::remove_cvref_t<BType>> &&
                      IsTiledTensorV<std::remove_cvref_t<CType>>,
                  "cg::einsum with a tiled operand currently requires all of A, B, C to be TiledRuntimeTensor "
                  "(mixed tiled/dense is not supported yet)");

    auto parse_result = parse_einsum_spec(static_cast<std::string_view>(spec));
    if (!parse_result) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "{}", parse_result.error().message);
    }
    auto &parsed = parse_result.value();
    if (parsed.conj_a || parsed.conj_b) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                "cg::einsum: conjugation (conj(...) in the spec) is not yet supported for tiled operands");
    }

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
        detail::tiled_runtime_einsum<T>(indices->spec, as<T>(params->c_pf), static_cast<CType *>(c_slot->ptr), as<T>(params->ab_pf),
                                        *static_cast<AType const *>(a_slot->ptr), *static_cast<BType const *>(b_slot->ptr));
    };

    // OpKind::Custom on purpose: the node participates in dependency ordering and
    // lifecycle, but stays invisible to the einsum-rewriting passes (contraction
    // planning, GEMM batching, stream fusion), which would otherwise synthesize
    // *dense* intermediates for operands that have no single buffer.
    //
    // It DOES carry a TiledEinsumDescriptor holding the same live indices/params
    // the executor reads, so a pass that explicitly asks for a tiled einsum can
    // inspect and rewrite it -- the tile-expansion pass needs the spec and the
    // prefactors, and a bare closure hides both. The descriptor type is distinct
    // from EinsumDescriptor precisely so that no dense-einsum pass picks it up.
    TiledEinsumDescriptor tdesc;
    tdesc.indices = indices;
    tdesc.params  = params;
    ctx.record(OpKind::Custom, std::move(label), {a_id, b_id}, {c_id}, std::move(executor), std::move(tdesc));
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
/// Complex prefactors are fully supported for complex dtypes:
/// Graph::create_params and the einsum node store PrefactorScalar (a
/// variant covering float/double/complex<float>/complex<double>), so a
/// complex ``c_pf``/``ab_pf`` round-trips through capture and replay
/// without narrowing.
///
/// ``conj_a`` / ``conj_b`` conjugate A / B inside the contraction for complex
/// dtypes, and are a no-op for real. They conjugate only the elements. A
/// conjugate-transpose is the conj flag combined with transposed index placement
/// in the spec, just like any transpose in einsum. To compute ``C = A^H @ B``,
/// store A as ``(k, i)`` so its row index k is the one contracted:
/// @code
/// // (A^H @ B)[i,j] = sum_k conj(A[k,i]) * B[k,j]
/// einsum("ij <- ki ; kj", C, A, B, conj_a=True);   // A^H @ B
/// einsum("ij <- ik ; jk", C, A, B, conj_b=True);   // A @ B^H
/// einsum("ij <- ik ; kj", C, A, B, conj_a=True);   // conj(A) @ B  (no transpose)
/// @endcode
/// Equivalently, wrap an operand in ``conj(...)`` directly in the spec; it ORs
/// with the kwargs, so ``einsum("ij <- conj(ki) ; kj", C, A, B)`` is also A^H @ B.
/// Native for GEMM-shaped contractions via PackedGemm, with no operand copy. The
/// generic loop conjugates per element otherwise. Not supported for tiled operands.
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
                       typename AType::ValueType ab_pf = typename AType::ValueType{1}, bool conj_a = false, bool conj_b = false) {
    einsum(EinsumFormatString(std::string_view{spec}), c_pf, C, ab_pf, A, B, conj_a, conj_b);
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

EINSUMS_NAMESPACE_END(compute_graph)
