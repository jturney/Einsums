//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/CaptureContext.hpp>
#include <Einsums/ComputeGraph/Detail/ErasedEinsum.hpp>
#include <Einsums/ComputeGraph/Detail/ErasedOperations.hpp>
#include <Einsums/ComputeGraph/EinsumSpec.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/Prefactor.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/PackedGemm/ContractionKey.hpp>
#include <Einsums/Profile.hpp>
#include <Einsums/TensorPermute/Permute.hpp>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <complex>
#include <memory>
#include <span>
#include <variant>

EINSUMS_NAMESPACE_BEGIN(compute_graph::detail)

template <typename T>
using Impl = einsums::detail::TensorImpl<T>;

// ── permute ───────────────────────────────────────────────────────────────────

template <typename T>
void eager_permute(ParsedPermuteSpec const &parsed, T beta, Impl<T> &C, T alpha, Impl<T> const &A) {
    LabeledSection("permute eager");
    dispatch::string_permute_impl<T>(parsed, beta, &C, alpha, A);
}

template <typename T>
void capture_permute(CaptureContext &ctx, ParsedPermuteSpec const &parsed, T beta, T alpha, TensorId a_id, TensorId c_id,
                     std::size_t rank) {
    LabeledSection("permute capture");

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
    desc.operators = parsed.operators;
    // Live scalars in the operands' own type, so a pass that rewrites a
    // prefactor is obeyed on the next replay. The executor used to bake
    // copies of these and ignore the descriptor entirely.
    desc.params        = std::make_shared<ElementwiseParams>();
    desc.params->alpha = PrefactorScalar{alpha};
    desc.params->beta  = PrefactorScalar{beta};

    auto label = fmt::format("permute: C[{}] = A[{}]", fmt::join(parsed.c_indices, ","), fmt::join(parsed.a_indices, ","));

    ctx.record_built(OpKind::Permute, std::move(label), packed_gemm::get_scalar_type<T>(), rank, std::move(desc),
                     std::span<TensorId const>{&a_id, 1}, std::span<TensorId const>{&c_id, 1}, {a_id}, {c_id});
}

// ── transpose ─────────────────────────────────────────────────────────────────

template <typename T>
void eager_transpose(Impl<T> &C, Impl<T> const &A) {
    LabeledSection("transpose eager");
    tensor_permute::transpose(&C, A);
}

template <typename T>
void capture_transpose(CaptureContext &ctx, TensorId a_id, TensorId c_id, std::size_t rank) {
    LabeledSection("transpose capture");

    // No descriptor, deliberately: a transpose is a fixed permutation with no
    // scalars, so (kind, dtype, rank, operand ids) is its complete content and
    // an empty descriptor alternative would record nothing. See build_executor.
    ctx.record_built(OpKind::Transpose, "transpose", packed_gemm::get_scalar_type<T>(), rank, std::monostate{},
                     std::span<TensorId const>{&a_id, 1}, std::span<TensorId const>{&c_id, 1}, {a_id}, {c_id});
}

#define EINSUMS_PERMUTE_OPERATIONS(T)                                                                                                      \
    template EINSUMS_EXPORT void eager_permute<T>(ParsedPermuteSpec const &, T, Impl<T> &, T, Impl<T> const &);                            \
    template EINSUMS_EXPORT void capture_permute<T>(CaptureContext &, ParsedPermuteSpec const &, T, T, TensorId, TensorId, std::size_t);   \
    template EINSUMS_EXPORT void eager_transpose<T>(Impl<T> &, Impl<T> const &);                                                           \
    template EINSUMS_EXPORT void capture_transpose<T>(CaptureContext &, TensorId, TensorId, std::size_t);

EINSUMS_PERMUTE_OPERATIONS(float)
EINSUMS_PERMUTE_OPERATIONS(double)
EINSUMS_PERMUTE_OPERATIONS(std::complex<float>)
EINSUMS_PERMUTE_OPERATIONS(std::complex<double>)
#undef EINSUMS_PERMUTE_OPERATIONS

EINSUMS_NAMESPACE_END(compute_graph::detail)
