//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/CaptureContext.hpp>
#include <Einsums/ComputeGraph/Detail/ErasedOperations.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/PackedGemm/ContractionKey.hpp>
#include <Einsums/Profile.hpp>

#include <fmt/format.h>

#include <array>
#include <cstddef>
#include <optional>
#include <span>
#include <stdexcept>
#include <string_view>

#include "MaskKernels.hpp"
#include "Record.hpp"

EINSUMS_NAMESPACE_BEGIN(compute_graph::detail)

CmpOp compare_op_from_name(char const *api, std::string_view name) {
    if (auto const op = cmp_op_from_name(name)) {
        return *op;
    }
    EINSUMS_THROW_EXCEPTION(std::invalid_argument, "{}: '{}' is not a comparison; the comparisons are gt, ge, lt, le, eq, ne", api, name);
}

// ── compare ───────────────────────────────────────────────────────────────────

template <typename T>
void eager_compare(CmpOp op, Impl<T> const &A, Impl<T> const &B, Impl<T> &out) {
    WAGGLE_ZONE("compare eager");
    run_compare<T>(op, A, B, out);
}

template <typename T>
void eager_compare_scalar(CmpOp op, Impl<T> const &A, double scalar, Impl<T> &out) {
    WAGGLE_ZONE("compare eager");
    run_compare_scalar<T>(op, A, scalar, out);
}

void capture_compare(CaptureContext &ctx, CmpOp op, TensorId a_id, std::optional<TensorId> b_id, std::optional<double> scalar,
                     TensorId out_id, packed_gemm::ScalarType dtype, std::size_t rank) {
    WAGGLE_ZONE("compare capture");
    // The mask is overwritten in full, so it is the output only. The scalar form has one input:
    // its right-hand side is data in the descriptor, not a tensor.
    std::array<TensorId, 2> const inputs{a_id, b_id.value_or(TensorId{0})};
    std::size_t const             count = b_id.has_value() ? 2 : 1;
    std::vector<TensorId>         listed(inputs.begin(), inputs.begin() + static_cast<std::ptrdiff_t>(count));
    ctx.record_built(OpKind::Compare, fmt::format("compare({})", cmp_op_name(op)), dtype, rank,
                     CompareDescriptor{.op = op, .scalar = scalar}, std::span<TensorId const>{inputs.data(), count},
                     std::span<TensorId const>{&out_id, 1}, std::move(listed), {out_id});
}

// ── where ─────────────────────────────────────────────────────────────────────

template <typename T>
void eager_select(Impl<T> const &mask, Impl<T> const &A, Impl<T> const &B, Impl<T> &out) {
    WAGGLE_ZONE("where eager");
    run_select<T>(mask, A, B, out);
}

void capture_select(CaptureContext &ctx, TensorId mask_id, TensorId a_id, TensorId b_id, TensorId out_id, packed_gemm::ScalarType dtype,
                    std::size_t rank) {
    WAGGLE_ZONE("where capture");
    std::array<TensorId, 3> const inputs{mask_id, a_id, b_id};
    ctx.record_built(OpKind::Select, "where", dtype, rank, SelectDescriptor{}, inputs, std::span<TensorId const>{&out_id, 1},
                     {mask_id, a_id, b_id}, {out_id});
}

// Comparisons order their operands, so every op here is real-only; the select is real-only with
// them, because its mask is one of their 1/0 tensors.
#define EINSUMS_MASK_OPERATIONS(T)                                                                                                         \
    template EINSUMS_EXPORT void eager_compare<T>(CmpOp, Impl<T> const &, Impl<T> const &, Impl<T> &);                                     \
    template EINSUMS_EXPORT void eager_compare_scalar<T>(CmpOp, Impl<T> const &, double, Impl<T> &);                                       \
    template EINSUMS_EXPORT void eager_select<T>(Impl<T> const &, Impl<T> const &, Impl<T> const &, Impl<T> &);

EINSUMS_CG_REAL_ELEMENT_TYPES(EINSUMS_MASK_OPERATIONS)
#undef EINSUMS_MASK_OPERATIONS

EINSUMS_NAMESPACE_END(compute_graph::detail)
