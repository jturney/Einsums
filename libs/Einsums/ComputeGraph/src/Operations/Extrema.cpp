//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/CaptureContext.hpp>
#include <Einsums/ComputeGraph/Detail/ErasedOperations.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraphTypes/Descriptors.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/PackedGemm/ContractionKey.hpp>
#include <Einsums/Profile.hpp>

#include <fmt/format.h>

#include <array>
#include <cstddef>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "ExtremaKernels.hpp"
#include "Record.hpp"

EINSUMS_NAMESPACE_BEGIN(compute_graph::detail)

// ── maximum / minimum ─────────────────────────────────────────────────────────

template <typename T>
void eager_elementwise_max_min(MaxMinOp op, Impl<T> const &A, Impl<T> const &B, Impl<T> &C) {
    WAGGLE_ZONE(op == MaxMinOp::Max ? "maximum eager" : "minimum eager");
    run_elementwise_max_min<T>(op, A, B, C);
}

void capture_elementwise_max_min(CaptureContext &ctx, MaxMinOp op, TensorId a_id, TensorId b_id, TensorId c_id,
                                 packed_gemm::ScalarType dtype, std::size_t rank) {
    WAGGLE_ZONE(op == MaxMinOp::Max ? "maximum capture" : "minimum capture");
    // Recorded with a descriptor, never a closure, so ExecutorBuilder rebuilds the executor and
    // a saved graph says what the node computes. C is overwritten in full, so it is the output
    // only; when C is A or B it is already listed as that operand.
    std::array<TensorId, 2> const inputs{a_id, b_id};
    ctx.record_built(OpKind::ElementwiseMaxMin, op == MaxMinOp::Max ? "maximum" : "minimum", dtype, rank,
                     ElementwiseMaxMinDescriptor{.op = op}, inputs, std::span<TensorId const>{&c_id, 1}, {a_id, b_id}, {c_id});
}

// ── max_axes / min_axes ───────────────────────────────────────────────────────

template <typename T>
void eager_reduce_axes(MaxMinOp op, Impl<T> &out, Impl<T> const &A, std::vector<std::size_t> const &kept) {
    WAGGLE_ZONE(op == MaxMinOp::Max ? "max_axes eager" : "min_axes eager");
    run_reduce_axes<T>(op, A, kept, out);
}

void capture_reduce_axes(CaptureContext &ctx, MaxMinOp op, TensorId a_id, TensorId out_id, packed_gemm::ScalarType dtype,
                         std::size_t out_rank, std::vector<std::size_t> kept) {
    WAGGLE_ZONE(op == MaxMinOp::Max ? "max_axes capture" : "min_axes capture");
    ctx.record_built(OpKind::ReduceAxes, op == MaxMinOp::Max ? "max_axes" : "min_axes", dtype, out_rank,
                     ReduceAxesDescriptor{.op = op, .kept = std::move(kept)}, std::span<TensorId const>{&a_id, 1},
                     std::span<TensorId const>{&out_id, 1}, {a_id}, {out_id});
}

// Ordering is not defined for complex numbers, so every op here is real-only.
#define EINSUMS_EXTREMA_OPERATIONS(T)                                                                                                      \
    template EINSUMS_EXPORT void eager_elementwise_max_min<T>(MaxMinOp, Impl<T> const &, Impl<T> const &, Impl<T> &);                      \
    template EINSUMS_EXPORT void eager_reduce_axes<T>(MaxMinOp, Impl<T> &, Impl<T> const &, std::vector<std::size_t> const &);

EINSUMS_CG_REAL_ELEMENT_TYPES(EINSUMS_EXTREMA_OPERATIONS)
#undef EINSUMS_EXTREMA_OPERATIONS

EINSUMS_NAMESPACE_END(compute_graph::detail)
