//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/Prefactor.hpp>
#include <Einsums/ComputeGraphTypes/Enums.hpp>
#include <Einsums/ComputeGraphTypes/Ids.hpp>

#include <complex>
#include <cstddef>
#include <unordered_set>
#include <variant>
#include <vector>

namespace einsums::compute_graph::passes {

/**
 * @brief True when @p nd overwrites its destination without reading it.
 *
 * A prefactor-bearing op (Einsum/Permute/BatchedGemm) whose destination
 * prefactor is zero writes C fresh, making any prior write to that tensor dead.
 * Shared by ScaleAbsorption (drop a scale the next op overwrites) and CSE
 * (only a pure-overwrite result is a safe common-subexpression survivor).
 */
[[nodiscard]] inline bool pure_overwrite(Node const &nd) {
    if (auto const *e = std::get_if<EinsumDescriptor>(&nd.op_data)) {
        return is_zero(e->c_prefactor);
    }
    if (auto const *p = std::get_if<PermuteDescriptor>(&nd.op_data)) {
        return p->beta == 0.0;
    }
    if (auto const *b = std::get_if<BatchedGemmDescriptor>(&nd.op_data)) {
        return b->beta == std::complex<double>{0.0, 0.0};
    }
    return false;
}

/**
 * @brief True when @p nd reads its destination (accumulates into it).
 *
 * The always-accumulating kinds (Scale/Axpy/Axpby/ElementTransform), or a
 * prefactor-bearing op with a non-zero destination prefactor. Used by
 * LoopInvariantHoisting to refuse to hoist a self-modifying update out of a loop.
 *
 * @note This is NOT the negation of pure_overwrite: an op with no prefactor
 *       descriptor (e.g. Gemm, Dot) returns false from both.
 */
[[nodiscard]] inline bool reads_destination(Node const &nd) {
    switch (nd.kind) {
    case OpKind::Scale:
    case OpKind::Axpy:
    case OpKind::Axpby:
    case OpKind::ElementTransform:
        return true;
    default:
        break;
    }
    if (auto const *e = std::get_if<EinsumDescriptor>(&nd.op_data)) {
        return !is_zero(e->c_prefactor);
    }
    if (auto const *p = std::get_if<PermuteDescriptor>(&nd.op_data)) {
        return p->beta != 0.0;
    }
    if (auto const *b = std::get_if<BatchedGemmDescriptor>(&nd.op_data)) {
        return b->beta != std::complex<double>{0.0, 0.0};
    }
    return false;
}

/**
 * @brief Interference gate shared by the reassociation passes
 *        (DistributiveFactoring, LinearCombinationContractionFolding).
 *
 * These passes move a combined op into the first member's slot, which is sound
 * only if nothing between the first and last member disturbs the partial sum or
 * a factor. Returns true when some node strictly between positions @p first and
 * @p last (exclusive), other than the group members flagged in @p is_member:
 *   - writes @p output_id or any tensor in @p operand_ids (clobbering the partial
 *     sum or a factor mid-fold),
 *   - reads @p output_id (observing the partial sum before it is complete), or
 *   - when @p reject_control_flow is set, is a control-flow node whose hidden
 *     sub-graph I/O cannot be inspected.
 *
 * @p reject_control_flow lets DistributiveFactoring disqualify a span containing
 * a Loop/Conditional while LinearCombinationContractionFolding keeps its existing
 * (control-flow-agnostic) behavior.
 */
[[nodiscard]] inline bool span_interferes(std::vector<Node> const &nodes, std::size_t first, std::size_t last,
                                          std::vector<bool> const &is_member, TensorId output_id,
                                          std::unordered_set<TensorId> const &operand_ids, bool reject_control_flow) {
    for (std::size_t n = first + 1; n < last; ++n) {
        if (is_member[n]) {
            continue;
        }
        Node const &other = nodes[n];
        if (reject_control_flow && is_control_flow(other.kind)) {
            return true;
        }
        for (auto const out : other.outputs) {
            if (out == output_id || operand_ids.count(out) != 0) {
                return true;
            }
        }
        for (auto const in : other.inputs) {
            if (in == output_id) {
                return true;
            }
        }
    }
    return false;
}

} // namespace einsums::compute_graph::passes
