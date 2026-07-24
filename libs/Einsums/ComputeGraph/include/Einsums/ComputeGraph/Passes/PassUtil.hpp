//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/ComputeGraph/Prefactor.hpp>
#include <Einsums/ComputeGraphTypes/Enums.hpp>

#include <complex>
#include <variant>

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

} // namespace einsums::compute_graph::passes
