//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/BLAS/ThreadControl.hpp>
#include <Einsums/ComputeGraph/Node.hpp>
#include <Einsums/Config/Namespace.hpp>

EINSUMS_NAMESPACE_BEGIN(compute_graph)

/**
 * @brief Whether a BLAS call's thread count can be dictated by its caller.
 *
 * Two vendors can be told, in different ways: MKL through the per-thread setter
 * it exposes, OpenBLAS through the OpenMP ICV it reads when it was built
 * against OpenMP. Accelerate reads neither - it schedules through GCD - and a
 * reference BLAS is serial and has nothing to dictate. Both answer false.
 *
 * A false answer is not a correctness matter: a node planned wide on such a
 * vendor still computes the right thing, it just does not get the threads the
 * plan assumed, which makes every width above 1 a lie the plan is built on.
 */
[[nodiscard]] inline bool blas_route_is_moldable() {
    return blas::has_per_thread_control() || blas::threads_with_openmp();
}

/**
 * @brief Whether this node's kernel will actually see the width it is planned at.
 *
 * The executor's wrap sets the OpenMP ICV and the vendor BLAS thread count
 * around the node, so the question is whether the kernel the node runs consumes
 * one of those. Everything einsums threads itself - HPTT, the elementwise
 * kernels, the batched-GEMM entry loop - forks from the ICV and so is always
 * moldable; a node whose work is a vendor BLAS call is moldable only where the
 * vendor can be told (@ref blas_route_is_moldable).
 *
 * Deliberately conservative in one direction only. The route a contraction
 * takes is a run-time decision (@c dispatch::last_dispatch_route), so a node
 * that MIGHT reach BLAS is classified as though it will; the cost of being
 * wrong is a node planned at width 1 that could have been wider, never a node
 * given a width it cannot use.
 *
 * Intended for a width-planning pass, which is the only caller that has to
 * choose; the executor honors whatever width it finds without asking.
 */
[[nodiscard]] inline bool kernel_moldability(Node const &node) {
    switch (node.kind) {
    // Kernels that are a vendor call, or that lower to one often enough that
    // planning them as anything else would be guessing.
    case OpKind::Einsum:
    case OpKind::Gemm:
    case OpKind::Gemv:
    case OpKind::Ger:
    case OpKind::Dot:
    case OpKind::GroupedDot:
    case OpKind::BatchedGemm:
    case OpKind::GroupedBatchedGemm:
    case OpKind::SymmGemm:
    case OpKind::Scale:
    case OpKind::Axpby:
    case OpKind::GroupedAxpby:
    case OpKind::DirectProduct:
    case OpKind::DirectDivision:
    case OpKind::Norm:
    case OpKind::Trace:
    case OpKind::SVD:
    case OpKind::SVD_DD:
    case OpKind::TruncatedSVD:
    case OpKind::QR:
    case OpKind::Syev:
    case OpKind::Heev:
    case OpKind::Geev:
    case OpKind::TruncatedSyev:
    case OpKind::Gesv:
    case OpKind::Getrf:
    case OpKind::Getri:
    case OpKind::Invert:
    case OpKind::Pseudoinverse:
    case OpKind::Det:
    case OpKind::Pow:
    case OpKind::SolveLyapunov:
        return blas_route_is_moldable();

    // Control flow runs a nested execute(), whose nodes carry their own widths;
    // giving the container one would count the same machine twice.
    case OpKind::Conditional:
    case OpKind::Loop:
        return false;

    default:
        return true;
    }
}

EINSUMS_NAMESPACE_END(compute_graph)
