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
 * Only a vendor with a per-thread setter qualifies, which today means MKL: its
 * setter scopes the request to the calling thread, so concurrent callers
 * wanting different widths is a regime the library supports.
 *
 * An OpenMP-built OpenBLAS OBEYS the caller's ICV but does not SUPPORT
 * concurrent callers whose ICVs differ, and treating obedience as moldability
 * is what let planned mixed widths corrupt results. Its server syncs a
 * process-global thread count to each caller's ``omp_get_max_threads()``,
 * frees the shared pack buffers whenever that global shrinks - under whatever
 * GEMM is in flight - and sizes its work queue from the global while forking
 * the team from the caller, so two callers disagreeing about the width
 * corrupt results or deadlock the position ring. Verified against 0.3.32
 * through 0.3.34 and the development head as of 2026-08; no upstream fix
 * covers it, so the ICV route deliberately answers false. With BLAS-route
 * nodes planned width 1 and the `VendorWidthFence` in the BLAS wrappers
 * clamping any vendor call made under a node-scoped width, at most one width
 * above 1 ever reaches the vendor - the unpinned calling thread's baseline -
 * which is the regime every executor ran in before widths existed.
 *
 * Accelerate reads neither knob - it schedules through GCD - and a reference
 * BLAS is serial and has nothing to dictate. Both answer false as well.
 *
 * For the vendors that answer false for lack of a knob, a wide plan is merely
 * ineffective: the node still computes the right thing, it just does not get
 * the threads the plan assumed, which makes every width above 1 a lie the
 * plan is built on.
 */
[[nodiscard]] inline bool blas_route_is_moldable() {
    return blas::has_per_thread_control();
}

/**
 * @brief Whether this node's kernel will actually see the width it is planned at.
 *
 * The executor's wrap sets the OpenMP ICV and the vendor BLAS thread count
 * around the node, so the question is whether the kernel the node runs consumes
 * one of those. Everything einsums threads itself - HPTT, the elementwise
 * kernels, the packed-GEMM loops, the batched-GEMM entry loop - forks from the
 * ICV and so is always moldable; a node whose work is a vendor BLAS call is
 * moldable only where the vendor can be told (@ref blas_route_is_moldable).
 *
 * Three kinds sit on the einsums-threaded side of that line for reasons worth
 * stating.
 *
 * A contraction routes through PackedGemm, whose engine declines to hand the
 * work to a single vendor GEMM when the caller holds a node width the wrappers'
 * fence would clamp; it packs and runs its own tiled loops instead, forking
 * from the ICV the width raised. Not every einsum shape does: GEMV- and
 * dot-shaped specs still end in a fenced vendor call and merely waste the width
 * they were given. That is the conservative direction being spent rather than
 * saved, and the planner's own size floors keep small nodes at width 1 anyway,
 * so what is wasted is a share of the budget on a node too big to be free.
 *
 * The two batched-GEMM kinds are einsums' own OpenMP loops over the batch, with
 * the vendor called nested-serial inside them. They fork from the ICV on every
 * vendor, which is why their wrappers carry no fence, and they have consumed
 * their caller's width since before widths were planned.
 *
 * Intended for a width-planning pass, which is the only caller that has to
 * choose; the executor honors whatever width it finds without asking.
 */
[[nodiscard]] inline bool kernel_moldability(Node const &node) {
    switch (node.kind) {
    // Contractions and batched GEMMs: einsums-threaded, see above.
    case OpKind::Einsum:
    case OpKind::BatchedGemm:
    case OpKind::GroupedBatchedGemm:
        return true;

    // Kernels that are a vendor call, or that lower to one often enough that
    // planning them as anything else would be guessing.
    case OpKind::Gemm:
    case OpKind::Gemv:
    case OpKind::Ger:
    case OpKind::Dot:
    case OpKind::GroupedDot:
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
