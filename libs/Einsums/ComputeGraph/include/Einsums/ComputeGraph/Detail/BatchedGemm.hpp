//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/BLAS.hpp>
#include <Einsums/ComputeGraphTypes/Descriptors.hpp>
#include <Einsums/Config/Namespace.hpp>

#include <complex>
#include <functional>
#include <utility>
#include <vector>

#ifdef _OPENMP
#    include <omp.h>
#endif

EINSUMS_NAMESPACE_BEGIN(compute_graph::detail)

/**
 * @brief Shared executor for @ref OpKind::BatchedGemm nodes.
 *
 * Two producers build these nodes and must agree exactly on how they run: the
 * GEMMBatching pass, which fuses independent 2D einsums after the fact, and
 * ``cg::batched_gemm``, which lets a caller emit the fused form directly. The
 * dispatch lives here so the two cannot drift.
 *
 * Pointers are re-extracted on every call rather than baked in, because
 * ``Graph::rebind`` and the MemoryPlanning arena can both move a tensor's
 * storage between executions.
 */

/// @{
/// Type the erased pointers and dispatch the matching `blas::gemm_batch`.
template <typename T>
void run_batched_gemm(BatchedGemmDescriptor const &d, std::vector<void const *> const &a_vs, std::vector<void const *> const &b_vs,
                      std::vector<void *> const &c_vs) {
    std::vector<T const *> a_arr(d.batch_count);
    std::vector<T const *> b_arr(d.batch_count);
    std::vector<T *>       c_arr(d.batch_count);
    for (int i = 0; i < d.batch_count; ++i) {
        a_arr[i] = static_cast<T const *>(a_vs[i]);
        b_arr[i] = static_cast<T const *>(b_vs[i]);
        c_arr[i] = static_cast<T *>(c_vs[i]);
    }
    blas::gemm_batch<T>(d.trans_a, d.trans_b, d.m, d.n, d.k, static_cast<T>(d.alpha.real()), a_arr.data(), d.lda, b_arr.data(), d.ldb,
                        static_cast<T>(d.beta.real()), c_arr.data(), d.ldc, d.batch_count);
}

/// The descriptor carries the full complex prefactor; preserve both parts, a
/// complex prefactor such as a phase factor must not be truncated to its real
/// part.
template <typename Complex>
void run_batched_gemm_complex(BatchedGemmDescriptor const &d, std::vector<void const *> const &a_vs, std::vector<void const *> const &b_vs,
                              std::vector<void *> const &c_vs) {
    using T = typename Complex::value_type;
    std::vector<Complex const *> a_arr(d.batch_count);
    std::vector<Complex const *> b_arr(d.batch_count);
    std::vector<Complex *>       c_arr(d.batch_count);
    for (int i = 0; i < d.batch_count; ++i) {
        a_arr[i] = static_cast<Complex const *>(a_vs[i]);
        b_arr[i] = static_cast<Complex const *>(b_vs[i]);
        c_arr[i] = static_cast<Complex *>(c_vs[i]);
    }
    Complex const alpha{static_cast<T>(d.alpha.real()), static_cast<T>(d.alpha.imag())};
    Complex const beta{static_cast<T>(d.beta.real()), static_cast<T>(d.beta.imag())};
    blas::gemm_batch<Complex>(d.trans_a, d.trans_b, d.m, d.n, d.k, alpha, a_arr.data(), d.lda, b_arr.data(), d.ldb, beta, c_arr.data(),
                              d.ldc, d.batch_count);
}
/// @}

/// Per-slice pointer extractors, one entry per batch member.
using BatchedGemmExtractors       = std::vector<std::function<std::pair<void const *, int>()>>;
using BatchedGemmOutputExtractors = std::vector<std::function<std::pair<void *, int>()>>;

/**
 * @brief Extract every member's pointer and issue the batch, in one pass.
 *
 * Each member's pointer is three dependent loads - slot, tensor, impl - over
 * tensors that are separate allocations, so a batch of a few thousand walks a
 * few thousand cache misses with nothing to prefetch. On a 2048-member 8x8x8
 * double batch that is around 10% of the node, which is small but is also
 * entirely serial work sitting under a call whose whole point is to be
 * parallel, so it is worth not paying.
 *
 * The members are independent, so the walk runs on an OpenMP team when there is
 * one to be had. Guarded on @c omp_in_parallel because the OpenMP executor
 * already replays whole graphs as a team, and nesting a second one inside a node
 * is how the OpenMP-built OpenBLAS gets miscomputed.
 *
 * And the pointers land in typed arrays directly. They used to be extracted into
 * `void const *` vectors and copied into typed ones, which is a second
 * allocation and a second pass over the batch for nothing.
 */
template <typename T, typename Alpha>
void extract_and_run(BatchedGemmDescriptor const &d, BatchedGemmExtractors const &a_exs, BatchedGemmExtractors const &b_exs,
                     BatchedGemmOutputExtractors const &c_exs, Alpha alpha, Alpha beta) {
    std::vector<T const *> a_arr(d.batch_count);
    std::vector<T const *> b_arr(d.batch_count);
    std::vector<T *>       c_arr(d.batch_count);

#ifdef _OPENMP
#    pragma omp parallel for schedule(static) if (!omp_in_parallel())
#endif
    for (int i = 0; i < d.batch_count; ++i) {
        a_arr[i] = static_cast<T const *>(a_exs[i]().first);
        b_arr[i] = static_cast<T const *>(b_exs[i]().first);
        c_arr[i] = static_cast<T *>(c_exs[i]().first);
    }

    blas::gemm_batch<T>(d.trans_a, d.trans_b, d.m, d.n, d.k, alpha, a_arr.data(), d.lda, b_arr.data(), d.ldb, beta, c_arr.data(), d.ldc,
                        d.batch_count);
}

/**
 * @brief Build the node executor for a @ref BatchedGemmDescriptor.
 *
 * @param d Shared BLAS parameters (m/n/k, leading dims, trans flags, prefactors,
 *          batch count, element type).
 * @param a_exs,b_exs,c_exs Per-member extractors, ordered so index @c i of each
 *          refers to the same contraction.
 */
inline std::function<void()> make_batched_gemm_executor(BatchedGemmDescriptor d, BatchedGemmExtractors a_exs, BatchedGemmExtractors b_exs,
                                                        BatchedGemmOutputExtractors c_exs) {
    return [d, a_exs = std::move(a_exs), b_exs = std::move(b_exs), c_exs = std::move(c_exs)]() {
        switch (d.scalar) {
        case BlasScalar::Float:
            extract_and_run<float>(d, a_exs, b_exs, c_exs, static_cast<float>(d.alpha.real()), static_cast<float>(d.beta.real()));
            break;
        case BlasScalar::Double:
            extract_and_run<double>(d, a_exs, b_exs, c_exs, d.alpha.real(), d.beta.real());
            break;
        // The descriptor carries the full complex prefactor; preserve both
        // parts, a phase factor must not be truncated to its real part.
        case BlasScalar::ComplexFloat:
            extract_and_run<std::complex<float>>(
                d, a_exs, b_exs, c_exs, std::complex<float>{static_cast<float>(d.alpha.real()), static_cast<float>(d.alpha.imag())},
                std::complex<float>{static_cast<float>(d.beta.real()), static_cast<float>(d.beta.imag())});
            break;
        case BlasScalar::ComplexDouble:
            extract_and_run<std::complex<double>>(d, a_exs, b_exs, c_exs, d.alpha, d.beta);
            break;
        }
    };
}

EINSUMS_NAMESPACE_END(compute_graph::detail)
