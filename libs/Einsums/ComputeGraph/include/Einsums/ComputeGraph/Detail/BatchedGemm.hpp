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
        std::vector<void const *> a_vs(d.batch_count);
        std::vector<void const *> b_vs(d.batch_count);
        std::vector<void *>       c_vs(d.batch_count);
        for (int i = 0; i < d.batch_count; ++i) {
            a_vs[i] = a_exs[i]().first;
            b_vs[i] = b_exs[i]().first;
            c_vs[i] = c_exs[i]().first;
        }
        switch (d.scalar) {
        case BlasScalar::Float:
            run_batched_gemm<float>(d, a_vs, b_vs, c_vs);
            break;
        case BlasScalar::Double:
            run_batched_gemm<double>(d, a_vs, b_vs, c_vs);
            break;
        case BlasScalar::ComplexFloat:
            run_batched_gemm_complex<std::complex<float>>(d, a_vs, b_vs, c_vs);
            break;
        case BlasScalar::ComplexDouble:
            run_batched_gemm_complex<std::complex<double>>(d, a_vs, b_vs, c_vs);
            break;
        }
    };
}

EINSUMS_NAMESPACE_END(compute_graph::detail)
