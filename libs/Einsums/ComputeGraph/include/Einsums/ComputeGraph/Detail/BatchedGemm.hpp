//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/BLAS.hpp>
#include <Einsums/ComputeGraph/ExecutorBuilder.hpp>
#include <Einsums/ComputeGraphTypes/Descriptors.hpp>
#include <Einsums/Config/Namespace.hpp>

#include <complex>
#include <cstddef>
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

/**
 * @brief One batch member's operand: where to read it from, and how far in.
 *
 * Data, not a closure. The accessor reaches the operand's live geometry through
 * the graph's slot, so ``Graph::rebind``, ``Graph::redirect_slot`` and the
 * MemoryPlanning arena are all honored without the producer of the node having
 * baked a pointer; @ref offset is how the *blocked* forms address one block of
 * a shared destination without materializing a view per member.
 *
 * @see OperandAccessor
 */
struct BatchedGemmOperand {
    OperandAccessor accessor;  ///< Resolves the operand's live @c TensorImpl.
    std::ptrdiff_t  offset{0}; ///< Element offset into that impl's buffer; nonzero only for the blocked forms.

    /// This member's live element pointer.
    template <typename T>
    [[nodiscard]] T *data() const {
        return accessor.impl<T>()->data() + offset;
    }

    /// This member's live leading dimension.
    template <typename T>
    [[nodiscard]] int leading_dim() const {
        return static_cast<int>(accessor.impl<T>()->get_lda());
    }
};

/// Per-slice operands, one entry per batch member.
using BatchedGemmOperands = std::vector<BatchedGemmOperand>;

/**
 * @brief Resolve every member's pointer and issue the batch, in one pass.
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
 *
 * @par Leading dimensions come from the live operands
 * The descriptor's @c lda / @c ldb / @c ldc were recorded when the node was
 * built, and ``Graph::rebind`` accepts any tensor of matching rank and dims -
 * a column slice of a larger store has the same shape and a different leading
 * dimension. Reading them back off the live impls costs three loads for the
 * whole batch and is what makes such a rebind compute the right answer instead
 * of striding through the wrong rows. The descriptor's copies stay as the
 * planning record every analysis reads; see @ref GemmOperand.
 *
 * A rebind that moves only SOME members to a different layout breaks the
 * premise of the batch itself - one call carries one leading dimension - and
 * nothing here can see it. Re-running the batching pass after such a rebind
 * re-forms the groups against the new layouts.
 */
template <typename T, typename Alpha>
void extract_and_run(BatchedGemmDescriptor const &d, BatchedGemmOperands const &a_ops, BatchedGemmOperands const &b_ops,
                     BatchedGemmOperands const &c_ops, Alpha alpha, Alpha beta) {
    if (d.batch_count <= 0) {
        return;
    }

    std::vector<T const *> a_arr(d.batch_count);
    std::vector<T const *> b_arr(d.batch_count);
    std::vector<T *>       c_arr(d.batch_count);

#ifdef _OPENMP
#    pragma omp parallel for schedule(static) if (!omp_in_parallel())
#endif
    for (int i = 0; i < d.batch_count; ++i) {
        a_arr[i] = a_ops[i].data<T>();
        b_arr[i] = b_ops[i].data<T>();
        c_arr[i] = c_ops[i].data<T>();
    }

    int const lda = a_ops[0].leading_dim<T>();
    int const ldb = b_ops[0].leading_dim<T>();
    int const ldc = c_ops[0].leading_dim<T>();

    blas::gemm_batch<T>(d.trans_a, d.trans_b, d.m, d.n, d.k, alpha, a_arr.data(), lda, b_arr.data(), ldb, beta, c_arr.data(), ldc,
                        d.batch_count);
}

/**
 * @brief Build the node executor for a @ref BatchedGemmDescriptor.
 *
 * @param d Shared BLAS parameters (m/n/k, leading dims, trans flags, prefactors,
 *          batch count, element type).
 * @param a_ops,b_ops,c_ops Per-member operands, ordered so index @c i of each
 *          refers to the same contraction.
 */
inline std::function<void()> make_batched_gemm_executor(BatchedGemmDescriptor d, BatchedGemmOperands a_ops, BatchedGemmOperands b_ops,
                                                        BatchedGemmOperands c_ops) {
    return [d, a_ops = std::move(a_ops), b_ops = std::move(b_ops), c_ops = std::move(c_ops)]() {
        switch (d.scalar) {
        case BlasScalar::Float:
            extract_and_run<float>(d, a_ops, b_ops, c_ops, static_cast<float>(d.alpha.real()), static_cast<float>(d.beta.real()));
            break;
        case BlasScalar::Double:
            extract_and_run<double>(d, a_ops, b_ops, c_ops, d.alpha.real(), d.beta.real());
            break;
        // The descriptor carries the full complex prefactor; preserve both
        // parts, a phase factor must not be truncated to its real part.
        case BlasScalar::ComplexFloat:
            extract_and_run<std::complex<float>>(
                d, a_ops, b_ops, c_ops, std::complex<float>{static_cast<float>(d.alpha.real()), static_cast<float>(d.alpha.imag())},
                std::complex<float>{static_cast<float>(d.beta.real()), static_cast<float>(d.beta.imag())});
            break;
        case BlasScalar::ComplexDouble:
            extract_and_run<std::complex<double>>(d, a_ops, b_ops, c_ops, d.alpha, d.beta);
            break;
        }
    };
}

EINSUMS_NAMESPACE_END(compute_graph::detail)
