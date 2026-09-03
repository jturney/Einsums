//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/**
 * @file TensorAlgebra.hpp
 * @brief Reusable tensor algebra blueprints for computation graphs.
 *
 * These functions compose standard ComputeGraph operations into common
 * patterns. They work both inside capture blocks (recorded into the graph)
 * and outside (executed immediately).
 *
 * Each blueprint runs ONE sequence of ``cg::`` calls, which already choose
 * between recording and executing; the only thing capture changes here is who
 * owns a temporary, so that is the only thing these branch on.
 *
 * @code
 * cg::Graph graph("example");
 * {
 *     cg::CaptureGuard guard(graph);
 *     cg::blueprints::symmetrize(&A);
 *     cg::blueprints::tensor_trace(&result, A);
 * }
 * graph.execute();
 * @endcode
 */

#include <Einsums/ComputeGraph/Operations.hpp>
#include <Einsums/Config/Namespace.hpp>

#include <optional>

EINSUMS_NAMESPACE_BEGIN(compute_graph::blueprints)

namespace detail {

/// A scratch matrix a blueprint can hand to the ``cg::`` ops either way.
///
/// A captured node outlives the call that recorded it, so a temporary it writes
/// through has to outlive it too: under capture the graph owns the tensor, and
/// outside capture a local is enough. Owning the choice here is what lets the
/// blueprint below run one code path instead of two.
template <typename T>
class ScratchMatrix {
  public:
    ScratchMatrix(CaptureContext &ctx, char const *name, size_t n) {
        if (ctx.is_capturing()) {
            _ptr = &ctx.graph()->template create_tensor<T, 2>(name, n, n);
        } else {
            _owned.emplace(name, n, n);
            _ptr = &*_owned;
        }
    }

    Tensor<T, 2> &operator*() const { return *_ptr; }
    Tensor<T, 2> *get() const { return _ptr; }

  private:
    std::optional<Tensor<T, 2>> _owned;
    Tensor<T, 2>               *_ptr{nullptr};
};

} // namespace detail

/**
 * @brief Symmetrize a rank-2 tensor in-place: A = 0.5 * (A + A^T).
 *
 * @tparam MatType Matrix tensor type.
 * @param[in,out] A The matrix to symmetrize. Must be square.
 *
 * @code
 * cg::blueprints::symmetrize(&A);
 * @endcode
 */
template <MatrixConcept MatType>
void symmetrize(MatType *A) {
    using T        = typename MatType::ValueType;
    size_t const n = A->dim(0);

    detail::ScratchMatrix<T> const At(CaptureContext::current(), "_sym_tmp", n);
    permute("ji <- ij", T{0}, At.get(), T{1}, *A);
    scale(T{0.5}, A);
    axpy(T{0.5}, *At, A);
}

/**
 * @brief Antisymmetrize a rank-2 tensor in-place: A = 0.5 * (A - A^T).
 *
 * @tparam MatType Matrix tensor type.
 * @param[in,out] A The matrix to antisymmetrize. Must be square.
 *
 * @code
 * cg::blueprints::antisymmetrize(&A);
 * @endcode
 */
template <MatrixConcept MatType>
void antisymmetrize(MatType *A) {
    using T        = typename MatType::ValueType;
    size_t const n = A->dim(0);

    detail::ScratchMatrix<T> const At(CaptureContext::current(), "_asym_tmp", n);
    permute("ji <- ij", T{0}, At.get(), T{1}, *A);
    scale(T{0.5}, A);
    axpy(T{-0.5}, *At, A);
}

/**
 * @brief Compute the trace of a rank-2 tensor: result = sum_i A(i,i).
 *
 * Stores the result in a rank-1 tensor with one element.
 *
 * @tparam ResultType Rank-1 destination tensor type.
 * @tparam MatType Matrix tensor type.
 * @param[out] result Rank-1 tensor with at least 1 element. result(0) = Tr(A).
 * @param[in] A The matrix to trace.
 *
 * @code
 * auto result = Tensor<double, 1>("trace", 1);
 * cg::blueprints::tensor_trace(&result, A);
 * // result(0) contains Tr(A)
 * @endcode
 */
template <VectorConcept ResultType, MatrixConcept MatType>
void tensor_trace(ResultType *result, MatType const &A) {
    // cg::trace is this operation. Writing the diagonal walk out again here
    // also baked the operand addresses into the recorded closure, so a rebind
    // moved the tensors out from under the node; going through cg::trace picks
    // up its slot-based executor as well.
    trace(result->data(), A);
}

/**
 * @brief Compute the matrix exponential via Taylor series: expA = sum_{k=0}^{order} A^k / k!
 *
 * @tparam MatType Matrix tensor type.
 * @param[out] expA The result matrix (same dimensions as A).
 * @param[in] A The input matrix. Must be square.
 * @param[in] order Number of Taylor terms (default 10).
 *
 * @code
 * cg::blueprints::matrix_exponential(&expA, A, 10);
 * @endcode
 */
template <MatrixConcept MatType>
void matrix_exponential(MatType *expA, MatType const &A, size_t order = 10) {
    using T        = typename MatType::ValueType;
    size_t const n = A.dim(0);

    auto &ctx = CaptureContext::current();

    detail::ScratchMatrix<T> const term(ctx, "_exp_term", n);
    detail::ScratchMatrix<T> const tmp(ctx, "_exp_tmp", n);

    // Both series accumulators start at the identity, expA because the k = 0
    // term is I and term because it carries A^k / k! forward from there.
    auto set_identity = [n](auto *m) {
        m->zero();
        for (size_t ii = 0; ii < n; ii++) {
            (*m)(ii, ii) = T{1};
        }
    };
    if (ctx.is_capturing()) {
        TensorId const exp_id = ctx.get_or_register(*expA);
        ctx.record(OpKind::Custom, "set_identity", {}, {exp_id}, [expA, set_identity]() { set_identity(expA); });
        TensorId const term_id = ctx.get_or_register(*term);
        ctx.record(OpKind::Custom, "set_identity_term", {}, {term_id}, [t = term.get(), set_identity]() { set_identity(t); });
    } else {
        set_identity(expA);
        set_identity(term.get());
    }

    // For each Taylor term: term = term * A / k, expA += term
    for (size_t k = 1; k <= order; k++) {
        T const factor = T{1} / static_cast<T>(k);
        einsum("ik;kj->ij", tmp.get(), *term, A);
        permute("ij <- ij", T{0}, term.get(), factor, *tmp);
        axpy(T{1}, *term, expA);
    }
}

EINSUMS_NAMESPACE_END(compute_graph::blueprints)
