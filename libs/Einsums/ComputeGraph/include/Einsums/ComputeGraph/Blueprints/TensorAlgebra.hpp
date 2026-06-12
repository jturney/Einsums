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

namespace einsums::compute_graph::blueprints {

using einsums::Indices;
namespace index = einsums::index;

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

    // A = 0.5 * A + 0.5 * A^T
    // We need a temporary for A^T
    // Since blueprints should be self-contained, create a local temp
    // This works both inside and outside capture
    auto &ctx = CaptureContext::current();
    if (ctx.is_capturing()) {
        // Inside capture: use the graph's create_tensor for the temp
        auto &At = ctx.graph()->create_tensor<T, 2>("_sym_tmp", n, n);
        permute("ji <- ij", T{0}, &At, T{1}, *A);
        scale(T{0.5}, A);
        axpy(T{0.5}, At, A);
    } else {
        // Outside capture: use stack-allocated temp
        auto At = Tensor<T, 2>("_sym_tmp", n, n);
        tensor_algebra::permute(T{0}, Indices{index::j, index::i}, &At, T{1}, Indices{index::i, index::j}, *A);
        linear_algebra::scale(T{0.5}, A);
        linear_algebra::axpy(T{0.5}, At, A);
    }
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

    auto &ctx = CaptureContext::current();
    if (ctx.is_capturing()) {
        auto &At = ctx.graph()->create_tensor<T, 2>("_asym_tmp", n, n);
        permute("ji <- ij", T{0}, &At, T{1}, *A);
        scale(T{0.5}, A);
        axpy(T{-0.5}, At, A);
    } else {
        auto At = Tensor<T, 2>("_asym_tmp", n, n);
        tensor_algebra::permute(T{0}, Indices{index::j, index::i}, &At, T{1}, Indices{index::i, index::j}, *A);
        linear_algebra::scale(T{0.5}, A);
        linear_algebra::axpy(T{-0.5}, At, A);
    }
}

/**
 * @brief Compute the trace of a rank-2 tensor: result = sum_i A(i,i).
 *
 * Stores the result in a rank-1 tensor with one element.
 *
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
    // Tr(A) = sum_i A(i,i) = einsum(" <- ii", result, A) but we don't have
    // single-operand einsum. Use dot with identity-like approach:
    // Tr(A) = sum_i A(i,i), compute manually outside graph, or use
    // einsum with a unit vector trick.
    // Simplest correct approach: element_transform to sum diagonal
    using T        = typename MatType::ValueType;
    size_t const n = A.dim(0);

    auto &ctx = CaptureContext::current();
    if (ctx.is_capturing()) {
        // Inside capture: record a custom operation
        auto *res_ptr = result;
        auto *a_slot  = &A; // Will be captured by lambda

        // Create a custom node that computes the trace
        TensorId r_id = ctx.get_or_register(*result);
        TensorId a_id = ctx.get_or_register(A);

        auto executor = [res_ptr, a_slot, n]() {
            typename MatType::ValueType sum{0};
            for (size_t ii = 0; ii < n; ii++) {
                sum += (*a_slot)(ii, ii);
            }
            (*res_ptr)(0) = sum;
        };

        ctx.record(OpKind::Custom, "tensor_trace", {a_id}, {r_id}, std::move(executor));
    } else {
        T sum{0};
        for (size_t ii = 0; ii < n; ii++) {
            sum += A(ii, ii);
        }
        (*result)(0) = sum;
    }
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
    if (ctx.is_capturing()) {
        auto &term = ctx.graph()->create_tensor<T, 2>("_exp_term", n, n);
        auto &tmp  = ctx.graph()->create_tensor<T, 2>("_exp_tmp", n, n);

        // expA = I (identity)
        // We need to set expA to identity, use a custom node
        auto    *exp_ptr = expA;
        TensorId exp_id  = ctx.get_or_register(*expA);
        TensorId a_id    = ctx.get_or_register(A);
        ctx.record(OpKind::Custom, "set_identity", {}, {exp_id}, [exp_ptr, n]() {
            exp_ptr->zero();
            for (size_t ii = 0; ii < n; ii++) {
                (*exp_ptr)(ii, ii) = typename std::remove_pointer_t<decltype(exp_ptr)>::ValueType{1};
            }
        });

        // term = I (will hold A^k / k!)
        TensorId term_id = ctx.get_or_register(term);
        ctx.record(OpKind::Custom, "set_identity_term", {}, {term_id}, [&term, n]() {
            term.zero();
            for (size_t ii = 0; ii < n; ii++) {
                term(ii, ii) = T{1};
            }
        });

        // For each Taylor term: term = term * A / k, expA += term
        for (size_t k = 1; k <= order; k++) {
            T factor = T{1} / static_cast<T>(k);
            // tmp = term * A
            einsum("ik;kj->ij", &tmp, term, A);
            // term = tmp * factor
            permute("ij <- ij", T{0}, &term, factor, tmp);
            // expA += term
            axpy(T{1}, term, expA);
        }
    } else {
        // Direct computation
        expA->zero();
        for (size_t ii = 0; ii < n; ii++) {
            (*expA)(ii, ii) = T{1};
        }

        auto term = Tensor<T, 2>("_exp_term", n, n);
        auto tmp  = Tensor<T, 2>("_exp_tmp", n, n);
        term.zero();
        for (size_t ii = 0; ii < n; ii++) {
            term(ii, ii) = T{1};
        }

        for (size_t k = 1; k <= order; k++) {
            T factor = T{1} / static_cast<T>(k);
            tensor_algebra::einsum(Indices{index::i, index::j}, &tmp, Indices{index::i, index::k}, term, Indices{index::k, index::j}, A);
            tensor_algebra::permute(T{0}, Indices{index::i, index::j}, &term, factor, Indices{index::i, index::j}, tmp);
            linear_algebra::axpy(T{1}, term, expA);
        }
    }
}

} // namespace einsums::compute_graph::blueprints
