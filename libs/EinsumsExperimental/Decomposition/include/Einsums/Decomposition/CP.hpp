//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Concepts/NamedRequirements.hpp>
#include <Einsums/Concepts/SubscriptChooser.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/LinearAlgebra.hpp>
#include <Einsums/Profile.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorAlgebra.hpp>
#include <Einsums/TensorBase/Common.hpp>
#include <Einsums/TensorUtilities/CreateTensorLike.hpp>

EINSUMS_NAMESPACE_BEGIN(decomposition)

/**
 * "Weight" a tensor for weighted CANDECOMP/PARAFAC decompositions (returns a copy) by input weights
 */
template <TensorConcept TTensor, VectorConcept WTensor>
    requires requires {
        requires SameUnderlying<TTensor, WTensor>;
        requires InSamePlace<TTensor, WTensor>;
        requires BasicTensorConcept<TTensor>;
        requires BasicTensorConcept<WTensor>;
    }
auto weight_tensor(TTensor const &tensor, WTensor const &weights) -> Tensor<ValueTypeT<TTensor>, TensorRank<TTensor>> {
    using TType            = ValueTypeT<TTensor>;
    constexpr size_t TRank = TensorRank<TTensor>;
    LabeledSection0();

    if (tensor.dim(0) != weights.dim(0)) {
        EINSUMS_THROW_EXCEPTION(dimension_error, "The first dimension of the tensor and the dimension of the weight DO NOT match");
    }

    auto weighted_tensor = create_tensor_like(tensor);

    std::array<size_t, TRank> strides;

    size_t elements = dims_to_strides(tensor.dims(), strides);

#pragma omp parallel for
    for (size_t elem = 0; elem < elements; elem++) {
        thread_local std::array<size_t, TRank> target_combination;
        sentinel_to_indices(elem, strides, target_combination);
        TType const &source = subscript_tensor(tensor, target_combination);
        TType       &target = weighted_tensor.data()[elem];
        TType const &scale  = subscript_tensor(weights, std::get<0>(target_combination));

        target = scale * source;
    }

    return weighted_tensor;
}

/**
 * Reconstructs a tensor given a CANDECOMP/PARAFAC decomposition
 *
 *   factors = The decomposed CANDECOMP matrices (dimension: [dim[i], rank])
 */
template <size_t TRank, typename TType, typename Alloc>
auto parafac_reconstruct(std::vector<Tensor<TType, 2>, Alloc> const &factors) -> Tensor<TType, TRank> {
    LabeledSection0();

    size_t     rank = 0;
    Dim<TRank> dims;

    size_t i = 0;
    for (auto const &factor : factors) {
        dims[i] = factor.dim(0);
        if (!rank)
            rank = factor.dim(1);
        i++;
    }

    Tensor<TType, TRank> new_tensor(dims);
    new_tensor.zero();

    std::array<size_t, TRank> index_strides;

    size_t elements = dims_to_strides(dims, index_strides);

    for (auto it = 0; it < elements; it++) {
        std::array<size_t, TRank> idx_combo;
        sentinel_to_indices(it, index_strides, idx_combo);

        TType &target = subscript_tensor(new_tensor, idx_combo);
        for (size_t r = 0; r < rank; r++) {
            double temp = 1.0;
            for_sequence<TRank>([&](auto n) { temp *= subscript_tensor(factors[n], std::get<n>(idx_combo), r); });
            target += temp;
        }
    }

    return new_tensor;
}

template <size_t TRank, typename TType, typename Alloc>
auto initialize_cp(std::vector<Tensor<TType, 2>, Alloc> &folds, size_t rank) -> BufferVector<Tensor<TType, 2>> {
    LabeledSection0();

    using namespace einsums::tensor_algebra;

    BufferVector<Tensor<TType, 2>> factors;
    factors.reserve(TRank);

    // Perform compile-time looping.
    for_sequence<TRank>([&](auto i) {
        size_t m = folds[i].dim(0);

        // Multiply the fold by its transpose
        Tensor fold_squared = create_tensor<TType>("fold squared", m, m);
        einsum(0.0, Indices{index::M, index::N}, &fold_squared, 1.0, Indices{index::M, index::p}, folds[i], Indices{index::N, index::p},
               folds[i]);

        Tensor S = create_tensor<TType>("eigenvalues", m);

        // Diagonalize fold squared (akin to SVD)
        linear_algebra::syev(&fold_squared, &S);

        // Reorder into row major form
        Tensor U = create_tensor<TType>("Left Singular Vectors", m, m);
        permute(Indices{index::M, index::N}, &U, Indices{index::N, index::M}, fold_squared);

        // If (i == 0), Scale U by the singular values
        if (i == 0) {
            for (size_t v = 0; v < S.dim(0); v++) {
                TType const scaling_factor = std::sqrt(S(v));
                if (std::abs(scaling_factor) > 1.0e-14)
                    linear_algebra::scale_column(v, scaling_factor, &U);
            }
        }

        // println("After scaling");
        // println(U);

        if (folds[i].dim(0) < rank) {
            // EINSUMS_LOG_WARN("dimension {} size {} is less than the requested decomposition rank {}", i, folds[i].dim(0), rank);
            /// @todo Need to padd U up to rank
            Tensor<TType, 2> Unew  = create_random_tensor<TType>("Padded SVD Left Vectors", folds[i].dim(0), rank);
            Unew(All, Range{0, m}) = U(All, All);

            // Need to save the factors
            factors.push_back(Unew);
        } else {
            // Need to save the factors
            factors.emplace_back(Tensor<TType, 2>{U(All, Range{m - rank, m})});
        }

        // println("latest factor added");
        // println(factors[factors.size() - 1]);
        // Tensor<TType, 2> Unew = create_random_tensor("Padded SVD Left Vectors", folds[i].dim(0), rank);
        // factors.emplace_back(Unew);
    });

    return factors;
}

/**
 * CANDECOMP/PARAFAC decomposition via alternating least squares (ALS).
 * Computes a rank-`rank` decomposition of `tensor` such that:
 *
 *   tensor = [|weights; factor[0], ..., factors[-1] |].
 */
template <template <typename, size_t> typename TTensor, size_t TRank, typename TType = double>
auto parafac(TTensor<TType, TRank> const &tensor, size_t rank, int n_iter_max = 100, double tolerance = 1.e-8)
    -> BufferVector<Tensor<TType, 2>> {
    LabeledSection0();

    using namespace einsums::tensor_algebra;
    using namespace einsums::index;

    // Compute set of unfolded matrices
    BufferVector<Tensor<TType, 2>> unfolded_matrices;
    unfolded_matrices.reserve(TRank);
    for_sequence<TRank>([&](auto i) { unfolded_matrices.push_back(tensor_algebra::unfold<i>(tensor)); });

    // Perform SVD guess for parafac decomposition procedure
    BufferVector<Tensor<TType, 2>> factors = initialize_cp<TRank>(unfolded_matrices, rank);

    TType  tensor_norm = linear_algebra::vec_norm(tensor);
    size_t nelem       = 1;
    for_sequence<TRank>([&](auto i) { nelem *= tensor.dim(i); });
    tensor_norm /= std::sqrt((TType)nelem);

    int    iter       = 0;
    bool   converged  = false;
    double prev_error = 0.0;
    while (iter < n_iter_max) {
        for_sequence<TRank>([&](auto n_ind) {
            // Form V and Khatri-Rao product intermediates
            Tensor<TType, 2> V;
            Tensor<TType, 2> KR;
            bool             first = true;

            for_sequence<TRank>([&](auto m_ind) {
                if (m_ind != n_ind) {
                    Tensor<TType, 2> A_tA{"V", rank, rank};
                    // A_tA = A^T[j] @ A[j]
                    // println("iter {}, mind {}", iter, m_ind);
                    // println(factors[m_ind]);
                    einsum(0.0, Indices{r, s}, &A_tA, 1.0, Indices{I, r}, factors[m_ind], Indices{I, s}, factors[m_ind]);

                    if (first) {
                        V     = A_tA;
                        KR    = factors[m_ind];
                        first = false;
                    } else {
                        // Uses a Hamamard Contraction to build V
                        Tensor<TType, 2> Vcopy = V;
                        einsum(0.0, Indices{r, s}, &V, 1.0, Indices{r, s}, Vcopy, Indices{r, s}, A_tA);

                        // Perform a Khatri-Rao contraction
                        KR = tensor_algebra::khatri_rao(Indices{I, r}, KR, Indices{M, r}, factors[m_ind]);
                    }
                }
            });

            // Update factors[n_ind]
            size_t ndim = tensor.dim(n_ind);

            // Step 1: Matrix Multiplication
            einsum(0.0, Indices{I, r}, &factors[n_ind], 1.0, Indices{I, K}, unfolded_matrices[n_ind], Indices{K, r}, KR);

            // Step 2: Linear Solve (instead of inversion, for numerical stability, column-major ordering)
            linear_algebra::gesv(&V, &factors[n_ind]);
        });

        // Check for convergence
        // Reconstruct Tensor based on the factors
        Tensor<TType, TRank> rec_tensor = parafac_reconstruct<TRank>(factors);

        double const unnormalized_error = rmsd(rec_tensor, tensor);
        double const curr_error         = unnormalized_error / tensor_norm;
        double const delta              = std::abs(curr_error - prev_error);

        // printf("    @CP Iteration %d, ERROR: %8.8f, DELTA: %8.8f\n", iter, curr_error, delta);

        if (iter >= 2 && delta < tolerance) {
            converged = true;
            break;
        }

        prev_error = curr_error;
        iter += 1;
    }
    if (!converged) {
        EINSUMS_LOG_WARN("CP decomposition failed to converge in {} iterations", n_iter_max);
    }

    // Return **non-normalized** factors
    return factors;
}

/**
 * Weighted CANDECOMP/PARAFAC decomposition via alternating least squares (ALS).
 * Computes a rank-`rank` decomposition of `tensor` such that:
 *
 *   tensor = [| factor[0], ..., factors[-1] |].
 *   weights = The weights to multiply the tensor by
 */
template <template <typename, size_t> typename TTensor, size_t TRank, typename TType = double>
auto weighted_parafac(TTensor<TType, TRank> const &tensor, TTensor<TType, 1> const &weights, size_t rank, int n_iter_max = 100,
                      double tolerance = 1.e-8) -> BufferVector<Tensor<TType, 2>> {
    LabeledSection0();

    using namespace einsums::tensor_algebra;

    // Compute set of unfolded matrices (unweighted)
    BufferVector<Tensor<TType, 2>> unfolded_matrices;
    unfolded_matrices.reserve(TRank);
    for_sequence<TRank>([&](auto i) { unfolded_matrices.push_back(tensor_algebra::unfold<i>(tensor)); });

    // Perform SVD guess for parafac decomposition procedure
    BufferVector<Tensor<TType, 2>> factors = initialize_cp<TRank>(unfolded_matrices, rank);

    { // Define new scope (for memory optimization)
        // Create the weighted tensor
        Tensor<TType, 1> square_weights("square_weights", weights.dim(0));
        einsum(0.0, Indices{index::P}, &square_weights, 1.0, Indices{index::P}, weights, Indices{index::P}, weights);
        Tensor<TType, TRank> weighted_tensor = weight_tensor(tensor, square_weights);
        for_sequence<TRank>([&](auto i) {
            if (i != 0)
                unfolded_matrices[i] = tensor_algebra::unfold<i>(weighted_tensor);
        });
    }

    double tensor_norm = norm(tensor);
    size_t nelem       = 1;
    for_sequence<TRank>([&](auto i) { nelem *= tensor.dim(i); });
    tensor_norm /= std::sqrt((double)nelem);

    int    iter       = 0;
    bool   converged  = false;
    double prev_error = 0.0;
    while (iter < n_iter_max) {
        size_t n = 0; // NOLINT
        for_sequence<TRank>([&](auto n_ind) {
            // Form V and Khatri-Rao product intermediates
            Tensor<TType, 2> V;
            Tensor<TType, 2> KR;
            bool             first = true;

            size_t m = 0; // NOLINT
            for_sequence<TRank>([&](auto m_ind) {
                if (m_ind != n_ind) {
                    Tensor<TType, 2> A_tA{"V", rank, rank};
                    // A_tA = A^T[j] @ A[j]
                    if (m == 0) {
                        Tensor<TType, 2> weighted_factor = weight_tensor(factors[m_ind], weights);
                        einsum(0.0, Indices{index::r, index::s}, &A_tA, 1.0, Indices{index::I, index::r}, weighted_factor,
                               Indices{index::I, index::s}, weighted_factor);
                    } else {
                        einsum(0.0, Indices{index::r, index::s}, &A_tA, 1.0, Indices{index::I, index::r}, factors[m_ind],
                               Indices{index::I, index::s}, factors[m_ind]);
                    }

                    if (first) {
                        V     = A_tA;
                        KR    = factors[m_ind];
                        first = false;
                    } else {
                        // Uses a Hamamard Contraction to build V
                        Tensor<TType, 2> Vcopy = V;
                        einsum(0.0, Indices{index::r, index::s}, &V, 1.0, Indices{index::r, index::s}, Vcopy, Indices{index::r, index::s},
                               A_tA);

                        // Perform a Khatri-Rao contraction
                        KR = tensor_algebra::khatri_rao(Indices{index::I, index::r}, KR, Indices{index::M, index::r}, factors[m_ind]);
                    }
                }
                m += 1;
            });

            // Update factors[n_ind]
            size_t ndim = tensor.dim(n_ind);

            // Step 1: Matrix Multiplication
            einsum(0.0, Indices{index::I, index::r}, &factors[n_ind], 1.0, Indices{index::I, index::K}, unfolded_matrices[n_ind],
                   Indices{index::K, index::r}, KR);

            // Step 2: Linear Solve (instead of inversion, for numerical stability, column-major ordering)
            linear_algebra::gesv(&V, &factors[n_ind]);

            n += 1;
        });

        // Check for convergence
        // Reconstruct Tensor based on the factors
        Tensor<TType, TRank> rec_tensor = parafac_reconstruct<TRank>(factors);

        double const unnormalized_error = rmsd(rec_tensor, tensor);
        double const curr_error         = unnormalized_error / tensor_norm;
        double const delta              = std::abs(curr_error - prev_error);

        // printf("    @CP Iteration %d, ERROR: %8.8f, DELTA: %8.8f\n", iter, curr_error, delta);

        if (iter >= 1 && delta < tolerance) {
            converged = true;
            break;
        }

        prev_error = curr_error;
        iter += 1;
    }
    if (!converged) {
        EINSUMS_LOG_WARN("CP decomposition failed to converge in {} iterations", n_iter_max);
    }

    // Return **non-normalized** factors
    return factors;
}

EINSUMS_NAMESPACE_END(decomposition)