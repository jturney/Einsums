//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/// @file CP.hpp
/// @brief CANDECOMP/PARAFAC decompositions, their contractions written as string specs.
///
/// The same algorithms as <Einsums/Decomposition/CP.hpp>, which spells its contractions with
/// compile-time index types.

#include <Einsums/ComputeGraph/KhatriRao.hpp>
#include <Einsums/ComputeGraph/Operations.hpp>
#include <Einsums/Concepts/NamedRequirements.hpp>
#include <Einsums/Concepts/SubscriptChooser.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Decomposition/StringSpec/Unfold.hpp>
#include <Einsums/LinearAlgebra.hpp>
#include <Einsums/Profile.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorBase/Common.hpp>
#include <Einsums/TensorPermute/Permute.hpp>
#include <Einsums/TensorUtilities/CreateTensorLike.hpp>

#include <stdexcept>

EINSUMS_NAMESPACE_BEGIN(decomposition::string_spec)

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
        EINSUMS_THROW_EXCEPTION(DimensionError, "The first dimension of the tensor and the dimension of the weight DO NOT match");
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

    BufferVector<Tensor<TType, 2>> factors;
    factors.reserve(TRank);

    // Perform compile-time looping.
    for_sequence<TRank>([&](auto i) {
        size_t m = folds[i].dim(0);

        // Multiply the fold by its transpose
        Tensor fold_squared = create_tensor<TType>("fold squared", m, m);
        compute_graph::einsum("mn <- mp ; np", TType{0}, &fold_squared, TType{1}, folds[i], folds[i]);

        Tensor S = create_tensor<TType>("eigenvalues", m);

        // Diagonalize fold squared (akin to SVD)
        linear_algebra::syev(&fold_squared, &S);

        // syev leaves the eigenvectors in the columns, in ascending order of eigenvalue.
        Tensor<TType, 2> U = fold_squared;

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
template <size_t TRank, typename TType>
auto parafac(Tensor<TType, TRank> const &tensor, size_t rank, int n_iter_max = 100, double tolerance = 1.e-8)
    -> BufferVector<Tensor<TType, 2>> {
    LabeledSection0();

    // Compute set of unfolded matrices
    BufferVector<Tensor<TType, 2>> unfolded_matrices;
    unfolded_matrices.reserve(TRank);
    for (size_t i = 0; i < TRank; i++) {
        unfolded_matrices.push_back(detail::unfolded(i, tensor));
    }

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
                    compute_graph::einsum("rs <- ir ; is", TType{0}, &A_tA, TType{1}, factors[m_ind], factors[m_ind]);

                    if (first) {
                        V     = A_tA;
                        KR    = factors[m_ind];
                        first = false;
                    } else {
                        // Uses a Hamamard Contraction to build V
                        Tensor<TType, 2> Vcopy = V;
                        compute_graph::einsum("rs <- rs ; rs", TType{0}, &V, TType{1}, Vcopy, A_tA);

                        // Perform a Khatri-Rao contraction
                        KR = compute_graph::khatri_rao("ir", KR, "mr", factors[m_ind]);
                    }
                }
            });

            // Update factors[n_ind]
            size_t ndim = tensor.dim(n_ind);

            // Step 1: Matrix Multiplication
            compute_graph::einsum("ir <- ik ; kr", TType{0}, &factors[n_ind], TType{1}, unfolded_matrices[n_ind], KR);

            // Step 2: Linear Solve (instead of inversion, for numerical stability). The update solves
            // factor * V = M; V is symmetric, so it is V * factor^T = M^T, which is gesv's form.
            Tensor<TType, 2> factor_t{"factor^T", rank, ndim};
            tensor_permute::permute("ri <- ir", &factor_t, factors[n_ind]);
            // A nonzero info means V is exactly singular and factor_t is not solved; continuing would
            // carry garbage into every later mode.
            if (int const info = linear_algebra::gesv(&V, &factor_t); info != 0) {
                EINSUMS_THROW_EXCEPTION(std::runtime_error,
                                        "parafac: the ALS update for mode {} could not be solved (gesv info {}); a positive info "
                                        "means the system is singular, and a rank-{} decomposition may be more than this tensor "
                                        "supports",
                                        static_cast<size_t>(n_ind), info, rank);
            }
            tensor_permute::permute("ir <- ri", &factors[n_ind], factor_t);
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
template <size_t TRank, typename TType>
auto weighted_parafac(Tensor<TType, TRank> const &tensor, Tensor<TType, 1> const &weights, size_t rank, int n_iter_max = 100,
                      double tolerance = 1.e-8) -> BufferVector<Tensor<TType, 2>> {
    LabeledSection0();

    // Compute set of unfolded matrices (unweighted)
    BufferVector<Tensor<TType, 2>> unfolded_matrices;
    unfolded_matrices.reserve(TRank);
    for (size_t i = 0; i < TRank; i++) {
        unfolded_matrices.push_back(detail::unfolded(i, tensor));
    }

    // Perform SVD guess for parafac decomposition procedure
    BufferVector<Tensor<TType, 2>> factors = initialize_cp<TRank>(unfolded_matrices, rank);

    { // Define new scope (for memory optimization)
        // Create the weighted tensor
        Tensor<TType, 1> square_weights("square_weights", weights.dim(0));
        compute_graph::einsum("p <- p ; p", TType{0}, &square_weights, TType{1}, weights, weights);
        Tensor<TType, TRank> weighted_tensor = weight_tensor(tensor, square_weights);
        for (size_t i = 1; i < TRank; i++) {
            unfolded_matrices[i] = detail::unfolded(i, weighted_tensor);
        }
    }

    double tensor_norm = linear_algebra::vec_norm(tensor);
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
                        compute_graph::einsum("rs <- ir ; is", TType{0}, &A_tA, TType{1}, weighted_factor, weighted_factor);
                    } else {
                        compute_graph::einsum("rs <- ir ; is", TType{0}, &A_tA, TType{1}, factors[m_ind], factors[m_ind]);
                    }

                    if (first) {
                        V     = A_tA;
                        KR    = factors[m_ind];
                        first = false;
                    } else {
                        // Uses a Hamamard Contraction to build V
                        Tensor<TType, 2> Vcopy = V;
                        compute_graph::einsum("rs <- rs ; rs", TType{0}, &V, TType{1}, Vcopy, A_tA);

                        // Perform a Khatri-Rao contraction
                        KR = compute_graph::khatri_rao("ir", KR, "mr", factors[m_ind]);
                    }
                }
                m += 1;
            });

            // Update factors[n_ind]
            size_t ndim = tensor.dim(n_ind);

            // Step 1: Matrix Multiplication
            compute_graph::einsum("ir <- ik ; kr", TType{0}, &factors[n_ind], TType{1}, unfolded_matrices[n_ind], KR);

            // Step 2: Linear Solve (instead of inversion, for numerical stability). The update solves
            // factor * V = M; V is symmetric, so it is V * factor^T = M^T, which is gesv's form.
            Tensor<TType, 2> factor_t{"factor^T", rank, ndim};
            tensor_permute::permute("ri <- ir", &factor_t, factors[n_ind]);
            // A nonzero info means V is exactly singular and factor_t is not solved; continuing would
            // carry garbage into every later mode.
            if (int const info = linear_algebra::gesv(&V, &factor_t); info != 0) {
                EINSUMS_THROW_EXCEPTION(std::runtime_error,
                                        "weighted_parafac: the ALS update for mode {} could not be solved (gesv info {}); a positive info "
                                        "means the system is singular, and a rank-{} decomposition may be more than this tensor "
                                        "supports",
                                        static_cast<size_t>(n_ind), info, rank);
            }
            tensor_permute::permute("ir <- ri", &factors[n_ind], factor_t);

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

EINSUMS_NAMESPACE_END(decomposition::string_spec)