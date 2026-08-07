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
#include <Einsums/TensorBase/IndexUtilities.hpp>

EINSUMS_NAMESPACE_BEGIN(decomposition)

template <TensorConcept TTensor, ContainerOf<TensorLike<TTensor, ValueTypeT<TTensor>, 2>> Factors>
auto tucker_reconstruct(TTensor const &g_tensor, Factors const &factors) -> TTensor {
    LabeledSection(__func__;

    // Dimension workspace for temps
    Dim dims_buffer = g_tensor.dims();
    // Buffers to hold intermediates while rebuilding the tensor
    TTensor *old_tensor_buffer = new TTensor(dims_buffer);
    TTensor *new_tensor_buffer;

    *old_tensor_buffer = g_tensor;

    // Reform the tensor (with all its intermediates)
    for_sequence<TensorRank<TTensor>>([&](auto i) {
        size_t full_idx   = factors[i].dim(0);
        dims_buffer[i]    = full_idx;
        new_tensor_buffer = new TTensor(dims_buffer);
        new_tensor_buffer->zero();

        std::array<size_t, TTensor::Rank> index_strides;

        size_t elements = dims_to_strides(old_tensor_buffer->dims(), index_strides);

        for (size_t element = 0; element < elements; element++) {
            std::array<size_t, TTensor::Rank> source_combination;
            sentinel_to_indices(element, index_strides, source_combination);
            for (size_t n = 0; n < full_idx; n++) {
                auto target_combination         = source_combination;
                std::get<i>(target_combination) = n;

                ValueTypeT<TTensor> &source = subscript_tensor(*old_tensor_buffer, source_combination);
                ValueTypeT<TTensor> &target = subscript_tensor(*new_tensor_buffer, target_combination);

                target += source * factors[i](n, std::get<i>(source_combination));
            }
        }

        delete old_tensor_buffer;
        old_tensor_buffer = new_tensor_buffer;
    });

    TTensor new_tensor = *(new_tensor_buffer);
    // Only delete one of the buffers, to avoid a double free
    delete new_tensor_buffer;

    return new_tensor;
}

template <size_t Rank, MatrixConcept TTensor, typename Alloc, ContainerOf<size_t> Ranks>
auto initialize_tucker(std::vector<TTensor, Alloc> &folds, Ranks &ranks) -> BufferVector<TTensor> {
    LabeledSection0();

    BufferVector<TTensor> factors;
    factors.reserve(TensorRank<TTensor>);

    // Perform compile-time looping.
    for_sequence<Rank>([&](auto i) {
        size_t rank    = ranks[i];
        auto [U, S, _] = linear_algebra::svd_dd(folds[i]);

        // println(tensor_algebra::unfold<i>(tensor));
        // println(S);

        if (folds[i].dim(0) < rank) {
            // i is an std::integral_constant the "()" obtains the underlying value.
            EINSUMS_LOG_WARN("dimension {} size {} is less than the requested decomposition rank {}", i(), folds[i].dim(0), rank);
            /// @todo Need to pad U up to rank
        }

        // Need to save the factors
        factors.emplace_back(std::move(Tensor{U(All, Range{0, rank})}));
    });

    return factors;
}

/**
 * Tucker decomposition of a tensor via Higher-Order SVD (HO-SVD).
 * Computes a rank-`rank` decomposition of `tensor` such that:
 *
 *   tensor = [|weights[r0][r1]...; factor[0][r0], ..., factors[-1][rn] |]
 */
template <template <typename, size_t> typename TTensor, size_t TRank, ContainerOf<size_t> Ranks, typename TType = double,
          ContainerOf<Tensor<TType, 2>> Folds>
auto tucker_ho_svd(TTensor<TType, TRank> const &tensor, Ranks &ranks, Folds const &folds)
    -> std::tuple<Tensor<TType, TRank>, BufferVector<Tensor<TType, 2>>> {
    LabeledSection0();

    // Compute set of unfolded matrices
    BufferVector<Tensor<TType, 2>> unfolded_matrices;
    unfolded_matrices.reserve(TRank);
    if (!folds.size()) {
        for_sequence<TRank>([&](auto i) { unfolded_matrices.push_back(tensor_algebra::unfold<i>(tensor)); });
    } else {
        unfolded_matrices = folds;
    }

    // Perform SVD guess for tucker decomposition procedure
    BufferVector<Tensor<TType, 2>> factors = initialize_tucker<TRank>(unfolded_matrices, ranks);

    // Get the dimension workspace for temps
    Dim<TRank> dims_buffer = tensor.dims();
    // Make buffers to hold intermediates while forming G
    Tensor<TType, TRank> *old_g_buffer;
    Tensor<TType, TRank> *new_g_buffer;

    old_g_buffer  = new Tensor<TType, TRank>(dims_buffer);
    *old_g_buffer = tensor;

    // Form G (with all of its intermediates)
    for_sequence<TRank>([&](auto i) {
        size_t rank    = ranks[i];
        dims_buffer[i] = rank;
        new_g_buffer   = new Tensor<TType, TRank>(dims_buffer);
        new_g_buffer->zero();

        std::array<size_t, TRank> index_strides;

        size_t elements = dims_to_strides(old_g_buffer->dims(), index_strides);

        for (size_t element = 0; element < elements; element++) {
            std::array<size_t, TRank> source_combination;
            sentinel_to_indices(element, index_strides, source_combination);
            for (size_t r = 0; r < rank; r++) {
                auto target_combination         = source_combination;
                std::get<i>(target_combination) = r;

                TType &source = subscript_tensor(*old_g_buffer, source_combination);
                TType &target = subscript_tensor(*new_g_buffer, target_combination);

                target += source * factors[i](std::get<i>(source_combination), r);
            }
        }

        delete old_g_buffer;
        old_g_buffer = new_g_buffer;
    });

    Tensor<TType, TRank> g_tensor = *(new_g_buffer);
    // ONLY delete one of the buffers, to avoid a double free
    delete new_g_buffer;

    return std::make_tuple(g_tensor, factors);
}

template <template <typename, size_t> typename TTensor, size_t TRank, ContainerOf<size_t> Ranks, typename TType = double>
auto tucker_ho_svd(TTensor<TType, TRank> const &tensor, Ranks &ranks) -> std::tuple<Tensor<TType, TRank>, BufferVector<Tensor<TType, 2>>> {
    return tucker_ho_svd(tensor, ranks, BufferVector<TTensor<TType, 2>>());
}

/**
 * Tucker decomposition via Higher-Order Orthogonal Inversion (HO-OI).
 * Computes a rank-`rank` decomposition of `tensor` such that:
 *
 *   tensor = [|weights[r0][r1]...; factor[0][r1], ..., factors[-1][rn] |].
 */
template <template <typename, size_t> typename TTensor, size_t TRank, ContainerOf<size_t> Ranks, typename TType = double>
auto tucker_ho_oi(TTensor<TType, TRank> const &tensor, Ranks &ranks, int n_iter_max = 100, double tolerance = 1.e-8)
    -> std::tuple<TTensor<TType, TRank>, BufferVector<Tensor<TType, 2>>> {
    LabeledSection0(__func__);

    // Use HO SVD as a starting guess
    auto ho_svd_guess = tucker_ho_svd(tensor, ranks);
    auto g_tensor     = std::get<0>(ho_svd_guess);
    auto factors      = std::get<1>(ho_svd_guess);

    int  iter      = 0;
    bool converged = false;
    while (iter < n_iter_max) {
        BufferVector<Tensor<TType, 2>> new_folds;
        new_folds.reserve(TRank);

        for_sequence<TRank>([&](auto i) {
            // Make the workspace for the contraction
            Dim<TRank> dims_buffer = tensor.dims();
            // Make buffers to form intermediates while forming new folds
            TTensor<TType, TRank> *old_fold_buffer;
            TTensor<TType, TRank> *new_fold_buffer;

            // Initialize old fold buffer to the tensor
            BufferAllocator<TTensor<TType, TRank>> allocator;
            old_fold_buffer  = std::construct_at(allocator.allocate(1), dims_buffer);
            *old_fold_buffer = tensor;

            for_sequence<TRank>([&](auto j) {
                if (j != i) {
                    size_t rank     = ranks[j];
                    dims_buffer[j]  = rank;
                    new_fold_buffer = std::construct_at(allocator.allocate(1), dims_buffer);
                    new_fold_buffer->zero();

                    std::array<size_t, TRank> index_strides;

                    size_t elements = dims_to_strides(old_fold_buffer->dims(), index_strides);

                    for (size_t element = 0; element < elements; element++) {
                        std::array<size_t, TRank> source_combination;
                        sentinel_to_indices(element, index_strides, source_combination);
                        for (size_t r = 0; r < rank; r++) {
                            auto target_combination         = source_combination;
                            std::get<j>(target_combination) = r;

                            TType &source = subscript_tensor(*old_fold_buffer, source_combination);
                            TType &target = subscript_tensor(*new_fold_buffer, target_combination);

                            target += source * factors[j](std::get<j>(source_combination), r);
                        }
                    }

                    allocator.deallocate(old_fold_buffer, 1);
                    old_fold_buffer = new_fold_buffer;
                }
            });

            Tensor<TType, 2> new_fold = tensor_algebra::unfold<i>(*new_fold_buffer);
            new_folds.push_back(new_fold);

            // Only delete once to avoid a double free
            allocator.deallocate(new_fold_buffer, 1);
        });

        // Reformulate guess based on HO SVD of new_folds
        auto new_ho_svd   = tucker_ho_svd(tensor, ranks, new_folds);
        auto new_g_tensor = std::get<0>(new_ho_svd);
        auto new_factors  = std::get<1>(new_ho_svd);

        // Check for convergence
        TType rmsd_max = rmsd(new_g_tensor, g_tensor);
        for_sequence<TRank>([&](auto n) { rmsd_max = std::max(rmsd_max, rmsd(new_factors[n], factors[n])); });

        // Update G and factors
        g_tensor = new_g_tensor;
        factors  = new_factors;

        if (rmsd_max < tolerance) {
            converged = true;
            break;
        }

        iter += 1;
    }
    if (!converged) {
        EINSUMS_LOG_WARN("Tucker HO-OI decomposition failed to converge in {} iterations", n_iter_max);
    }

    return std::make_tuple(g_tensor, factors);
}

EINSUMS_NAMESPACE_END(decomposition)