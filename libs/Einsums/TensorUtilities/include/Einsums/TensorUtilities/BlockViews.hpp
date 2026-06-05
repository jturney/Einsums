//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Errors/Error.hpp>
#include <Einsums/Tensor/BlockTensor.hpp>
#include <Einsums/Tensor/TiledTensor.hpp>

namespace einsums {

/**
 * @brief Create a view of a block tensor where each member has a different view.
 *
 * This function takes in a vector of ranges and applies the ranges to each dimension of each block. For instance:

 * @code
 * BlockTensor<double, 2> tensor("tensor", 5, 5);
 * apply_view(tensor, std::vector<Range>{Range{0, 3}, Range{1, 4}});
 * @endcode
 *
 * This will create a view where the first block has been restricted to the top-left 3x3 block and the second has
 * been restricted to the middle 3x3 block.
 *
 * @param[in] tensor The tensor to view.
 * @param[in] spec A vector of ranges that will be passed to the internal tensors to construct the view.
 *
 * @return A TiledTensorView that contains the resulting views.
 *
 * @versionadded{1.1.0}
 */
template <CoreBlockTensorConcept TensorType, ContainerOf<Range> Spec>
auto apply_view(TensorType const &tensor, Spec const &spec) {
    if (spec.size() > tensor.num_blocks()) {
        EINSUMS_THROW_EXCEPTION(num_argument_error, "Can not apply view specification to block tensor. Incorrect number of indices given.");
    }

    BufferVector<int64_t> sizes(spec.size());

    for (int i = 0; i < spec.size(); i++) {
        if (spec[i][1] < 0 && tensor.block_dim(i) != 0) {
            sizes[i] = tensor.block_dim(i) + spec[i][1] - spec[i][0];
        } else if (tensor.block_dim(i) == 0) {
            sizes[i] = 0;
        } else {
            sizes[i] = spec[i][1] - spec[i][0];
        }
    }

    auto out = TiledTensorView<typename TensorType::ValueType, TensorType::Rank>("unnamed view", sizes);

    std::array<int, TensorType::Rank> position;

    auto index = std::array<Range, TensorType::Rank>();

    for (int i = 0; i < spec.size(); i++) {
        position.fill(i);

        if (out.has_zero_size(position)) {
            continue;
        }

        index.fill(spec.at(i));

        out.insert_tile(position, std::apply(tensor[i], index));
    }

    return out;
}

/**
 * @brief Create a view of a block tensor where each member has a different view.
 *
 * This function takes in several vectors of ranges and applies the ranges to the respective dimension of each block. For instance:

 * @code
 * BlockTensor<double, 2> tensor("tensor", 5, 5);
 * apply_view(tensor, std::vector<Range>{Range{0, 3}, Range{1, 4}}, std::vector<Range>{Range{1, 4}, Range{0, 3}});
 * @endcode
 *
 * This will create a view where the first block has been restricted to the first three rows and middle three columns,
 * and the second block has been restricted to the middle three rows and the first three columns.
 *
 * @param[in] tensor The tensor to view.
 * @param[in] spec Several vectors of ranges that will be passed to the internal tensors to construct the view.
 *
 * @return A TiledTensorView that contains the resulting views.
 *
 * @versionadded{1.1.0}
 */
template <CoreBlockTensorConcept TensorType, ContainerOf<Range>... ViewSpec>
    requires requires {
        requires sizeof...(ViewSpec) == TensorType::Rank;
        requires sizeof...(ViewSpec) > 1;
    }
auto apply_view(TensorType const &tensor, ViewSpec &&...spec) {
    if (((spec.size() > tensor.num_blocks()) || ...)) {
        EINSUMS_THROW_EXCEPTION(num_argument_error, "Can not apply view specification to block tensor. Incorrect number of indices given.");
    }

    auto spec_array = std::array{std::forward<ViewSpec>(spec)...};

    std::array<BufferVector<int>, TensorType::Rank> sizes;

    for (int i = 0; i < spec_array.size(); i++) {
        sizes[i].resize(spec_array[i].size());
        for (int j = 0; j < spec_array[i].size(); j++) {
            if (spec_array[i][j][1] < 0 && tensor.block_dim(j) != 0) {
                sizes[i][j] = tensor.block_dim(j) + spec_array[i][j][1] - spec_array[i][j][0];
            } else if (tensor.block_dim(j) == 0) {
                sizes[i][j] = 0;
            } else {
                sizes[i][j] = spec_array[i][j][1] - spec_array[i][j][0];
            }
        }
    }

    auto out = TiledTensorView<typename TensorType::ValueType, TensorType::Rank>("unnamed view", sizes);

    std::array<int, TensorType::Rank> position;

    auto index = std::array<Range, TensorType::Rank>();

    for (int i = 0; i < tensor.num_blocks(); i++) {
        position.fill(i);

        if (out.has_zero_size(position)) {
            continue;
        }

        for (int j = 0; j < index.size(); j++) {
            index[j] = spec_array[j][i];
        }

        out.insert_tile(position, std::apply(tensor[i], index));
    }

    return out;
}

/**
 * @brief Create a view of a tiled tensor where each member has a different view.
 *
 * This function takes in several vectors of ranges and applies the ranges to the respective dimension of each tile. For instance:

 * @code
 * TiledTensor<double, 2> tensor("tensor", std::array{5, 5});
 * apply_view(tensor, std::vector<Range>{Range{0, 3}, Range{1, 4}}, std::vector<Range>{Range{1, 4}, Range{0, 3}});
 * @endcode
 *
 * This will create a view where:
 * - The (0,0) tile is restricted to the first three rows and middle three columns.
 * - The (0,1) tile is restricted to the top left 3x3 block.
 * - The (1,0) tile is restricted to the middle 3x3 block.
 * - The (1,1) tile is restricted to the middle three columns and first three rows.
 *
 * @param[in] tensor The tensor to view.
 * @param[in] spec Several vectors of ranges that will be passed to the internal tensors to construct the view.
 *
 * @return A TiledTensorView that contains the resulting views.
 *
 * @versionadded{1.1.0}
 */
template <CoreTiledTensorConcept TensorType, ContainerOf<Range>... ViewSpec>
    requires requires {
        requires sizeof...(ViewSpec) == TensorType::Rank;
        requires sizeof...(ViewSpec) > 1;
    }
auto apply_view(TensorType const &tensor, ViewSpec &&...spec) {
    auto spec_array = std::array{std::forward<ViewSpec>(spec)...};

    for (int i = 0; i < TensorType::Rank; i++) {
        if (spec_array[i].size() > tensor.grid_size(i)) {
            EINSUMS_THROW_EXCEPTION(num_argument_error,
                                    "Can not apply view specification to tiled tensor. Incorrect number of indices given.");
        }
    }

    std::array<BufferVector<int>, TensorType::Rank> sizes;

    for (int i = 0; i < spec_array.size(); i++) {
        sizes[i].resize(spec_array[i].size());
        for (int j = 0; j < spec_array[i].size(); j++) {
            if (spec_array[i][j][1] < 0 && tensor.tile_size(i)[j] != 0) {
                sizes[i][j] = tensor.tile_size(i)[j] + spec_array[i][j][1] - spec_array[i][j][0];
            } else if (tensor.tile_size(i)[j] == 0) {
                sizes[i][j] = 0;
            } else {
                sizes[i][j] = spec_array[i][j][1] - spec_array[i][j][0];
            }
        }
    }

    auto out = TiledTensorView<typename TensorType::ValueType, TensorType::Rank>("unnamed view", sizes);

    auto index = std::array<Range, TensorType::Rank>();

    for (auto const &pair : tensor.tiles()) {

        if (out.has_zero_size(pair.first)) {
            continue;
        }

        for (int j = 0; j < index.size(); j++) {
            index[j] = spec_array[j][pair.first[j]];
        }

        out.insert_tile(pair.first, std::apply(pair.second, index));
    }

    return out;
}

} // namespace einsums
