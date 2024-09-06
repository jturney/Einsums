//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <cstddef>

namespace einsums {

/**
 * @brief Represents options and default options for printing tensors.
 */
struct tensor_print_options {
    int  width{7};          /// How many columns of tensor data are printed per line.
    bool full_output{true}; /// Print the tensor data (true) or just name and data span information (false).
};

// Forward declarations of tensors.
template <typename T, size_t Rank>
struct tensor;

template <typename T, size_t Rank>
struct block_tensor;

template <typename T, size_t Rank>
struct tiled_tensor;

#if defined(EINSUMS_COMPUTE_CODE)
template <typename T, size_t Rank>
struct device_tensor;

template <typename T, size_t Rank>
struct block_device_tensor;

template <typename T, size_t Rank>
struct tiled_device_tensor;
#endif

} // namespace einsums