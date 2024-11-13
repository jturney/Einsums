//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <cstddef>

namespace einsums {

/**
 * @brief Represents options and default options for printing tensors.
 */
struct TensorPrintOptions {
    int  width{7};          /// How many columns of tensor data are printed per line.
    bool full_output{true}; /// Print the tensor data (true) or just name and data span information (false).
};

// Forward declarations of tensors.
template <typename T, size_t Rank>
struct Tensor;

template <typename T, size_t Rank>
struct BlockTensor;

template <typename T, size_t Rank>
struct TiledTensor;

#if defined(EINSUMS_COMPUTE_CODE)
template <typename T, size_t Rank>
struct DeviceTensor;

template <typename T, size_t Rank>
struct BlockDeviceTensor;

template <typename T, size_t Rank>
struct TiledDeviceTensor;
#endif

template <typename T, size_t Rank>
struct TensorView;

template <typename T, size_t ViewRank, size_t Rank>
struct DiskView;

template <typename T, size_t Rank>
struct DiskTensor;

} // namespace einsums