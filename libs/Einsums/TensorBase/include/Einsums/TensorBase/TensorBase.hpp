//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/TensorBase/Common.hpp>

#include <complex>
#include <memory>
#include <mutex>

namespace einsums::tensor_base {

/*==================
 * Location-based.
 *==================*/

/**
 * @struct CoreTensor
 *
 * @brief Represents a tensor only available to the core.
 */
struct EINSUMS_EXPORT CoreTensor {};

/**
 * @struct DiskTensor
 *
 * @brief Represents a tensor stored on disk.
 */
struct EINSUMS_EXPORT DiskTensor {};

/*===================
 * Other properties.
 *===================*/

/**
 * @struct TiledTensorNoExtra
 *
 * @brief Specifies that a tensor is a tiled tensor without needing to specify type parameters.
 *
 * Only used internally. Use TiledTensorBase in your code.
 */
struct TiledTensorNoExtra {};

// Large class. See TiledTensor.hpp for code.
template <typename T, size_t Rank, typename TensorType>
struct TiledTensor;

/**
 * @struct BlockTensorNoExtra
 *
 * @brief Specifies that a tensor is a block tensor. Internal use only. Use BlockTensorBase instead.
 *
 * Specifies that a tensor is a block tensor without needing template parameters. Internal use only.
 * Use BlockTensorBase in your code.
 */
struct EINSUMS_EXPORT BlockTensorNoExtra {};

// Large class. See BlockTensor.hpp for code.
template <typename T, size_t Rank, typename TensorType>
struct BlockTensor;

/**
 * @struct AlgebraOptimizedTensor
 *
 * @brief Specifies that the tensor type can be used by einsum to select different routines other than the generic algorithm.
 */
struct EINSUMS_EXPORT AlgebraOptimizedTensor {};

class EINSUMS_EXPORT RuntimeTensorNoType {};

class EINSUMS_EXPORT RuntimeTensorViewNoType {};

} // namespace einsums::tensor_base
