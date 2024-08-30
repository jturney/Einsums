//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <cstddef>

namespace einsums {

// Forward declarations of tensors.
template <typename T, size_t Rank>
struct tensor;

template <typename T, size_t Rank>
struct block_tensor;

template <typename T, size_t Rank>
struct tiled_tensor;

#ifdef __HIP__
template <typename T, size_t Rank>
struct device_tensor;

template <typename T, size_t Rank>
struct block_device_tensor;

template <typename T, size_t Rank>
struct tiled_device_tensor;
#endif

} // namespace einsums