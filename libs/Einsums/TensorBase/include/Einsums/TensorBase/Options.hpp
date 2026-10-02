//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Options/Get.hpp>

// The storage-order option, in the lowest module that reads it.

EINSUMS_NAMESPACE_BEGIN(option)

/// Build tensors row-major rather than the column-major default.
inline constinit cl::ConfigOption<bool> RowMajor =
    cl::config_flag("einsums:row-major", "Construct tensors in row-major order rather than column-major", "Tensor Options", false);

EINSUMS_NAMESPACE_END(option)

EINSUMS_NAMESPACE_BEGIN()

/// @brief Whether tensors default to row-major. Reads @ref option::RowMajor.
///
/// Out of line on purpose: a TU outside the library gets its own, unregistered copy of the
/// descriptor, so an inline read there would always return the default.
EINSUMS_EXPORT bool default_row_major();

EINSUMS_NAMESPACE_END()

EINSUMS_NAMESPACE_BEGIN()

/**
 * @brief Give the storage-order option its command-line presence. Idempotent.
 */
EINSUMS_EXPORT int register_Einsums_TensorBase_options();

namespace detail {
[[maybe_unused]] static int const register_options_Einsums_TensorBase = register_Einsums_TensorBase_options();
}

EINSUMS_NAMESPACE_END()
