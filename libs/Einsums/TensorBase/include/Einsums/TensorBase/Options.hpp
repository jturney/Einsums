//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Options/Get.hpp>

/*
 * The storage-order option, declared in the lowest module that reads it.
 *
 * It is read on the tensor-construction path, so it lives here rather than in
 * Tensor: TensorImpl, TensorBase's own index helpers, Tensor, and
 * TensorUtilities all ask for it, and TensorBase is the only one of those they
 * all depend on.
 */

EINSUMS_NAMESPACE_BEGIN(option)

/// Build tensors row-major rather than the column-major default.
inline constinit cl::ConfigOption<bool> RowMajor =
    cl::config_flag("einsums:row-major", "Construct tensors in row-major order rather than column-major", "Tensor Options", false);

EINSUMS_NAMESPACE_END(option)

EINSUMS_NAMESPACE_BEGIN()

/// @brief Whether tensors default to row-major. Reads @ref option::RowMajor.
///
/// Defined in the library rather than read from the descriptor at each call
/// site, and that is load-bearing. A ConfigOption caches the address of its
/// registry entry inside itself, filled in when its module registers it. These
/// headers are compiled by consumers outside the library, and such a
/// translation unit gets its OWN copy of the descriptor, whose entry is never
/// filled - so an inline read there quietly returns the default however the
/// option was set. For this option that means tensors built in those
/// translation units silently disagree about their storage order with tensors
/// built inside the library.
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
