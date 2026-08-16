//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/CommandLine/Get.hpp>
#include <Einsums/Config/Namespace.hpp>

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

/**
 * @brief Give the storage-order option its command-line presence. Idempotent.
 */
EINSUMS_EXPORT int register_Einsums_TensorBase_options();

namespace detail {
[[maybe_unused]] static int const register_options_Einsums_TensorBase = register_Einsums_TensorBase_options();
}

EINSUMS_NAMESPACE_END()
