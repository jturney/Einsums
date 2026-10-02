//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Config/Namespace.hpp>

#include <functional>

EINSUMS_NAMESPACE_BEGIN()
/// The type of a function which is registered to be executed as a
/// shutdown or pre-shutdown function.
/// @versionadded{1.0.0}
using ShutdownFunctionType = std::function<void()>;

/// \brief Add a function to run in \a einsums::finalize(), before every shutdown function.
///
/// \param f  [in] The function to register.
///
/// \note Throws once the pre-shutdown functions have started running.
///
/// \see    \a einsums::register_shutdown_function()
/// @versionadded{1.0.0}
EINSUMS_EXPORT void register_pre_shutdown_function(ShutdownFunctionType f);

/// \brief Add a function to run in \a einsums::finalize(), after every pre-shutdown function.
///
/// \param f  [in] The function to register.
///
/// \note Throws once the shutdown functions have started running.
///
/// \see    \a einsums::register_pre_shutdown_function()
/// @versionadded{1.0.0}
EINSUMS_EXPORT void register_shutdown_function(ShutdownFunctionType f);

namespace detail {

/**
 * @brief Registers a pointer to be freed at program exit.
 *
 * For pointers that cannot be freed when the main thread is done with them.
 *
 * @param f The function that deletes the pointer.
 *
 * @versionadded{1.0.0}
 */
EINSUMS_EXPORT void register_free_pointer(std::function<void()> f);

} // namespace detail

EINSUMS_NAMESPACE_END()
