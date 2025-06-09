//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <functional>

namespace einsums {
/// The type of a function which is registered to be executed as a
/// shutdown or pre-shutdown function.
using ShutdownFunctionType = std::function<void()>;

/// \brief Add a function to be executed during
/// \a einsums::finalize() but guaranteed before any shutdown function is
/// executed (system-wide)
///
/// Any of the functions registered with \a register_pre_shutdown_function
/// are guaranteed to be executed during the execution of
/// \a einsums::finalize() before any of the registered shutdown functions are
/// executed (see: \a einsums::register_shutdown_function()).
///
/// \param f  [in] The function to be registered as
///           a pre-shutdown function.
///
/// \note If this function is called while the pre-shutdown functions are
///       being executed, or after that point, it will raise a invalid_status
///       exception.
///
/// \see    \a einsums::register_shutdown_function()
EINSUMS_EXPORT void register_pre_shutdown_function(ShutdownFunctionType f);

/// \brief Add a function to be executed during
/// \a einsums::finalize() but guaranteed after any pre-shutdown function is
/// executed (system-wide)
///
/// Any of the functions registered with \a register_shutdown_function
/// are guaranteed to be executed during the execution of
/// \a einsums::finalize() after any of the registered pre-shutdown functions
/// are executed (see: \a einsums::register_pre_shutdown_function()).
///
/// \param f  [in] The function to be registered to run as
///           a shutdown function.
///
/// \note If this function is called while the shutdown functions are
///       being executed, or after that point, it will raise a invalid_status
///       exception.
///
/// \see    \a einsums::register_pre_shutdown_function()
EINSUMS_EXPORT void register_shutdown_function(ShutdownFunctionType f);

namespace detail {

/**
 * @brief Registers a pointer to be freed at program exit.
 *
 * There are certain cases where a pointer can't necessarily be freed when it is no longer in use by the main thread.
 * This function makes the shutdown routine aware of these pointers so that they can be freed.
 *
 * @param f The function that deletes the pointer.
 */
EINSUMS_EXPORT void register_free_pointer(std::function<void()> f);

} // namespace detail

} // namespace einsums
