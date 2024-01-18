//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <functional>

namespace einsums {

///////////////////////////////////////////////////////////////////////////////////////////////////
/// The type of a function which is registered to be executed as a startup or pre-startup function.
using StartupFunctionType = std::function<void()>;

/// \brief Add a function to be executed by an einsums thread before einsums_main but guaranteed before any startup
/// function is executed (system-wide).
///
/// Any functions registered with \a register_pre_startup_function are guaranteed to be executed by an einsums thread
/// before any of the registered startup functions are executed (see \a einsums::register_startup_function()).
///
/// \param f The function to be registered to run by an einsums thread as a pre-startup function
///
/// \note If this function is called while the pre-startup functions are being executed or after that point, it will
/// raise an invalid_status exception.
///
/// This function is one of the few API functions which can be called before the runtime system has been fully
/// initialized. It will automatically stage the provided startup function to the runtime system during its
/// initialization, if necessary.
///
/// \see \a einsums::register_startup_function()
EINSUMS_EXPORT void register_pre_startup_function(StartupFunctionType f);

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Add a function to be executed by a pika thread before pika_main but guaranteed after any pre-startup function
/// is executed (system-wide).
///
/// Any of the functions registered with \a register_startup_function are guaranteed to be executed by a einsums thread
/// after any of the registered pre-startup functions are executed (see: \a einsums::register_pre_startup_function()),
/// but before \a einsums_main is being called.
///
/// \param f  [in] The function to be registered to run by a einsums thread as a startup function.
///
/// \note If this function is called while the startup functions are  being executed or after that point, it will raise
///       a invalid_status exception.
///
///       This function is one of the few API functions which can be called before the runtime system has been fully
///       initialized. It will automatically stage the provided startup function to the runtime system during its
///       initialization (if necessary).
///
/// \see    \a einsums::register_pre_startup_function()
EINSUMS_EXPORT void register_startup_function(StartupFunctionType f);

} // namespace einsums