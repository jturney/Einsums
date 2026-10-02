//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Config/Namespace.hpp>

#include <functional>

EINSUMS_NAMESPACE_BEGIN()

/// The type of a function which is registered to be executed as a startup
/// or pre-startup function.
using StartupFunctionType = std::function<void()>;

/// \brief Add a function to run before einsums_main and before every startup function.
///
/// \param f  [in] The function to register.
///
/// \note Callable before initialization; throws once the pre-startup functions have started.
///
/// \see    \a einsums::register_startup_function()
/// @versionadded{1.0.0}
EINSUMS_EXPORT void register_pre_startup_function(StartupFunctionType f);

/// \brief Add a function to run before einsums_main, after every pre-startup function.
///
/// \param f  [in] The function to register.
///
/// \note Callable before initialization; throws once the startup functions have started.
///
/// \see    \a einsums::register_pre_startup_function()
/// @versionadded{1.0.0}
EINSUMS_EXPORT void register_startup_function(StartupFunctionType f);

EINSUMS_NAMESPACE_END()
