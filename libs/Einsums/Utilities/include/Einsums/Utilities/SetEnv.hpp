//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Config/Namespace.hpp>

#include <string>

EINSUMS_NAMESPACE_BEGIN()

/**
 * @brief Set (or overwrite) an environment variable, portably.
 *
 * setenv on POSIX, _putenv_s on MSVC.
 *
 * @warning Racing std::getenv on another thread is undefined behavior. Call during startup or
 *          test setup, before other threads exist.
 *
 * @param[in] name The variable name.
 * @param[in] value The value to set.
 */
EINSUMS_EXPORT void set_env_var(std::string const &name, std::string const &value);

/**
 * @brief Remove an environment variable, portably.
 *
 * std::getenv returns nullptr afterwards on both platforms. Same warning as set_env_var().
 *
 * @param[in] name The variable name.
 */
EINSUMS_EXPORT void unset_env_var(std::string const &name);

EINSUMS_NAMESPACE_END()
