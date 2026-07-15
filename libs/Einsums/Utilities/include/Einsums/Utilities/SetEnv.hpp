//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <string>

namespace einsums {

/**
 * @brief Set (or overwrite) an environment variable, portably.
 *
 * There is no standard C++ way to SET an environment variable: std::getenv is
 * the whole standard surface, setenv/unsetenv are POSIX, and the MSVC CRT
 * spells it _putenv_s. This wraps the difference.
 *
 * @warning The environment block is process-global mutable state with no
 *          synchronization: mutating it while another thread calls
 *          std::getenv (directly or through a library) is undefined behavior
 *          on every platform. Call during startup/configuration, before
 *          threads that might read the environment exist - the runtime's
 *          initialization path and test setup qualify; steady-state code
 *          generally should not.
 *
 * @param[in] name The variable name.
 * @param[in] value The value to set.
 */
EINSUMS_EXPORT void set_env_var(std::string const &name, std::string const &value);

/**
 * @brief Remove an environment variable, portably.
 *
 * On Windows the variable is set to the empty string, which the CRT treats
 * as removal (_putenv_s semantics); std::getenv returns nullptr afterwards
 * on both platforms. The same thread-safety warning as set_env_var()
 * applies.
 *
 * @param[in] name The variable name.
 */
EINSUMS_EXPORT void unset_env_var(std::string const &name);

} // namespace einsums
