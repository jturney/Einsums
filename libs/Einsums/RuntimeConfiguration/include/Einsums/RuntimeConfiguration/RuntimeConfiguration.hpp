//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Logging.hpp>
#include <Einsums/TypeSupport/Observable.hpp>

#if defined(EINSUMS_HAVE_UNISTD_H)
#    include <unistd.h>
#endif

#if defined(EINSUMS_WINDOWS)
#    include <process.h>
#endif

#include <functional>
#include <string>
#include <vector>

namespace einsums {

/**
 * @brief Add a function to the list of startup functions to add module-specific command line arguments.
 *
 * @versionadded{1.0.0}
 * @versionchanged{2.0.0} parameter changed to const&
 */
EINSUMS_EXPORT void register_arguments(std::function<void()> const &);

/**
 * @struct RuntimeConfiguration
 *
 * Handles the current configuration state of the running instance.
 *
 * Currently, defaults are handled in pre_initialize. Eventually,
 * some kind of configuration file will be implemented that will
 * override the defaults set in pre_initialize.
 *
 * The current instance of the RuntimeConfiguration can be obtained
 * from Runtime::config() or from runtime_config() functions.
 *
 * @versionadded{1.0.0}
 */
struct EINSUMS_EXPORT RuntimeConfiguration {
    /**
     * @property original
     *
     * @todo Document.
     *
     * @versionadded{1.0.0}
     */
    std::vector<std::string> original;

    /**
     * Constructor of the runtime configuration object of einsums.
     *
     * @param[in] argc the argc argument from main
     * @param[in] argv the argv argument from main
     * @param[in] user_command_line callback function that can be used to register additional command-line options
     *
     * @versionadded{1.0.0}
     */
    RuntimeConfiguration(int argc, char const *const *argv, std::function<void()> const &user_command_line = {});

    /**
     * Constructor of the runtime configuration object of einsums. This is used when argv has been packaged into a vector.
     *
     * @param[in] argv The argv that has been packaged up.
     * @param[in] user_command_line callback function that can be used to register additional command-line options
     *
     * @versionadded{1.0.0}
     */
    explicit RuntimeConfiguration(std::vector<std::string> const &argv, std::function<void()> const &user_command_line = {});

    RuntimeConfiguration() = delete;

    [[nodiscard]] std::vector<std::string> const &unknown_arguments() const { return _unknown_arguments; }

  private:
    /**
     * Currently sets reasonable defaults for the development of Einsums.
     *
     * @versionadded{1.0.0}
     */
    void pre_initialize();

    /**
     * Parse the command line arguments provided in argc and argv. Returns unknown command line arguments.
     *
     * @param[in] user_command_line Callbiack function that can be used to register additional command-line arguments.
     *
     * @versionadded{1.0.0}
     */
    void parse_command_line(std::function<void()> const &user_command_line = {});

    std::vector<std::string> _unknown_arguments;
};

} // namespace einsums