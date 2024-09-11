//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <einsums/errors/error_code.hpp>
#include <einsums/ini/ini.hpp>
#include <einsums/runtime_configuration/runtime_configuration_fwd.hpp>

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace einsums::util {

// The runtime_configuration class is a wrapper for the runtime
// configuration data allowing to extract configuration information in a
// more convenient way
struct runtime_configuration : public einsums::detail::section {

    // initialize and load configuration information
    explicit runtime_configuration(char const *argv0, std::vector<std::string> const &extra_static_ini_defs = {});

    // re-initialize all entries based on the additional information from the
    // given configuration file.
    void reconfigure(std::string const &ini_file);

    // re-initialize all entries based on the additional information from
    // any explicit command line options
    void reconfigure(std::vector<std::string> const &ini_defs);

    // Load application specific configuration and merge it with the
    // default configuration loaded from pika.ini
    bool load_application_configuration(char const *filename, error_code &ec = throws);

  private:
    void pre_initialize_ini();
    void post_initialize_ini(std::string &pika_ini_file, std::vector<std::string> const &cmdline_ini_defs);
    void pre_initialize_logging_ini();

    void reconfigure();

  private:
    std::string              einsums_ini_file;
    std::vector<std::string> cmdline_ini_defs;
    std::vector<std::string> extra_static_ini_defs;

    bool need_to_call_pre_initialize;
#if defined(__linux) || defined(linx) || defined(__linux__)
    char const *argv0;
#endif
};

} // namespace einsums::util