//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <einsums/config.hpp>

#include <einsums/logging.hpp>
#include <einsums/modules/errors.hpp>
#include <einsums/runtime_configuration/init_ini_data.hpp>
#include <einsums/string_util/tokenize.hpp>

#include <cstdlib>
#include <filesystem>
#include <iostream>

namespace einsums::util {

bool handle_ini_file(einsums::detail::section &ini, std::string const &loc) {
    try {
        std::error_code ec;
        if (!std::filesystem::exists(loc, ec) || ec) {
            return false;
        }
        ini.read(loc);
    } catch (einsums::exception const &) {
        return false;
    }
    return true;
}

bool handle_ini_file_env(einsums::detail::section &ini, char const *env_var, char const *file_suffix) {
    char const *env = getenv(env_var);
    if (env != nullptr) {
        std::filesystem::path inipath(env);
        if (file_suffix != nullptr) {
            inipath /= std::filesystem::path(file_suffix);
        }

        if (handle_ini_file(ini, inipath.string())) {
            EINSUMS_LOG(info, "loaded configuration (${{{}}}): {}", env_var, inipath.string());
            return true;
        }
    }
    return false;
}

bool init_ini_data_base(einsums::detail::section &ini, std::string &einsums_ini_file) {
    std::string ini_path(ini.get_entry("einsums.master_ini_path"));
    std::string ini_paths_suffixes(ini.get_entry("einsums.master_ini_path_suffixes"));

    auto tok_paths    = string_util::tokenize(ini_path, ':');
    auto tok_suffixes = string_util::tokenize(ini_paths_suffixes, ':');

    bool result = false;
    for (auto i : tok_paths) {
        for (auto j : tok_suffixes) {
            std::string path(i);
            path += j;
            bool result2 = handle_ini_file(ini, path + "/einsums.ini");
            if (result2) {
                EINSUMS_LOG(info, "loaded configuration: {}/einsums.ini", path);
            }
            result = result2 || result;
        }
    }

    // look in the current directory first
    std::string cwd = std::filesystem::current_path().string() + "/.einsums.ini";
    {
        bool result2 = handle_ini_file(ini, cwd);
        if (result2) {
            EINSUMS_LOG(info, "loaded configuration: {}", cwd);
        }
        result = result2 || result;
    }

    // look for master ini in the EINSUMS_INI environment
    result = handle_ini_file_env(ini, "EINSUMS_INI") || result;

    // afterwards in the standard locations
#if !defined(EINSUMS_WINDOWS) // /etc/einsums.ini doesn't make sense for Windows
    {
        bool result2 = handle_ini_file(ini, "/etc/einsums.ini");
        if (result2) {
            EINSUMS_LOG(info, "loaded configuration: /etc/einsums.ini");
        }
        result = result2 || result;
    }
#endif

    result = handle_ini_file_env(ini, "HOME", ".einsums.ini") || result;
    result = handle_ini_file_env(ini, "PWD", ".einsums.ini") || result;

    if (!einsums_ini_file.empty()) {
        std::error_code ec;
        if (!std::filesystem::exists(einsums_ini_file, ec) || ec) {
            std::cerr << "einsums::init: command line warning: file specified using --einsums:config "
                         "does not exist ("
                      << einsums_ini_file << ")." << std::endl;
            einsums_ini_file.clear();
            result = false;
        } else {
            bool result2 = handle_ini_file(ini, einsums_ini_file);
            if (result2) {
                EINSUMS_LOG(info, "loaded configuration: {}", einsums_ini_file);
            }
            return result || result2;
        }
    }
    return result;
}

void merge_component_inis(einsums::detail::section &ini) {
    // now merge all information into one global structure
    std::string                   ini_path(ini.get_entry("einsums.ini_path"));
    std::vector<std::string_view> ini_paths;

    // split off the separate paths from the given path list
    ini_paths = string_util::tokenize(ini_path, ':');

    // have all path elements, now find ini files in there...
    std::vector<std::string_view>::iterator ini_end = ini_paths.end();
    for (std::vector<std::string_view>::iterator it = ini_paths.begin(); it != ini_end; ++it) {
        try {
            std::filesystem::directory_iterator nodir;
            std::filesystem::path               this_path(*it);

            std::error_code ec;
            if (!std::filesystem::exists(this_path, ec) || ec)
                continue;

            for (std::filesystem::directory_iterator dir(this_path); dir != nodir; ++dir) {
                if (dir->path().extension() != ".ini")
                    continue;

                // read and merge the ini file into the main ini hierarchy
                try {
                    ini.merge(dir->path().string());
                    EINSUMS_LOG(info, "loaded configuration: {}", dir->path().string());
                } catch (einsums::exception const & /*e*/) {
                    ;
                }
            }
        } catch (std::filesystem::filesystem_error const & /*e*/) {
            ;
        }
    }
}

} // namespace einsums::util