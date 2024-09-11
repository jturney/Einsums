//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <einsums/assert.hpp>
#include <einsums/command_line_handling/late_command_line_handling.hpp>
#include <einsums/command_line_handling/parse_command_line.hpp>
#include <einsums/program_options/options_description.hpp>
#include <einsums/program_options/variables_map.hpp>
#include <einsums/runtime_configuration/runtime_configuration.hpp>
#include <einsums/string_util/from_string.hpp>
#include <einsums/type_support/unused.hpp>

#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace einsums::detail {
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
void decode(std::string &str, char const *s, char const *r) {
    std::string::size_type pos = 0;
    while ((pos = str.find(s, pos)) != std::string::npos) {
        str.replace(pos, 2, r);
    }
}

std::string decode_string(std::string str) {
    decode(str, "\\n", "\n");
    return str;
}

int handle_late_commandline_options(einsums::util::runtime_configuration &ini, einsums::program_options::options_description const &options,
                                    void (*handle_print_bind)(std::size_t)) {
    // do secondary command line processing, check validity of options only
    try {
        std::string unknown_cmd_line(ini.get_entry("einsums.unknown_cmd_line", ""));
        if (!unknown_cmd_line.empty()) {
            einsums::program_options::variables_map vm;

            commandline_error_mode mode = commandline_error_mode::rethrow_on_error;
            std::string            allow_unknown(ini.get_entry("einsums.commandline.allow_unknown", "0"));
            if (allow_unknown != "0")
                mode = commandline_error_mode::allow_unregistered;

            std::vector<std::string> still_unregistered_options;
            parse_commandline(ini, options, unknown_cmd_line, vm, mode, nullptr, &still_unregistered_options);

            std::string still_unknown_commandline;
            for (std::size_t i = 1; i < still_unregistered_options.size(); ++i) {
                if (i != 1) {
                    still_unknown_commandline += " ";
                }
                still_unknown_commandline += enquote(still_unregistered_options[i]);
            }

            if (!still_unknown_commandline.empty()) {
                einsums::detail::section *s = ini.get_section("einsums");
                EINSUMS_ASSERT(s != nullptr);
                s->add_entry("unknown_cmd_line_option", still_unknown_commandline);
            }
        }

        std::string fullhelp(ini.get_entry("einsums.cmd_line_help", ""));
        if (!fullhelp.empty()) {
            std::string help_option(ini.get_entry("einsums.cmd_line_help_option", ""));
            if (0 == std::string("full").find(help_option)) {
                std::cout << decode_string(fullhelp);
                std::cout << options << std::endl;
            } else {
                throw einsums::detail::command_line_error("unknown help option: " + help_option);
            }
            return 1;
        }

        // secondary command line handling, looking for --exit and other
        // options
        std::string cmd_line = ini.get_entry("einsums.commandline.command", "") + " " +
                               ini.get_entry("einsums.commandline.prepend_options", "") + ini.get_entry("einsums.commandline.options", "") +
                               ini.get_entry("einsums.commandline.config_options", "");

        if (!cmd_line.empty()) {
            einsums::program_options::variables_map vm;

            parse_commandline(ini, options, cmd_line, vm,
                              commandline_error_mode::allow_unregistered | commandline_error_mode::report_missing_config_file);

            if (vm.count("einsums:print-bind") || (std::getenv("EINSUMS_PRINT_BIND") != nullptr)) {
                std::size_t num_threads = einsums::string_util::from_string<std::size_t>(ini.get_entry("einsums.os_threads", 1));
                handle_print_bind(num_threads);
            }

            if (vm.count("einsums:exit")) {
                return 1;
            }
        }
    } catch (std::exception const &e) {
        std::cerr << "handle_late_commandline_options: " << "command line processing: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
} // namespace einsums::detail
