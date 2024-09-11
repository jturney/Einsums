//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <einsums/config.hpp>

#include <einsums/assert.hpp>
#include <einsums/command_line_handling/command_line_handling.hpp>
#include <einsums/command_line_handling/parse_command_line.hpp>
#include <einsums/debugging/attach_debugger.hpp>
// #include <einsums/functional/detail/reset_function.hpp>
#include <einsums/preprocessor/stringize.hpp>
#include <einsums/program_options/options_description.hpp>
#include <einsums/program_options/variables_map.hpp>
#include <einsums/runtime_configuration/runtime_configuration.hpp>
#include <einsums/string_util/from_string.hpp>
#include <einsums/string_util/tokenize.hpp>
#include <einsums/topology/cpu_mask.hpp>
#include <einsums/topology/topology.hpp>
#include <einsums/type_support/unused.hpp>
// #include <einsums/util/get_entry_as.hpp>
#include <einsums/util/manage_config.hpp>
#include <einsums/version.hpp>

#include <fmt/ostream.h>
#include <fmt/printf.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <memory>
#include <spdlog/common.h>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace einsums::detail {
std::string runtime_configuration_string(command_line_handling const &cfg) {
    std::ostringstream strm;

    // default scheduler used for this run
    strm << "  {scheduler}: " << cfg._queuing << "\n";

    // amount of threads and cores configured for this run
    strm << "  {os-threads}: " << cfg._num_threads << "\n";
    strm << "  {cores}: " << cfg._num_cores << "\n";

    return strm.str();
}

///////////////////////////////////////////////////////////////////////
int print_version(std::ostream &out) {
    out << std::endl << einsums::copyright() << std::endl;
    out << einsums::complete_version() << std::endl;
    return 1;
}

int print_info(std::ostream &out, command_line_handling const &cfg) {
    out << "Static configuration:\n---------------------\n";
    out << einsums::configuration_string() << std::endl;

    out << "Runtime configuration:\n----------------------\n";
    out << runtime_configuration_string(cfg) << std::endl;

    return 1;
}

///////////////////////////////////////////////////////////////////////
inline void encode(std::string &str, char s, char const *r, std::size_t inc = 1ull) {
    std::string::size_type pos = 0;
    while ((pos = str.find_first_of(s, pos)) != std::string::npos) {
        str.replace(pos, 1, r);
        pos += inc;
    }
}

inline std::string encode_string(std::string str) {
    encode(str, '\n', "\\n");
    return str;
}

inline std::string encode_and_enquote(std::string str) {
    encode(str, '\"', "\\\"", 2);
    return enquote(std::move(str));
}

///////////////////////////////////////////////////////////////////////
std::string handle_process_mask(detail::manage_config &cfgmap, einsums::program_options::variables_map &vm, bool use_process_mask) {
    std::string mask_string = cfgmap.get_value<std::string>("einsums.process_mask", "");

    char const *mask_env = std::getenv("EINSUMS_PROCESS_MASK");
    if (nullptr != mask_env) {
        mask_string = mask_env;
    }

    if (vm.count("einsums:process-mask")) {
        mask_string = vm["einsums:process-mask"].as<std::string>();
    }

#if defined(__APPLE__)
    EINSUMS_UNUSED(use_process_mask);

    if (!mask_string.empty()) {
        fmt::print(std::cerr, "Explicit process mask is set with --einsums:process-mask or EINSUMS_PROCESS_MASK, but "
                              "thread binding is not supported on macOS. The process mask will be ignored.");
        mask_string = "";
    }
#else
    if (!mask_string.empty() && !use_process_mask) {
        fmt::print(std::cerr, "Explicit process mask is set with --einsums:process-mask or EINSUMS_PROCESS_MASK, but "
                              "--einsums:ignore-process-mask is also set. The process mask will be ignored.\n");
    }
#endif

    return mask_string;
}

std::string handle_queuing(detail::manage_config &cfgmap, einsums::program_options::variables_map &vm, std::string const &default_) {
    // command line options is used preferred
    if (vm.count("einsums:queuing"))
        return vm["einsums:queuing"].as<std::string>();

    // use either cfgmap value or default
    return cfgmap.get_value<std::string>("einsums.scheduler", default_);
}

std::string handle_affinity(detail::manage_config &cfgmap, einsums::program_options::variables_map &vm, std::string const &default_) {
    // command line options is used preferred
    if (vm.count("einsums:affinity"))
        return vm["einsums:affinity"].as<std::string>();

    // use either cfgmap value or default
    return cfgmap.get_value<std::string>("einsums.affinity", default_);
}

std::string handle_affinity_bind(detail::manage_config &cfgmap, einsums::program_options::variables_map &vm, std::string const &default_) {
    // command line options is used preferred
    if (vm.count("einsums:bind")) {
        std::string affinity_desc;

        std::vector<std::string> bind_affinity = vm["einsums:bind"].as<std::vector<std::string>>();
        for (std::string const &s : bind_affinity) {
            if (!affinity_desc.empty())
                affinity_desc += ";";
            affinity_desc += s;
        }

        return affinity_desc;
    }

    // use either cfgmap value or default
    return cfgmap.get_value<std::string>("einsums.bind", default_);
}

std::size_t handle_pu_step(detail::manage_config &cfgmap, einsums::program_options::variables_map &vm, std::size_t default_) {
    // command line options is used preferred
    if (vm.count("einsums:pu-step"))
        return vm["einsums:pu-step"].as<std::size_t>();

    // use either cfgmap value or default
    return cfgmap.get_value<std::size_t>("einsums.pu_step", default_);
}

std::size_t handle_pu_offset(detail::manage_config &cfgmap, einsums::program_options::variables_map &vm, std::size_t default_) {
    // command line options is used preferred
    if (vm.count("einsums:pu-offset"))
        return vm["einsums:pu-offset"].as<std::size_t>();

    // use either cfgmap value or default
    return cfgmap.get_value<std::size_t>("einsums.pu_offset", default_);
}

std::size_t handle_numa_sensitive(detail::manage_config &cfgmap, einsums::program_options::variables_map &vm, std::size_t default_) {
    if (vm.count("einsums:numa-sensitive") != 0) {
        std::size_t numa_sensitive = vm["einsums:numa-sensitive"].as<std::size_t>();
        if (numa_sensitive > 2) {
            throw einsums::detail::command_line_error("Invalid argument value for --einsums:numa-sensitive. Allowed values are 0, 1, or "
                                                      "2");
        }
        return numa_sensitive;
    }

    // use either cfgmap value or default
    return cfgmap.get_value<std::size_t>("einsums.numa_sensitive", default_);
}

///////////////////////////////////////////////////////////////////////
std::size_t get_number_of_default_threads(bool use_process_mask) {
    if (use_process_mask) {
        threads::detail::topology &top = threads::detail::get_topology();
        return threads::detail::count(top.get_cpubind_mask_main_thread());
    } else {
        return threads::detail::hardware_concurrency();
    }
}

std::size_t get_number_of_default_cores(bool use_process_mask) {
    threads::detail::topology &top = threads::detail::get_topology();

    std::size_t num_cores = top.get_number_of_cores();

    if (use_process_mask) {
        threads::detail::mask_type proc_mask           = top.get_cpubind_mask_main_thread();
        std::size_t                num_cores_proc_mask = 0;

        for (std::size_t num_core = 0; num_core < num_cores; ++num_core) {
            threads::detail::mask_type core_mask = top.init_core_affinity_mask_from_core(num_core);
            if (threads::detail::bit_and(core_mask, proc_mask)) {
                ++num_cores_proc_mask;
            }
        }

        return num_cores_proc_mask;
    }

    return num_cores;
}

///////////////////////////////////////////////////////////////////////
std::size_t handle_num_threads(detail::manage_config &cfgmap, einsums::util::runtime_configuration const &rtcfg,
                               einsums::program_options::variables_map &vm, bool use_process_mask) {
    // If using the process mask we override "cores" and "all" options but
    // keep explicit numeric values.
    const std::size_t init_threads = get_number_of_default_threads(use_process_mask);
    const std::size_t init_cores   = get_number_of_default_cores(use_process_mask);

    std::size_t default_threads = init_threads;

    std::string threads_str =
        cfgmap.get_value<std::string>("einsums.os_threads", rtcfg.get_entry("einsums.os_threads", std::to_string(default_threads)));

    if ("cores" == threads_str) {
        default_threads = init_cores;
    } else if ("all" == threads_str) {
        default_threads = init_threads;
    } else {
        default_threads = einsums::string_util::from_string<std::size_t>(threads_str);
    }

    std::size_t threads = cfgmap.get_value<std::size_t>("einsums.os_threads", default_threads);

    if (vm.count("einsums:threads")) {
        threads_str = vm["einsums:threads"].as<std::string>();
        if ("all" == threads_str) {
            threads = init_threads;
        } else if ("cores" == threads_str) {
            threads = init_cores;
        } else {
            threads = einsums::string_util::from_string<std::size_t>(threads_str);
        }

        if (threads == 0) {
            throw einsums::detail::command_line_error("Number of --einsums:threads must be greater than 0");
        }

#if defined(EINSUMS_HAVE_MAX_CPU_COUNT)
        if (threads > EINSUMS_HAVE_MAX_CPU_COUNT) {
            // clang-format off
                    throw einsums::detail::command_line_error("Requested more than "
                        EINSUMS_PP_STRINGIZE(EINSUMS_HAVE_MAX_CPU_COUNT)" --einsums:threads "
                        "to use for this application, use the option "
                        "-DEINSUMS_WITH_MAX_CPU_COUNT=<N> when configuring einsums.");
            // clang-format on
        }
#endif
    }

    // make sure minimal requested number of threads is observed
    std::size_t min_os_threads = cfgmap.get_value<std::size_t>("einsums.force_min_os_threads", threads);

    if (min_os_threads == 0) {
        throw einsums::detail::command_line_error("Number of einsums.force_min_os_threads must be greater than 0");
    }

#if defined(EINSUMS_MAX_CPU_COUNT)
    if (min_os_threads > EINSUMS_MAX_CPU_COUNT) {
        throw einsums::detail::command_line_error("Requested more than " EINSUMS_PP_STRINGIZE(
            EINSUMS_HAVE_MAX_CPU_COUNT) " einsums.force_min_os_threads to use for this application, "
                                        "use the option -DEINSUMS_WITH_MAX_CPU_COUNT=<N> when "
                                        "configuring einsums.");
    }
#endif

    threads = (std::max)(threads, min_os_threads);

    return threads;
}

std::size_t handle_num_cores(detail::manage_config &cfgmap, einsums::program_options::variables_map &vm, std::size_t num_threads,
                             bool use_process_mask) {
    std::string cores_str = cfgmap.get_value<std::string>("einsums.cores", "");
    if ("all" == cores_str) {
        cfgmap.config_["einsums.cores"] = std::to_string(get_number_of_default_cores(use_process_mask));
    }

    std::size_t num_cores = cfgmap.get_value<std::size_t>("einsums.cores", num_threads);
    if (vm.count("einsums:cores")) {
        cores_str = vm["einsums:cores"].as<std::string>();
        if ("all" == cores_str) {
            num_cores = get_number_of_default_cores(use_process_mask);
        } else {
            num_cores = einsums::string_util::from_string<std::size_t>(cores_str);
        }
    }

    return num_cores;
}

///////////////////////////////////////////////////////////////////////
void command_line_handling::check_affinity_domain() const {
    if (_affinity_domain != "pu") {
        if (0 != std::string("pu").find(_affinity_domain) && 0 != std::string("core").find(_affinity_domain) &&
            0 != std::string("numa").find(_affinity_domain) && 0 != std::string("machine").find(_affinity_domain)) {
            throw einsums::detail::command_line_error("Invalid command line option --einsums:affinity, value must be one of: pu, core, "
                                                      "numa, or machine.");
        }
    }
}

void command_line_handling::check_affinity_description() const {
    if (_affinity_bind.empty()) {
        return;
    }

    if (!(_pu_offset == std::size_t(-1) || _pu_offset == std::size_t(0)) || _pu_step != 1 || _affinity_domain != "pu") {
        throw einsums::detail::command_line_error("Command line option --einsums:bind should not be used with --einsums:pu-step, "
                                                  "--einsums:pu-offset, or --einsums:affinity.");
    }
}

void command_line_handling::check_pu_offset() const {
    if (_pu_offset != std::size_t(-1) && _pu_offset >= einsums::threads::detail::hardware_concurrency()) {
        throw einsums::detail::command_line_error("Invalid command line option --einsums:pu-offset, value must be smaller than number "
                                                  "of available processing units.");
    }
}

void command_line_handling::check_pu_step() const {
    if (einsums::threads::detail::hardware_concurrency() > 1 &&
        (_pu_step == 0 || _pu_step >= einsums::threads::detail::hardware_concurrency())) {
        throw einsums::detail::command_line_error("Invalid command line option --einsums:pu-step, value must be non-zero and smaller "
                                                  "than number of available processing units.");
    }
}

///////////////////////////////////////////////////////////////////////////
void command_line_handling::handle_arguments(detail::manage_config &cfgmap, einsums::program_options::variables_map &_vm,
                                             std::vector<std::string> &ini_config) {
    bool debug_clp = _vm.count("einsums:debug-clp");

    if (_vm.count("einsums:ini")) {
        std::vector<std::string> cfg = _vm["einsums:ini"].as<std::vector<std::string>>();
        std::copy(cfg.begin(), cfg.end(), std::back_inserter(ini_config));
        cfgmap.add(cfg);
    }

    _use_process_mask =
#if defined(__APPLE__)
        false;
#else
        !((cfgmap.get_value<int>("einsums.ignore_process_mask", 0) > 0) || (_vm.count("einsums:ignore-process-mask") > 0));
#endif

    ini_config.emplace_back("einsums.ignore_process_mask!=" + std::to_string(!_use_process_mask));

    _process_mask = handle_process_mask(cfgmap, _vm, _use_process_mask);
    ini_config.emplace_back("einsums.process_mask!=" + _process_mask);
    if (!_process_mask.empty()) {
        auto const m = string_util::from_string<threads::detail::mask_type>(_process_mask);
        threads::detail::get_topology().set_cpubind_mask_main_thread(m);
    }

    // handle setting related to schedulers
    _queuing = detail::handle_queuing(cfgmap, _vm, "local-priority-fifo");
    ini_config.emplace_back("einsums.scheduler=" + _queuing);

    _affinity_domain = detail::handle_affinity(cfgmap, _vm, "pu");
    ini_config.emplace_back("einsums.affinity=" + _affinity_domain);

    check_affinity_domain();

    _affinity_bind = detail::handle_affinity_bind(cfgmap, _vm, "");
    if (!_affinity_bind.empty()) {
#if defined(__APPLE__)
        if (_affinity_bind != "none") {
            std::cerr << "Warning: thread binding set to \"" << _affinity_bind
                      << "\" but thread binding is not supported on macOS. Ignoring option." << std::endl;
        }
        _affinity_bind = "";
#else
        ini_config.emplace_back("einsums.bind!=" + _affinity_bind);
#endif
    }

    _pu_step = detail::handle_pu_step(cfgmap, _vm, 1);
#if defined(__APPLE__)
    if (_pu_step != 1) {
        std::cerr << "Warning: PU step set to \"" << _pu_step << "\" but thread binding is not supported on macOS. Ignoring option."
                  << std::endl;
        _pu_step = 1;
    }
#endif
    ini_config.emplace_back("einsums.pu_step=" + std::to_string(_pu_step));

    check_pu_step();

    _pu_offset = detail::handle_pu_offset(cfgmap, _vm, std::size_t(-1));

    // NOLINTNEXTLINE(bugprone-branch-clone)
    if (_pu_offset != std::size_t(-1)) {
#if defined(__APPLE__)
        std::cerr << "Warning: PU offset set to \"" << _pu_offset << "\" but thread binding is not supported on macOS. Ignoring option."
                  << std::endl;
        _pu_offset = std::size_t(-1);
        ini_config.emplace_back("einsums.pu_offset=0");
#else
        ini_config.emplace_back("einsums.pu_offset=" + std::to_string(_pu_offset));
#endif
    } else {
        ini_config.emplace_back("einsums.pu_offset=0");
    }

    check_pu_offset();

    _numa_sensitive = detail::handle_numa_sensitive(cfgmap, _vm, _affinity_bind.empty() ? 0 : 1);
    ini_config.emplace_back("einsums.numa_sensitive=" + std::to_string(_numa_sensitive));

    // default affinity mode is now 'balanced' (only if no pu-step or
    // pu-offset is given)
    if (_pu_step == 1 && _pu_offset == std::size_t(-1) && _affinity_bind.empty()) {
#if defined(__APPLE__)
        _affinity_bind = "none";
#else
        _affinity_bind = "balanced";
#endif
        ini_config.emplace_back("einsums.bind!=" + _affinity_bind);
    }

    check_affinity_description();

    // handle number of cores and threads
    _num_threads = detail::handle_num_threads(cfgmap, _rtcfg, _vm, _use_process_mask);
    _num_cores   = detail::handle_num_cores(cfgmap, _vm, _num_threads, _use_process_mask);

    // Set number of cores and OS threads in configuration.
    ini_config.emplace_back("einsums.os_threads=" + std::to_string(_num_threads));
    ini_config.emplace_back("einsums.cores=" + std::to_string(_num_cores));

    if (_vm.count("einsums:high-priority-threads")) {
        std::size_t num_high_priority_queues = _vm["einsums:high-priority-threads"].as<std::size_t>();
        if (num_high_priority_queues != std::size_t(-1) && num_high_priority_queues > _num_threads) {
            throw einsums::detail::command_line_error("Invalid command line option: number of high priority threads "
                                                      "(--einsums:high-priority-threads), should not be larger than number of threads "
                                                      "(--einsums:threads)");
        }

        if (!(_queuing == "local-priority" || _queuing == "abp-priority")) {
            throw einsums::detail::command_line_error("Invalid command line option --einsums:high-priority-threads, valid for "
                                                      "--einsums:queuing=local-priority and --einsums:queuing=abp-priority only");
        }

        ini_config.emplace_back("einsums.thread_queue.high_priority_queues!=" + std::to_string(num_high_priority_queues));
    }

#if defined(EINSUMS_HAVE_MPI)
    std::size_t const mpi_completion_mode = detail::handle_mpi_completion_mode(cfgmap, _rtcfg, vm);
    ini_config.emplace_back("einsums.mpi.completion_mode=" + std::to_string(mpi_completion_mode));
#endif

    update_logging_settings(_vm, ini_config);

    if (debug_clp) {
        std::cerr << "Configuration before runtime start:\n";
        std::cerr << "-----------------------------------\n";
        for (std::string const &s : ini_config) {
            std::cerr << s << std::endl;
        }
        std::cerr << "-----------------------------------\n";
    }
}

///////////////////////////////////////////////////////////////////////////
void command_line_handling::update_logging_settings(einsums::program_options::variables_map &vm, std::vector<std::string> &ini_config) {
    if (vm.count("einsums:log-destination")) {
        ini_config.emplace_back("einsums.log.destination=" + vm["einsums:log-destination"].as<std::string>());
    }

    if (vm.count("einsums:log-level")) {
        ini_config.emplace_back("einsums.log.level=" +
                                std::to_string(vm["einsums:log-level"].as<std::underlying_type_t<spdlog::level::level_enum>>()));
    }

    if (vm.count("einsums:log-format")) {
        ini_config.emplace_back("einsums.log.format=" + vm["einsums:log-format"].as<std::string>());
    }
}

///////////////////////////////////////////////////////////////////////////
void command_line_handling::store_command_line(int argc, const char *const *argv) {
    // Collect the command line for diagnostic purposes.
    std::string command;
    std::string cmd_line;
    std::string options;
    for (int i = 0; i < argc; ++i) {
        // quote only if it contains whitespace
        std::string arg = detail::encode_and_enquote(argv[i]); //-V108

        cmd_line += arg;
        if (i == 0) {
            command = arg;
        } else {
            options += " " + arg;
        }

        if ((i + 1) != argc) {
            cmd_line += " ";
        }
    }

    // Store the program name and the command line.
    _ini_config.emplace_back("einsums.cmd_line!=" + cmd_line);
    _ini_config.emplace_back("einsums.commandline.command!=" + command);
    _ini_config.emplace_back("einsums.commandline.options!=" + options);
}

///////////////////////////////////////////////////////////////////////////
void command_line_handling::store_unregistered_options(std::string const &cmd_name, std::vector<std::string> const &unregistered_options) {
    std::string unregistered_options_cmd_line;

    if (!unregistered_options.empty()) {
        using iterator_type = std::vector<std::string>::const_iterator;

        iterator_type end = unregistered_options.end();
        // Silence bogus warning from GCC 12:
        // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=105329
#if defined(EINSUMS_GCC_VERSION) && EINSUMS_GCC_VERSION >= 120000
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wrestrict"
#endif
        for (iterator_type it = unregistered_options.begin(); it != end; ++it)
            unregistered_options_cmd_line += " " + detail::encode_and_enquote(*it);
#if defined(EINSUMS_GCC_VERSION) && EINSUMS_GCC_VERSION >= 120000
#    pragma GCC diagnostic pop
#endif

        _ini_config.emplace_back("einsums.unknown_cmd_line!=" + detail::encode_and_enquote(cmd_name) + unregistered_options_cmd_line);
    }

    _ini_config.emplace_back("einsums.program_name!=" + cmd_name);
    _ini_config.emplace_back("einsums.reconstructed_cmd_line!=" + encode_and_enquote(cmd_name) + " " + reconstruct_command_line(_vm) + " " +
                             unregistered_options_cmd_line);
}

///////////////////////////////////////////////////////////////////////////
bool command_line_handling::handle_help_options(einsums::program_options::options_description const &help) {
    if (_vm.count("einsums:help")) {
        std::cout << help << std::endl;
        return true;
    }
    return false;
}

void command_line_handling::handle_attach_debugger() {
#if defined(_POSIX_VERSION) || defined(EINSUMS_WINDOWS)
    if (_vm.count("einsums:attach-debugger")) {
        std::string option = _vm["einsums:attach-debugger"].as<std::string>();
        if (option != "off" && option != "startup" && option != "exception" && option != "test-failure") {
            // clang-format off
                std::cerr <<
                    "einsums::init: command line warning: --einsums:attach-debugger: "
                    "invalid option: " << option << ". Allowed values are "
                    "'off', 'startup', 'exception' or 'test-failure'" << std::endl;
            // clang-format on
        } else {
            if (option == "startup") {
                debug::detail::attach_debugger();
            } else if (option == "exception") {
                // Signal handlers need to be installed to be able to attach
                // the debugger on uncaught exceptions
                _ini_config.emplace_back("einsums.install_signal_handlers!=1");
            }

            _ini_config.emplace_back("einsums.attach_debugger!=" + option);
        }
    }
#endif
}

///////////////////////////////////////////////////////////////////////////
// separate command line arguments from configuration settings
std::vector<std::string> command_line_handling::preprocess_config_settings(int argc, const char *const *argv) {
    std::vector<std::string> options;
    options.reserve(static_cast<std::size_t>(argc) + _ini_config.size());

    // extract all command line arguments from configuration settings and
    // remove them from this list
    auto it = std::stable_partition(_ini_config.begin(), _ini_config.end(), [](std::string const &e) { return e.find("--einsums:") != 0; });

    std::move(it, _ini_config.end(), std::back_inserter(options));
    _ini_config.erase(it, _ini_config.end());

    // store the command line options that came from the configuration
    // settings in the registry
    if (!options.empty()) {
        std::string config_options;
        for (auto const &option : options) {
            config_options += " " + option;
        }

        _rtcfg.add_entry("einsums.commandline.config_options", config_options);
    }

    // now append all original command line options
    for (int i = 1; i != argc; ++i) {
        options.emplace_back(argv[i]);
    }

    return options;
}

///////////////////////////////////////////////////////////////////////////
std::vector<std::string> prepend_options(std::vector<std::string> &&args, std::string &&options) {
    if (options.empty()) {
        return std::move(args);
    }

    std::vector<std::string> result = string_util::split_escaped_list(options);
    std::move(args.begin(), args.end(), std::back_inserter(result));
    return result;
}

///////////////////////////////////////////////////////////////////////////
command_line_handling_result command_line_handling::call(einsums::program_options::options_description const &desc_cmdline, int argc,
                                                         const char *const *argv) {
    // set the flag signaling that command line parsing has been done
    _cmd_line_parsed = true;

    // separate command line arguments from configuration settings
    std::vector<std::string> args = preprocess_config_settings(argc, argv);

    detail::manage_config cfgmap(_ini_config);

    // insert the pre-configured ini settings before loading modules
    for (std::string const &e : _ini_config)
        _rtcfg.parse("<user supplied config>", e, true, false);

    // support re-throwing command line exceptions for testing purposes
    commandline_error_mode error_mode = commandline_error_mode::allow_unregistered;
    if (cfgmap.get_value("einsums.commandline.rethrow_errors", 0) != 0) {
        error_mode |= commandline_error_mode::rethrow_on_error;
    }

    // The cfg registry may hold command line options to prepend to the
    // real command line.
    std::string prepend_command_line = _rtcfg.get_entry("einsums.commandline.prepend_options");

    args = prepend_options(std::move(args), std::move(prepend_command_line));

    // Initial analysis of the command line options. This is
    // preliminary as it will not take into account any aliases as
    // defined in any of the runtime configuration files.
    {
        // Boost V1.47 and before do not properly reset a variables_map
        // when calling vm.clear(). We work around that problems by
        // creating a separate instance just for the preliminary
        // command line handling.
        einsums::program_options::variables_map prevm;
        parse_commandline(_rtcfg, desc_cmdline, argv[0], args, prevm, error_mode);

        // handle all --einsums:foo options
        std::vector<std::string> ini_config; // discard
        handle_arguments(cfgmap, prevm, ini_config);

        // re-initialize runtime configuration object
        if (prevm.count("einsums:config"))
            _rtcfg.reconfigure(prevm["einsums:config"].as<std::string>());
        else
            _rtcfg.reconfigure("");

        // Make sure any aliases defined on the command line get used
        // for the option analysis below.
        std::vector<std::string> cfg;
        if (prevm.count("einsums:ini")) {
            cfg = prevm["einsums:ini"].as<std::vector<std::string>>();
            cfgmap.add(cfg);
        }

        // append ini options from command line
        std::copy(_ini_config.begin(), _ini_config.end(), std::back_inserter(cfg));

        // set logging options from command line options
        std::vector<std::string> ini_config_logging;
        update_logging_settings(prevm, ini_config_logging);

        std::copy(ini_config_logging.begin(), ini_config_logging.end(), std::back_inserter(cfg));

        _rtcfg.reconfigure(cfg);
    }

    // Re-run program option analysis, ini settings (such as aliases)
    // will be considered now.

    // Now re-parse the command line using the node number (if given).
    // This will additionally detect any --einsums:N:foo options.
    einsums::program_options::options_description help;
    std::vector<std::string>                      unregistered_options;

    parse_commandline(_rtcfg, desc_cmdline, argv[0], args, _vm, error_mode | commandline_error_mode::report_missing_config_file, &help,
                      &unregistered_options);

    // break into debugger, if requested
    handle_attach_debugger();

    // handle all --einsums:foo and --einsums:*:foo options
    handle_arguments(cfgmap, _vm, _ini_config);

    // store unregistered command line and arguments
    store_command_line(argc, argv);
    store_unregistered_options(argv[0], unregistered_options);

    // add all remaining ini settings to the global configuration
    _rtcfg.reconfigure(_ini_config);

    // help can be printed only after the runtime mode has been set
    if (handle_help_options(help)) {
        return command_line_handling_result::exit;
    }

    // print version/copyright information
    if (_vm.count("einsums:version")) {
        if (!_version_printed) {
            detail::print_version(std::cout);
            _version_printed = true;
        }

        return command_line_handling_result::exit;
    }

    // print configuration information (static and dynamic)
    if (_vm.count("einsums:info")) {
        if (!_info_printed) {
            detail::print_info(std::cout, *this);
            _info_printed = true;
        }

        return command_line_handling_result::exit;
    }

    // Print a warning about process masks resulting in only one worker
    // thread, but only do so if that would not be the default on the given
    // system and it was not given explicitly to --einsums:threads.
    if (_use_process_mask) {
        bool const command_line_arguments_given = _vm.count("einsums:threads") != 0 || _vm.count("einsums:cores") != 0;
        if (_num_threads == 1 && get_number_of_default_threads(false) != 1 && !command_line_arguments_given) {
            std::cerr << "The einsums runtime will be started with only one worker thread because the "
                         "process mask has restricted the available resources to only one thread. If "
                         "this is unintentional make sure the process mask contains the resources "
                         "you need or use --einsums:ignore-process-mask to use all resources. Use "
                         "--einsums:print-bind to print the thread bindings used by einsums.\n";
        } else if (_num_cores == 1 && get_number_of_default_cores(false) != 1 && !command_line_arguments_given) {
            fmt::print(std::cerr,
                       "The einsums runtime will be started on only one core with {} worker threads "
                       "because the process mask has restricted the available resources to only one "
                       "core. If this is unintentional make sure the process mask contains the "
                       "resources you need or use --einsums:ignore-process-mask to use all resources. "
                       "Use --einsums:print-bind to print the thread bindings used by einsums.\n",
                       _num_threads);
        }
    }

    return command_line_handling_result::success;
}
} // namespace einsums::detail
