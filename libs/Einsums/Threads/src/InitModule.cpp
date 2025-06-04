//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include "InitModule.hpp"

#include <Einsums/Runtime.hpp>
#include <Einsums/Threads.hpp>

#include <argparse/argparse.hpp>

namespace einsums {

/*
 * Set up the internal state of the module. If the module does not need to be set up, then this
 * file can be safely deleted. Make sure that if you do, you also remove its reference in the CMakeLists.txt,
 * as well as the initialization header for the module and the dependence on Einsums_Runtime, assuming these
 * aren't being used otherwise.
 *
 * Logging will not be available by the time the initialization routines are run.
 */

int setup_Einsums_Threads() {
    // Auto-generated code. Do not touch if you are unsure of what you are doing.
    // Instead, modify the other functions below.
    // If you don't need a function, you may remove its respective line from the
    // if statement below.
    static bool is_initialized = false;

    if (!is_initialized) {
        register_arguments(add_Einsums_Threads_arguments);
        register_startup_function(initialize_Einsums_Threads);
        register_shutdown_function(finalize_Einsums_Threads);

        is_initialized = true;
    }

    return 0;
}

EINSUMS_EXPORT void add_Einsums_Threads_arguments(argparse::ArgumentParser &parser) {
    /** @todo Fill in.
     *
     * If you are not using one of the following maps, you may remove references to it.
     * The maps may not need to be locked before use, but it should still be done just in
     * case.
     */
    auto &global_config = GlobalConfigMap::get_singleton();
    auto &global_string = global_config.get_string_map()->get_value();
    auto &global_double = global_config.get_double_map()->get_value();
    auto &global_int    = global_config.get_int_map()->get_value();
    auto &global_bool   = global_config.get_bool_map()->get_value();

    /* Examples:
     * String argument:
     * parser.add_argument("--einsums:dummy-str")
     *   .default_value("default value")
     *   .help("Dummy string argument.")
     *   .store_into(global_string["dummy-str"]);
     *
     * Double argument:
     * parser.add_argument("--einsums:dummy-double")
     *   .default_value(1.23)
     *   .help("Dummy double argument.")
     *   .store_into(global_double["dummy-double"]);
     *
     * Integer argument: Be careful! These are int64_t, not int, so
     * literals need to be handled with care.
     *
     * parser.add_argument("--einsums:dummy-int")
     *   .default_value<int64_t>(1)
     *   .help("Dummy int argument.")
     *   .store_into(global_int["dummy-int"]);
     *
     * parser.add_argument("--einsums:dummy-int2")
     *   .default_value(1L)
     *   .help("Dummy int argument, another way.")
     *   .store_into(global_int["dummy-int2"]);
     *
     * Boolean argument:
     * This one adds a flag that is normally false, but is set when provided.
     * parser.add_argument("--einsums:dummy-bool")
     *   .flag()
     *   .help("Dummy int argument.")
     *   .store_into(global_int["dummy-bool"]);
     *
     * This one adds a flag that is normally true, but is unset when provided.
     * parser.add_argument("--einsums:no-dummy-bool2")
     *   .default_value(true)
     *   .implicit_value(false)
     *   .help("Dummy int argument.")
     *   .store_into(global_int["dummy-bool2"]);
     */
    parser.add_argument("--einsums:no-lock-detection")
        .default_value(true)
        .implicit_value(false)
        .help("Do not perform mutex lock detection")
        .store_into(global_bool["einsums:lock-detection"]);
    parser.add_argument("--einsums:trace-depth")
        .default_value(static_cast<int64_t>(EINSUMS_HAVE_THREAD_BACKTRACE_DEPTH))
        .help("How many lines to include in backtrace on various thread detections")
        .store_into(global_int["einsums:trace-depth"]);
}

void initialize_Einsums_Threads() {
    auto &global_config = GlobalConfigMap::get_singleton();
    auto &global_int    = global_config.get_int_map()->get_value();
    auto &global_bool   = global_config.get_bool_map()->get_value();

    if (global_bool["einsums::lock-detection"]) {
        threads::enable_lock_detection();
        threads::trace_depth_lock_detection(global_int["einsums:trace-depth"]);
    } else {
        threads::disable_lock_detection();
    }
}

void finalize_Einsums_Threads() {
    /// @todo Fill in.
}

} // namespace einsums