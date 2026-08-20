//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/BufferAllocator/InitModule.hpp>
#include <Einsums/BufferAllocator/ModuleVars.hpp>
#include <Einsums/BufferAllocator/Options.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Logging.hpp>
#include <Einsums/Options.hpp>
#include <Einsums/Options/Options.hpp>
#include <Einsums/Runtime.hpp>
#include <Einsums/StringUtil/MemoryString.hpp>

EINSUMS_NAMESPACE_BEGIN()

/*
 * Set up the internal state of the module. If the module does not need to be set up, then this
 * file can be safely deleted. Make sure that if you do, you also remove its reference in the CMakeLists.txt,
 * as well as the initialization header for the module and the dependence on Einsums_Runtime, assuming these
 * aren't being used otherwise.
 */

int init_Einsums_BufferAllocator() {
    // Auto-generated code. Do not touch if you are unsure of what you are doing.
    // Instead, modify the other functions below.
    static bool is_initialized = false;

    if (!is_initialized) {
        einsums::register_arguments(einsums::add_Einsums_BufferAllocator_arguments);
        einsums::register_pre_startup_function(einsums::initialize_Einsums_BufferAllocator);
        einsums::register_shutdown_function(einsums::finalize_Einsums_BufferAllocator);
        is_initialized = true;
    }

    return 0;
}

EINSUMS_EXPORT int register_Einsums_BufferAllocator_options() {
    cl::register_option(option::BufferSize);
    cl::register_option(option::WorkBufferSize);
    cl::register_option(option::MaxMemory);
    return 0;
}

EINSUMS_EXPORT void add_Einsums_BufferAllocator_arguments() {
    // Registration itself happens at load time; what is left here is the part
    // that needs the module's singleton, which does not exist that early.
    cl::on_change(option::BufferSize, &detail::Einsums_BufferAllocator_vars::update_max_size);
    cl::on_change(option::WorkBufferSize, &detail::Einsums_BufferAllocator_vars::update_max_size);

    auto &singleton = detail::Einsums_BufferAllocator_vars::get_singleton();
    auto  lock      = std::lock_guard(singleton);
    singleton.set_max_size(string_util::memory_string("4MB"));
}

void initialize_Einsums_BufferAllocator() {
    // Create the singleton instance early.
    auto &singleton = detail::Einsums_BufferAllocator_vars::get_singleton();
}

void finalize_Einsums_BufferAllocator() {
}

EINSUMS_NAMESPACE_END()