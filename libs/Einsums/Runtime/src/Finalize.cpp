//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/Assert.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Errors/Error.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/Logging.hpp>
#include <Einsums/Profile.hpp>
#include <Einsums/Runtime/InitRuntime.hpp>
#include <Einsums/Runtime/Runtime.hpp>

#include <H5public.h>
#include <cstdlib>

EINSUMS_NAMESPACE_BEGIN()

namespace detail {

namespace {
std::list<std::function<void()>> __deleters{};
}

void register_free_pointer(std::function<void()> f) {
    __deleters.push_back(std::move(f));
}

} // namespace detail

int finalize() {
    // The Runtime destructor now handles shutdown: running shutdown functions,
    // profiler shutdown, deinit_global_data. This function remains for backward
    // compatibility but is effectively a no-op; the destructor runs when the
    // Runtime unique_ptr goes out of scope in run_impl().
    //
    // If called explicitly before the destructor, it will run the cleanup early.
    auto *rt = runtime_ptr();
    if (!rt) {
        return EXIT_SUCCESS; // Already finalized (destructor ran).
    }

    rt->call_shutdown_functions(true);
    EINSUMS_LOG_INFO("ran pre-shutdown functions");
    rt->call_shutdown_functions(false);
    EINSUMS_LOG_INFO("ran shutdown functions");

#if defined(EINSUMS_HAVE_PROFILER)
    profile::Profiler::instance().shutdown();

    try {
        if (config::get(option::ProfileReport)) {
            std::ofstream out(config::get(option::ProfileFilename), config::get(option::ProfileAppend) ? std::ios::ate : std::ios::trunc);
            profile::Profiler::instance().print(config::get(option::ProfileDetailed), out);
        }
    } catch (...) {
        EINSUMS_LOG_INFO("Exception thrown by the profiler during shutdown. Ignoring.");
    }
#endif

    rt->deinit_global_data();

    // Free lost pointers.
    for (auto const &fn : detail::__deleters) {
        fn();
    }

    EINSUMS_LOG_INFO("einsums shutdown completed");

    return EXIT_SUCCESS;
}

EINSUMS_NAMESPACE_END()