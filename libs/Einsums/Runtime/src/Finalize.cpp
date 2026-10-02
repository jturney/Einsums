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
#include <fstream>

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
    // ~Runtime does the shutdown; calling this first just runs it early.
    auto *rt = runtime_ptr();
    if (!rt) {
        return EXIT_SUCCESS; // Already finalized (destructor ran).
    }

    rt->call_shutdown_functions(true);
    EINSUMS_LOG_INFO("ran pre-shutdown functions");
    rt->call_shutdown_functions(false);
    EINSUMS_LOG_INFO("ran shutdown functions");

    detail::shutdown_profiler_and_report();

    rt->deinit_global_data();

    // Free lost pointers.
    for (auto const &fn : detail::__deleters) {
        fn();
    }

    EINSUMS_LOG_INFO("einsums shutdown completed");

    return EXIT_SUCCESS;
}

EINSUMS_NAMESPACE_END()