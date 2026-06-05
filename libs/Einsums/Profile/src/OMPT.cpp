//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/Logging.hpp>
#include <Einsums/Print.hpp>
#include <Einsums/StringUtil/FromString.hpp>

#include <cstdlib>

// The CMake file should prevent this entire file from being compiled.
// We do this check just to make sure.
#if defined(EINSUMS_HAVE_OMP_TOOLS_H)
#    include <omp-tools.h>
#endif

#if defined(EINSUMS_HAVE_PROFILER) && defined(EINSUMS_HAVE_OMP_TOOLS_H)
#    include <Einsums/Profile.hpp>

#    include <print>
#endif

namespace einsums {

#if defined(EINSUMS_HAVE_PROFILER) && defined(EINSUMS_HAVE_OMP_TOOLS_H)

namespace {

void on_parallel_begin(ompt_data_t * /*encountering_task_data*/, ompt_frame_t const * /*encountering_task_frame*/,
                       ompt_data_t * /*parallel_data*/, unsigned int /*requested_parallelism*/, int /*flags*/,
                       void const * /*codeptr_ra*/) {
    profile::Profiler::instance().push("omp_parallel");
}

void on_parallel_end(ompt_data_t * /*parallel_data*/, ompt_data_t * /*encountering_task_data*/, int /*flags*/,
                     void const * /*codeptr_ra*/) {
    profile::Profiler::instance().pop();
}

void on_implicit_task(ompt_scope_endpoint_t endpoint, ompt_data_t * /*parallel_data*/, ompt_data_t * /*task_data*/,
                      unsigned int /*actual_parallelism*/, unsigned int /*index*/, int /*flags*/) {
    if (endpoint == ompt_scope_begin) {
        profile::Profiler::instance().push("omp_implicit_task");
    } else if (endpoint == ompt_scope_end) {
        profile::Profiler::instance().pop();
    }
}

} // anonymous namespace

#endif // EINSUMS_HAVE_PROFILER && EINSUMS_HAVE_OMP_TOOLS_H

namespace {

int ompt_initialize(ompt_function_lookup_t lookup, int /*initial_device_num*/, ompt_data_t * /*tool_data*/) {
#if defined(EINSUMS_HAVE_PROFILER) && defined(EINSUMS_HAVE_OMP_TOOLS_H)
    auto set_callback = reinterpret_cast<ompt_set_callback_t>(lookup("ompt_set_callback"));
    if (set_callback) {
        set_callback(ompt_callback_parallel_begin, reinterpret_cast<ompt_callback_t>(on_parallel_begin));
        set_callback(ompt_callback_parallel_end, reinterpret_cast<ompt_callback_t>(on_parallel_end));
        set_callback(ompt_callback_implicit_task, reinterpret_cast<ompt_callback_t>(on_implicit_task));
    }
    EINSUMS_LOG_INFO("OMPT profiler callbacks registered.");
#else
    EINSUMS_LOG_INFO("OMPT initialized (profiler callbacks not available).");
#endif
    return 1;
}

void ompt_finalize(ompt_data_t * /* tool_data */) {
    EINSUMS_LOG_INFO("OpenMP runtime is shutting down...\n");
}

} // anonymous namespace

extern "C" {
ompt_start_tool_result_t *ompt_start_tool(unsigned int omp_version, char const *runtime_version) {
    char const *optstr   = std::getenv("EINSUMS_USE_OMPT");
    bool const  use_ompt = optstr != nullptr ? from_string<bool>(optstr, false) : false;

    // Einsums println function uses an OpenMP function to check if it's running in a parallel
    // section. Unfortunately, within this function OpenMP is still initializing and that function
    // may hang.
    if (use_ompt)
        std::println(stdout, "ompt_start_tool: running on omp_version {}, runtime_version {}", omp_version, runtime_version);

    static ompt_start_tool_result_t result;
    result.initialize      = &ompt_initialize;
    result.finalize        = &ompt_finalize;
    result.tool_data.value = 0L;
    result.tool_data.ptr   = nullptr;

    return use_ompt ? &result : nullptr;
}
}

} // namespace einsums