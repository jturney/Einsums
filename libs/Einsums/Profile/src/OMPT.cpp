//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

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

namespace einsums {

namespace {

ompt_set_callback_t  ompt_set_callback{nullptr};
ompt_finalize_tool_t ompt_finalize_tool{nullptr};

} // namespace

int einsums_ompt_register(ompt_callbacks_t e, ompt_callback_t c, char const *name) {
    ompt_set_result_t rc = ompt_set_callback(e, c);
    switch (rc) {
    case ompt_set_error:
        println_warn("Failed to register OMPT callback {}!", name);
        break;
    case ompt_set_never:
        println_warn("OMPT callback {} never supported by this runtime.", name);
        break;
    case ompt_set_impossible:
        println_warn("OMPT callback {} impossible from this runtime", name);
        break;
    case ompt_set_sometimes:
        println_warn("OMPT callback {} sometimes supported by this runtime", name);
        break;
    case ompt_set_sometimes_paired:
        println_warn("OMPT callback {} sometimes paired by this runtime.", name);
        break;
    case ompt_set_always:
    default:
        break;
    }
    return 0;
}

void thread_begin(ompt_thread_t thread_type, ompt_data_t *thread_data) {
    switch (thread_type) {
    case ompt_thread_initial:
        println("OpenMP Initial Thread {}", thread_data->ptr);
        break;
    case ompt_thread_worker:
        println("OpenMP Worker Thread {}", thread_data->ptr);
        break;
    case ompt_thread_other:
        println("OpenMP Other Thread {}", thread_data->ptr);
        break;
    case ompt_thread_unknown:
        println("OpenMP Unknown Thread {}", thread_data->ptr);
        break;
    }
}

void thread_end(ompt_data_t *thread_data) {
    printf("OpenMP Thread End %p\n", thread_data->ptr);
}

void parallel_begin(ompt_data_t        *encountering_task_data,  /* data of encountering task           */
                    ompt_frame_t const *encountering_task_frame, /* frame data of encountering task     */
                    ompt_data_t        *parallel_data,           /* data of parallel region             */
                    unsigned int        requested_team_size,     /* requested number of threads in team */
                    int                 flags,                   /* flags */
                    void const         *codeptr_ra               /* return address of runtime call      */
) {
    EINSUMS_LOG_TRACE("OpenMP Parallel Region: Parallel Region Begin {}", requested_team_size);
}

void parallel_end(ompt_data_t *parallel_data,          /* data of parallel region             */
                  ompt_data_t *encountering_task_data, /* data of encountering task           */
                  int          flags,                  /* flags              */
                  void const  *codeptr_ra              /* return address of runtime call      */
) {
    EINSUMS_LOG_TRACE("OpenMP Parallel Region: Parallel Region End");
}

int ompt_initialize(ompt_function_lookup_t lookup, int initial_device_num, ompt_data_t *tool_data) {
    // It appears that the Einsums printing routines are available at this point. Logging is not
    // available as the Einsums runtime isn't initialized yet.
    // println("Initializing OMPT");
    ompt_finalize_tool = reinterpret_cast<ompt_finalize_tool_t>(lookup("ompt_finalize_tool"));
    ompt_set_callback  = reinterpret_cast<ompt_set_callback_t>(lookup("ompt_set_callback"));

    // Register mandatory events
    einsums_ompt_register(ompt_callback_thread_begin, reinterpret_cast<ompt_callback_t>(thread_begin), "thread_begin");
    einsums_ompt_register(ompt_callback_thread_end, reinterpret_cast<ompt_callback_t>(thread_end), "thread_end");
    einsums_ompt_register(ompt_callback_parallel_begin, reinterpret_cast<ompt_callback_t>(parallel_begin), "parallel_begin");
    einsums_ompt_register(ompt_callback_parallel_end, reinterpret_cast<ompt_callback_t>(parallel_end), "parallel_end");

    return 1;
}

void ompt_finalize(ompt_data_t * /* tool_data */) {
    // The Einsums runtime could have already been shutdown at this point.
    // fprintf(stdout, "OpenMP runtime is shutting down...\n");
}

extern "C" {
ompt_start_tool_result_t *ompt_start_tool(unsigned int omp_version, char const *runtime_version) {
    char const *optstr = std::getenv("EINSUMS_USE_OMPT");
    // if (optstr) {
    // fprintf(stdout, "EINSUMS_USE_OMPT: %s\n", optstr);
    // }
    bool use_ompt = optstr != nullptr ? from_string<bool>(optstr, false) : false;

    // Einsums println function uses an OpenMP function to check if it's running in a parallel
    // section. Unfortunately, within this function OpenMP is still initializing and that function
    // may hang.
    if (use_ompt)
        fprintf(stdout, "ompt_start_tool: running on omp_version %d, runtime_version %s\n", omp_version, runtime_version);

    static ompt_start_tool_result_t result;
    result.initialize      = &ompt_initialize;
    result.finalize        = &ompt_finalize;
    result.tool_data.value = 0L;
    result.tool_data.ptr   = nullptr;

    return use_ompt ? &result : nullptr;
}
}

} // namespace einsums