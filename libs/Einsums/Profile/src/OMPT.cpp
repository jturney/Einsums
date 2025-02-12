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

int ompt_initialize(ompt_function_lookup_t lookup, int initial_device_num, ompt_data_t *tool_data) {
    // It appears that the Einsums runtime is up and running at this point. But refrain from
    // using Einsums printing routines just in case.
    // println("Initializing OMPT");
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