//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Options/Declare.hpp>
#include <Einsums/Profile/Options.hpp>

#if defined(EINSUMS_HAVE_PROFILER)
#    include <Einsums/Config/Version.hpp>
#    include <Einsums/Profile/Profile.hpp>

#    include <fmt/format.h>
#endif

EINSUMS_NAMESPACE_BEGIN()

void configure_profiler_from_options() {
#if defined(EINSUMS_HAVE_PROFILER)
    waggle::SettingsUpdate update;
    if (auto const disable = config::try_get(option::ProfileDisable)) {
        update.record = !*disable;
    }
    update.report                = config::try_get(option::ProfileReport);
    update.report_file           = config::try_get(option::ProfileFilename);
    update.report_append         = config::try_get(option::ProfileAppend);
    update.report_detailed       = config::try_get(option::ProfileDetailed);
    update.save                  = config::try_get(option::ProfileSave);
    update.server                = config::try_get(option::ProfileServer);
    update.port                  = config::try_get(option::ProfilePort);
    update.wait_for_viewer       = config::try_get(option::ProfileWaitForViewer);
    update.max_distinct_children = config::try_get(option::ProfileMaxDistinctChildren);

    waggle::configure(update);
    // EINSUMS_BUILD_TYPE is a bare identifier (release, debug), so stringify it.
#    define EINSUMS_PROFILE_STRINGIFY_(x) #x
#    define EINSUMS_PROFILE_STRINGIFY(x)  EINSUMS_PROFILE_STRINGIFY_(x)
    waggle::init(
        {.name       = "einsums",
         .version    = fmt::format("{}.{}.{}{}", EINSUMS_VERSION_MAJOR, EINSUMS_VERSION_MINOR, EINSUMS_VERSION_PATCH, EINSUMS_VERSION_TAG),
         .git_commit = std::string(git_commit()),
         .git_branch = std::string(git_branch()),
         .git_dirty  = git_dirty(),
         .build_type = EINSUMS_PROFILE_STRINGIFY(EINSUMS_BUILD_TYPE)});
#    undef EINSUMS_PROFILE_STRINGIFY
#    undef EINSUMS_PROFILE_STRINGIFY_
#endif
}

int register_Einsums_Profile_options() {
    cl::register_option(option::ProfileDisable);
    cl::register_option(option::ProfileReport);
    cl::register_option(option::ProfileFilename);
    cl::register_option(option::ProfileAppend);
    cl::register_option(option::ProfileDetailed);
    cl::register_option(option::ProfileSave);
    cl::register_option(option::ProfileServer);
    cl::register_option(option::ProfilePort);
    cl::register_option(option::ProfileWaitForViewer);
    cl::register_option(option::ProfileMaxDistinctChildren);
    return 0;
}

EINSUMS_NAMESPACE_END()
