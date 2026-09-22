//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Options/Declare.hpp>
#include <Einsums/Profile/Options.hpp>

EINSUMS_NAMESPACE_BEGIN()

bool profile_recording_disabled() {
    return config::get(option::ProfileDisable);
}

bool profile_server_enabled() {
    return config::get(option::ProfileServer);
}

std::int64_t profile_server_port() {
    return config::get(option::ProfilePort);
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
    return 0;
}

EINSUMS_NAMESPACE_END()
