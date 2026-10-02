//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Options/Declare.hpp>
#include <Einsums/Runtime/Options.hpp>

EINSUMS_NAMESPACE_BEGIN()

int register_Einsums_Runtime_options() {
    // Set at registration so each option's variable is known before initialize().
    cl::Registry::instance().set_env_prefix("EINSUMS");
    cl::register_option(option::InstallSignalHandlers);
    cl::register_option(option::AttachDebugger);
    cl::register_option(option::DiagnosticsOnTerminate);
    return 0;
}

EINSUMS_NAMESPACE_END()
