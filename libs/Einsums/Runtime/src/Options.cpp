//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Options/Declare.hpp>
#include <Einsums/Runtime/Options.hpp>

EINSUMS_NAMESPACE_BEGIN()

int register_Einsums_Runtime_options() {
    // The environment prefix is a fact about the library, not about one start-up, so it is
    // set where the options register: the registry can then name each option's variable
    // (EINSUMS_LOG_LEVEL) before initialize(), for `einsums options` and the Python layer.
    // RuntimeConfiguration::parse_command_line sets the same value again.
    cl::Registry::instance().set_env_prefix("EINSUMS");
    cl::register_option(option::InstallSignalHandlers);
    cl::register_option(option::AttachDebugger);
    cl::register_option(option::DiagnosticsOnTerminate);
    return 0;
}

EINSUMS_NAMESPACE_END()
