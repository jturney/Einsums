//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/CommandLine/Declare.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Runtime/Options.hpp>

EINSUMS_NAMESPACE_BEGIN()

int register_Einsums_Runtime_options() {
    cl::register_option(option::InstallSignalHandlers);
    cl::register_option(option::AttachDebugger);
    cl::register_option(option::DiagnosticsOnTerminate);
    return 0;
}

EINSUMS_NAMESPACE_END()
