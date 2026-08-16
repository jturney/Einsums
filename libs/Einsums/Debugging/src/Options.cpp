//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Debugging/Options.hpp>
#include <Einsums/Options/Declare.hpp>

EINSUMS_NAMESPACE_BEGIN()

int register_Einsums_Debugging_options() {
    cl::register_option(option::CrashHandler);
    cl::register_option(option::CrashDumpDir);
    return 0;
}

EINSUMS_NAMESPACE_END()
