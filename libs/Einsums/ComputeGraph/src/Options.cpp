//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/CommandLine/Declare.hpp>
#include <Einsums/ComputeGraph/Options.hpp>
#include <Einsums/Config/Namespace.hpp>

EINSUMS_NAMESPACE_BEGIN()

int register_Einsums_ComputeGraph_options() {
    cl::register_option(option::PassDisable);
    cl::register_option(option::PassAnalyze);
    cl::register_option(option::PassVerbose);
    cl::register_option(option::GraphProfileGroups);
    cl::register_option(option::GraphVerifyLevels);
    cl::register_option(option::GpuDisable);
    return 0;
}

EINSUMS_NAMESPACE_END()
