//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Hardware/Options.hpp>
#include <Einsums/Options/Declare.hpp>

EINSUMS_NAMESPACE_BEGIN()

int register_Einsums_Hardware_options() {
    cl::register_option(option::HardwareL1CacheSize);
    cl::register_option(option::HardwareL2CacheSize);
    cl::register_option(option::HardwareL3CacheSize);
    cl::register_option(option::CacheDir);
    cl::register_option(option::HardwareCalibration);
    cl::register_option(option::HardwareOmpRegionCostNs);
    cl::register_option(option::HardwareOmpMinParallelElements);
    cl::register_option(option::HardwareOmpMinParallelFlops);
    return 0;
}

EINSUMS_NAMESPACE_END()
