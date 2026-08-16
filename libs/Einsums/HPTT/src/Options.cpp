//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config/Namespace.hpp>
#include <Einsums/HPTT/Options.hpp>
#include <Einsums/Options/Declare.hpp>

EINSUMS_NAMESPACE_BEGIN()

int register_Einsums_HPTT_options() {
    cl::register_option(option::HpttSelectionMethod);
    return 0;
}

EINSUMS_NAMESPACE_END()
