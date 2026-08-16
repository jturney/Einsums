//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/CommandLine/Declare.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/TensorBase/Options.hpp>

EINSUMS_NAMESPACE_BEGIN()

int register_Einsums_TensorBase_options() {
    cl::register_option(option::RowMajor);
    return 0;
}

EINSUMS_NAMESPACE_END()
