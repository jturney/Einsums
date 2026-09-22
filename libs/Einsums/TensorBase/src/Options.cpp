//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Options/Declare.hpp>
#include <Einsums/TensorBase/Options.hpp>

EINSUMS_NAMESPACE_BEGIN()

bool default_row_major() {
    return config::get(option::RowMajor);
}

int register_Einsums_TensorBase_options() {
    cl::register_option(option::RowMajor);
    return 0;
}

EINSUMS_NAMESPACE_END()
