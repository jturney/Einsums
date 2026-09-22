//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Options/Declare.hpp>
#include <Einsums/PackedGemm/Options.hpp>

EINSUMS_NAMESPACE_BEGIN()

int register_Einsums_PackedGemm_options() {
    cl::register_option(option::PackedGemmFlattenBudget);
    return 0;
}

EINSUMS_NAMESPACE_END()
