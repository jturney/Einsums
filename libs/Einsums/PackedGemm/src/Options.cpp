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
    cl::register_option(option::PackedGemmComplex1m);
    cl::register_option(option::PackedGemmCoresPerL3);
    cl::register_option(option::PackedGemmCTempBudget);
    cl::register_option(option::PackedGemmOuterProductFloor);
    cl::register_option(option::PackedGemmOuterGemmMax);
    cl::register_option(option::PackedGemmStreamMinElems);
    cl::register_option(option::PackedGemmStreamElemsPerThread);
    cl::register_option(option::PackedGemmDumpPlan);
    return 0;
}

EINSUMS_NAMESPACE_END()
