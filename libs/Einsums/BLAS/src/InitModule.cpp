//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/BLAS/InitModule.hpp>
#include <Einsums/BLASVendor/Vendor.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Runtime.hpp>

/*
 * Set up the internal state of the module. If the module does not need to be set up, then this
 * file can be safely deleted. Make sure that if you do, you also remove its reference in the CMakeLists.txt,
 * as well as the initialization header for the module and the dependence on Einsums_Runtime, assuming these
 * aren't being used otherwise.
 */
EINSUMS_NAMESPACE_BEGIN()
int setup_Einsums_BLAS() {
    // Auto-generated code. Do not touch if you are unsure of what you are doing.
    // Instead, modify the other functions below.
    static bool is_initialized = false;

    if (!is_initialized) {
        einsums::register_startup_function(einsums::initialize_Einsums_BLAS);
        einsums::register_shutdown_function(einsums::finalize_Einsums_BLAS);
        is_initialized = true;
    }

    return 0;
}

void initialize_Einsums_BLAS() {
    blas::vendor::initialize();
}

void finalize_Einsums_BLAS() {
    blas::vendor::finalize();
}
EINSUMS_NAMESPACE_END()