//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Logging.hpp>
#include <Einsums/PackedGemm/InitModule.hpp>
#include <Einsums/Runtime.hpp>

namespace einsums {

void initialize_Einsums_PackedGemm() {
    EINSUMS_LOG_INFO("PackedGemm: initializing module");
}

void finalize_Einsums_PackedGemm() {
    EINSUMS_LOG_INFO("PackedGemm: finalizing module");
}

int setup_Einsums_PackedGemm() {
    static bool is_initialized = false;
    if (!is_initialized) {
        einsums::register_startup_function(initialize_Einsums_PackedGemm);
        einsums::register_shutdown_function(finalize_Einsums_PackedGemm);
        is_initialized = true;
    }
    return 0;
}

} // namespace einsums
