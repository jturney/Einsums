//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Runtime/Runtime.hpp>
#include <Einsums/Runtime/ShutdownFunction.hpp>
#include <Einsums/Runtime/StartupFunction.hpp>
#include <Einsums/Utilities/InitModule.hpp>
#include <Einsums/Utilities/Random.hpp>

#include <chrono>
#include <random>

/*
 * Set up the internal state of the module. If the module does not need to be set up, then this
 * file can be safely deleted. Make sure that if you do, you also remove its reference in the CMakeLists.txt,
 * as well as the initialization header for the module and the dependence on Einsums_Runtime, assuming these
 * aren't being used otherwise.
 */

int einsums::init_Einsums_Utilities() {
    // Auto-generated code. Do not touch if you are unsure of what you are doing.
    // Instead, modify the other functions below.
    static bool is_initialized = false;

    if (!is_initialized) {
        einsums::register_pre_startup_function(einsums::initialize_Einsums_Utilities);
        is_initialized = true;
    }
    return 0;
}

void einsums::initialize_Einsums_Utilities() {
    einsums::random_engine = std::default_random_engine(std::chrono::system_clock::now().time_since_epoch().count());
}