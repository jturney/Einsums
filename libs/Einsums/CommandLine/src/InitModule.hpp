//--------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All Rights Reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//--------------------------------------------------------------------------------------------

#pragma once

#include <argparse/argparse.hpp>

/*
 * Exported definitions for initialization. If the module does not need to be initialized,
 * this header can be safely deleted. Just make sure to remove the reference in CMakeLists.txt,
 * as well as the initialization source file and the reference to Einsums_Runtime, if no other
 * symbols from this are being used.
 *
 * If this module does need to be initialized, make sure this header is referenced at least somewhere
 * in your public code. If it is not, then the initialization routines will not occur. Feel free
 * to change the namespaces of these functions. Just remember to also change the namespaces in the
 * corresponding code files.
 */

namespace einsums {

int setup_Einsums_CommandLine();

void add_Einsums_CommandLine_arguments(argparse::ArgumentParser &);
void initialize_Einsums_CommandLine();
void finalize_Einsums_CommandLine();

namespace detail {

static int initialize_module_Einsums_CommandLine = setup_Einsums_CommandLine();

}
}
