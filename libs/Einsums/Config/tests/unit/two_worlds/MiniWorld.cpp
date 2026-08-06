//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include "MiniWorld.hpp"

// Compiled once per copy, with a different MINIWORLD_VALUE each time. The
// default is deliberately a value no test expects, so a build that forgets to
// define it fails loudly rather than accidentally matching a real copy.
#if !defined(MINIWORLD_VALUE)
#    define MINIWORLD_VALUE (-1)
#endif

extern "C" {

int miniworld_value(void) {
    return MINIWORLD_VALUE;
}

void const *miniworld_identity(void) {
    // One per loaded copy. Never read, only compared.
    static char const marker = 0;
    return &marker;
}
}
