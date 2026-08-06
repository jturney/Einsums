//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// Stands in for a compiled stage module: it does nothing but call into the
// miniworld it was linked against. Built once per copy, from this one source,
// so any difference in what it reports comes from the linker rather than from
// the code.

#include "MiniWorld.hpp"

#if defined(_WIN32)
#    define CONSUMER_EXPORT __declspec(dllexport)
#else
#    define CONSUMER_EXPORT __attribute__((visibility("default")))
#endif

extern "C" {

CONSUMER_EXPORT int consumer_value(void) {
    return miniworld_value();
}

CONSUMER_EXPORT void const *consumer_identity(void) {
    return miniworld_identity();
}
}
