//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// A minimal stage module, compiled twice: once at the project's language level
// and once at C++23, so the two disagree about config_fingerprint().
//
// One build must be accepted and the other refused. Testing only the accepted
// build would prove nothing about the part of EINSUMS_STAGE_MODULE that is easy
// to get wrong. The macro is supposed to publish the fingerprints the HEADERS
// computed in this translation unit; if it instead read them back from the
// runtime world() record, every module would agree with every library by
// construction and the handshake would pass for the wrong reason. Only a build
// whose headers genuinely differ can tell those two implementations apart.
//
// __cplusplus is the discriminator for the same reason BuildModeProbe.cpp uses
// it: nothing can intercept it, and it is a genuine input to the fold rather
// than a hook added for the test.

#include <Einsums/Python/StageModule.hpp>

#include <pybind11/pybind11.h>

#if !defined(STAGE_PROBE_NAME)
#    error "STAGE_PROBE_NAME must name the module"
#endif

#define STAGE_PROBE_STR2(x) #x
#define STAGE_PROBE_STR(x)  STAGE_PROBE_STR2(x)

namespace {

/// A stage in name only. What is under test is the handshake, not the work, and
/// a stage that does arithmetic would only add ways for this test to fail for
/// reasons that are not about the handshake.
int stage_probe_add(int a, int b) {
    return a + b;
}

} // namespace

PYBIND11_MODULE(STAGE_PROBE_NAME, m) {
    EINSUMS_STAGE_MODULE(m, STAGE_PROBE_STR(STAGE_PROBE_NAME));

    m.def("stage_probe_add", &stage_probe_add, "Trivial stage entry point.");

    // The language level this module was ACTUALLY compiled at. Exported so the
    // test can separate "the macro published the wrong fingerprint" from "the
    // build system never varied the input", which otherwise have the same
    // symptom: two modules that agree when they should not.
    m.attr("__probe_cplusplus__") = static_cast<long>(__cplusplus);
}
