//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// One source, compiled twice with opposite NDEBUG settings, so the test can
// compare two config fingerprints that differ in exactly one way.
//
// This is the only honest way to check that the build mode reaches the
// fingerprint. Asserting anything about a single compiled value would only
// restate whatever the header happens to do; recompiling the same source under
// a different mode and requiring a different answer is a fact about the
// fingerprint rather than a restatement of it.

#include <Einsums/Config/ABI.hpp>

#include <cstdint>

#if !defined(BUILD_MODE_PROBE_NAME)
#    error "BUILD_MODE_PROBE_NAME must name the exported probe"
#endif

extern "C" std::uint64_t BUILD_MODE_PROBE_NAME() {
    return einsums::sealed::config_fingerprint();
}
