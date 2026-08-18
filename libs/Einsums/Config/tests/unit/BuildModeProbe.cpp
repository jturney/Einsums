//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// One source, compiled twice at different language levels, so the test can
// compare two config fingerprints that differ in exactly one folded input.
//
// Asserting anything about a single compiled value would only restate whatever
// the header happens to do; recompiling the same source under a different
// setting and requiring a different answer is a fact about the fingerprint.
//
// __cplusplus is the discriminator because nothing can intercept it. The first
// two attempts used standard-library debug macros: NDEBUG, which turned out not
// to be an ABI axis at all, and then _GLIBCXX_DEBUG, which reaches
// config_fingerprint() under gcc and clang but not under icx - the flag is
// passed and the fold does not see it. A core-language macro set by -std has no
// such failure mode, and it is a genuine fold input rather than a hook added
// for the test.

#include <Einsums/Config/ABI.hpp>

#include <cstdint>

#if !defined(BUILD_MODE_PROBE_NAME) || !defined(BUILD_MODE_PROBE_STD_NAME)
#    error "BUILD_MODE_PROBE_NAME and BUILD_MODE_PROBE_STD_NAME must name the exported probes"
#endif

/// The fingerprint as this translation unit computes it.
extern "C" std::uint64_t BUILD_MODE_PROBE_NAME() {
    // Forced through a constexpr VARIABLE, which the standard requires to be
    // initialized by constant evaluation right here. Returning the call
    // directly only works while the optimizer folds it: config_fingerprint()
    // is constexpr and therefore inline, so an unfolded call leaves a weak
    // definition in each probe object, the linker keeps exactly one, and both
    // probes then answer with the same __cplusplus - the two TUs having been
    // compiled at different levels notwithstanding. That is not hypothetical.
    // At -O2 clang folds it to an immediate; add -fsanitize=address,undefined
    // and it does not, and the suite fails claiming the fold ignores its input
    // when the fold never ran.
    constexpr std::uint64_t fingerprint = einsums::sealed::config_fingerprint();
    return fingerprint;
}

/// The language level this translation unit was ACTUALLY compiled at.
///
/// Exported separately so the test can tell two very different failures apart.
/// If the fingerprints match, the question is whether the fold ignored its
/// input or whether the build system never varied the input in the first
/// place - and those have the same symptom. This makes the second one say so.
extern "C" long BUILD_MODE_PROBE_STD_NAME() {
    return __cplusplus;
}
