//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// Test launcher for per-rung SIMD dispatch tests (einsums_add_simd_rung_tests).
//
// Usage: simd_rung_guard <rung> <command> [args...]
//
// If the host CPU cannot execute <rung>, exits with 77 - registered as the
// test's SKIP_RETURN_CODE - so ctest reports an honest "Skipped" instead of
// silently rerunning another rung (EINSUMS_SIMD_ARCH replaces an unsupported
// rung with a supported one). Otherwise replaces itself with <command>.

#include <Einsums/SIMD/RuntimeFeatures.hpp>

#include <cstdio>

#if defined(_WIN32)
#    include <process.h>
#else
#    include <unistd.h>
#endif

int main(int argc, char **argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: simd_rung_guard <rung> <command> [args...]\n");
        return 2;
    }

    auto const requested = einsums::simd::parse_instruction_set(argv[1]);
    if (!requested.has_value()) {
        std::fprintf(stderr, "simd_rung_guard: unknown rung '%s'\n", argv[1]);
        return 2;
    }

    auto const &features = einsums::simd::cpu_features();
    if (!einsums::simd::supports(features, *requested)) {
        std::fprintf(stderr, "simd_rung_guard: host cannot run %s (best supported: %s); skipping the test\n",
                     einsums::simd::to_string(*requested), einsums::simd::to_string(einsums::simd::highest_supported(features)));
        return 77; // SKIP_RETURN_CODE
    }

#if defined(_WIN32)
    auto const rc = _execv(argv[2], argv + 2);
#else
    auto const rc = execv(argv[2], argv + 2);
#endif
    std::fprintf(stderr, "simd_rung_guard: failed to exec '%s' (rc=%d)\n", argv[2], static_cast<int>(rc));
    return 2;
}
