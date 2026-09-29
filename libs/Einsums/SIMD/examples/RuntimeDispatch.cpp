//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file RuntimeDispatch.cpp
/// @brief One kernel, compiled for every rung, chosen when the program runs.
///
/// A binary built for the x86-64 baseline runs everywhere but uses 128-bit SSE2 even on a CPU with
/// AVX-512. A binary built with -march=native uses the full width but crashes on an older CPU. The
/// dispatch ladder gets both: RuntimeDispatchKernel.cpp is compiled once per rung into its own
/// namespace, and this file picks the best copy the running CPU supports.
///
/// Three pieces make it work:
///
///   - EINSUMS_SIMD_FOR_EACH_BUILT_RUNG(X) calls X(arch_<rung>) for every rung this build compiled,
///     which is how the declarations below name every copy without knowing which exist.
///   - EINSUMS_SIMD_LADDER(fn) expands to the five slots select() takes, one per rung, with nullptr
///     for a rung that was not built.
///   - select<Fn>(...) returns the slot of the rung selected_arch() chose: the best the CPU and the
///     operating system support, lowered by --einsums:simd:arch if given.
///
/// Resolve the pointer once and keep it, here in a function-local static. The choice cannot
/// change while the program runs, and resolving it per call would put a lookup in the hot path.
///
/// Try it on an x86 machine with --einsums:simd:arch=baseline, v2, v3 or v4 and watch the width
/// change. On other architectures the ladder has one rung, arch_native.

#include <Einsums/Runtime.hpp>
#include <Einsums/SIMD/RungLadder.hpp>
#include <Einsums/SIMD/RuntimeFeatures.hpp>

#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

namespace simd_example {

// Declare every copy of the kernel that this build compiled.
#define SIMD_EXAMPLE_DECLARE_RUNG(ns)                                                                                                      \
    namespace ns {                                                                                                                         \
    float sum_of_squares(float const *x, std::size_t n);                                                                                   \
    int   float_lanes();                                                                                                                   \
    }
EINSUMS_SIMD_FOR_EACH_BUILT_RUNG(SIMD_EXAMPLE_DECLARE_RUNG)
#undef SIMD_EXAMPLE_DECLARE_RUNG

using SumOfSquaresFn = float (*)(float const *, std::size_t);
using FloatLanesFn   = int (*)();

/// The copy of sum_of_squares for the selected rung, resolved on first use.
float sum_of_squares(float const *x, std::size_t n) {
    static SumOfSquaresFn const kernel = einsums::simd::select<SumOfSquaresFn>(EINSUMS_SIMD_LADDER(sum_of_squares));
    return kernel(x, n);
}

int float_lanes() {
    static FloatLanesFn const kernel = einsums::simd::select<FloatLanesFn>(EINSUMS_SIMD_LADDER(float_lanes));
    return kernel();
}

} // namespace simd_example

int einsums_main() {
    using namespace einsums::simd;
    std::cout << "Selected rung: " << to_string(selected_arch()) << "; the kernel's Vec<float> holds " << simd_example::float_lanes()
              << " floats.\n";

    std::vector<float> x(1001);
    double             expect = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        x[i] = static_cast<float>(i % 13) * 0.5f;
        expect += static_cast<double>(x[i]) * x[i];
    }
    float const got = simd_example::sum_of_squares(x.data(), x.size());
    std::cout << "sum of squares: " << got << " (exact " << expect << ")\n";
    // Quarter-integer squares below 36 in float: every partial sum is exact to 2^24.
    return std::abs(static_cast<double>(got) - expect) < 1e-6 * expect ? 0 : 1;
}

int main(int argc, char **argv) {
    return einsums::start(einsums_main, argc, argv);
}
