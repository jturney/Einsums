//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file PlatformInfo.cpp
/// @brief What this build compiled for, and what the running CPU can execute.
///
/// Two questions with different answers. The inline Vec<T> operations are fixed when a translation
/// unit is compiled: this file's Vec<float> is as wide as its compiler flags allow, whatever CPU it
/// later runs on. Runtime-dispatched kernels, compiled once per instruction-set rung (see
/// RuntimeDispatch.cpp), pick a rung when the program runs, from what the CPU and operating system
/// support and from --einsums:simd:arch, which can only lower it.

#include <Einsums/Runtime.hpp>
#include <Einsums/SIMD/Platform.hpp>
#include <Einsums/SIMD/RuntimeFeatures.hpp>
#include <Einsums/SIMD/Vec.hpp>

#include <iostream>

using namespace einsums::simd;

int einsums_main() {
    // Compile time: the width of every inline operation in this file.
    std::cout << "This file was compiled for " << native_bits << "-bit vectors: " << lanes<float> << " floats or "
              << lanes<double> << " doubles per Vec.\n";
    // aarch64 always has fused multiply-add; x86 has it from FMA3.
    std::cout << "  fused multiply-add: " << (has_fma || has_neon ? "yes" : "no") << ", NEON: " << has_neon << ", AVX2: " << has_avx2
              << ", AVX-512: " << has_avx512 << "\n\n";

    // Run time: what this CPU can execute, rung by rung, best first.
    CpuFeatures const &cpu = cpu_features();
    std::cout << "Rungs of this architecture, best first:\n";
    for (InstructionSet const set : preference_order(cpu.arch)) {
        std::cout << "  " << to_string(set) << " (" << vector_bits(set) << "-bit): " << (supports(cpu, set) ? "supported" : "not supported")
                  << "\n";
    }
    std::cout << "\nHighest rung this CPU supports: " << to_string(highest_supported(cpu)) << "\n";
    std::cout << "Rung the dispatched kernels use:  " << to_string(selected_arch()) << "\n";
    std::cout << "(Run with --einsums:simd:arch=baseline to see the second line drop.)\n";

    // The selected rung can never exceed what the CPU supports.
    return supports(cpu, selected_arch()) ? 0 : 1;
}

int main(int argc, char **argv) {
    return einsums::start(einsums_main, argc, argv);
}
