//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// cpu_info()'s SIMD width must describe the rung the process dispatches to.
//
// It once described the flags the library was compiled with, so a build for
// the x86-64 baseline reported two doubles per vector on every AVX2 machine
// and PackedGemm, which sizes its register tile from the field, ran SSE-width
// kernels there. Registered per rung through the SIMD rung tests, so the
// STRIPES_ARCH those registrations set has to move the width along with the
// kernels.

#include <Einsums/Hardware/CpuInfo.hpp>

#include <Stripes/Platform.hpp>
#include <Stripes/RuntimeFeatures.hpp>

#include <Einsums/Testing.hpp>

using namespace einsums;

TEST_CASE("Hardware - simd_width_f64 follows the selected rung", "[Hardware][SIMD]") {
    auto const &hw   = hardware::cpu_info();
    auto const  rung = stripes::selected_arch();

    REQUIRE(hw.simd_width_f64 == stripes::vector_bits(rung) / 64);
    REQUIRE(hw.simd_width_f32 == 2 * hw.simd_width_f64);

    // The rung ladder never selects a rung the CPU cannot run, so the runtime
    // width is at least the width the library itself was compiled at.
    REQUIRE(hw.simd_width_f64 >= hw.compiled_simd_width_f64);
    REQUIRE(hw.compiled_simd_width_f64 == stripes::native_lanes<double>);
}

TEST_CASE("Hardware - rung widths are the register widths", "[Hardware][SIMD]") {
    REQUIRE(stripes::vector_bits(stripes::InstructionSet::Baseline) == 128);
    REQUIRE(stripes::vector_bits(stripes::InstructionSet::V2) == 128);
    REQUIRE(stripes::vector_bits(stripes::InstructionSet::V3) == 256);
    REQUIRE(stripes::vector_bits(stripes::InstructionSet::V4) == 512);
    REQUIRE(stripes::vector_bits(stripes::InstructionSet::Sme) == 128);
}
