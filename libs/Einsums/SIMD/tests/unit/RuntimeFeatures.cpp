//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/SIMD/Platform.hpp>
#include <Einsums/SIMD/RuntimeFeatures.hpp>

#include <catch2/catch_all.hpp>

using einsums::simd::CpuFeatures;
using einsums::simd::InstructionSet;

namespace {

// Synthetic feature sets for the pure ladder functions. Field-by-field
// construction mirrors what detect() produces on real hardware.
CpuFeatures features_v2() {
    CpuFeatures f;
    f.sse2       = true;
    f.sse3       = true;
    f.ssse3      = true;
    f.sse41      = true;
    f.sse42      = true;
    f.popcnt     = true;
    f.cmpxchg16b = true;
    f.lahf_sahf  = true;
    return f;
}

CpuFeatures features_v3() {
    CpuFeatures f = features_v2();
    f.osxsave     = true;
    f.os_avx      = true;
    f.avx         = true;
    f.avx2        = true;
    f.fma         = true;
    f.bmi1        = true;
    f.bmi2        = true;
    f.f16c        = true;
    f.lzcnt       = true;
    f.movbe       = true;
    return f;
}

CpuFeatures features_v4() {
    CpuFeatures f = features_v3();
    f.os_avx512   = true;
    f.avx512f     = true;
    f.avx512vl    = true;
    f.avx512bw    = true;
    f.avx512dq    = true;
    f.avx512cd    = true;
    return f;
}

} // namespace

TEST_CASE("cpu_features: sane on the host", "[simd][runtime-features]") {
    auto const &f = einsums::simd::cpu_features();

#if defined(__x86_64__) || defined(_M_X64)
    // SSE2 is part of the x86-64 baseline; every x86-64 CPU has it.
    REQUIRE(f.sse2);
    REQUIRE_FALSE(f.neon);
#elif defined(__aarch64__) || defined(_M_ARM64)
    REQUIRE(f.neon);
    REQUIRE_FALSE(f.sse2);
#endif

    // Feature-implication sanity: these hold on all real silicon.
    if (f.sse42) {
        CHECK(f.sse41);
        CHECK(f.ssse3);
        CHECK(f.sse2);
    }
    if (f.avx2) {
        CHECK(f.avx);
        CHECK(f.os_avx);
    }
    if (f.avx512f) {
        CHECK(f.os_avx512);
        CHECK(f.avx2);
    }

    // Detection is cached: same object every call.
    CHECK(&einsums::simd::cpu_features() == &f);
}

TEST_CASE("cpu_features: agrees with the compile-time baseline", "[simd][runtime-features]") {
    auto const &f = einsums::simd::cpu_features();

    // Whatever ISA this test was COMPILED for must be present at runtime,
    // or the binary could not be running. This cross-checks detect()
    // against Platform.hpp on every machine the test suite runs on.
    if constexpr (einsums::simd::has_sse42) {
        CHECK(f.sse42);
    }
    if constexpr (einsums::simd::has_avx2) {
        CHECK(f.avx2);
    }
    if constexpr (einsums::simd::has_avx512) {
        CHECK(f.avx512f);
        CHECK(f.avx512vl);
    }
    if constexpr (einsums::simd::has_neon) {
        CHECK(f.neon);
    }
}

TEST_CASE("highest_supported: full psABI gates", "[simd][runtime-features]") {
    CHECK(einsums::simd::highest_supported(CpuFeatures{}) == InstructionSet::Baseline);
    CHECK(einsums::simd::highest_supported(features_v2()) == InstructionSet::V2);
    CHECK(einsums::simd::highest_supported(features_v3()) == InstructionSet::V3);
    CHECK(einsums::simd::highest_supported(features_v4()) == InstructionSet::V4);

    SECTION("one missing extension drops the whole level") {
        auto f = features_v3();
        f.bmi2 = false;
        CHECK(einsums::simd::highest_supported(f) == InstructionSet::V2);
    }

    SECTION("CPU support without OS state does not qualify") {
        auto f      = features_v4();
        f.os_avx512 = false;
        CHECK(einsums::simd::highest_supported(f) == InstructionSet::V3);

        auto g   = features_v3();
        g.os_avx = false;
        CHECK(einsums::simd::highest_supported(g) == InstructionSet::V2);
    }
}

TEST_CASE("parse_instruction_set / to_string", "[simd][runtime-features]") {
    using einsums::simd::parse_instruction_set;

    CHECK(parse_instruction_set("baseline") == InstructionSet::Baseline);
    CHECK(parse_instruction_set("v2") == InstructionSet::V2);
    CHECK(parse_instruction_set("v3") == InstructionSet::V3);
    CHECK(parse_instruction_set("v4") == InstructionSet::V4);
    CHECK(parse_instruction_set("x86-64-v3") == InstructionSet::V3);
    CHECK(parse_instruction_set("AVX2") == InstructionSet::V3);
    CHECK(parse_instruction_set("avx512") == InstructionSet::V4);
    CHECK(parse_instruction_set("sse4.2") == InstructionSet::V2);
    CHECK(parse_instruction_set("sse2") == InstructionSet::Baseline);
    CHECK_FALSE(parse_instruction_set("pentium3").has_value());
    CHECK_FALSE(parse_instruction_set("").has_value());

    // Round trip through the canonical names.
    for (auto set : {InstructionSet::Baseline, InstructionSet::V2, InstructionSet::V3, InstructionSet::V4}) {
        CHECK(parse_instruction_set(einsums::simd::to_string(set)) == set);
    }
}

TEST_CASE("resolve_arch: override semantics", "[simd][runtime-features]") {
    using einsums::simd::resolve_arch;

    auto const v4 = features_v4();

    SECTION("no override selects the hardware ceiling") {
        CHECK(resolve_arch(v4, std::nullopt) == InstructionSet::V4);
        CHECK(resolve_arch(features_v2(), std::nullopt) == InstructionSet::V2);
    }

    SECTION("override can lower the rung") {
        CHECK(resolve_arch(v4, "baseline") == InstructionSet::Baseline);
        CHECK(resolve_arch(v4, "v2") == InstructionSet::V2);
        CHECK(resolve_arch(v4, "avx2") == InstructionSet::V3);
    }

    SECTION("override above the ceiling clamps") {
        CHECK(resolve_arch(features_v2(), "v4") == InstructionSet::V2);
        CHECK(resolve_arch(CpuFeatures{}, "avx2") == InstructionSet::Baseline);
    }

    SECTION("unparseable override is ignored") {
        CHECK(resolve_arch(v4, "fastest-please") == InstructionSet::V4);
    }
}

TEST_CASE("selected_arch: cached and within the host ceiling", "[simd][runtime-features]") {
    auto const first = einsums::simd::selected_arch();
    CHECK(first <= einsums::simd::highest_supported(einsums::simd::cpu_features()));
    CHECK(einsums::simd::selected_arch() == first);
}

TEST_CASE("select: falls down the ladder through missing rungs", "[simd][runtime-features]") {
    using Fn = int (*)();

    static constexpr auto ret_baseline = +[]() { return 0; };
    static constexpr auto ret_v2       = +[]() { return 2; };
    static constexpr auto ret_v3       = +[]() { return 3; };
    static constexpr auto ret_v4       = +[]() { return 4; };

    int const rung = static_cast<int>(einsums::simd::selected_arch());

    SECTION("full ladder picks the selected rung exactly") {
        Fn const  fn       = einsums::simd::select<Fn>(ret_baseline, ret_v2, ret_v3, ret_v4);
        int const expected = rung == 0 ? 0 : rung + 1;
        CHECK(fn() == expected);
    }

    SECTION("missing rungs fall through to the next built one") {
        Fn const fn = einsums::simd::select<Fn>(ret_baseline, nullptr, nullptr, nullptr);
        CHECK(fn() == 0);

        Fn const  fn2      = einsums::simd::select<Fn>(ret_baseline, ret_v2, nullptr, nullptr);
        int const expected = rung >= 1 ? 2 : 0;
        CHECK(fn2() == expected);
    }

    SECTION("baseline only requires one argument") {
        Fn const fn = einsums::simd::select<Fn>(ret_baseline);
        CHECK(fn() == 0);
    }
}
