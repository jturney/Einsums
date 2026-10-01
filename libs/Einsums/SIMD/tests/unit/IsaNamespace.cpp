//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// Two translation units compiled for different instruction sets must not share the SIMD headers'
// entities. This file is built at the project's baseline and IsaNamespaceWide.cpp at the v3
// rung's flags, as a per-rung kernel and the code that dispatches to it are. Before the headers
// declared everything inside an instruction-set inline namespace, both units named the same
// einsums::simd::native_lanes<float>; the linker kept one definition, and the unit whose lane
// count it did not match read the other's. Any non-inlined entity (Vec's operator[] at -O0, a
// helper MSVC declined to __forceinline) was exposed the same way.

#include <Einsums/SIMD/RuntimeFeatures.hpp>

#include <string_view>

#include "IsaNamespaceProbe.hpp"

#include <catch2/catch_all.hpp>

TEST_CASE("translation units built for different instruction sets keep separate SIMD entities", "[simd][isa-namespace]") {
    using namespace einsums::simd;
    if (!supports(cpu_features(), InstructionSet::V3)) {
        SKIP("the wide translation unit needs the v3 rung");
    }
    IsaProbe const base = isa_probe_here();
    IsaProbe const wide = isa_probe_wide();
    INFO("baseline tag " << base.tag << ", v3 tag " << wide.tag);

    REQUIRE(base.compiled_lanes != wide.compiled_lanes);
    CHECK(std::string_view(base.tag) != std::string_view(wide.tag));
    CHECK(base.linked_object != wide.linked_object);
    CHECK(base.linked_lanes == base.compiled_lanes);
    CHECK(wide.linked_lanes == wide.compiled_lanes);
}
