//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// Enumerations format as their names through format_as, found by argument-dependent lookup,
// so a message can pass the value itself. These pin that for enums from three places - one
// declared in ComputeGraphTypes, one from PackedGemm whose name ComputeGraph owns, and the
// runtime's own - and that the format spec reaches the name.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/Runtime/Runtime.hpp>

#include <fmt/format.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string_view>

#include <Einsums/Testing.hpp>

using namespace einsums;
namespace cg = einsums::compute_graph;

TEST_CASE("EnumFormatting - an enum formats as its name", "[ComputeGraph][EnumFormatting]") {
    CHECK(fmt::format("{}", cg::OpKind::Einsum) == cg::op_kind_name(cg::OpKind::Einsum));
    CHECK(fmt::format("{}", cg::PassPhase::Tuning) == "tuning");
    CHECK(fmt::format("{}", packed_gemm::ScalarType::Float64) == cg::scalar_type_name(packed_gemm::ScalarType::Float64));
    CHECK(fmt::format("{}", RuntimeState::Running) == "Running");
}

TEST_CASE("EnumFormatting - the format spec applies to the name", "[ComputeGraph][EnumFormatting]") {
    CHECK(fmt::format("[{:>8}]", cg::PassPhase::Tuning) == "[  tuning]");
    CHECK(fmt::format("[{:<9}]", RuntimeState::Running) == "[Running  ]");
}

TEST_CASE("EnumFormatting - a value outside the enumeration names as the fallback", "[ComputeGraph][EnumFormatting]") {
    CHECK(fmt::format("{}", static_cast<RuntimeState>(250)) == "Unknown");
}

TEST_CASE("EnumFormatting - every op kind's name parses back to that kind", "[ComputeGraph][EnumFormatting]") {
    // The parse once stopped at OpKind::Custom, so the tile kinds declared after it named
    // themselves in a file and could not be read back. Every value of the underlying type is
    // tried, so a kind added anywhere in the enumeration is covered without editing this case.
    std::size_t named = 0;
    for (unsigned value = 0; value <= std::numeric_limits<std::uint8_t>::max(); ++value) {
        auto const             kind = static_cast<cg::OpKind>(value);
        std::string_view const name = cg::op_kind_name(kind);
        if (name == "Unknown") {
            continue;
        }
        ++named;
        INFO("kind " << name);
        auto const parsed = cg::op_kind_from_name(name);
        REQUIRE(parsed.has_value());
        CHECK(*parsed == kind);
    }
    CHECK(cg::op_kind_from_name("TileElementwise") == cg::OpKind::TileElementwise);
    CHECK(named > static_cast<std::size_t>(cg::OpKind::Custom));
    CHECK_FALSE(cg::op_kind_from_name("NoSuchKind").has_value());
}
