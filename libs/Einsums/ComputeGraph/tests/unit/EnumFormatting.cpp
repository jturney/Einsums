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
