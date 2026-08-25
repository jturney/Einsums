//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file SpaceAnnotation.cpp
/// @brief Per-slot index-space annotations and the letter binding capture derives from them.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/ComputeGraphTypes/Spaces.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <array>
#include <cstddef>
#include <new>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums;
namespace cg = einsums::compute_graph;

namespace {

/// The three spaces every case below draws from, registered in a registry the case owns so no
/// two tests can see each other's declarations.
struct Spaces {
    cg::SpaceRegistry registry;
    cg::SpaceId       occ;
    cg::SpaceId       virt;
    cg::SpaceId       aux;

    Spaces() {
        occ  = registry.register_space(cg::IndexSpace{.name = "occ", .scale_symbol = "o", .typical_extent = 4.0});
        virt = registry.register_space(cg::IndexSpace{.name = "virt", .scale_symbol = "v", .typical_extent = 8.0});
        aux  = registry.register_space(cg::IndexSpace{.name = "aux", .scale_symbol = "x", .typical_extent = 16.0});
    }
};

/// The descriptor of the graph's single einsum node, or null when it has none.
cg::EinsumDescriptor const *einsum_descriptor(cg::Graph const &graph) {
    for (auto const &node : graph.nodes()) {
        if (auto const *desc = std::get_if<cg::EinsumDescriptor>(&node.op_data)) {
            return desc;
        }
    }
    return nullptr;
}

} // namespace

TEST_CASE("SpaceAnnotation - annotate and read back", "[ComputeGraph][Spaces]") {
    Spaces spaces;

    cg::Graph graph("annotate");
    graph.set_space_registry(spaces.registry);

    auto &T = graph.create_zero_tensor<double, 2>("T", 4, 8);

    graph.annotate_spaces(T, {spaces.occ, spaces.virt});

    auto const id = graph.find_tensor_id_by_ptr(&T);
    REQUIRE(id != 0);
    CHECK(graph.tensor_spaces(id) == std::vector<cg::SpaceId>{spaces.occ, spaces.virt});

    // Clearing is the empty vector, and it returns the tensor to unannotated.
    graph.annotate_spaces(id, {});
    CHECK(graph.tensor_spaces(id).empty());
}

TEST_CASE("SpaceAnnotation - rejects a bad annotation", "[ComputeGraph][Spaces]") {
    Spaces spaces;

    cg::Graph graph("annotate-errors");
    graph.set_space_registry(spaces.registry);

    auto &T = graph.create_zero_tensor<double, 2>("T", 4, 8);

    SECTION("count must match the rank") {
        CHECK_THROWS_AS(graph.annotate_spaces(T, {spaces.occ}), std::invalid_argument);
        CHECK_THROWS_AS(graph.annotate_spaces(T, {spaces.occ, spaces.virt, spaces.aux}), std::invalid_argument);
    }

    SECTION("a default-constructed id names nothing") {
        CHECK_THROWS_AS(graph.annotate_spaces(T, {spaces.occ, cg::SpaceId{}}), std::invalid_argument);
    }

    SECTION("an id from another registry does not resolve") {
        cg::SpaceRegistry other;
        for (int i = 0; i < 12; ++i) {
            other.register_space(cg::IndexSpace{.name = "space" + std::to_string(i), .scale_symbol = "s"});
        }
        auto const foreign = other.register_space(cg::IndexSpace{.name = "foreign", .scale_symbol = "f"});
        CHECK_THROWS_AS(graph.annotate_spaces(T, {spaces.occ, foreign}), std::invalid_argument);
    }

    SECTION("an unknown tensor id is out of range") {
        CHECK_THROWS_AS(graph.annotate_spaces(cg::TensorId{9999}, {spaces.occ, spaces.virt}), std::out_of_range);
        CHECK_THROWS_AS(graph.tensor_spaces(cg::TensorId{9999}), std::out_of_range);
    }
}

TEST_CASE("SpaceAnnotation - capture binds letters from annotated operands", "[ComputeGraph][Spaces]") {
    Spaces spaces;

    auto A = create_random_tensor<double>("A", 4, 8); // occ x virt
    auto B = create_random_tensor<double>("B", 8, 4); // virt x occ

    cg::Graph graph("bind");
    graph.set_space_registry(spaces.registry);
    auto &C = graph.create_zero_tensor<double, 2>("C", 4, 4);

    graph.annotate_spaces(A, {spaces.occ, spaces.virt});
    graph.annotate_spaces(B, {spaces.virt, spaces.occ});

    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ij <- ia ; aj", &C, A, B);
    }

    auto const *desc = einsum_descriptor(graph);
    REQUIRE(desc != nullptr);
    CHECK(desc->letter_spaces.size() == 3);
    CHECK(desc->space_for_letter("i") == std::optional<cg::SpaceId>{spaces.occ});
    CHECK(desc->space_for_letter("a") == std::optional<cg::SpaceId>{spaces.virt});
    CHECK(desc->space_for_letter("j") == std::optional<cg::SpaceId>{spaces.occ});
}

TEST_CASE("SpaceAnnotation - one letter over two spaces is a capture error", "[ComputeGraph][Spaces]") {
    Spaces spaces;

    auto A = create_random_tensor<double>("A", 4, 8);
    auto B = create_random_tensor<double>("B", 8, 4);

    cg::Graph graph("conflict");
    graph.set_space_registry(spaces.registry);
    auto &C = graph.create_zero_tensor<double, 2>("C", 4, 4);

    // 'a' is the link letter, and the two operands disagree about what it ranges over.
    graph.annotate_spaces(A, {spaces.occ, spaces.occ});
    graph.annotate_spaces(B, {spaces.virt, spaces.occ});

    cg::CaptureGuard const guard(graph);
    CHECK_THROWS_WITH(cg::einsum("ij <- ia ; aj", &C, A, B), Catch::Matchers::ContainsSubstring("'a'") &&
                                                                 Catch::Matchers::ContainsSubstring("occ") &&
                                                                 Catch::Matchers::ContainsSubstring("virt"));
}

TEST_CASE("SpaceAnnotation - a repeated letter must agree with itself", "[ComputeGraph][Spaces]") {
    Spaces spaces;

    auto A = create_random_tensor<double>("A", 4, 4); // diagonal read: both axes are occ
    auto B = create_random_tensor<double>("B", 4, 8);

    cg::Graph graph("diagonal");
    graph.set_space_registry(spaces.registry);
    auto &C = graph.create_zero_tensor<double, 1>("C", 8);

    SECTION("consistent annotation binds once") {
        graph.annotate_spaces(A, {spaces.occ, spaces.occ});
        graph.annotate_spaces(B, {spaces.occ, spaces.virt});

        {
            cg::CaptureGuard const guard(graph);
            cg::einsum("a <- ii ; ia", &C, A, B);
        }

        auto const *desc = einsum_descriptor(graph);
        REQUIRE(desc != nullptr);
        CHECK(desc->space_for_letter("i") == std::optional<cg::SpaceId>{spaces.occ});
        CHECK(desc->space_for_letter("a") == std::optional<cg::SpaceId>{spaces.virt});
    }

    SECTION("the two slots of the repeated letter disagreeing is an error") {
        graph.annotate_spaces(A, {spaces.occ, spaces.aux});
        graph.annotate_spaces(B, {spaces.occ, spaces.virt});

        cg::CaptureGuard const guard(graph);
        CHECK_THROWS_WITH(cg::einsum("a <- ii ; ia", &C, A, B),
                          Catch::Matchers::ContainsSubstring("'i'") && Catch::Matchers::ContainsSubstring("aux"));
    }
}

TEST_CASE("SpaceAnnotation - a half-annotated contraction maps only what it knows", "[ComputeGraph][Spaces]") {
    Spaces spaces;

    auto A = create_random_tensor<double>("A", 4, 8);
    auto B = create_random_tensor<double>("B", 8, 4);

    cg::Graph graph("half");
    graph.set_space_registry(spaces.registry);
    auto &C = graph.create_zero_tensor<double, 2>("C", 4, 4);

    graph.annotate_spaces(A, {spaces.occ, spaces.virt});

    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ij <- ia ; aj", &C, A, B);
    }

    auto const *desc = einsum_descriptor(graph);
    REQUIRE(desc != nullptr);
    CHECK(desc->space_for_letter("i") == std::optional<cg::SpaceId>{spaces.occ});
    CHECK(desc->space_for_letter("a") == std::optional<cg::SpaceId>{spaces.virt});
    // 'j' only ever met unannotated slots, so it gets no entry rather than a guess.
    CHECK(desc->space_for_letter("j") == std::nullopt);
    CHECK(desc->letter_spaces.size() == 2);

    // 'a' resolved from A, but nothing was written back to B: inheritance is a later pass's
    // decision, and capture spreading it would be exactly how one wrong annotation propagates.
    auto const b_id = graph.find_tensor_id_by_ptr(&B);
    REQUIRE(b_id != 0);
    CHECK(graph.tensor_spaces(b_id).empty());

    // C's output letters are not all bound, so its annotation stays empty too.
    auto const c_id = graph.find_tensor_id_by_ptr(&C);
    REQUIRE(c_id != 0);
    CHECK(graph.tensor_spaces(c_id).empty());
}

TEST_CASE("SpaceAnnotation - a graph-owned intermediate inherits its output letters", "[ComputeGraph][Spaces]") {
    Spaces spaces;

    auto A = create_random_tensor<double>("A", 4, 8);
    auto B = create_random_tensor<double>("B", 8, 16);

    cg::Graph graph("inherit");
    graph.set_space_registry(spaces.registry);
    auto &C = graph.create_zero_tensor<double, 2>("C", 4, 16);

    graph.annotate_spaces(A, {spaces.occ, spaces.virt});
    graph.annotate_spaces(B, {spaces.virt, spaces.aux});

    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ix <- ia ; ax", &C, A, B);
    }

    auto const c_id = graph.find_tensor_id_by_ptr(&C);
    REQUIRE(c_id != 0);
    CHECK(graph.tensor_spaces(c_id) == std::vector<cg::SpaceId>{spaces.occ, spaces.aux});

    // A caller-owned output is never written back to, whatever its letters resolve to.
    auto D = create_zero_tensor<double>("D", 4, 16);
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ix <- ia ; ax", &D, A, B);
    }
    auto const d_id = graph.find_tensor_id_by_ptr(&D);
    REQUIRE(d_id != 0);
    CHECK(graph.tensor_spaces(d_id).empty());
}

TEST_CASE("SpaceAnnotation - an unannotated program is unchanged", "[ComputeGraph][Spaces]") {
    auto A = create_random_tensor<double>("A", 4, 8);
    auto B = create_random_tensor<double>("B", 8, 4);

    cg::Graph graph("unannotated");
    auto     &C = graph.create_zero_tensor<double, 2>("C", 4, 4);

    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ij <- ia ; aj", &C, A, B);
    }

    auto const *desc = einsum_descriptor(graph);
    REQUIRE(desc != nullptr);
    CHECK(desc->letter_spaces.empty());

    graph.execute();

    // The contraction summed by hand, so the end-to-end result is checked against something
    // that shares no code with the path under test.
    for (int row = 0; row < 4; ++row) {
        for (int col = 0; col < 4; ++col) {
            double expected = 0.0;
            for (int link = 0; link < 8; ++link) {
                expected += A(row, link) * B(link, col);
            }
            CHECK_THAT(C(row, col), Catch::Matchers::WithinAbs(expected, 1.0e-12));
        }
    }
}

TEST_CASE("SpaceAnnotation - the string-spec capture route binds the same way", "[ComputeGraph][Spaces]") {
    Spaces spaces;

    auto A = create_random_tensor<double>("A", 4, 8);
    auto B = create_random_tensor<double>("B", 8, 16);

    cg::Graph graph("string-spec");
    graph.set_space_registry(spaces.registry);
    auto &C = graph.create_zero_tensor<double, 2>("C", 4, 16);

    graph.annotate_spaces(A, {spaces.occ, spaces.virt});
    graph.annotate_spaces(B, {spaces.virt, spaces.aux});

    // A spec built at run time: no consteval validation, straight through
    // EinsumFormatString(string_view), which is the route the Python bindings take.
    std::string const spec = "ix <- ia ; ax";
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum(cg::EinsumFormatString(std::string_view{spec}), 0.0, &C, 1.0, A, B);
    }

    auto const *desc = einsum_descriptor(graph);
    REQUIRE(desc != nullptr);
    CHECK(desc->space_for_letter("i") == std::optional<cg::SpaceId>{spaces.occ});
    CHECK(desc->space_for_letter("a") == std::optional<cg::SpaceId>{spaces.virt});
    CHECK(desc->space_for_letter("x") == std::optional<cg::SpaceId>{spaces.aux});
}

TEST_CASE("SpaceAnnotation - a rebuilt node re-derives its letter map", "[ComputeGraph][Spaces]") {
    Spaces spaces;

    cg::Graph graph("rebuilt");
    graph.set_space_registry(spaces.registry);

    auto &A = graph.create_runtime_tensor<double>("A", {4, 8});
    auto &B = graph.create_runtime_tensor<double>("B", {8, 16});
    auto &C = graph.create_runtime_tensor<double>("C", {4, 16});

    auto const a_id = graph.find_tensor_id_by_ptr(&A);
    auto const b_id = graph.find_tensor_id_by_ptr(&B);
    auto const c_id = graph.find_tensor_id_by_ptr(&C);
    REQUIRE(a_id != 0);
    REQUIRE(b_id != 0);
    REQUIRE(c_id != 0);

    graph.annotate_spaces(a_id, {spaces.occ, spaces.virt});
    graph.annotate_spaces(b_id, {spaces.virt, spaces.aux});

    auto const parsed = cg::parse_einsum_spec("ix <- ia ; ax");
    REQUIRE(parsed.has_value());

    auto const node = graph.make_einsum_node(a_id, b_id, c_id, parsed.value(), cg::PrefactorScalar{0.0}, cg::PrefactorScalar{1.0});

    auto const *desc = std::get_if<cg::EinsumDescriptor>(&node.op_data);
    REQUIRE(desc != nullptr);
    CHECK(desc->space_for_letter("i") == std::optional<cg::SpaceId>{spaces.occ});
    CHECK(desc->space_for_letter("a") == std::optional<cg::SpaceId>{spaces.virt});
    CHECK(desc->space_for_letter("x") == std::optional<cg::SpaceId>{spaces.aux});

    // The output is graph-owned and unannotated, so it inherits here too.
    CHECK(graph.tensor_spaces(c_id) == std::vector<cg::SpaceId>{spaces.occ, spaces.aux});
}

TEST_CASE("SpaceAnnotation - an address freed and reused is not the same tensor", "[ComputeGraph][Spaces]") {
    Spaces spaces;

    cg::Graph graph("reused_address");
    graph.set_space_registry(spaces.registry);

    // The by-object annotation entry is keyed on the caller's address, and an address is only an
    // identity for as long as the object at it lives. A tensor built on top of a destroyed one
    // would otherwise inherit its handle, and the annotation meant for the new one would land on
    // the old one's slots. Placement-new into one buffer reproduces that reuse deterministically
    // rather than hoping the allocator obliges.
    using Rank2 = Tensor<double, 2>;
    alignas(Rank2) std::array<std::byte, sizeof(Rank2)> storage{};

    auto *first = new (storage.data()) Rank2("first", 4, 8);
    graph.annotate_spaces(*first, {spaces.occ, spaces.virt});
    cg::TensorId const first_id = graph.find_tensor_id_by_ptr(first);
    REQUIRE(first_id != 0);

    first->~Rank2();
    auto *second = new (storage.data()) Rank2("second", 4, 8);
    REQUIRE(static_cast<void const *>(second) == static_cast<void const *>(storage.data()));

    graph.annotate_spaces(*second, {spaces.virt, spaces.occ});
    cg::TensorId const second_id = graph.find_tensor_id_by_ptr(second);

    CHECK(second_id != first_id);
    CHECK(graph.tensor_spaces(first_id) == std::vector<cg::SpaceId>{spaces.occ, spaces.virt});
    CHECK(graph.tensor_spaces(second_id) == std::vector<cg::SpaceId>{spaces.virt, spaces.occ});

    second->~Rank2();
}
