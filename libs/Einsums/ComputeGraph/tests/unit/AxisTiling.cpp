//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file AxisTiling.cpp
/// @brief What AxisTiling decides, on the shape it was designed against.
///
/// The program below is the density-fitted MP2 correlation energy written over all four of its
/// orbital indices, at extents small enough to run in a unit test. It is the one shape where
/// every part of the decision has a knowable right answer: the occupied axes are the ones to
/// slice, the virtual ones are not, and the reason the virtual ones are not is structural
/// rather than a matter of their size.
///
/// Each case asserts the MECHANISM. "The pass fired" would pass on a decision that sliced the
/// wrong axes at the wrong depth, so what is asserted is which axes, at what extents, into how
/// many slices, in chunks of what, and what the largest intermediate measures on both sides of
/// the rewrite.

#include <Einsums/ComputeGraph.hpp>
// Not reached through the umbrella header, which does not carry this pass.
#include <Einsums/ComputeGraph/Options.hpp>
#include <Einsums/ComputeGraph/Passes/AxisTiling.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums;
namespace cg = einsums::compute_graph;

namespace {

constexpr std::size_t nocc = 4;
constexpr std::size_t nvir = 6;
constexpr std::size_t naux = 7;

/// Bytes one rank-four tensor over the four orbital indices costs.
constexpr std::size_t whole_bytes = nocc * nvir * nocc * nvir * sizeof(double);
/// Bytes one occupied PAIR of that tensor costs.
constexpr std::size_t pair_bytes = nvir * nvir * sizeof(double);
/// Bytes one occupied ROW of it costs, which is what slicing a single axis leaves.
constexpr std::size_t row_bytes = nocc * nvir * nvir * sizeof(double);

/// The buffers the caller owns, kept alive for as long as the graph that reads them.
struct Problem {
    Tensor<double, 3> fitted{create_random_tensor<double>("B", naux, nocc, nvir)};
    Tensor<double, 4> denominator{create_random_tensor<double>("D", nocc, nvir, nocc, nvir)};
    Tensor<double, 1> energy{"E", 1};
};

/// E = sum_iajb (2 K[i,a,j,b] - K[i,b,j,a]) K[i,a,j,b] D[i,a,j,b], with K = sum_Q B B.
///
/// The integral is formed twice on purpose, exactly as the Laplace proving ground forms it: the
/// exchange combination needs its own copy. Everything here is a graph-owned intermediate over
/// all four indices, which is what makes the program the one that needs tiling.
void capture(cg::Graph &graph, Problem &problem) {
    auto &integral    = graph.scratch<double, 4>("K", nocc, nvir, nocc, nvir);
    auto &amplitude   = graph.scratch<double, 4>("T", nocc, nvir, nocc, nvir);
    auto &again       = graph.scratch<double, 4>("K_again", nocc, nvir, nocc, nvir);
    auto &exchange    = graph.scratch<double, 4>("K_exchange", nocc, nvir, nocc, nvir);
    auto &combination = graph.scratch<double, 4>("Kbar", nocc, nvir, nocc, nvir);

    cg::CaptureGuard const guard(graph);
    cg::einsum("Q,i,a ; Q,j,b -> i,a,j,b", &integral, problem.fitted, problem.fitted);
    cg::direct_product(1.0, integral, problem.denominator, 0.0, &amplitude);
    cg::einsum("Q,i,a ; Q,j,b -> i,a,j,b", &again, problem.fitted, problem.fitted);
    cg::permute("i,a,j,b <- i,b,j,a", &exchange, again);
    cg::axpby(2.0, again, 0.0, &combination);
    cg::axpby(-1.0, exchange, 1.0, &combination);
    // The rank-one destination is the graph-native scalar handle; a bare double would register
    // a rank-zero slot the accumulation has no axpby to write through.
    cg::dot_python(&problem.energy, combination, amplitude);
}

/// The same energy with the exchange term dropped, so nothing permutes the virtual axes.
///
/// Without that permutation BOTH pairs of axes carry a slice labelling, and which one the pass
/// takes is decided by traffic alone. That is the half of the decision the four-index program
/// above cannot exercise, because there the virtual pair is rejected before any size is looked
/// at.
void capture_without_exchange(cg::Graph &graph, Problem &problem) {
    auto &integral    = graph.scratch<double, 4>("K", nocc, nvir, nocc, nvir);
    auto &amplitude   = graph.scratch<double, 4>("T", nocc, nvir, nocc, nvir);
    auto &combination = graph.scratch<double, 4>("Kbar", nocc, nvir, nocc, nvir);

    cg::CaptureGuard const guard(graph);
    cg::einsum("Q,i,a ; Q,j,b -> i,a,j,b", &integral, problem.fitted, problem.fitted);
    cg::direct_product(1.0, integral, problem.denominator, 0.0, &amplitude);
    cg::axpby(2.0, integral, 0.0, &combination);
    cg::dot_python(&problem.energy, combination, amplitude);
}

/// Run the pass alone at @p cap bytes and hand back what it decided.
std::shared_ptr<cg::passes::AxisTiling> decide(cg::Graph &graph, std::int64_t cap) {
    auto tiling = std::make_shared<cg::passes::AxisTiling>();
    tiling->set_memory_cap(cap);
    cg::PassManager manager;
    manager.add(tiling);
    graph.apply(manager);
    return tiling;
}

/// The reasons the pass reported, as a set of prefixes a case can look for.
bool declined_because(cg::passes::AxisTiling const &tiling, std::string_view fragment) {
    for (auto const &[reason, count] : tiling.skip_reasons()) {
        if (reason.find(fragment) != std::string::npos) {
            return count > 0;
        }
    }
    return false;
}

} // namespace

TEST_CASE("AxisTiling picks the occupied axes of the four-index energy", "[ComputeGraph][Pass][AxisTiling]") {
    Problem   problem;
    cg::Graph graph("mp2 full axis");
    capture(graph, problem);

    // A cap between one occupied pair and one occupied row: the pair fits, the row does not, so
    // the only feasible depth is one and the only feasible set is both occupied axes.
    REQUIRE(pair_bytes < 400);
    REQUIRE(row_bytes > 400);
    auto const tiling = decide(graph, 400);

    // The axes, by the letter the contraction that writes the seed names them with. i and j are
    // the occupied ones; a and b are rejected because the exchange permutation exchanges them,
    // so a slice of the permuted tensor would need a slice of its source at a different pair.
    CHECK(tiling->axis_letters() == std::vector<std::string>{"i", "j"});
    CHECK(tiling->axis_names() == std::vector<std::string>{"K[0]", "K[2]"});
    CHECK(tiling->axis_extents() == std::vector<std::int64_t>{nocc, nocc});

    CHECK(tiling->slice_count() == nocc * nocc);
    CHECK(tiling->depth() == 1);
    CHECK(tiling->iterations() == nocc * nocc);

    // The water number in miniature: o^2 v^2 becomes v^2.
    CHECK(tiling->largest_before() == whole_bytes);
    CHECK(tiling->largest_after() == pair_bytes);

    // Every four-index intermediate streams, and the three-index integral streams too, because
    // one of its axes is an occupied one. Nothing is left whole but the energy.
    std::vector<std::string> const streamed = tiling->streamed();
    for (auto const &name : {"K", "T", "K_again", "K_exchange", "Kbar", "B", "D"}) {
        INFO("expected '" << name << "' among the streamed tensors");
        CHECK(std::ranges::find(streamed, std::string{name}) != streamed.end());
    }
    CHECK(tiling->whole() == std::vector<std::string>{"E"});
    CHECK(tiling->accumulator() == "E");
}

TEST_CASE("AxisTiling declines a program whose largest intermediate already fits", "[ComputeGraph][Pass][AxisTiling]") {
    Problem   problem;
    cg::Graph graph("mp2 full axis");
    capture(graph, problem);

    auto const tiling = decide(graph, static_cast<std::int64_t>(whole_bytes));
    CHECK(tiling->slice_count() == 0);
    CHECK(tiling->largest_before() == whole_bytes);
    CHECK(declined_because(*tiling, "already fits the cap"));
}

TEST_CASE("AxisTiling is off when the cap is zero", "[ComputeGraph][Pass][AxisTiling]") {
    Problem   problem;
    cg::Graph graph("mp2 full axis");
    capture(graph, problem);

    auto const tiling = decide(graph, 0);
    CHECK(tiling->slice_count() == 0);
    CHECK(declined_because(*tiling, "memory cap is zero"));
}

TEST_CASE("AxisTiling takes the chunk the cap allows", "[ComputeGraph][Pass][AxisTiling]") {
    Problem   problem;
    cg::Graph graph("mp2 full axis");
    capture(graph, problem);

    // Three pairs would fit, but a chunk has to divide the slice count so every chunk is full;
    // sixteen slices in chunks of three is not a schedule, so the depth walks down to two.
    auto const tiling = decide(graph, 3 * static_cast<std::int64_t>(pair_bytes));
    CHECK(tiling->axis_letters() == std::vector<std::string>{"i", "j"});
    CHECK(tiling->depth() == 2);
    CHECK(tiling->iterations() == (nocc * nocc) / 2);
    CHECK(tiling->largest_after() == 2 * pair_bytes);
}

TEST_CASE("AxisTiling prefers the axes that stream the fewest bytes", "[ComputeGraph][Pass][AxisTiling]") {
    Problem   problem;
    cg::Graph graph("mp2 no exchange");
    capture_without_exchange(graph, problem);

    // At this cap the occupied pair fits (v^2 doubles) and so does the virtual pair (o^2), so
    // both are candidates and the decision is not the cap's. The occupied pair wins because
    // each of its slices reads a Q-by-v slab of the integral o^2 times, where the virtual pair
    // reads a Q-by-o slab v^2 times, and o is the smaller of the two.
    REQUIRE(nvir * nvir * sizeof(double) < 400);
    REQUIRE(nocc * nocc * sizeof(double) < 400);
    auto const tiling = decide(graph, 400);

    CHECK(tiling->axis_letters() == std::vector<std::string>{"i", "j"});
    CHECK(tiling->largest_after() == pair_bytes);
}

TEST_CASE("AxisTiling reads the cap the caller states over the option", "[ComputeGraph][Pass][AxisTiling]") {
    cg::passes::AxisTiling tiling;
    CHECK(tiling.memory_cap() == config::get(option::GraphTilingMemoryCap));
    tiling.set_memory_cap(4096);
    CHECK(tiling.memory_cap() == 4096);
}
