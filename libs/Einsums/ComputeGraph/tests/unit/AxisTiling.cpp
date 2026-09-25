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
#include <Einsums/Tensor/RuntimeTensor.hpp>

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

/// A reproducible filling, so two arms of one comparison see identical inputs.
RuntimeTensor<double> filled(std::string name, std::vector<std::size_t> dims, unsigned seed) {
    RuntimeTensor<double> out(std::move(name), std::move(dims));
    for (std::size_t i = 0; i < out.size(); ++i) {
        seed          = (seed * 1103515245U) + 12345U;
        out.data()[i] = (static_cast<double>((seed >> 8U) % 2048U) / 1024.0) - 1.0;
    }
    return out;
}

/// The buffers the caller owns, kept alive for as long as the graph that reads them.
///
/// Rank-erased, because the emitted body names its operands' static types and a rank-erased
/// operand is the one it can name. A statically ranked capture is declined with that reason.
struct Problem {
    RuntimeTensor<double> fitted{filled("B", {naux, nocc, nvir}, 7U)};
    RuntimeTensor<double> denominator{filled("D", {nocc, nvir, nocc, nvir}, 19U)};
    RuntimeTensor<double> energy{"E", std::vector<std::size_t>{1}};
};

/// E = sum_iajb (2 K[i,a,j,b] - K[i,b,j,a]) K[i,a,j,b] D[i,a,j,b], with K = sum_Q B B.
///
/// The integral is formed twice on purpose, and here the duplicate is load-bearing rather than
/// inherited: two contractions are what give the emitted chunk two grouped families, and the
/// exchange permutation reads the copy rather than the tensor the amplitude reads, which is what
/// the occupied-exchange case below is about. Everything here is a graph-owned intermediate over
/// all four indices, which is what makes the program the one that needs tiling.
void capture(cg::Graph &graph, Problem &problem) {
    std::vector<std::size_t> const shape{nocc, nvir, nocc, nvir};
    auto                          &integral    = graph.scratch_runtime<double>("K", shape);
    auto                          &amplitude   = graph.scratch_runtime<double>("T", shape);
    auto                          &again       = graph.scratch_runtime<double>("K_again", shape);
    auto                          &exchange    = graph.scratch_runtime<double>("K_exchange", shape);
    auto                          &combination = graph.scratch_runtime<double>("Kbar", shape);

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
    std::vector<std::size_t> const shape{nocc, nvir, nocc, nvir};
    auto                          &integral    = graph.scratch_runtime<double>("K", shape);
    auto                          &amplitude   = graph.scratch_runtime<double>("T", shape);
    auto                          &combination = graph.scratch_runtime<double>("Kbar", shape);

    cg::CaptureGuard const guard(graph);
    cg::einsum("Q,i,a ; Q,j,b -> i,a,j,b", &integral, problem.fitted, problem.fitted);
    cg::direct_product(1.0, integral, problem.denominator, 0.0, &amplitude);
    cg::axpby(2.0, integral, 0.0, &combination);
    cg::dot_python(&problem.energy, combination, amplitude);
}

/// The four-index energy whose exchange term permutes the OCCUPIED pair rather than the virtual
/// one.
///
/// Not a physical program: it is the shape the symmetry-partner sentence of the design is about.
/// At a fixed pair the body holds the ``(i,j)`` slab, and this exchange wants the ``(j,i)`` one,
/// which the body is not at. So the occupied pair stops being a candidate and the pass says why.
void capture_with_occupied_exchange(cg::Graph &graph, Problem &problem) {
    std::vector<std::size_t> const shape{nocc, nvir, nocc, nvir};
    auto                          &integral    = graph.scratch_runtime<double>("K", shape);
    auto                          &amplitude   = graph.scratch_runtime<double>("T", shape);
    auto                          &again       = graph.scratch_runtime<double>("K_again", shape);
    auto                          &exchange    = graph.scratch_runtime<double>("K_exchange", shape);
    auto                          &combination = graph.scratch_runtime<double>("Kbar", shape);

    cg::CaptureGuard const guard(graph);
    cg::einsum("Q,i,a ; Q,j,b -> i,a,j,b", &integral, problem.fitted, problem.fitted);
    cg::direct_product(1.0, integral, problem.denominator, 0.0, &amplitude);
    cg::einsum("Q,i,a ; Q,j,b -> i,a,j,b", &again, problem.fitted, problem.fitted);
    cg::permute("i,a,j,b <- j,a,i,b", &exchange, again);
    cg::axpby(2.0, again, 0.0, &combination);
    cg::axpby(-1.0, exchange, 1.0, &combination);
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

// ── The rewrite ─────────────────────────────────────────────────────────────

namespace {

/// The energy the program computes with no tiling on it, which every tiled arm is held to.
double untiled_energy(Problem &problem) {
    cg::Graph graph("mp2 untiled");
    capture(graph, problem);
    auto manager = cg::PassManager::create_default();
    graph.apply(manager);
    graph.execute();
    return problem.energy.data()[0];
}

/// The loop node the rewrite emitted, or nothing when it emitted none.
cg::Graph const *loop_body(cg::Graph const &graph) {
    for (auto const &node : graph.nodes()) {
        if (auto const *loop = node.op_data.get_if<cg::LoopDescriptor>(); loop != nullptr && loop->body) {
            return loop->body.get();
        }
    }
    return nullptr;
}

/// A body-declared tensor's dims, by name.
std::vector<std::size_t> body_dims(cg::Graph const &body, std::string_view name) {
    for (auto const &[id, handle] : body.tensors_map()) {
        if (handle.name == name) {
            return handle.dims;
        }
    }
    return {};
}

} // namespace

TEST_CASE("AxisTiling replays the same energy from the loop it emits", "[ComputeGraph][Pass][AxisTiling]") {
    Problem      problem;
    double const reference = untiled_energy(problem);

    cg::Graph graph("mp2 tiled");
    capture(graph, problem);
    auto const tiling = decide(graph, 400);
    REQUIRE(tiling->num_tiled() == 1);

    auto manager = cg::PassManager::create_default();
    graph.apply(manager);
    problem.energy.data()[0] = 0.0;
    graph.execute();

    // Re-associating: the reduction is summed pair by pair rather than over the whole
    // four-index tensor, so the two agree to the tier's bound and not to the bit.
    double const tiled = problem.energy.data()[0];
    INFO("untiled " << reference << " against tiled " << tiled);
    CHECK(std::abs(tiled - reference) <= cg::tier_bound(cg::PassTier::ReAssociating, 1e-16) * std::abs(reference) + 1e-14);
    CHECK(tiled != 0.0);

    // Replays restart the sweep rather than continuing it.
    problem.energy.data()[0] = 0.0;
    graph.execute();
    CHECK(problem.energy.data()[0] == Catch::Approx(tiled));
}

TEST_CASE("AxisTiling keeps a contraction's permutation operator", "[ComputeGraph][Pass][AxisTiling]") {
    // Defends the member contractions the pass emits. It re-rendered each one's spec from the
    // index lists and conjugation flags alone, so an operator the contraction carried was not in
    // the text it captured, and a tiled P(ab) contraction summed one term instead of two.
    //
    // Returns the energy and, when tiled, the pass that decided.
    auto const run = [](std::string const &spec, bool tiled) {
        Problem                                 problem;
        std::vector<std::size_t> const          shape{nocc, nvir, nocc, nvir};
        cg::Graph                               graph(tiled ? "operator tiled" : "operator untiled");
        std::shared_ptr<cg::passes::AxisTiling> tiling;
        {
            auto                  &integral    = graph.scratch_runtime<double>("K", shape);
            auto                  &amplitude   = graph.scratch_runtime<double>("T", shape);
            auto                  &combination = graph.scratch_runtime<double>("Kbar", shape);
            cg::CaptureGuard const guard(graph);
            cg::einsum(cg::EinsumFormatString(spec), &integral, problem.fitted, problem.fitted);
            cg::direct_product(1.0, integral, problem.denominator, 0.0, &amplitude);
            cg::axpby(2.0, integral, 0.0, &combination);
            cg::dot_python(&problem.energy, combination, amplitude);
        }
        if (tiled) {
            tiling = decide(graph, 400);
        }
        auto manager = cg::PassManager::create_default();
        graph.apply(manager);
        problem.energy.data()[0] = 0.0;
        graph.execute();
        return std::pair{problem.energy.data()[0], tiling};
    };
    auto const agrees = [](double tiled, double reference) {
        INFO("untiled " << reference << " against tiled " << tiled);
        REQUIRE(reference != 0.0);
        CHECK(std::abs(tiled - reference) <= cg::tier_bound(cg::PassTier::ReAssociating, 1e-16) * std::abs(reference) + 1e-14);
    };

    SECTION("an operator on axes the pass does not slice travels with each member") {
        std::string const spec         = "i,a,j,b <- P(ab) Q,i,a ; Q,j,b";
        auto const [reference, unused] = run(spec, false);
        auto const [tiled, tiling]     = run(spec, true);
        REQUIRE(tiling->num_tiled() == 1);
        agrees(tiled, reference);
    }
    SECTION("an operator on the axes it would slice keeps them whole, with the reason") {
        std::string const spec         = "i,a,j,b <- P(ij) Q,i,a ; Q,j,b";
        auto const [reference, unused] = run(spec, false);
        auto const [tiled, tiling]     = run(spec, true);
        CHECK(declined_because(*tiling, "a permutation operator exchanges one of the candidate's sliced axes"));
        agrees(tiled, reference);
    }
}

TEST_CASE("AxisTiling emits a loop whose body declares the slice, not the slab", "[ComputeGraph][Pass][AxisTiling]") {
    Problem   problem;
    cg::Graph graph("mp2 tiled");
    capture(graph, problem);
    auto const tiling = decide(graph, 400);
    REQUIRE(tiling->num_tiled() == 1);

    // The parent holds the zeroing of the accumulation and the loop, and nothing of the seven
    // nodes the region was.
    REQUIRE(graph.num_nodes() == 2);
    CHECK(graph.nodes()[0].kind == cg::OpKind::Scale);
    CHECK(graph.nodes()[1].kind == cg::OpKind::Loop);

    auto const *body = loop_body(graph);
    REQUIRE(body != nullptr);

    // Every four-index intermediate is re-declared at ONE pair. The sliced axes are gone rather
    // than one element wide: a loop variable is not an axis, and dropping them is what makes the
    // body's contraction the ordinary matrix product the hand-written pair loop performs.
    std::vector<std::size_t> const pair{nvir, nvir};
    for (auto const &name : {"K#0", "T#0", "K_again#0", "K_exchange#0", "Kbar#0"}) {
        INFO("body intermediate '" << name << "'");
        CHECK(body_dims(*body, name) == pair);
    }
    // The accumulation's partial is one element, and the caller's energy is not re-declared.
    CHECK(body_dims(*body, "axtile_partial#0") == std::vector<std::size_t>{1});

    // A slice of a caller's tensor is a VIEW of the caller's buffer, never a copy: every View
    // node in the body names a parent, and the integral's views name the caller's integral.
    std::size_t integral_views = 0;
    for (auto const &node : body->nodes()) {
        if (node.kind != cg::OpKind::View) {
            continue;
        }
        auto const *desc = node.op_data.get_if<cg::ViewDescriptor>();
        REQUIRE(desc != nullptr);
        auto const *parent = body->find_tensor(desc->parent_id);
        REQUIRE(parent != nullptr);
        if (parent->name == "B") {
            ++integral_views;
            CHECK(parent->data_ptr == problem.fitted.data());
        }
    }
    // Two: one per operand of the contraction, at two different occupied indices.
    CHECK(integral_views == 2);
}

TEST_CASE("AxisTiling leaves the storage invariants intact", "[ComputeGraph][Pass][AxisTiling]") {
    Problem   problem;
    cg::Graph graph("mp2 tiled");
    capture(graph, problem);
    REQUIRE(decide(graph, 400)->num_tiled() == 1);

    auto manager = cg::PassManager::create_default();
    graph.apply(manager);

    CHECK(cg::passes::duplicate_materializations(graph).empty());
    CHECK(cg::passes::stranded_materializations(graph).empty());

    // A body-declared intermediate's lifecycle is HOISTED to the parent, ahead of the loop,
    // which is where the placement rules put a loop body's workspace: a buffer the body reuses
    // on every iteration is allocated once. A setup body's workspace is the case that stays
    // inside, and this is not one.
    auto const *body = loop_body(graph);
    REQUIRE(body != nullptr);
    for (auto const &node : body->nodes()) {
        INFO("body node '" << node.label << "'");
        CHECK(node.kind != cg::OpKind::Materialize);
    }

    std::map<std::string, std::size_t> allocated;
    std::size_t                        loop_position = graph.num_nodes();
    for (std::size_t i = 0; i < graph.num_nodes(); ++i) {
        auto const &node = graph.nodes()[i];
        if (node.kind == cg::OpKind::Loop) {
            loop_position = i;
        }
        if (node.kind != cg::OpKind::Materialize) {
            continue;
        }
        for (auto const tid : node.outputs) {
            auto const *handle = graph.find_tensor(tid);
            if (handle != nullptr && handle->name.find('#') != std::string::npos) {
                INFO("'" << handle->name << "' is materialized behind the loop it feeds");
                CHECK(i < loop_position);
                ++allocated[handle->name];
            }
        }
    }
    // One lifecycle each, and the buffer that is allocated is the PAIR rather than the slab.
    for (auto const &name : {"K#0", "T#0", "K_again#0", "K_exchange#0", "Kbar#0", "axtile_partial#0"}) {
        INFO("body intermediate '" << name << "'");
        CHECK(allocated[name] == 1);
    }
    CHECK(body_dims(*body, "K#0") == std::vector<std::size_t>{nvir, nvir});
}

TEST_CASE("AxisTiling leaves a pair a permutation exchanges whole, with the reason", "[ComputeGraph][Pass][AxisTiling]") {
    Problem   problem;
    cg::Graph graph("mp2 occupied exchange");
    capture_with_occupied_exchange(graph, problem);

    auto const tiling = decide(graph, 400);

    // The occupied pair is gone from the decision, and the tally says what took it out. The
    // pass then takes the virtual pair, which this program's permutation does leave in place:
    // a partner contraction is not emitted and the candidate is simply not offered.
    CHECK(declined_because(*tiling, "a permutation exchanges two of the candidate's sliced axes"));
    CHECK(tiling->axis_letters() == std::vector<std::string>{"a", "b"});
    CHECK(tiling->largest_after() == tiling->depth() * nocc * nocc * sizeof(double));
}

// ── Chunks ──────────────────────────────────────────────────────────────────

namespace {

/// How many nodes of each kind the loop body holds.
std::map<cg::OpKind, std::size_t> body_kinds(cg::Graph const &graph) {
    std::map<cg::OpKind, std::size_t> out;
    auto const                       *body = loop_body(graph);
    if (body == nullptr) {
        return out;
    }
    for (auto const &node : body->nodes()) {
        ++out[node.kind];
    }
    return out;
}

/// Capture, tile at @p cap, optimize, replay, and hand back the energy.
double tiled_energy(Problem &problem, std::int64_t cap, std::shared_ptr<cg::passes::AxisTiling> *out = nullptr) {
    cg::Graph graph("mp2 tiled");
    capture(graph, problem);
    auto const tiling = decide(graph, cap);
    if (out != nullptr) {
        *out = tiling;
    }
    auto manager = cg::PassManager::create_default();
    graph.apply(manager);
    problem.energy.data()[0] = 0.0;
    graph.execute();
    return problem.energy.data()[0];
}

} // namespace

TEST_CASE("AxisTiling emits a chunk as one grouped node per family", "[ComputeGraph][Pass][AxisTiling]") {
    Problem   problem;
    cg::Graph graph("mp2 chunked");
    capture(graph, problem);

    // Three pairs would fit; the chunk has to divide the sixteen slices, so the depth is two.
    auto const tiling = decide(graph, 3 * static_cast<std::int64_t>(pair_bytes));
    REQUIRE(tiling->depth() == 2);
    REQUIRE(tiling->num_tiled() == 1);

    auto const kinds = body_kinds(graph);
    // Two contractions, one permutation, two accumulations, one reduction and the accumulation
    // into the energy: seven captured nodes, and at a chunk of two each becomes ONE grouped node
    // rather than two ungrouped ones. Nothing of the ungrouped families survives.
    CHECK(kinds.at(cg::OpKind::GroupedBatchedGemm) == 2);
    CHECK(kinds.at(cg::OpKind::GroupedPermute) == 1);
    CHECK(kinds.at(cg::OpKind::GroupedDirectProduct) == 1);
    CHECK(kinds.at(cg::OpKind::GroupedDot) == 1);
    // Two combinations plus the accumulation into the energy.
    CHECK(kinds.at(cg::OpKind::GroupedAxpby) == 3);
    CHECK(kinds.count(cg::OpKind::Einsum) == 0);
    CHECK(kinds.count(cg::OpKind::Dot) == 0);

    // The chunk's members each hold their own pair, so the store scales with the chunk.
    CHECK(tiling->largest_after() == 2 * pair_bytes);
    CHECK(body_dims(*loop_body(graph), "K#1") == std::vector<std::size_t>{nvir, nvir});
}

TEST_CASE("AxisTiling gets the same energy from a chunk of pairs as from one", "[ComputeGraph][Pass][AxisTiling]") {
    Problem      problem;
    double const reference = untiled_energy(problem);

    std::shared_ptr<cg::passes::AxisTiling> per_pair;
    std::shared_ptr<cg::passes::AxisTiling> per_chunk;
    double const                            one  = tiled_energy(problem, 400, &per_pair);
    double const                            many = tiled_energy(problem, 3 * static_cast<std::int64_t>(pair_bytes), &per_chunk);

    REQUIRE(per_pair->depth() == 1);
    REQUIRE(per_chunk->depth() == 2);
    CHECK(per_chunk->largest_after() == 2 * per_pair->largest_after());

    double const bound = cg::tier_bound(cg::PassTier::ReAssociating, 1e-16) * std::abs(reference) + 1e-14;
    INFO("per pair " << one << ", per chunk " << many << ", untiled " << reference);
    CHECK(std::abs(many - one) <= bound);
    CHECK(std::abs(many - reference) <= bound);
}
