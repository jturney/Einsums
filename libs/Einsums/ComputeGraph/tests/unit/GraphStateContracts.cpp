//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file GraphStateContracts.cpp
/// @brief Sequences of the Graph's stateful operations, and the invariants each step keeps.
///
/// A Graph carries state beyond its nodes: a pending multi-slot bind, rebound slots, adopted
/// cleanups, its entry in the profiler registry, and whatever optimization and save/load did
/// to it. Each feature has tests of its own. What those tests cannot see is an ordering: a
/// move in the middle of an open bind, a move-assignment over a graph that still holds
/// cleanups, an optimized graph moved before it runs. Every case here walks such a sequence,
/// checks the numbers against a brute-force reference after each execute, and requires
/// ``Graph::verify()`` to come back empty after each step.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/ComputeGraph/GraphIR.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>
#include <Einsums/Testing/ReferenceEinsum.hpp>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Einsums/Testing.hpp>

using einsums::testing::reference_einsum;

using namespace einsums;
namespace cg = einsums::compute_graph;

namespace {

/// Graph::verify() reports nothing.
void require_well_formed(cg::Graph const &graph) {
    auto const problems = graph.verify();
    INFO(fmt::format("{}", fmt::join(problems, "\n")));
    REQUIRE(problems.empty());
}

/// Every element of @p got within a relative tolerance of @p want.
template <typename TensorType>
void require_close(TensorType const &got, TensorType const &want) {
    REQUIRE(got.size() == want.size());
    double worst = 0.0;
    for (std::size_t n = 0; n < got.size(); ++n) {
        double const scale = std::max(1.0, std::abs(want.data()[n]));
        worst              = std::max(worst, std::abs(got.data()[n] - want.data()[n]) / scale);
    }
    REQUIRE(worst < 1.0e-12);
}

size_t count_occurrences(std::string const &haystack, std::string const &needle) {
    size_t count = 0;
    for (size_t pos = haystack.find(needle); pos != std::string::npos; pos = haystack.find(needle, pos + needle.size())) {
        ++count;
    }
    return count;
}

/// How many times the registry lists a graph of this name.
size_t registry_entries(std::string const &graph_name) {
    return count_occurrences(cg::registered_graphs_json(), fmt::format(R"("name":"{}")", graph_name));
}

/// ``out = (amp * amp) contracted against amp`` over a deferred intermediate whose extents
/// follow the dim symbols ``no`` and ``nv``, so a bind at a new size must re-derive it.
struct SymbolicChain {
    cg::SpaceRegistry registry;
    cg::SpaceId       occ;
    cg::SpaceId       virt;

    SymbolicChain(std::string const &prefix)
        : occ(registry.register_space(cg::IndexSpace{.name = prefix + "_occ", .scale_symbol = "o", .dim_symbol = "no"})),
          virt(registry.register_space(cg::IndexSpace{.name = prefix + "_virt", .scale_symbol = "v", .dim_symbol = "nv"})) {}

    void capture(cg::Graph &graph, RuntimeTensor<double> &amp, RuntimeTensor<double> &out) {
        graph.set_space_registry(registry);
        graph.annotate_spaces(amp, {occ, virt});
        graph.annotate_dims(amp, {"no", "nv"});
        graph.annotate_spaces(out, {occ, occ});
        graph.annotate_dims(out, {"no", "no"});
        auto                  &tmp = graph.declare_zero_runtime_tensor<double>("tmp", {cg::SpaceDim{occ}, cg::SpaceDim{virt}}, true);
        cg::CaptureGuard const guard(graph);
        cg::einsum("ia;ia->ia", &tmp, amp, amp);
        cg::einsum("ia;ja->ij", &out, tmp, amp);
    }

    static RuntimeTensor<double> reference(RuntimeTensor<double> const &amp) {
        std::size_t const     no = amp.dim(0);
        std::size_t const     nv = amp.dim(1);
        RuntimeTensor<double> tmp("tmp", std::vector<std::size_t>{no, nv});
        RuntimeTensor<double> out("out", std::vector<std::size_t>{no, no});
        reference_einsum("ia <- ia ; ia", 0.0, &tmp, 1.0, amp, amp);
        reference_einsum("ij <- ia ; ja", 0.0, &out, 1.0, tmp, amp);
        return out;
    }
};

RuntimeTensor<double> random_runtime(std::string const &name, std::size_t rows, std::size_t cols) {
    return RuntimeTensor<double>(create_random_tensor<double>(name, rows, cols));
}

/// Allocate what the graph deferred, which a bind leaves to the resource phase.
void materialize(cg::Graph &graph) {
    cg::PassManager pm;
    pm.add<cg::passes::Materialization>();
    graph.apply(pm);
}

} // namespace

// Defends: an open multi-slot bind carried across a move construction. The pending list used
// to be left behind by a move, and each pending step captured the graph it was added to. The
// existing move test in SymbolicExtents.cpp checks only the re-derived extents; this one runs
// the moved-to graph and checks it computes the NEW problem.
TEST_CASE("Graph state - an open bind survives a move and the moved-to graph computes the new problem",
          "[ComputeGraph][GraphState][Bind][Move]") {
    SymbolicChain chain("gsc_bind_move");
    auto          amp = random_runtime("amp", 4, 6);
    auto          out = random_runtime("out", 4, 4);

    cg::Graph graph("gsc_bind_move");
    chain.capture(graph, amp, out);
    require_well_formed(graph);

    auto amp2 = random_runtime("amp2", 3, 5);
    auto out2 = random_runtime("out2", 3, 3);
    graph.bind_begin();
    graph.bind_add("amp", amp2);
    graph.bind_add("out", out2);

    cg::Graph moved(std::move(graph));
    require_well_formed(moved);
    REQUIRE_NOTHROW(moved.bind_commit());
    require_well_formed(moved);

    materialize(moved);
    require_well_formed(moved);
    moved.execute();
    require_close(out2, SymbolicChain::reference(amp2));
}

// Defends: an open bind carried across a move ASSIGNMENT, onto a graph that was already
// holding a different program. The commit must land on the program the pending slots were
// added against.
TEST_CASE("Graph state - an open bind survives a move-assignment", "[ComputeGraph][GraphState][Bind][Move]") {
    SymbolicChain chain("gsc_bind_assign");
    auto          amp = random_runtime("amp", 4, 6);
    auto          out = random_runtime("out", 4, 4);

    cg::Graph graph("gsc_bind_assign");
    chain.capture(graph, amp, out);

    auto amp2 = random_runtime("amp2", 2, 7);
    auto out2 = random_runtime("out2", 2, 2);
    graph.bind_begin();
    graph.bind_add("amp", amp2);
    graph.bind_add("out", out2);

    auto      X = create_random_tensor<double>("X", 3, 3);
    auto      Y = create_zero_tensor<double>("Y", 3, 3);
    cg::Graph target("gsc_bind_assign_target");
    {
        cg::CaptureGuard const guard(target);
        cg::einsum("ik;kj->ij", &Y, X, X);
    }

    target = std::move(graph);
    require_well_formed(target);
    REQUIRE_NOTHROW(target.bind_commit());
    materialize(target);
    require_well_formed(target);
    target.execute();
    require_close(out2, SymbolicChain::reference(amp2));

    // The replaced program never ran.
    auto zero = create_zero_tensor<double>("zero", 3, 3);
    require_close(Y, zero);
}

// Defends: a rebound slot followed through a move. A rebind repoints the slot the executors
// read through; a move carries the slot table, so the moved-to graph must read the rebound
// operand, and a second rebind on the new owner must take effect in turn.
TEST_CASE("Graph state - rebind, move, execute, move-assign, rebind, execute", "[ComputeGraph][GraphState][Rebind][Move]") {
    auto A1 = create_random_tensor<double>("A1", 4, 3);
    auto A2 = create_random_tensor<double>("A2", 4, 3);
    auto A3 = create_random_tensor<double>("A3", 4, 3);
    auto B  = create_random_tensor<double>("B", 3, 5);
    auto C  = create_zero_tensor<double>("C", 4, 5);

    cg::Graph graph("gsc_rebind_move");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A1, B);
    }
    graph.execute();
    auto expected = create_zero_tensor<double>("expected", 4, 5);
    reference_einsum("ij <- ik ; kj", &expected, A1, B);
    require_close(C, expected);

    graph.rebind(A1, A2);
    require_well_formed(graph);

    cg::Graph moved(std::move(graph));
    require_well_formed(moved);
    C.zero();
    moved.execute();
    reference_einsum("ij <- ik ; kj", &expected, A2, B);
    require_close(C, expected);

    cg::Graph owner("gsc_rebind_owner");
    owner = std::move(moved);
    require_well_formed(owner);
    owner.rebind(A2, A3);
    require_well_formed(owner);
    C.zero();
    owner.execute();
    reference_einsum("ij <- ik ; kj", &expected, A3, B);
    require_close(C, expected);
}

// Defends: move-assignment releasing what the assigned-over graph held, exactly as its
// destructor would. Its adopted cleanups run once, newest first, at the assignment and never
// again; its open bind is discarded rather than committed against the program that replaced
// it. Both graphs name their operands A, B and C, so a pending slot that leaked across the
// assignment would repoint the new program's A and change the answer.
TEST_CASE("Graph state - move-assigning over a graph runs its cleanups once and drops its open bind",
          "[ComputeGraph][GraphState][Adopt][Bind][Move]") {
    int              target_cleanups = 0;
    int              source_cleanups = 0;
    std::vector<int> order;

    auto A_src = create_random_tensor<double>("A", 4, 3);
    auto B_src = create_random_tensor<double>("B", 3, 5);
    auto C_src = create_zero_tensor<double>("C", 4, 5);

    {
        auto A_tgt   = create_random_tensor<double>("A", 4, 3);
        auto B_tgt   = create_random_tensor<double>("B", 3, 5);
        auto C_tgt   = create_zero_tensor<double>("C", 4, 5);
        auto A_other = create_random_tensor<double>("A", 4, 3);

        cg::Graph target("gsc_assign_target");
        {
            cg::CaptureGuard const guard(target);
            cg::einsum("ik;kj->ij", &C_tgt, A_tgt, B_tgt);
        }
        target.adopt([&] {
            ++target_cleanups;
            order.push_back(1);
        });
        target.adopt([&] {
            ++target_cleanups;
            order.push_back(2);
        });
        target.bind_begin();
        target.bind_add("A", A_other);

        {
            cg::Graph source("gsc_assign_source");
            {
                cg::CaptureGuard const guard(source);
                cg::einsum("ik;kj->ij", &C_src, A_src, B_src);
            }
            source.adopt([&] { ++source_cleanups; });

            target = std::move(source);
            CHECK(target_cleanups == 2);
            CHECK(order == std::vector<int>{2, 1});
            CHECK(source_cleanups == 0);
            require_well_formed(target);
        }
        // The moved-from source has died; the cleanup it handed over has not run.
        CHECK(source_cleanups == 0);

        // The target's open bind went with the program it was opened against.
        REQUIRE_NOTHROW(target.bind_commit());
        require_well_formed(target);
        target.execute();

        auto expected = create_zero_tensor<double>("expected", 4, 5);
        reference_einsum("ij <- ik ; kj", &expected, A_src, B_src);
        require_close(C_src, expected);
        auto zero = create_zero_tensor<double>("zero", 4, 5);
        require_close(C_tgt, zero);
    }
    CHECK(target_cleanups == 2);
    CHECK(source_cleanups == 1);
}

// Defends: adopted cleanups travelling through a chain of moves. Every moved-from graph dies
// before the final owner; none of them may run what it handed over, and the final owner runs
// it exactly once.
TEST_CASE("Graph state - adopted cleanups run once, when the last owner dies", "[ComputeGraph][GraphState][Adopt][Move]") {
    int  runs = 0;
    auto A    = create_random_tensor<double>("A", 3, 3);
    auto C    = create_zero_tensor<double>("C", 3, 3);
    {
        std::optional<cg::Graph> last;
        {
            cg::Graph first("gsc_cleanup_chain");
            {
                cg::CaptureGuard const guard(first);
                cg::einsum("ik;kj->ij", &C, A, A);
            }
            first.adopt([&] { ++runs; });
            cg::Graph second(std::move(first));
            cg::Graph third("gsc_cleanup_chain_third");
            third = std::move(second);
            last.emplace(std::move(third));
        }
        CHECK(runs == 0);
        require_well_formed(*last);
        last->execute();
        auto expected = create_zero_tensor<double>("expected", 3, 3);
        reference_einsum("ij <- ik ; kj", &expected, A, A);
        require_close(C, expected);
        CHECK(runs == 0);
    }
    CHECK(runs == 1);
}

// Defends: an optimized graph moved before and between replays. optimize() leaves Materialize,
// Free and planned storage in the graph; a move must carry all of it, and a move-assignment over
// another optimized graph must not free storage the survivor still uses.
TEST_CASE("Graph state - execute, optimize, move, execute, move-assign, execute", "[ComputeGraph][GraphState][Optimize][Move]") {
    auto A  = create_random_tensor<double>("A", 6, 4);
    auto B  = create_random_tensor<double>("B", 4, 5);
    auto D  = create_random_tensor<double>("D", 5, 3);
    auto C  = create_zero_tensor<double>("C", 6, 3);
    auto R  = create_random_tensor<double>("R", 6, 5);
    auto R0 = R;

    auto expected_c = create_zero_tensor<double>("expected_c", 6, 3);
    {
        auto T = create_zero_tensor<double>("T", 6, 5);
        reference_einsum("ij <- ik ; kj", &T, A, B);
        reference_einsum("ij <- ik ; kj", &expected_c, T, D);
    }
    // R += 0.5 A B on every execute.
    auto accumulate_r = [&](Tensor<double, 2> &r) { reference_einsum("ij <- ik ; kj", 1.0, &r, 0.5, A, B); };
    auto expected_r   = R0;

    cg::Graph graph("gsc_optimize_move");
    {
        auto                  &T = graph.scratch<double, 2>("T", 6, 5);
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", 0.0, &T, 1.0, A, B);
        cg::einsum("ik;kj->ij", 0.0, &C, 1.0, T, D);
        cg::einsum("ik;kj->ij", 1.0, &R, 0.5, A, B);
    }
    materialize(graph);
    graph.execute();
    accumulate_r(expected_r);
    require_close(C, expected_c);
    require_close(R, expected_r);

    graph.optimize();
    require_well_formed(graph);
    C.zero();
    graph.execute();
    accumulate_r(expected_r);
    require_close(C, expected_c);
    require_close(R, expected_r);

    cg::Graph moved(std::move(graph));
    require_well_formed(moved);
    C.zero();
    moved.execute();
    accumulate_r(expected_r);
    require_close(C, expected_c);
    require_close(R, expected_r);

    // A second optimized graph, assigned over by the first.
    auto      X = create_random_tensor<double>("X", 5, 5);
    auto      Y = create_zero_tensor<double>("Y", 5, 5);
    cg::Graph other("gsc_optimize_other");
    {
        auto                  &U = other.scratch<double, 2>("U", 5, 5);
        cg::CaptureGuard const guard(other);
        cg::einsum("ik;kj->ij", 0.0, &U, 1.0, X, X);
        cg::einsum("ik;kj->ij", 0.0, &Y, 1.0, U, X);
    }
    other.optimize();
    other.execute();

    other = std::move(moved);
    require_well_formed(other);
    C.zero();
    other.execute();
    accumulate_r(expected_r);
    require_close(C, expected_c);
    require_close(R, expected_r);
}

// Defends: executors that passes bake keep reaching their graph after it moves. LCCF's
// L-builder node looks its tensors up by id on every run, and it used to capture the graph's
// address to do it, so executing a folded graph after moving it read the moved-from object and
// crashed on a null tensor. graph.optimize() runs this pass, so returning an optimized graph from
// a function or storing it in a container was enough. The executor now holds the graph's
// GraphAnchor, which every move re-points.
TEST_CASE("Graph state - a graph folded by LCCF survives a move", "[ComputeGraph][GraphState][Move][LCCF]") {
    RuntimeTensor<double> v(create_random_tensor<double>("v", 4));
    RuntimeTensor<double> W(create_random_tensor<double>("W", 4, 3, 3));
    RuntimeTensor<double> out("out", std::vector<std::size_t>{3, 3});
    out.zero();

    RuntimeTensor<double> expected("expected", std::vector<std::size_t>{3, 3});
    reference_einsum("ij <- k ; kij", 0.0, &expected, 2.0, v, W);
    reference_einsum("ij <- k ; kji", 1.0, &expected, -1.0, v, W);

    std::optional<cg::Graph> holder;
    {
        cg::Graph graph("gsc_lccf_move");
        {
            cg::CaptureGuard const guard(graph);
            cg::einsum("i,j <- k ; k,i,j", 0.0, &out, 2.0, v, W);
            cg::einsum("i,j <- k ; k,j,i", 1.0, &out, -1.0, v, W);
        }
        auto [modified, pass] = graph.apply<cg::passes::LinearCombinationContractionFolding>();
        REQUIRE(modified);
        REQUIRE(pass.num_groups() == 1);
        holder.emplace(std::move(graph));
    } // the graph the pass ran on is destroyed here

    require_well_formed(*holder);
    holder->execute();
    require_close(out, expected);

    // And once more through a move assignment over a live graph.
    cg::Graph other("gsc_lccf_other");
    other = std::move(*holder);
    holder.reset();
    out.zero();
    other.execute();
    require_close(out, expected);
}

// Defends: the same move safety for StreamContractionFusion's fused node, which resolves the
// streamed operand, the weights and the outputs by id on every run and captured the graph's
// address to do it, exactly as LCCF did.
TEST_CASE("Graph state - a graph fused by StreamContractionFusion survives a move", "[ComputeGraph][GraphState][Move][StreamFusion]") {
    constexpr std::size_t kN = 40; // above the pass's stream threshold

    RuntimeTensor<double> TEI(create_random_tensor<double>("TEI", kN, kN, kN, kN));
    RuntimeTensor<double> D(create_random_tensor<double>("D", kN, kN));
    RuntimeTensor<double> J("J", std::vector<std::size_t>{kN, kN});
    RuntimeTensor<double> K("K", std::vector<std::size_t>{kN, kN});
    J.zero();
    K.zero();

    RuntimeTensor<double> J_ref("J_ref", std::vector<std::size_t>{kN, kN});
    RuntimeTensor<double> K_ref("K_ref", std::vector<std::size_t>{kN, kN});
    reference_einsum("ij <- ijkl ; kl", 0.0, &J_ref, 2.0, TEI, D);
    reference_einsum("ij <- ikjl ; kl", 0.0, &K_ref, -1.0, TEI, D);

    std::optional<cg::Graph> holder;
    {
        cg::Graph graph("gsc_stream_move");
        {
            cg::CaptureGuard const guard(graph);
            cg::einsum("i,j <- i,j,k,l ; k,l", 0.0, &J, 2.0, TEI, D);
            cg::einsum("i,j <- i,k,j,l ; k,l", 0.0, &K, -1.0, TEI, D);
        }
        auto [modified, pass] = graph.apply<cg::passes::StreamContractionFusion>();
        REQUIRE(modified);
        REQUIRE(pass.num_groups() == 1);
        holder.emplace(std::move(graph));
    }

    require_well_formed(*holder);
    holder->execute();
    require_close(J, J_ref);
    require_close(K, K_ref);
}

// Defends: the GraphAnchor contract every pass-baked executor relies on. It must name the graph
// wherever the graph has been moved, by construction or by assignment, and once the graph is
// gone it must throw rather than hand back a dangling reference.
TEST_CASE("Graph state - an anchor follows its graph and outlives it safely", "[ComputeGraph][GraphState][Move][Anchor]") {
    std::shared_ptr<cg::GraphAnchor const> anchor;
    {
        cg::Graph graph("gsc_anchor");
        anchor = graph.anchor();
        CHECK(&anchor->graph() == &graph);
        CHECK(graph.anchor() == anchor);

        cg::Graph moved(std::move(graph));
        CHECK(&anchor->graph() == &moved);

        cg::Graph  assigned("gsc_anchor_target");
        auto const replaced = assigned.anchor();
        assigned            = std::move(moved);
        CHECK(&anchor->graph() == &assigned);
        CHECK_THROWS_AS(replaced->graph(), std::logic_error);
    }
    CHECK_THROWS_AS(anchor->graph(), std::logic_error);
}

// Defends: the profiler registry entry following a graph through two moves. The registry
// names a graph by address, so each move must hand the entry over rather than add one or
// leave a stale one, a graph that was assigned over must drop out, and the entry must go when
// the final owner dies.
TEST_CASE("Graph state - a graph moved twice is listed once and unlisted when its last owner dies",
          "[ComputeGraph][GraphState][Registry][Move]") {
    auto A = create_random_tensor<double>("A", 4, 4);
    auto C = create_zero_tensor<double>("C", 4, 4);
    auto E = create_zero_tensor<double>("E", 4, 4);

    std::string const name   = "gsc_registry_moved_twice";
    std::string const target = "gsc_registry_assign_target";
    {
        cg::Graph graph(name);
        {
            cg::CaptureGuard const guard(graph);
            cg::einsum("ik;kj->ij", &C, A, A);
        }
        CHECK(registry_entries(name) == 1);

        cg::Graph first(std::move(graph));
        CHECK(registry_entries(name) == 1);
        require_well_formed(first);

        cg::Graph second(target);
        {
            cg::CaptureGuard const guard(second);
            cg::einsum("ik;kj->ij", &E, A, A);
        }
        CHECK(registry_entries(target) == 1);

        second = std::move(first);
        CHECK(registry_entries(name) == 1);
        CHECK(registry_entries(target) == 0);
        require_well_formed(second);

        second.execute();
        CHECK(registry_entries(name) == 1);
        auto expected = create_zero_tensor<double>("expected", 4, 4);
        reference_einsum("ij <- ik ; kj", &expected, A, A);
        require_close(C, expected);
    }
    CHECK(registry_entries(name) == 0);
    CHECK(registry_entries(target) == 0);
}

// Defends: the save/load round trip taken from an OPTIMIZED graph rather than a freshly
// captured one, then bound to new operands and run from a moved-to graph. Every tensor is the
// caller's, so nothing optimize() adds needs a lifecycle node, and what it does rewrite (the
// accumulation pair, the node order) must come back from the file computing the same thing.
TEST_CASE("Graph state - save after optimize, load, bind, move, execute", "[ComputeGraph][GraphState][SaveLoad][Optimize]") {
    auto A = create_random_tensor<double>("A", 6, 4);
    auto B = create_random_tensor<double>("B", 4, 5);
    auto D = create_random_tensor<double>("D", 5, 3);
    auto T = create_zero_tensor<double>("T", 6, 5);
    auto C = create_zero_tensor<double>("C", 6, 3);
    auto R = create_random_tensor<double>("R", 6, 5);

    cg::Graph graph("gsc_save_optimized");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", 0.0, &T, 1.0, A, B);
        cg::einsum("ik;kj->ij", 0.0, &C, 1.0, T, D);
        cg::scale(2.0, &R);
        cg::einsum("ik;kj->ij", 1.0, &R, 0.5, A, B);
    }
    graph.optimize();
    require_well_formed(graph);
    graph.execute();

    auto const text = cg::save_graph_string(graph);
    INFO((text ? std::string{} : text.error().message));
    REQUIRE(text.has_value());

    auto loaded = cg::load_graph_string(*text);
    INFO((loaded ? std::string{} : loaded.error().message));
    REQUIRE(loaded.has_value());
    require_well_formed(*loaded);

    auto A2 = create_random_tensor<double>("A2", 6, 4);
    auto B2 = create_random_tensor<double>("B2", 4, 5);
    auto D2 = create_random_tensor<double>("D2", 5, 3);
    auto T2 = create_zero_tensor<double>("T2", 6, 5);
    auto C2 = create_zero_tensor<double>("C2", 6, 3);
    auto R2 = create_random_tensor<double>("R2", 6, 5);
    auto R0 = R2;
    loaded->bind("A", A2, "B", B2, "D", D2, "T", T2, "C", C2, "R", R2);
    require_well_formed(*loaded);

    // Moved once more before it runs, so the loaded graph's state also crosses a move.
    cg::Graph runner(std::move(*loaded));
    require_well_formed(runner);
    runner.execute();

    auto T_ref = create_zero_tensor<double>("T_ref", 6, 5);
    auto C_ref = create_zero_tensor<double>("C_ref", 6, 3);
    auto R_ref = R0;
    reference_einsum("ij <- ik ; kj", &T_ref, A2, B2);
    reference_einsum("ij <- ik ; kj", &C_ref, T_ref, D2);
    reference_einsum("ij <- ik ; kj", 2.0, &R_ref, 0.5, A2, B2);
    require_close(T2, T_ref);
    require_close(C2, C_ref);
    require_close(R2, R_ref);

    // A replay reads the bound operands again, and accumulates R once more.
    C2.zero();
    runner.execute();
    reference_einsum("ij <- ik ; kj", 2.0, &R_ref, 0.5, A2, B2);
    require_close(C2, C_ref);
    require_close(R2, R_ref);
}

// Placeholder for a capability that does not exist yet: saving a graph after optimize() when
// it holds deferred scratch. The Materialization pass optimize() runs emits a Materialize node
// for the scratch, and the writer cannot rebuild that kind from data, so the save refuses.
// The refusal is the right behaviour today (a file that silently dropped the node would load
// into a graph whose scratch is never allocated), and this case pins that it stays a refusal
// that names the node. The replacement test, once Materialize is reconstructible: save after
// optimize, load, bind new operands, execute, and compare against reference_einsum, as the
// case above does for a graph without scratch.
TEST_CASE("Graph state - save after optimize refuses a graph with deferred scratch, naming the node",
          "[ComputeGraph][GraphState][SaveLoad][Optimize]") {
    auto A = create_random_tensor<double>("A", 6, 4);
    auto B = create_random_tensor<double>("B", 4, 5);
    auto D = create_random_tensor<double>("D", 5, 3);
    auto C = create_zero_tensor<double>("C", 6, 3);

    cg::Graph graph("gsc_save_optimized_scratch");
    {
        auto                  &T = graph.scratch<double, 2>("T", 6, 5);
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", 0.0, &T, 1.0, A, B);
        cg::einsum("ik;kj->ij", 0.0, &C, 1.0, T, D);
    }
    graph.optimize();
    require_well_formed(graph);

    auto const text = cg::save_graph_string(graph);
    REQUIRE_FALSE(text.has_value());
    CHECK_THAT(text.error().message, Catch::Matchers::ContainsSubstring("Materialize"));
    CHECK_THAT(text.error().message, Catch::Matchers::ContainsSubstring("materialize(T)"));

    // The refused graph is untouched by the attempt and still runs.
    require_well_formed(graph);
    graph.execute();
    auto T_ref = create_zero_tensor<double>("T_ref", 6, 5);
    auto C_ref = create_zero_tensor<double>("C_ref", 6, 3);
    reference_einsum("ij <- ik ; kj", &T_ref, A, B);
    reference_einsum("ij <- ik ; kj", &C_ref, T_ref, D);
    require_close(C, C_ref);
}
