//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file EscapeAnalysis.cpp
/// @brief Who writes a tensor, who reads it, and what a region may dissolve.
///
/// The component exists because two passes had already built most of it, each
/// for its own question and each in its own translation unit, and the region
/// framework needs a third phrasing of the same facts. A third copy is how a
/// module ends up with three derivations of one relation that disagree in the
/// corner nobody tested, which in this module is not hypothetical: the
/// full-cover alias bug and the 32-hop `resolve_alias` cap were both an
/// incomplete alias relation, and both surfaced as a race or a wrong number
/// rather than as an error.
///
/// So the cases here cover the facts and each consumer's phrasing of them.
/// `Pass_SymmetryPropagation.cpp` and `Pass_LoopInvariantHoisting.cpp` still
/// hold their own behavioural cases, and those passing unchanged after the
/// factoring is the evidence it preserved behaviour; this file covers what the
/// shared component says on its own terms, including the region rule neither
/// pass exercises.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <unordered_set>
#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums;
namespace cg = einsums::compute_graph;

namespace {

/// Every node id of @p graph, which is the region a whole-graph rewrite covers.
std::unordered_set<cg::NodeId> all_nodes(cg::Graph const &graph) {
    std::unordered_set<cg::NodeId> out;
    for (auto const &node : graph.nodes()) {
        out.insert(node.id);
    }
    return out;
}

/// The nodes of @p graph other than those writing @p name, which is the region
/// "everything except this tensor's producer".
std::unordered_set<cg::NodeId> nodes_except(cg::Graph const &graph, cg::TensorId tid) {
    std::unordered_set<cg::NodeId> out;
    for (auto const &node : graph.nodes()) {
        bool writes = false;
        for (auto const out_tid : node.outputs) {
            writes = writes || out_tid == tid;
        }
        if (!writes) {
            out.insert(node.id);
        }
    }
    return out;
}

/// The graph tensor named @p name, or an invalid id.
cg::TensorId id_of(cg::Graph const &graph, std::string_view name) {
    for (auto const &[id, handle] : graph.tensors_map()) {
        if (handle.name == name) {
            return id;
        }
    }
    return cg::TensorId{};
}

} // namespace

TEST_CASE("value-writers are counted and lifecycle nodes are not", "[ComputeGraph][EscapeAnalysis]") {
    // A freshly allocated or zeroed tensor is then filled by exactly one real
    // op, so counting the Alloc as a writer would make every deferred
    // intermediate look overwritten and decline every rewrite over a graph built
    // from declare_*, which is every graph the save/load path produces.
    auto A = create_random_tensor<double>("A", 4, 3);
    auto B = create_random_tensor<double>("B", 3, 5);
    auto D = create_random_tensor<double>("D", 5, 2);
    auto C = create_zero_tensor<double>("C", 4, 2);

    // scratch is consumed, so DeadNodeElimination keeps its producer; an
    // unread intermediate would be dropped and the case would prove nothing.
    cg::Graph graph("writers");
    auto     &scratch = graph.declare_zero_runtime_tensor<double>("scratch", {4, 5}, true);
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &scratch, A, B);
        cg::einsum("ij;jl->il", &C, scratch, D);
    }
    auto pm = cg::PassManager::create_for(cg::OptLevel::O1);
    graph.apply(pm); // Materialization gives scratch its lifecycle nodes

    auto const escapes = cg::EscapeAnalysis::over(graph);
    auto const tid     = id_of(graph, "scratch");
    REQUIRE(tid != cg::TensorId{});
    // The lifecycle nodes really are there, so the exclusion is doing work
    // rather than the case passing on a graph that never grew any.
    std::size_t lifecycle = 0;
    for (auto const &node : graph.nodes()) {
        for (auto const out_tid : node.outputs) {
            if (out_tid == tid && is_lifecycle(node.kind)) {
                ++lifecycle;
            }
        }
    }
    CHECK(lifecycle > 0);

    CHECK(escapes.writer_count(tid) == 1);
    CHECK(escapes.stable(tid));
}

TEST_CASE("a second writer makes a tensor unstable", "[ComputeGraph][EscapeAnalysis]") {
    auto A = create_random_tensor<double>("A", 4, 3);
    auto B = create_random_tensor<double>("B", 3, 5);
    auto C = create_zero_tensor<double>("C", 4, 5);

    cg::Graph graph("two writers");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A, B);
        cg::scale(2.0, &C);
    }

    auto const escapes = cg::EscapeAnalysis::over(graph);
    auto const tid     = id_of(graph, "C");
    REQUIRE(tid != cg::TensorId{});
    CHECK(escapes.writer_count(tid) == 2);
    CHECK_FALSE(escapes.stable(tid));
}

TEST_CASE("a write through a view counts against the parent", "[ComputeGraph][EscapeAnalysis]") {
    // The whole point of resolving first. A component that counted the view
    // object would report the parent as single-writer while a different node
    // overwrote it, which is the shape of the full-cover alias bug.
    RuntimeTensor<double> parent("parent", std::vector<size_t>{8, 8});

    cg::Graph graph("view writes");
    {
        cg::CaptureGuard const guard(graph);
        cg::scale(0.0, &parent);
        auto &slice = cg::view_runtime(parent, {cg::ViewAxis::range(0, 4), cg::ViewAxis::full()});
        cg::scale(2.0, &slice);
    }

    auto const escapes = cg::EscapeAnalysis::over(graph);
    auto const tid     = id_of(graph, "parent");
    REQUIRE(tid != cg::TensorId{});
    // Three, not two: the full scale, the slice's scale, and the View node
    // itself, which lists the slice as an output and so resolves onto the
    // parent's buffer. A View writes no value, so counting it is a deliberate
    // conservatism rather than an accident - see the note on
    // EscapeAnalysis::writer_count. It is also what both passes this component
    // was factored out of already did, and the point of the factoring was to
    // keep their behaviour, not to improve it in the same step.
    CHECK(escapes.writer_count(tid) == 3);
    CHECK_FALSE(escapes.stable(tid));
    // And the two ids are one alias class, which is what a region has to range
    // over rather than asking about one of them.
    CHECK(escapes.aliases_of(tid).size() == 2);
}

TEST_CASE("a region may dissolve an intermediate nothing outside touches", "[ComputeGraph][EscapeAnalysis][Region]") {
    auto A = create_random_tensor<double>("A", 4, 3);
    auto B = create_random_tensor<double>("B", 3, 5);
    auto D = create_random_tensor<double>("D", 5, 2);
    auto C = create_zero_tensor<double>("C", 4, 2);

    // tmp is written by the first contraction and read by the second, and
    // nothing else in the graph mentions it. That is the shape a factorization
    // dissolves.
    cg::Graph graph("dissolvable");
    auto     &tmp = graph.create_zero_runtime_tensor<double>("tmp", {4, 5}, true);
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &tmp, A, B);
        cg::einsum("ij;jl->il", &C, tmp, D);
    }

    auto const escapes = cg::EscapeAnalysis::over(graph);
    auto const tid     = id_of(graph, "tmp");
    REQUIRE(tid != cg::TensorId{});
    CHECK(escapes.classify(tid, all_nodes(graph)) == cg::Escape::Dissolvable);

    // C is the region's output and the user holds it, so it is never dissolvable
    // however tightly the region is drawn.
    auto const c_tid = id_of(graph, "C");
    REQUIRE(c_tid != cg::TensorId{});
    CHECK(escapes.classify(c_tid, all_nodes(graph)) == cg::Escape::UserOwned);
}

TEST_CASE("an intermediate read outside the region is not dissolvable", "[ComputeGraph][EscapeAnalysis][Region]") {
    auto A  = create_random_tensor<double>("A", 4, 3);
    auto B  = create_random_tensor<double>("B", 3, 5);
    auto D  = create_random_tensor<double>("D", 5, 2);
    auto C  = create_zero_tensor<double>("C", 4, 2);
    auto C2 = create_zero_tensor<double>("C2", 4, 2);

    cg::Graph graph("read outside");
    auto     &tmp = graph.create_zero_runtime_tensor<double>("tmp", {4, 5}, true);
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &tmp, A, B);
        cg::einsum("ij;jl->il", &C, tmp, D);
        cg::einsum("ij;jl->il", &C2, tmp, D);
    }

    auto const escapes = cg::EscapeAnalysis::over(graph);
    auto const tid     = id_of(graph, "tmp");
    REQUIRE(tid != cg::TensorId{});

    // A region holding only the first two nodes leaves the third reading tmp.
    std::unordered_set<cg::NodeId> partial;
    for (std::size_t i = 0; i < 2; ++i) {
        partial.insert(graph.nodes()[i].id);
    }
    CHECK(escapes.classify(tid, partial) == cg::Escape::ReadOutside);

    // Grow the region to cover the reader and it becomes dissolvable, which is
    // the useful half: an outside reader usually means the region is too small.
    CHECK(escapes.classify(tid, all_nodes(graph)) == cg::Escape::Dissolvable);
}

TEST_CASE("an intermediate written outside the region is not dissolvable", "[ComputeGraph][EscapeAnalysis][Region]") {
    // Reported as WrittenOutside rather than ReadOutside, and the distinction is
    // the point: an outside writer means the region does not own the value at
    // all, while an outside reader means it merely needs to grow.
    auto A = create_random_tensor<double>("A", 4, 3);
    auto B = create_random_tensor<double>("B", 3, 5);
    auto D = create_random_tensor<double>("D", 5, 2);
    auto C = create_zero_tensor<double>("C", 4, 2);

    cg::Graph graph("written outside");
    auto     &tmp = graph.create_zero_runtime_tensor<double>("tmp", {4, 5}, true);
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &tmp, A, B);
        cg::einsum("ij;jl->il", &C, tmp, D);
    }

    auto const escapes = cg::EscapeAnalysis::over(graph);
    auto const tid     = id_of(graph, "tmp");
    REQUIRE(tid != cg::TensorId{});
    CHECK(escapes.classify(tid, nodes_except(graph, tid)) == cg::Escape::WrittenOutside);
}

TEST_CASE("a buffer a loop body touches is not dissolvable", "[ComputeGraph][EscapeAnalysis][Region]") {
    // A Loop node does not list its body's writes, so a tensor the body fills
    // looks untouched from the parent's node list. Anything reasoning from that
    // list has to ask before concluding it knows every access.
    auto A = create_random_tensor<double>("A", 4, 3);
    auto B = create_random_tensor<double>("B", 3, 5);
    auto C = create_zero_tensor<double>("C", 4, 5);

    // C is written in the parent AND in the body. From the parent's node list
    // alone it has one writer and looks settled; only the subtree question
    // reveals the second, which is exactly the blind spot a Loop node creates.
    cg::Graph graph("loop touches");
    {
        cg::CaptureGuard const guard(graph);
        cg::scale(0.0, &C);
    }
    auto &body = graph.add_loop("thrice", 3, [](size_t iter) { return iter < 2; });
    {
        cg::CaptureGuard const body_guard(body);
        cg::einsum("ik;kj->ij", &C, A, B);
    }

    auto const escapes = cg::EscapeAnalysis::over(graph);
    auto const tid     = id_of(graph, "C");
    REQUIRE(tid != cg::TensorId{});
    CHECK(escapes.writer_count(tid) == 1);         // what the parent's list shows
    CHECK(escapes.subtree_writer_count(tid) == 2); // what is actually there
    CHECK(escapes.touched_by_subtree(tid));
    CHECK_FALSE(escapes.stable(tid));
    CHECK(escapes.classify(tid, all_nodes(graph)) == cg::Escape::UserOwned);
}

TEST_CASE("every escape reason has a phrase note_skip can aggregate", "[ComputeGraph][EscapeAnalysis]") {
    // note_skip keys its tally on the reason string, so a phrase carrying a
    // tensor name would produce one line per candidate instead of one line with
    // a count. Shape-independence is checked here rather than assumed at each
    // call site.
    for (auto const reason : {cg::Escape::Dissolvable, cg::Escape::UserOwned, cg::Escape::ReadOutside, cg::Escape::WrittenOutside,
                              cg::Escape::AliasedFromOutside, cg::Escape::TouchedBySubgraph, cg::Escape::Unknown}) {
        auto const phrase = cg::escape_reason(reason);
        INFO("phrase: " << phrase);
        CHECK_FALSE(phrase.empty());
        CHECK(phrase != "unclassified");
        // No digits, which is the cheap proxy for "carries no extent or id".
        CHECK(phrase.find_first_of("0123456789") == std::string_view::npos);
    }
}
