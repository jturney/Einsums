//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file PassPhases.cpp
/// @brief Tests for OptimizerPass::phase, the phase-filtered PassManagers, and
///        the Graph::structure_version counter they are enforced with.

#include <Einsums/ComputeGraph.hpp>
#include <Einsums/ComputeGraphTypes/Spaces.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorUtilities/CreateRandomTensor.hpp>
#include <Einsums/TensorUtilities/CreateZeroTensor.hpp>

#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include <Einsums/Testing.hpp>

using namespace einsums;
namespace cg = einsums::compute_graph;

namespace {

/// The phase every built-in pass is expected to declare.
///
/// Hard-coded on purpose, and the reason is the monotonicity-test genre rather
/// than belt-and-braces: ``OptimizerPass::phase`` defaults to
/// ``PassPhase::Tuning``, so a new pass nobody classified would silently join
/// the phase whose output is never saved. Requiring the table to NAME it turns
/// "I forgot to think about this" into a failing test, and the failure message
/// is the question that had to be answered anyway - may a saved graph keep this
/// pass's output on another machine?
std::map<std::string, cg::PassPhase> const &expected_phases() {
    static std::map<std::string, cg::PassPhase> const table = {
        // Analysis: writes annotations, never the node set.
        {"SymmetryPropagation", cg::PassPhase::Analysis},
        {"SpacePropagation", cg::PassPhase::Analysis},
        {"ProvenancePropagation", cg::PassPhase::Analysis},

        // Diagnostic: read-only reporting.
        {"CrossSpaceValidation", cg::PassPhase::Diagnostic},
        {"ScalingAnalysis", cg::PassPhase::Diagnostic},
        {"GPUDiagnostics", cg::PassPhase::Diagnostic},

        // Structural-algebraic: machine-independent, and the only output a save keeps.
        {"DeltaElimination", cg::PassPhase::StructuralAlgebraic},
        {"ConstantFolding", cg::PassPhase::StructuralAlgebraic},
        {"ScaleAbsorption", cg::PassPhase::StructuralAlgebraic},
        {"PermuteFusion", cg::PassPhase::StructuralAlgebraic},
        {"CSE", cg::PassPhase::StructuralAlgebraic},
        {"DeadNodeElimination", cg::PassPhase::StructuralAlgebraic},
        {"SymmetrizedAccumulation", cg::PassPhase::StructuralAlgebraic},
        {"ElementWiseFusion", cg::PassPhase::StructuralAlgebraic},
        {"LinearCombinationContractionFolding", cg::PassPhase::StructuralAlgebraic},
        {"DistributiveFactoring", cg::PassPhase::StructuralAlgebraic},
        {"LoopInvariantHoisting", cg::PassPhase::StructuralAlgebraic},
        {"ContractionPlanning", cg::PassPhase::StructuralAlgebraic},

        // Structural-resource: node-set changes made for machine reasons.
        {"TiledExpansion", cg::PassPhase::StructuralResource},
        {"ScratchPrivatization", cg::PassPhase::StructuralResource},
        {"DistributionPlanning", cg::PassPhase::StructuralResource},
        {"GPUPlacement", cg::PassPhase::StructuralResource},
        {"TransferInsertion", cg::PassPhase::StructuralResource},
        {"TransferElimination", cg::PassPhase::StructuralResource},
        {"InputSlicing", cg::PassPhase::StructuralResource},
        {"SUMMAExpansion", cg::PassPhase::StructuralResource},
        {"CommunicationInsertion", cg::PassPhase::StructuralResource},
        {"CommunicationElimination", cg::PassPhase::StructuralResource},
        {"CommunicationScheduling", cg::PassPhase::StructuralResource},

        // Tuning: schedule, memory and batching over a final node set.
        {"GEMMBatching", cg::PassPhase::Tuning},
        {"Reorder", cg::PassPhase::Tuning},
        {"IOPrefetch", cg::PassPhase::Tuning},
        {"Materialization", cg::PassPhase::Tuning},
        {"StreamContractionFusion", cg::PassPhase::Tuning},
        {"StreamAssignment", cg::PassPhase::Tuning},
        {"InplaceOptimization", cg::PassPhase::Tuning},
        {"FreeInsertion", cg::PassPhase::Tuning},
        {"MemoryPlanning", cg::PassPhase::Tuning},
        // Not in the default pipeline, tagged anyway: the classification is a
        // property of the pass, not of whether create_default() happens to use it.
        {"ThreadPlanning", cg::PassPhase::Tuning},
        // A region rewrite that rewrites nothing. Structural-algebraic because it LOWERS, which
        // is a node-set change however faithful, and a pass whose phase said otherwise would be
        // refused by the manager's own read-only check.
        {"RegionIdentity", cg::PassPhase::StructuralAlgebraic},
    };
    return table;
}

std::vector<std::string> names_of(cg::PassManager const &pm) {
    std::vector<std::string> out;
    out.reserve(pm.passes().size());
    for (auto const &p : pm.passes()) {
        out.push_back(p->name());
    }
    return out;
}

/// Snapshot of everything about the node list that an executor can observe.
struct NodeShape {
    std::string               kind;
    std::string               label;
    std::vector<cg::TensorId> inputs;
    std::vector<cg::TensorId> outputs;

    bool operator==(NodeShape const &) const = default;
};

std::vector<NodeShape> shape_of(cg::Graph const &g) {
    std::vector<NodeShape> out;
    out.reserve(g.nodes().size());
    for (auto const &n : g.nodes()) {
        out.push_back(NodeShape{.kind = std::string(cg::op_kind_name(n.kind)), .label = n.label, .inputs = n.inputs, .outputs = n.outputs});
    }
    return out;
}

/// A pass that declares no phase at all, to pin the default the doc promises.
class UnclassifiedPass : public cg::OptimizerPass {
  public:
    [[nodiscard]] std::string name() const override { return "UnclassifiedPass"; }

    bool run(cg::Graph &) override { return false; }
};

/// Analysis phase, but it rewrites the node set. The manager must catch it.
class RogueAnalysisPass : public cg::OptimizerPass {
  public:
    [[nodiscard]] std::string   name() const override { return "RogueAnalysisPass"; }
    [[nodiscard]] cg::PassPhase phase() const override { return cg::PassPhase::Analysis; }

    bool run(cg::Graph &graph) override {
        cg::Node n;
        n.kind  = cg::OpKind::Scale;
        n.label = "rogue";
        graph.add_node(std::move(n));
        return false; // and it lies about it, which is the point
    }
};

/// Counts the nodes it saw, and never resets between the manager's re-runs, so
/// the LAST value it recorded is what the final sweep observed.
class NodeCountingAnalysis : public cg::OptimizerPass {
  public:
    [[nodiscard]] std::string   name() const override { return "NodeCountingAnalysis"; }
    [[nodiscard]] cg::PassPhase phase() const override { return cg::PassPhase::Analysis; }

    bool run(cg::Graph &graph) override {
        _last_seen = graph.num_nodes();
        _runs++;
        return false;
    }

    void reset_stats() override {
        _last_seen = 0;
        _runs      = 0;
    }

    [[nodiscard]] size_t last_seen() const { return _last_seen; }
    [[nodiscard]] size_t runs() const { return _runs; }

  private:
    size_t _last_seen{0};
    size_t _runs{0};
};

/// Structural-algebraic, and it really does change the node set.
class AppendNodePass : public cg::OptimizerPass {
  public:
    explicit AppendNodePass(bool active) : _active(active) {}

    [[nodiscard]] std::string   name() const override { return "AppendNodePass"; }
    [[nodiscard]] cg::PassPhase phase() const override { return cg::PassPhase::StructuralAlgebraic; }

    bool run(cg::Graph &graph) override {
        if (!_active) {
            return false;
        }
        cg::Node n;
        n.kind    = cg::OpKind::Scale;
        n.label   = "appended";
        n.execute = []() {};
        graph.add_node(std::move(n));
        graph.mark_sorted();
        return true;
    }

  private:
    bool _active;
};

} // namespace

// ══════════════════════════════════════════════════════════════════════════════
// The phase table
// ══════════════════════════════════════════════════════════════════════════════

TEST_CASE("pass phases - every default pass declares the phase the table names", "[ComputeGraph][Phases]") {
    auto const pm = cg::PassManager::create_default();

    for (auto const &[name, phase] : pm.phase_of_each()) {
        INFO("pass '" << name << "' reports phase '" << cg::pass_phase_name(phase) << "'");
        auto const hit = expected_phases().find(name);
        REQUIRE(hit != expected_phases().end()); // a new pass must be classified here
        CHECK(hit->second == phase);
    }
}

TEST_CASE("pass phases - a pass outside the default pipeline is classified too", "[ComputeGraph][Phases]") {
    cg::passes::ThreadPlanning const planner;
    CHECK(planner.phase() == cg::PassPhase::Tuning);
    CHECK(expected_phases().at(planner.name()) == planner.phase());
}

TEST_CASE("pass phases - the unclassified default is the never-saved one", "[ComputeGraph][Phases]") {
    // A pass that forgot to answer the question must not have its output written
    // into a saved graph, so the default is the phase that is always re-derived.
    UnclassifiedPass const pass;
    CHECK(pass.phase() == cg::PassPhase::Tuning);
}

TEST_CASE("pass phases - names round-trip", "[ComputeGraph][Phases]") {
    CHECK(std::string(cg::pass_phase_name(cg::PassPhase::Analysis)) == "analysis");
    CHECK(std::string(cg::pass_phase_name(cg::PassPhase::StructuralAlgebraic)) == "structural-algebraic");
    CHECK(std::string(cg::pass_phase_name(cg::PassPhase::StructuralResource)) == "structural-resource");
    CHECK(std::string(cg::pass_phase_name(cg::PassPhase::Tuning)) == "tuning");
    CHECK(std::string(cg::pass_phase_name(cg::PassPhase::Diagnostic)) == "diagnostic");
}

// ══════════════════════════════════════════════════════════════════════════════
// The named managers
// ══════════════════════════════════════════════════════════════════════════════

TEST_CASE("named managers - the four partition the default pipeline exactly", "[ComputeGraph][Phases]") {
    auto const def        = names_of(cg::PassManager::create_default());
    auto const analysis   = names_of(cg::PassManager::analysis_pass_manager());
    auto const structural = names_of(cg::PassManager::structural_pass_manager());
    auto const resource   = names_of(cg::PassManager::resource_pass_manager());
    auto const tuning     = names_of(cg::PassManager::tuning_pass_manager());

    CHECK(analysis.size() + structural.size() + resource.size() + tuning.size() == def.size());

    std::vector<std::string> union_of;
    for (auto const *part : {&analysis, &structural, &resource, &tuning}) {
        union_of.insert(union_of.end(), part->begin(), part->end());
    }
    auto sorted_def = def;
    std::ranges::sort(sorted_def);
    std::ranges::sort(union_of);
    CHECK(union_of == sorted_def);
}

TEST_CASE("named managers - each view preserves the default's relative order", "[ComputeGraph][Phases]") {
    auto const def = names_of(cg::PassManager::create_default());

    // A filtered view is a SUBSEQUENCE of the default order. That is what makes
    // it a view rather than a re-plan: the hand-ordered sequence in
    // populate_default() carries ordering constraints (LCCF before
    // LoopInvariantHoisting, GEMMBatching before DistributionPlanning) that a
    // phase-sorted rebuild would quietly drop.
    auto is_subsequence = [&def](std::vector<std::string> const &part) {
        auto it = def.begin();
        for (auto const &name : part) {
            it = std::find(it, def.end(), name);
            if (it == def.end()) {
                return false;
            }
            ++it;
        }
        return true;
    };

    CHECK(is_subsequence(names_of(cg::PassManager::analysis_pass_manager())));
    CHECK(is_subsequence(names_of(cg::PassManager::structural_pass_manager())));
    CHECK(is_subsequence(names_of(cg::PassManager::resource_pass_manager())));
    CHECK(is_subsequence(names_of(cg::PassManager::tuning_pass_manager())));
}

TEST_CASE("named managers - every member reports the phase its manager filtered on", "[ComputeGraph][Phases]") {
    for (auto const &[name, phase] : cg::PassManager::structural_pass_manager().phase_of_each()) {
        INFO(name);
        CHECK(phase == cg::PassPhase::StructuralAlgebraic);
    }
    for (auto const &[name, phase] : cg::PassManager::resource_pass_manager().phase_of_each()) {
        INFO(name);
        CHECK(phase == cg::PassPhase::StructuralResource);
    }
    for (auto const &[name, phase] : cg::PassManager::tuning_pass_manager().phase_of_each()) {
        INFO(name);
        CHECK(phase == cg::PassPhase::Tuning);
    }
    for (auto const &[name, phase] : cg::PassManager::analysis_pass_manager().phase_of_each()) {
        INFO(name);
        CHECK((phase == cg::PassPhase::Analysis || phase == cg::PassPhase::Diagnostic));
    }
}

TEST_CASE("named managers - the read-only manager changes neither structure nor numbers", "[ComputeGraph][Phases]") {
    size_t const N = 6;
    auto         A = create_random_tensor<double>("A", N, N);
    auto         B = create_random_tensor<double>("B", N, N);
    auto         C = create_zero_tensor<double>("C", N, N);

    cg::Graph graph("read_only_manager");
    auto     &tmp = graph.create_tensor<double, 2>("tmp", N, N);
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &tmp, A, B);
        cg::einsum("ik;kj->ij", &C, tmp, B);
    }

    graph.execute();
    auto const reference = C;

    auto const json_before      = graph.to_json();
    auto const shape_before     = shape_of(graph);
    auto const structure_before = graph.structure_version();

    auto pm = cg::PassManager::analysis_pass_manager();
    CHECK_FALSE(pm.run(graph)); // analysis and diagnostic passes never modify

    CHECK(graph.to_json() == json_before);
    CHECK(shape_of(graph) == shape_before);
    CHECK(graph.structure_version() == structure_before);

    C.zero();
    graph.execute();
    for (size_t ii = 0; ii < N; ii++) {
        for (size_t jj = 0; jj < N; jj++) {
            REQUIRE(std::abs(C(ii, jj) - reference(ii, jj)) < 1e-13);
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Graph::structure_version
// ══════════════════════════════════════════════════════════════════════════════

TEST_CASE("structure_version - bumps once per captured node", "[ComputeGraph][Phases]") {
    auto A = create_random_tensor<double>("A", 4, 4);
    auto B = create_random_tensor<double>("B", 4, 4);
    auto C = create_zero_tensor<double>("C", 4, 4);

    cg::Graph  graph("structure_version_capture");
    auto const empty = graph.structure_version();
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A, B);
    }
    CHECK(graph.num_nodes() == 1);
    CHECK(graph.structure_version() == empty + 1);
}

TEST_CASE("structure_version - annotation, rebind and execute leave it alone", "[ComputeGraph][Phases]") {
    cg::SpaceRegistry registry;
    auto const        occ = registry.register_space(cg::IndexSpace{.name = "occ", .scale_symbol = "o", .typical_extent = 4.0});

    auto A1 = create_random_tensor<double>("A1", 4, 4);
    auto A2 = create_random_tensor<double>("A2", 4, 4);
    auto B  = create_random_tensor<double>("B", 4, 4);
    auto C  = create_zero_tensor<double>("C", 4, 4);

    cg::Graph graph("structure_version_stable");
    graph.set_space_registry(registry);
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A1, B);
    }
    auto const captured = graph.structure_version();

    // An annotation is metadata about what an axis ranges over: it changes no
    // node, so a plan built over the node set survives it.
    graph.annotate_spaces(A1, {occ, occ});
    CHECK(graph.structure_version() == captured);

    graph.execute();
    CHECK(graph.structure_version() == captured);

    graph.rebind(A1, A2);
    CHECK(graph.structure_version() == captured);

    graph.execute();
    CHECK(graph.structure_version() == captured);
}

TEST_CASE("structure_version - a real structural pass moves it", "[ComputeGraph][Phases]") {
    size_t const N = 4;
    auto         A = create_random_tensor<double>("A", N, N);
    auto         C = create_zero_tensor<double>("C", N, N);

    cg::Graph graph("structure_version_cse");
    auto     &tmp = graph.create_tensor<double, 2>("tmp", N, N);
    auto     &dup = graph.create_tensor<double, 2>("dup", N, N);
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &tmp, A, A);
        cg::einsum("ik;kj->ij", &dup, A, A); // the duplicate CSE folds away
        cg::einsum("ik;kj->ij", &C, tmp, dup);
    }

    auto const before    = graph.structure_version();
    auto [modified, cse] = graph.apply<cg::passes::CSE>();
    REQUIRE(modified);
    CHECK(graph.structure_version() > before);
}

TEST_CASE("structure_version - dead-node removal moves it", "[ComputeGraph][Phases]") {
    size_t const N = 4;
    auto         A = create_random_tensor<double>("A", N, N);
    auto         C = create_zero_tensor<double>("C", N, N);

    cg::Graph graph("structure_version_dne");
    auto     &dead = graph.create_tensor<double, 2>("dead", N, N);
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &dead, A, A); // nothing reads it
        cg::einsum("ik;kj->ij", &C, A, A);
    }

    auto const before       = graph.structure_version();
    auto const nodes_before = graph.num_nodes();
    auto [modified, dne]    = graph.apply<cg::passes::DeadNodeElimination>();
    INFO("nodes " << nodes_before << " -> " << graph.num_nodes());
    REQUIRE(modified);
    CHECK(graph.structure_version() > before);
    CHECK(graph.num_nodes() < nodes_before);
}

// ══════════════════════════════════════════════════════════════════════════════
// The read-only contract, enforced
// ══════════════════════════════════════════════════════════════════════════════

TEST_CASE("phase contract - an analysis pass that rewrites is rejected", "[ComputeGraph][Phases]") {
    auto A = create_random_tensor<double>("A", 4, 4);
    auto C = create_zero_tensor<double>("C", 4, 4);

    cg::Graph graph("rogue_analysis");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A, A);
    }

    cg::PassManager pm;
    pm.add(std::make_shared<RogueAnalysisPass>());

    // The pass returns false, so the program-order guard never looks at it. The
    // structure counter is what catches it, which is the whole reason the
    // counter is separate from analysis_version.
    CHECK_THROWS_AS(pm.run(graph), std::logic_error);
}

// ══════════════════════════════════════════════════════════════════════════════
// The post-structural analysis re-run
// ══════════════════════════════════════════════════════════════════════════════

TEST_CASE("re-run rule - analysis re-runs when a later pass changed the structure", "[ComputeGraph][Phases]") {
    auto A = create_random_tensor<double>("A", 4, 4);
    auto C = create_zero_tensor<double>("C", 4, 4);

    cg::Graph graph("rerun_analysis");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A, A);
    }

    auto counter = std::make_shared<NodeCountingAnalysis>();

    cg::PassManager pm;
    pm.add(counter);
    pm.add(std::make_shared<AppendNodePass>(true));

    CHECK(pm.run(graph));

    // Two invocations: the scheduled one at position 0 (one node), and the
    // consistency sweep after AppendNodePass grew the graph (two nodes).
    CHECK(counter->runs() == 2);
    CHECK(counter->last_seen() == graph.num_nodes());
    CHECK(counter->last_seen() == 2);
}

TEST_CASE("re-run rule - a structural pass that changed nothing triggers no re-run", "[ComputeGraph][Phases]") {
    auto A = create_random_tensor<double>("A", 4, 4);
    auto C = create_zero_tensor<double>("C", 4, 4);

    cg::Graph graph("no_rerun_analysis");
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &C, A, A);
    }

    auto counter = std::make_shared<NodeCountingAnalysis>();

    cg::PassManager pm;
    pm.add(counter);
    pm.add(std::make_shared<AppendNodePass>(false)); // reports no modification

    CHECK_FALSE(pm.run(graph));
    CHECK(counter->runs() == 1);
}

TEST_CASE("re-run rule - SpacePropagation sees the writer ScaleAbsorption removed", "[ComputeGraph][Phases]") {
    // The observable with a real analysis pass and a real structural one.
    // ``tmp`` starts with two writers - a Scale and the einsum that overwrites
    // it - and SpacePropagation declines a multiply-written tensor, because a
    // second writer could bind its slots to something else. ScaleAbsorption
    // deletes the dead Scale, leaving one writer, but it runs AFTER the analysis
    // pass here: without the end-of-pipeline re-run the annotation would never
    // appear even though the graph now supports it.
    size_t const no = 4;
    size_t const nv = 5;
    size_t const nx = 6;

    cg::SpaceRegistry registry;
    auto const        occ  = registry.register_space(cg::IndexSpace{.name = "occ", .scale_symbol = "o", .typical_extent = 4.0});
    auto const        virt = registry.register_space(cg::IndexSpace{.name = "virt", .scale_symbol = "v", .typical_extent = 5.0});
    auto const        aux  = registry.register_space(cg::IndexSpace{.name = "aux", .scale_symbol = "x", .typical_extent = 6.0});

    auto A   = create_random_tensor<double>("A", no, nv);
    auto B   = create_random_tensor<double>("B", nv, nx);
    auto out = create_zero_tensor<double>("out", no, no);

    cg::Graph graph("rerun_space_propagation");
    graph.set_space_registry(registry);
    auto &tmp = graph.create_zero_tensor<double, 2>("tmp", no, nx);
    {
        cg::CaptureGuard const guard(graph);
        cg::scale(3.0, &tmp);                          // writer 1, dead once absorbed
        cg::einsum("ia;ax->ix", 0.0, &tmp, 1.0, A, B); // writer 2
        cg::einsum("ix;jx->ij", &out, tmp, tmp);
    }

    // Annotated AFTER capture, so capture's own derivation had nothing to work
    // from and the inference is genuinely the pass's.
    graph.annotate_spaces(A, {occ, virt});
    graph.annotate_spaces(B, {virt, aux});

    auto const tmp_id = graph.find_tensor_id_by_ptr(&tmp);
    REQUIRE(tmp_id != 0);
    REQUIRE(graph.tensor_spaces(tmp_id).empty());

    auto prop = std::make_shared<cg::passes::SpacePropagation>();

    cg::PassManager pm;
    pm.add(prop);
    pm.add(std::make_shared<cg::passes::ScaleAbsorption>());
    size_t const nodes_before = graph.num_nodes();
    REQUIRE(pm.run(graph));

    // The Scale is gone; the allocation and the two einsums remain.
    INFO("nodes " << nodes_before << " -> " << graph.num_nodes());
    REQUIRE(graph.num_nodes() == nodes_before - 1);
    auto const spaces = graph.tensor_spaces(tmp_id);
    REQUIRE(spaces.size() == 2);
    CHECK(spaces[0] == occ);
    CHECK(spaces[1] == aux);
    CHECK(prop->num_inferred() == 1);
}

// ══════════════════════════════════════════════════════════════════════════════
// explain()
// ══════════════════════════════════════════════════════════════════════════════

TEST_CASE("explain - each reported line names the phase that produced it", "[ComputeGraph][Phases]") {
    size_t const N = 4;
    auto         A = create_random_tensor<double>("A", N, N);
    auto         C = create_zero_tensor<double>("C", N, N);

    cg::Graph graph("explain_phase");
    auto     &dead = graph.create_tensor<double, 2>("dead", N, N);
    {
        cg::CaptureGuard const guard(graph);
        cg::einsum("ik;kj->ij", &dead, A, A);
        cg::einsum("ik;kj->ij", &C, A, A);
    }

    cg::PassManager pm;
    pm.add<cg::passes::DeadNodeElimination>();
    REQUIRE(pm.run(graph));

    auto const report = pm.explain();
    INFO(report);
    CHECK(report.find("[structural-algebraic]") != std::string::npos);
}
