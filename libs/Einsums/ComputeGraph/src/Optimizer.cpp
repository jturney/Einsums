//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Comm/Platform.hpp>
#include <Einsums/ComputeGraph/CostModel.hpp>
#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/Optimizer.hpp>
#include <Einsums/ComputeGraph/Options.hpp>
#include <Einsums/ComputeGraph/Passes/CSE.hpp>
#include <Einsums/ComputeGraph/Passes/CommunicationElimination.hpp>
#include <Einsums/ComputeGraph/Passes/CommunicationInsertion.hpp>
#include <Einsums/ComputeGraph/Passes/CommunicationScheduling.hpp>
#include <Einsums/ComputeGraph/Passes/ConstantFolding.hpp>
#include <Einsums/ComputeGraph/Passes/ContractionPlanning.hpp>
#include <Einsums/ComputeGraph/Passes/CrossSpaceValidation.hpp>
#include <Einsums/ComputeGraph/Passes/DeadNodeElimination.hpp>
#include <Einsums/ComputeGraph/Passes/DeltaElimination.hpp>
#include <Einsums/ComputeGraph/Passes/DistributionPlanning.hpp>
#include <Einsums/ComputeGraph/Passes/DistributiveFactoring.hpp>
#include <Einsums/ComputeGraph/Passes/ElementWiseFusion.hpp>
#include <Einsums/ComputeGraph/Passes/FreeInsertion.hpp>
#include <Einsums/ComputeGraph/Passes/GEMMBatching.hpp>
#include <Einsums/ComputeGraph/Passes/GPUDiagnostics.hpp>
#include <Einsums/ComputeGraph/Passes/GPUPlacement.hpp>
#include <Einsums/ComputeGraph/Passes/IOPrefetch.hpp>
#include <Einsums/ComputeGraph/Passes/InplaceOptimization.hpp>
#include <Einsums/ComputeGraph/Passes/InputSlicing.hpp>
#include <Einsums/ComputeGraph/Passes/LinearCombinationContractionFolding.hpp>
#include <Einsums/ComputeGraph/Passes/LoopInvariantHoisting.hpp>
#include <Einsums/ComputeGraph/Passes/Materialization.hpp>
#include <Einsums/ComputeGraph/Passes/MemoryPlanning.hpp>
#include <Einsums/ComputeGraph/Passes/PermuteFusion.hpp>
#include <Einsums/ComputeGraph/Passes/ProvenancePropagation.hpp>
#include <Einsums/ComputeGraph/Passes/Reorder.hpp>
#include <Einsums/ComputeGraph/Passes/SUMMAExpansion.hpp>
#include <Einsums/ComputeGraph/Passes/ScaleAbsorption.hpp>
#include <Einsums/ComputeGraph/Passes/ScalingAnalysis.hpp>
#include <Einsums/ComputeGraph/Passes/ScratchPrivatization.hpp>
#include <Einsums/ComputeGraph/Passes/SpacePropagation.hpp>
#include <Einsums/ComputeGraph/Passes/StreamAssignment.hpp>
#include <Einsums/ComputeGraph/Passes/StreamContractionFusion.hpp>
#include <Einsums/ComputeGraph/Passes/SymmetrizedAccumulation.hpp>
#include <Einsums/ComputeGraph/Passes/SymmetryPropagation.hpp>
#include <Einsums/ComputeGraph/Passes/TiledExpansion.hpp>
#include <Einsums/ComputeGraph/Passes/TransferElimination.hpp>
#include <Einsums/ComputeGraph/Passes/TransferInsertion.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/GPU/Platform.hpp>
#include <Einsums/Logging.hpp>
#include <Einsums/Profile.hpp>

#include <algorithm>
#include <chrono>
#include <set>
#include <sstream>
#include <utility>

EINSUMS_NAMESPACE_BEGIN(compute_graph)

std::string_view pass_phase_name(PassPhase phase) {
    switch (phase) {
    case PassPhase::Analysis:
        return "analysis";
    case PassPhase::StructuralAlgebraic:
        return "structural-algebraic";
    case PassPhase::StructuralResource:
        return "structural-resource";
    case PassPhase::Tuning:
        return "tuning";
    case PassPhase::Diagnostic:
        return "diagnostic";
    }
    return "unknown";
}

void OptimizerPass::report(int level, std::string_view message) const {
    if (_verbosity >= level) {
        fmt::print(stderr, "[{}] {}\n", name(), message);
    }
}

void OptimizerPass::note_skip(std::string_view reason, std::string_view detail) const {
    auto hit = std::ranges::find_if(_skips, [&](auto const &e) { return e.first == reason; });
    if (hit == _skips.end()) {
        _skips.emplace_back(std::string(reason), 1);
    } else {
        hit->second++;
    }

    if (_verbosity >= 3) {
        if (detail.empty()) {
            report(3, fmt::format("declined: {}", reason));
        } else {
            report(3, fmt::format("declined: {} ({})", reason, detail));
        }
    }
}

bool OptimizerPass::approximate(Graph &graph, ApproximationRecord record) const {
    if (std::string reason = graph.can_approximate(record); !reason.empty()) {
        // The reason is already a full sentence naming this pass, so it goes in as the
        // aggregation key rather than being wrapped in one. Two different refusals of the
        // same pass are different keys, which is right: "cannot be bounded" and "over
        // budget" call for different fixes and a tally that merged them would say neither.
        note_skip("declined a lossy rewrite", reason);
        return false;
    }
    graph.note_approximation(std::move(record));
    return true;
}

std::vector<std::pair<std::string, std::size_t>> OptimizerPass::skip_reasons() const {
    auto out = _skips;
    // Most-frequent first: the dominant reason a pass stayed quiet is the one
    // worth reading, and a stable order keeps the report diffable.
    std::ranges::stable_sort(out, [](auto const &a, auto const &b) { return a.second > b.second; });
    return out;
}

namespace {

/// Parse a comma-separated list of pass names into a set.
std::set<std::string> parse_disabled_passes() {
    std::set<std::string> disabled;
    try {
        auto const disabled_str = config::get(option::PassDisable);
        if (!disabled_str.empty()) {
            std::istringstream ss(disabled_str);
            std::string        token;
            while (std::getline(ss, token, ',')) {
                // Trim whitespace
                auto start = token.find_first_not_of(' ');
                auto end   = token.find_last_not_of(' ');
                if (start != std::string::npos) {
                    disabled.insert(token.substr(start, end - start + 1));
                }
            }
        }
    } catch (...) { // NOLINT
    }
    return disabled;
}

} // namespace

namespace {

/// Run a single pass on @p graph and, when the pass opts in via
/// ``recurse_into_subgraphs()``, on every descendant loop body /
/// conditional branch in post-order (children before re-running on parent
/// is not required, passes either rewrite a level in isolation or hoist
/// from children into the parent in a single ``run()`` call on the
/// parent). Returns ``true`` if any invocation of @p pass modified its
/// graph.
bool run_pass_recursive(OptimizerPass &pass, Graph &graph) {
    bool modified = pass.run(graph);
    if (pass.recurse_into_subgraphs()) {
        graph.for_each_subgraph([&](Graph &sub) {
            if (run_pass_recursive(pass, sub)) {
                modified = true;
            }
        });
    }
    return modified;
}

/// Node-position hazard guard. Position is program order in this IR: the hazard
/// scan in topological_sort treats a read that appears before a write as a
/// legitimate WAR, so a pass that appends or moves a WRITER past a surviving
/// reader silently changes which value that reader observes - the reader now
/// legally runs first and sees the tensor's initial contents instead of the
/// written result (GEMMBatching did exactly this; LCCF was bitten by the same
/// trap earlier).
///
/// The invariant checked: for every (reader node, input tensor) pair that
/// exists both before and after a pass, "the read observes an in-graph
/// writer" may never flip to "the read observes the initial contents".
/// Pairs created by the pass (new nodes, redirected inputs) have no baseline
/// and are skipped, as is the reverse transition (a pass may legitimately
/// insert a writer, e.g. Materialization's Initialize, in front of a read of
/// a deferred tensor). Top-level graph only; sub-graph bodies are not walked.
std::unordered_map<NodeId, std::unordered_map<TensorId, bool>> observed_writes(Graph const &graph) {
    std::unordered_map<NodeId, std::unordered_map<TensorId, bool>> seen;
    std::unordered_set<TensorId>                                   written;
    for (auto const &node : graph.nodes()) {
        for (auto const tid : node.inputs) {
            seen[node.id][tid] = written.count(graph.resolve_alias(tid)) != 0;
        }
        for (auto const tid : node.outputs) {
            written.insert(graph.resolve_alias(tid));
        }
    }
    return seen;
}

void check_observed_writes(Graph const &graph, std::unordered_map<NodeId, std::unordered_map<TensorId, bool>> const &before,
                           std::string const &pass_name, std::vector<std::pair<NodeId, TensorId>> const &compensated) {
    auto const after = observed_writes(graph);
    for (auto const &[nid, per_tid] : after) {
        auto const nit = before.find(nid);
        if (nit == before.end())
            continue; // node created by the pass: no baseline
        for (auto const &[tid, sees_writer] : per_tid) {
            auto const tit = nit->second.find(tid);
            if (tit == nit->second.end())
                continue; // input redirected by the pass: no baseline
            if (tit->second && !sees_writer) {
                // The pass may declare a read whose writer it removed while
                // compensating the reader (e.g. folding a deleted scale into
                // this einsum's prefactor); the structural guard is waived for
                // exactly those (reader, tensor) pairs.
                if (std::ranges::find(compensated, std::pair{nid, tid}) != compensated.end()) {
                    continue;
                }
                std::string node_label  = "?";
                std::string tensor_name = "?";
                for (auto const &node : graph.nodes()) {
                    if (node.id == nid) {
                        node_label = node.label;
                        break;
                    }
                }
                if (auto const hit = graph.tensors_map().find(tid); hit != graph.tensors_map().end()) {
                    tensor_name = hit->second.name;
                }
                EINSUMS_THROW_EXCEPTION(
                    std::logic_error,
                    "Graph '{}': pass '{}' broke program order: node '{}' (id={}) read tensor '{}' (id={}) from an in-graph "
                    "writer before the pass, but the writer now sits AFTER the reader (or was removed), so the read would "
                    "observe the tensor's initial contents. Rewrite passes must place replacement writers at the first "
                    "replaced node's position, never append them (see GEMMBatching/LCCF).",
                    graph.name(), pass_name, node_label, nid, tensor_name, tid);
            }
        }
    }
}

/// A read-only phase that moved ``Graph::structure_version`` broke its own
/// contract, and the diagnosis has to name the pass rather than surface later as
/// a stale annotation or a re-planned schedule nobody asked for.
void check_read_only_phase(Graph const &graph, OptimizerPass const &pass, std::uint64_t before) {
    auto const phase = pass.phase();
    if (phase != PassPhase::Analysis && phase != PassPhase::Diagnostic) {
        return;
    }
    if (graph.structure_version() == before) {
        return;
    }
    EINSUMS_THROW_EXCEPTION(std::logic_error,
                            "Graph '{}': pass '{}' declares phase '{}', which may only write annotations, but it changed the "
                            "graph's structure (structure_version {} -> {}). Either the pass belongs in a structural phase or "
                            "the node-set change is a bug; a read-only phase is re-run after structural passes, so a rewrite "
                            "here would be applied more than once.",
                            graph.name(), pass.name(), pass_phase_name(phase), before, graph.structure_version());
}

} // namespace

PassManager &PassManager::disable(std::string pass_name) {
    _switches[std::move(pass_name)] = false;
    return *this;
}

PassManager &PassManager::enable(std::string pass_name) {
    _switches[std::move(pass_name)] = true;
    return *this;
}

bool PassManager::run(Graph &graph) {
    LabeledSection("PassManager::run({})", graph.name());

    // Two sources of "skip this pass", and the more specific one wins. The option
    // is what a user types with no rebuild; a switch is what this program said
    // about this pipeline. A program that called enable() and silently did not get
    // the pass would be the surprise the whole surface exists to remove.
    auto const from_option = parse_disabled_passes();
    auto const is_disabled = [&](std::string const &pass_name) {
        if (auto const hit = _switches.find(pass_name); hit != _switches.end()) {
            return !hit->second;
        }
        return from_option.count(pass_name) != 0;
    };

    // Every name either source mentions, so one that matches no pass in this
    // pipeline can be reported rather than doing nothing in silence.
    std::set<std::string> unmatched(from_option.begin(), from_option.end());
    for (auto const &[pass_name, on] : _switches) {
        unmatched.insert(pass_name);
    }
    for (auto const &pass : _passes) {
        unmatched.erase(pass->name());
    }
    _last_unmatched.assign(unmatched.begin(), unmatched.end());
    _last_skipped.clear();
    for (auto const &pass_name : _last_unmatched) {
        EINSUMS_LOG_WARN("PassManager: pass switch '{}' matches no pass in this pipeline, so it does nothing; check the "
                         "spelling, or whether this build has that pass",
                         pass_name);
    }

    bool const analyze = config::get(option::PassAnalyze);
    bool const verbose = config::get(option::PassVerbose);

    // A level nobody set programmatically comes from the option, so a report
    // can be turned on from a command line without editing the program. A
    // caller that chose its own level keeps it.
    if (_verbosity == 0) {
        if (auto const level = static_cast<int>(config::get(option::PassVerbosity)); level > 0) {
            set_verbosity(level);
        }
    }

    bool any_modified = false;
    // Has a structural pass changed the node set since the last time the
    // analysis phase looked at it? Cleared by each analysis pass that runs, set
    // by every structural pass that reports a modification. When it is still
    // set at the end of the pipeline, the annotations describe a node set that
    // no longer exists and the analysis passes are re-run over the final one.
    bool structure_stale_for_analysis = false;
    for (auto &pass : _passes) {
        if (is_disabled(pass->name())) {
            _last_skipped.push_back(pass->name());
            EINSUMS_LOG_INFO("PassManager: skipping disabled pass '{}'", pass->name());
            if (_verbosity >= 1) {
                fmt::print(stderr, "[PassManager] {}: DISABLED\n", pass->name());
            }
            continue;
        }

        LabeledSection("pass:{}", pass->name());

        size_t nodes_before = graph.num_nodes();
        auto   t0           = std::chrono::high_resolution_clock::now();

        // Zero the pass's counters ONCE per apply. run() must not do this
        // itself: the recursive driver calls it per subgraph, so a reset there
        // would leave the getters reporting only the last subgraph visited.
        pass->reset_all_stats();

        if (analyze) {
            // Analysis-only: save node list, run pass, log results, restore.
            // Sub-graph recursion is intentionally skipped here, we only
            // want to *measure* the top-level effect, not mutate
            // descendants (we don't snapshot them).
            auto       saved_nodes  = graph.nodes();
            bool const saved_sorted = true; // will be re-sorted anyway

            bool const modified = pass->run(graph);
            auto       t1       = std::chrono::high_resolution_clock::now();
            double     ms       = std::chrono::duration<double, std::milli>(t1 - t0).count();

            EINSUMS_LOG_INFO("PassManager [analyze]: pass '{}' {} the graph ({} -> {} nodes, {:.2f} ms)", pass->name(),
                             modified ? "would modify" : "did not modify", nodes_before, graph.num_nodes(), ms);

            // Restore original graph state. Putting the old node list back is
            // itself a node-set change, so declare it: a plan built against the
            // list the pass produced does not describe this one.
            graph.nodes() = std::move(saved_nodes);
            graph.note_structural_change();
            graph.topological_sort();
        } else {
            auto const baseline         = observed_writes(graph);
            auto const structure_before = graph.structure_version();
            bool const modified         = run_pass_recursive(*pass, graph);
            auto       t1               = std::chrono::high_resolution_clock::now();
            double     ms               = std::chrono::duration<double, std::milli>(t1 - t0).count();

            check_read_only_phase(graph, *pass, structure_before);

            switch (pass->phase()) {
            case PassPhase::Analysis:
                // Annotations now describe the current node set.
                structure_stale_for_analysis = false;
                break;
            case PassPhase::StructuralAlgebraic:
                // What a save persists is this phase's output, so this is the phase whose
                // membership the provenance block is about. Recorded here rather than asked of
                // the caller, which is what it used to be and what nobody remembered to supply.
                if (modified) {
                    graph.note_structural_pass(pass->name());
                }
                structure_stale_for_analysis = structure_stale_for_analysis || modified;
                break;
            case PassPhase::StructuralResource:
                structure_stale_for_analysis = structure_stale_for_analysis || modified;
                break;
            case PassPhase::Tuning:
            case PassPhase::Diagnostic:
                break;
            }

            if (modified) {
                any_modified = true;
                check_observed_writes(graph, baseline, pass->name(), pass->compensated_reads());
            }
            ProfileAnnotate("modified", modified ? "true" : "false");

            if (verbose || modified) {
                EINSUMS_LOG_INFO("PassManager: pass '{}' {} ({} -> {} nodes, {:.2f} ms)", pass->name(), modified ? "MODIFIED" : "no change",
                                 nodes_before, graph.num_nodes(), ms);
            }
            // Per-pass summary to stderr when verbosity is enabled (independent of
            // the logger's level so `pm.set_verbosity(1)` always shows it).
            if (_verbosity >= 1) {
                fmt::print(stderr, "[PassManager] {}: {} ({} -> {} nodes, {:.2f} ms)\n", pass->name(), modified ? "MODIFIED" : "no change",
                           nodes_before, graph.num_nodes(), ms);
            }
        }
    }

    // Post-run annotation consistency. The default pipeline places the analysis
    // passes at #19-20, ahead of the GPU, distributed and tail blocks, so a
    // structural-resource pass down there leaves SymmetryPropagation's and
    // SpacePropagation's output describing a node set that has since gained
    // transfers, slices or SUMMA loops. Re-running them once at the end is the
    // smallest thing that makes "after run(), the annotations match the graph"
    // true without moving any pass.
    //
    // Deliberately NOT a reset_all_stats() re-run: the counters accumulate
    // across the two invocations, so explain() and num_inferred() report what
    // the pipeline inferred in total rather than only what the second, usually
    // idempotent, sweep added. Skipped in analyze mode, where the graph is
    // restored after every pass and there is no final node set to annotate.
    if (structure_stale_for_analysis && !analyze) {
        for (auto &pass : _passes) {
            if (pass->phase() != PassPhase::Analysis || is_disabled(pass->name())) {
                continue;
            }
            LabeledSection("reanalyze:{}", pass->name());
            auto const structure_before = graph.structure_version();
            run_pass_recursive(*pass, graph);
            check_read_only_phase(graph, *pass, structure_before);
            EINSUMS_LOG_INFO("PassManager: re-ran analysis pass '{}' after a structural change", pass->name());
            if (_verbosity >= 1) {
                fmt::print(stderr, "[PassManager] {}: re-run (structure changed after the analysis phase)\n", pass->name());
            }
        }
    }

    return any_modified;
}

PassManager PassManager::create_default() {
    PassManager pm;
    pm.populate_default();
    return pm;
}

void PassManager::populate_filtered(std::initializer_list<PassPhase> keep) {
    for (auto &pass : build_default_passes()) {
        if (std::ranges::find(keep, pass->phase()) != keep.end()) {
            add(std::move(pass));
        }
    }
}

PassManager PassManager::filtered_default(std::initializer_list<PassPhase> keep) {
    PassManager pm;
    pm.populate_filtered(keep);
    return pm;
}

void PassManager::populate_analysis() {
    populate_filtered({PassPhase::Analysis, PassPhase::Diagnostic});
}

void PassManager::populate_structural() {
    populate_filtered({PassPhase::StructuralAlgebraic});
}

void PassManager::populate_resource() {
    populate_filtered({PassPhase::StructuralResource});
}

void PassManager::populate_tuning() {
    populate_filtered({PassPhase::Tuning});
}

PassManager PassManager::analysis_pass_manager() {
    return filtered_default({PassPhase::Analysis, PassPhase::Diagnostic});
}

PassManager PassManager::structural_pass_manager() {
    return filtered_default({PassPhase::StructuralAlgebraic});
}

PassManager PassManager::resource_pass_manager() {
    return filtered_default({PassPhase::StructuralResource});
}

PassManager PassManager::tuning_pass_manager() {
    return filtered_default({PassPhase::Tuning});
}

std::vector<std::pair<std::string, PassPhase>> PassManager::phase_of_each() const {
    std::vector<std::pair<std::string, PassPhase>> out;
    out.reserve(_passes.size());
    for (auto const &p : _passes) {
        out.emplace_back(p->name(), p->phase());
    }
    return out;
}

PassManager PassManager::create_for(OptLevel level) {
    PassManager pm;
    switch (level) {
    case OptLevel::O0:
        break;
    case OptLevel::O1:
        // Cleanup cluster only: reduce node count, no restructuring, no
        // memory planning. Matches the head of populate_default().
        pm.add<passes::ConstantFolding>();
        pm.add<passes::ScaleAbsorption>();
        pm.add<passes::PermuteFusion>();
        pm.add<passes::CSE>();
        pm.add<passes::DeadNodeElimination>();
        pm.add<passes::ElementWiseFusion>();
        // Materialization is correctness-enabling, not an optimization: a
        // graph that uses declare_tensor() cannot execute without it, and
        // the execute-time "still deferred" diagnostic tells users that
        // graph.optimize() fixes the problem, at every level above O0.
        // After DNE so dead deferred tensors are not allocated.
        pm.add<passes::Materialization>();
        break;
    case OptLevel::O2:
        pm.populate_default();
        break;
    }
    return pm;
}

std::string PassManager::explain() const {
    // Each pass reports its own statistics; see OptimizerPass::explain. A pass
    // that did nothing returns nothing, so a quiet report means a quiet
    // pipeline, and a pass defined outside this library is summarized like any
    // other rather than being invisible to a type switch here.
    std::string out;
    for (auto const &p : _passes) {
        // The phase leads each line: a rewrite that is right on one machine and
        // wrong on another is a phase question first, and the label is what says
        // whether a saved graph would have kept this decision or re-derived it.
        auto const tag = fmt::format("  - [{}] ", pass_phase_name(p->phase()));
        for (auto const &entry : p->explain()) {
            out += tag;
            out += entry;
            out += '\n';
        }
    }

    bool const applied_anything = !out.empty();

    // From verbosity 2 up, follow what the pipeline DID with what it DECLINED
    // and why. A pipeline that looks inert is the case where this matters most:
    // "no optimizations applied" alone cannot distinguish "this graph was
    // already optimal" from "every candidate was rejected by one gate you could
    // have satisfied", and those call for opposite responses from the user.
    if (_verbosity >= 2) {
        std::string skipped;
        for (auto const &p : _passes) {
            auto const reasons = p->skip_reasons();
            if (reasons.empty()) {
                continue;
            }
            skipped += "  - ";
            skipped += p->name();
            skipped += " declined:\n";
            for (auto const &[reason, count] : reasons) {
                skipped += fmt::format("      {} candidate(s): {}\n", count, reason);
            }
        }
        if (!skipped.empty()) {
            if (applied_anything) {
                out += '\n';
            }
            out += "  not applied:\n";
            out += skipped;
        }
    }

    if (!applied_anything && _last_skipped.empty() && _last_unmatched.empty()) {
        out.insert(0, "  (no optimizations applied)\n");
    }

    // A switched-off pass is reported at EVERY verbosity, unlike the skip tally
    // above. "No optimizations applied" and "the pass that would have applied one
    // was switched off" look identical in a report that omits this, and the second
    // is usually a leftover environment variable from the last bisect run.
    if (!_last_skipped.empty()) {
        if (!out.empty()) {
            out += '\n';
        }
        out += "  switched off for this run:\n";
        for (auto const &pass_name : _last_skipped) {
            out += fmt::format("  - {}\n", pass_name);
        }
    }

    // A name that matched nothing is the silent-typo mode, and it is worth more
    // noise than a skip: the user asked for something that did not happen and
    // nothing else in the run will say so.
    if (!_last_unmatched.empty()) {
        if (!out.empty()) {
            out += '\n';
        }
        out += "  switch names matching no pass in this pipeline:\n";
        for (auto const &pass_name : _last_unmatched) {
            out += fmt::format("  - {} (misspelled, or a pass this build does not have)\n", pass_name);
        }
    }
    return out;
}

void PassManager::populate_default() {
    // Appends, rather than replacing: the canonical list is built once by
    // build_default_passes() so populate_default() and the phase-filtered
    // factories cannot drift apart.
    for (auto &pass : build_default_passes()) {
        add(std::move(pass));
    }
}

std::vector<std::shared_ptr<OptimizerPass>> PassManager::build_default_passes() {
    std::vector<std::shared_ptr<OptimizerPass>> list;

    // Detect hardware once and share the cost_model across cost-model passes.
    auto cost_model = CostModel::detect_default();

    // Provenance first, and it really does have to be first: a tag says what a tensor IS, and
    // DeltaElimination below cannot recognize a delta somebody permuted before contracting
    // unless the tag has already travelled across that permute. Costs one walk of the node set
    // and writes annotations only.
    list.push_back(std::make_shared<passes::ProvenancePropagation>());

    // Lowering, so it comes before everything: a tiled op is one opaque Custom
    // node that no pass below can read, and expanding it into per-tile DENSE nodes
    // is what puts those tiles in front of CSE, ContractionPlanning, GEMMBatching,
    // InplaceOptimization and MemoryPlanning. Running it after any of them would
    // just mean those passes saw the unreadable form. Self-gating: it declines
    // above its node budget, and declines rather than guessing whenever tile
    // sparsity is not decidable, so a graph with no tiled operands is untouched.
    // Shares the detected cost model with the passes below, so the densify
    // decision and their planning are made against one profile.
    list.push_back(std::make_shared<passes::TiledExpansion>(4096, -1.0, passes::Densify::Auto, passes::FuseTiles::Auto, cost_model));

    // Delta elimination ahead of the cleanup cluster, because what it leaves behind is exactly
    // what that cluster is for: dissolving an intermediate strands the Alloc and Free that
    // named it, and DeadNodeElimination three entries below removes them. Running it after
    // would leave a dead allocation in every graph it fired on.
    //
    // Self-gating, and cheaply: it declines before forming a single region when no tensor in
    // the graph is declared an identity, which is every graph that has not been annotated.
    list.push_back(std::make_shared<passes::DeltaElimination>());

    // Graph-transforming passes (reduce node count first).
    // Order matters: PermuteFusion runs before CSE/DNE so duplicate
    // permute→einsum patterns collapse into the same fused node, and
    // before Materialization / GPU placement so those passes don't
    // allocate / place tensors that are about to be removed.
    list.push_back(std::make_shared<passes::ConstantFolding>());
    list.push_back(std::make_shared<passes::ScaleAbsorption>());
    list.push_back(std::make_shared<passes::PermuteFusion>());
    list.push_back(std::make_shared<passes::CSE>());
    list.push_back(std::make_shared<passes::DeadNodeElimination>());
    // Fold the CCSD symmetrization idiom (r2 += s*(tmp + P(tmp))) before
    // ElementWiseFusion, which would otherwise compose the two axpby into one
    // executor and hide the pattern. Recurses into loop bodies (the residual).
    list.push_back(std::make_shared<passes::SymmetrizedAccumulation>());
    list.push_back(std::make_shared<passes::ElementWiseFusion>());
    // Fold transpose-paired contractions (the CCSD 2J-K idiom) into one
    // contraction against L = sum_k a_k P_k(B), and do it BEFORE
    // LoopInvariantHoisting: LCCF emits the L construction as its own node, whose
    // only input is the paired operand, so when that operand is loop-invariant --
    // the common case, an integral block from one-time setup -- LIH lifts the
    // builder out of the loop and L is built once instead of every replay. Ordered
    // after ElementWiseFusion for the same reason CSE/DNE precede it: match on a
    // deduplicated, canonical node set.
    list.push_back(std::make_shared<passes::LinearCombinationContractionFolding>());
    // Factor a shared operand out of sibling accumulating contractions, next to
    // LCCF for the same reason and with the same ordering logic: it emits the sum
    // it builds as ordinary nodes, so when the summed operands are loop-invariant
    // LoopInvariantHoisting below lifts the build out of the iteration and the sum
    // is assembled once instead of every replay. Several consumers of one sum
    // share a single build, which is what turns a hand-named quantity like CCSD's
    // tau into one tensor. Self-gating on the shared cost model: it declines when
    // the axpy chain would cost more than the contractions it saves, which is the
    // bandwidth-bound case, so it is a no-op on graphs it cannot help.
    list.push_back(std::make_shared<passes::DistributiveFactoring>(cost_model));
    list.push_back(std::make_shared<passes::LoopInvariantHoisting>());
    // After the structural rewrites above have settled and hoisting has thinned
    // the bodies: rename reused scratch onto per-generation clones so false
    // WAR/WAW chains stop serializing otherwise independent work. Before the
    // planning cluster so ContractionPlanning/GEMMBatching/Reorder see the
    // widened dependency structure, and before Materialization so the clones
    // are allocated with everything else.
    list.push_back(std::make_shared<passes::ScratchPrivatization>());

    // Chain restructuring belongs in the planning phase: it rewrites GEMM
    // chains using the shared cost model and declares DEFERRED intermediates,
    // so it must precede GEMMBatching/Reorder (which schedule the final
    // node set) and DistributionPlanning/Materialization (which size and
    // allocate the intermediates it introduces). It used to run dead-last,
    // where its restructured nodes got no placement or memory management and
    // its eagerly-created intermediates leaked for the graph's lifetime.
    list.push_back(std::make_shared<passes::ContractionPlanning>(cost_model));
    // GEMMBatching collapses groups of independent, shape-compatible
    // 2D×2D→2D einsums into a single BatchedGemm node backed by
    // blas::gemm_batch. Runs after CSE/DNE so duplicates/unused nodes
    // are already gone, and before Reorder so the scheduler sees the
    // batched node as one unit. Must stay BEFORE DistributionPlanning
    // (which reads EinsumDescriptor on every node): BatchedGemm nodes
    // aren't inspected by the distribution/GPU passes, so any einsums
    // that need those optimizations should not be batched first. A
    // future commit can gate GEMMBatching on the absence of a
    // distribution requirement.
    list.push_back(std::make_shared<passes::GEMMBatching>(cost_model));
    list.push_back(std::make_shared<passes::Reorder>());
    list.push_back(std::make_shared<passes::IOPrefetch>());

    // Deferred allocation: decide distribution, then materialize.
    // Runs before GPU passes so GPUPlacement sees correct tensor sizes.
    list.push_back(std::make_shared<passes::DistributionPlanning>());
    list.push_back(std::make_shared<passes::Materialization>());

    // Symmetry propagation: now that tensors exist, infer descriptors on
    // graph-owned intermediates and push them to the backing tensors so
    // the rank-2 BLAS dispatch (Phase 2) fires at graph.execute(). Runs
    // here (after Materialization, before GPU placement) so downstream
    // passes and executions see the inferred symmetry.
    list.push_back(std::make_shared<passes::SymmetryPropagation>());

    // Space propagation: fill in the index spaces of graph-owned intermediates
    // from the annotations their producers' operands carry, so the algebraic
    // passes and the cross-space checks see a fully annotated graph after the
    // user has annotated only the inputs. Independent of SymmetryPropagation
    // (neither reads the other's output); it sits here because it is the same
    // shape of analysis and wants the same position, after Materialization and
    // before the backend passes.
    list.push_back(std::make_shared<passes::SpacePropagation>());

    // Cross-space validation: now that every intermediate carries whatever spaces
    // could be inferred, check that no contraction letter binds a slot of one
    // space against a slot of another. Immediately after SpacePropagation
    // because that pass is what makes the check see a whole program rather than
    // its inputs, and because SpacePropagation declines a conflicting operand
    // silently by design and leaves the diagnosis here. Read-only and silent
    // unless something is wrong: the findings reach graph.explain() and
    // print_report(), never stdout.
    list.push_back(std::make_shared<passes::CrossSpaceValidation>());

    // Scaling analysis: the cost layer delivered as a user-facing report. Runs
    // after the validation so a report is not built on letters the check just
    // called wrong, and after the restructuring passes so the polynomials
    // describe the graph that will actually execute. Read-only, and a no-op
    // report on a graph with no contractions.
    list.push_back(std::make_shared<passes::ScalingAnalysis>());

    // Stream fusion: merge sibling contractions that sweep one large tensor
    // into a single storage-order pass. After Materialization (its size
    // thresholds read real dims) and SymmetryPropagation (which inspects the
    // einsums it may consume), and before GPU placement and the liveness
    // passes, which treat the fused Custom node as one unit. Declines
    // distributed operands; measured >= 1.5x on every qualifying shape
    // (thresholds gate the rest to no-ops). The shared cost_model derives the
    // output-size cap from the cache hierarchy (thread-private accumulators
    // must stay cache-resident).
    list.push_back(std::make_shared<passes::StreamContractionFusion>(cost_model));

    // GPU passes, only included when a GPU backend (or mock) is available.
    // GPUPlacement uses the shared CostModel for its cost model.
    if constexpr (gpu::has_gpu || gpu::is_mock) {
        list.push_back(std::make_shared<passes::GPUPlacement>(cost_model));
        list.push_back(std::make_shared<passes::TransferInsertion>());
        list.push_back(std::make_shared<passes::TransferElimination>());
        list.push_back(std::make_shared<passes::GPUDiagnostics>());
        list.push_back(std::make_shared<passes::StreamAssignment>());
    }

    // Distributed communication passes (when MPI or mock is available).
    if constexpr (comm::has_mpi || comm::is_mock) {
        list.push_back(std::make_shared<passes::InputSlicing>());
        list.push_back(std::make_shared<passes::SUMMAExpansion>());
        list.push_back(std::make_shared<passes::CommunicationInsertion>());
        list.push_back(std::make_shared<passes::CommunicationElimination>());
        list.push_back(std::make_shared<passes::CommunicationScheduling>());
    }

    // Merge elementwise outputs into dying inputs BEFORE the liveness-based
    // passes: each merge removes a buffer, shortening the intervals
    // FreeInsertion and MemoryPlanning then work with.
    list.push_back(std::make_shared<passes::InplaceOptimization>());

    // Free intermediates after their last consumer to reduce peak memory.
    list.push_back(std::make_shared<passes::FreeInsertion>());

    // Analysis and planning passes (examine final graph).
    list.push_back(std::make_shared<passes::MemoryPlanning>());
    return list;
}

EINSUMS_NAMESPACE_END(compute_graph)
