//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Comm/Platform.hpp>
#include <Einsums/ComputeGraph/Graph.hpp>
#include <Einsums/ComputeGraph/HardwareProfile.hpp>
#include <Einsums/ComputeGraph/Optimizer.hpp>
#include <Einsums/ComputeGraph/Passes/CSE.hpp>
#include <Einsums/ComputeGraph/Passes/CommunicationElimination.hpp>
#include <Einsums/ComputeGraph/Passes/CommunicationInsertion.hpp>
#include <Einsums/ComputeGraph/Passes/CommunicationScheduling.hpp>
#include <Einsums/ComputeGraph/Passes/ConstantFolding.hpp>
#include <Einsums/ComputeGraph/Passes/ContractionPlanning.hpp>
#include <Einsums/ComputeGraph/Passes/DeadNodeElimination.hpp>
#include <Einsums/ComputeGraph/Passes/DistributionPlanning.hpp>
#include <Einsums/ComputeGraph/Passes/ElementWiseFusion.hpp>
#include <Einsums/ComputeGraph/Passes/FreeInsertion.hpp>
#include <Einsums/ComputeGraph/Passes/GEMMBatching.hpp>
#include <Einsums/ComputeGraph/Passes/GPUDiagnostics.hpp>
#include <Einsums/ComputeGraph/Passes/GPUPlacement.hpp>
#include <Einsums/ComputeGraph/Passes/IOPrefetch.hpp>
#include <Einsums/ComputeGraph/Passes/InplaceOptimization.hpp>
#include <Einsums/ComputeGraph/Passes/InputSlicing.hpp>
#include <Einsums/ComputeGraph/Passes/LoopInvariantHoisting.hpp>
#include <Einsums/ComputeGraph/Passes/Materialization.hpp>
#include <Einsums/ComputeGraph/Passes/MemoryPlanning.hpp>
#include <Einsums/ComputeGraph/Passes/PermuteFusion.hpp>
#include <Einsums/ComputeGraph/Passes/Reorder.hpp>
#include <Einsums/ComputeGraph/Passes/SUMMAExpansion.hpp>
#include <Einsums/ComputeGraph/Passes/ScaleAbsorption.hpp>
#include <Einsums/ComputeGraph/Passes/StreamAssignment.hpp>
#include <Einsums/ComputeGraph/Passes/SymmetryPropagation.hpp>
#include <Einsums/ComputeGraph/Passes/TransferElimination.hpp>
#include <Einsums/ComputeGraph/Passes/TransferInsertion.hpp>
#include <Einsums/Config/Types.hpp>
#include <Einsums/GPU/Platform.hpp>
#include <Einsums/Logging.hpp>
#include <Einsums/Profile.hpp>

#include <chrono>
#include <set>
#include <sstream>

namespace einsums::compute_graph {

void OptimizerPass::report(int level, std::string_view message) const {
    if (_verbosity >= level) {
        fmt::print(stderr, "[{}] {}\n", name(), message);
    }
}

namespace {

/// Parse a comma-separated list of pass names into a set.
std::set<std::string> parse_disabled_passes() {
    std::set<std::string> disabled;
    try {
        auto &gc           = GlobalConfigMap::get_singleton();
        auto  disabled_str = gc.get_string("pass-disable", "");
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

bool get_pass_flag(std::string const &key, bool default_val) {
    try {
        auto &gc = GlobalConfigMap::get_singleton();
        return gc.get_bool(key, default_val);
    } catch (...) {
        return default_val;
    }
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

} // namespace

bool PassManager::run(Graph &graph) {
    profile::Profiler::instance().push(fmt::format("PassManager::run({})", graph.name()));

    auto       disabled = parse_disabled_passes();
    bool const analyze  = get_pass_flag("pass-analyze", false);
    bool const verbose  = get_pass_flag("pass-verbose", false);

    bool any_modified = false;
    for (auto &pass : _passes) {
        // Check if this pass is disabled
        if (disabled.count(pass->name())) {
            EINSUMS_LOG_INFO("PassManager: skipping disabled pass '{}'", pass->name());
            continue;
        }

        profile::Profiler::instance().push(fmt::format("pass:{}", pass->name()));

        size_t nodes_before = graph.num_nodes();
        auto   t0           = std::chrono::high_resolution_clock::now();

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

            // Restore original graph state
            graph.nodes() = std::move(saved_nodes);
            graph.topological_sort();
        } else {
            bool const modified = run_pass_recursive(*pass, graph);
            auto       t1       = std::chrono::high_resolution_clock::now();
            double     ms       = std::chrono::duration<double, std::milli>(t1 - t0).count();

            if (modified) {
                any_modified = true;
            }
            profile::annotate("modified", modified ? "true" : "false");

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

        profile::Profiler::instance().pop();
    }

    profile::Profiler::instance().pop();
    return any_modified;
}

PassManager PassManager::create_default() {
    PassManager pm;
    pm.populate_default();
    return pm;
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
    // Harvest the statistics the passes already expose into one report.
    // dynamic_cast per known pass type: passes without counters contribute
    // nothing; unknown/user passes are simply not summarized.
    std::string out;
    auto        line = [&out](std::string text) {
        out += "  - ";
        out += text;
        out += '\n';
    };

    for (auto const &p : _passes) {
        if (auto const *sa = dynamic_cast<passes::ScaleAbsorption const *>(p.get()); sa && sa->num_absorbed() > 0) {
            line(fmt::format("ScaleAbsorption: removed {} dead scale(s)", sa->num_absorbed()));
        } else if (auto const *pf = dynamic_cast<passes::PermuteFusion const *>(p.get()); pf && pf->num_rewrites() > 0) {
            line(fmt::format("PermuteFusion: folded {} permute(s) into contractions", pf->num_rewrites()));
        } else if (auto const *dne = dynamic_cast<passes::DeadNodeElimination const *>(p.get()); dne && dne->num_eliminated() > 0) {
            line(fmt::format("DeadNodeElimination: removed {} dead node(s)", dne->num_eliminated()));
        } else if (auto const *ewf = dynamic_cast<passes::ElementWiseFusion const *>(p.get()); ewf && ewf->num_fused() > 0) {
            line(fmt::format("ElementWiseFusion: fused {} elementwise pair(s)", ewf->num_fused()));
        } else if (auto const *lih = dynamic_cast<passes::LoopInvariantHoisting const *>(p.get()); lih && lih->num_hoisted() > 0) {
            line(fmt::format("LoopInvariantHoisting: hoisted {} node(s) out of loops", lih->num_hoisted()));
        } else if (auto const *cp = dynamic_cast<passes::ContractionPlanning const *>(p.get()); cp && !cp->chain_reports().empty()) {
            line(fmt::format("ContractionPlanning: restructured {} of {} GEMM chain(s), {} intermediate(s)", cp->chains_restructured(),
                             cp->chain_reports().size(), cp->intermediates_created()));
            for (auto const &rep : cp->chain_reports()) {
                if (rep.speedup > 1.05) {
                    line(fmt::format("    chain of {}: est. {:.1f}us -> {:.1f}us ({:.2f}x)", rep.chain_length, rep.original_time_us,
                                     rep.optimal_time_us, rep.speedup));
                }
            }
        } else if (auto const *gb = dynamic_cast<passes::GEMMBatching const *>(p.get());
                   gb && (gb->num_batches() > 0 || gb->num_gate_skipped() > 0)) {
            line(fmt::format("GEMMBatching: {} batch(es) absorbing {} GEMM(s); {} group(s) left parallel by the profitability gate",
                             gb->num_batches(), gb->total_batched(), gb->num_gate_skipped()));
        } else if (auto const *ip = dynamic_cast<passes::InplaceOptimization const *>(p.get());
                   ip && (ip->num_merged() > 0 || ip->num_candidates() > 0)) {
            line(fmt::format("InplaceOptimization: merged {} buffer(s) into dying inputs ({} candidate(s))", ip->num_merged(),
                             ip->num_candidates()));
        } else if (auto const *fi = dynamic_cast<passes::FreeInsertion const *>(p.get()); fi && fi->num_freed() > 0) {
            line(fmt::format("FreeInsertion: {} intermediate(s) freed after their last consumer", fi->num_freed()));
        } else if (auto const *mp = dynamic_cast<passes::MemoryPlanning const *>(p.get()); mp && mp->total_memory() > 0) {
            line(fmt::format("MemoryPlanning: peak {:.2f} MB of {:.2f} MB total", static_cast<double>(mp->peak_memory()) / 1048576.0,
                             static_cast<double>(mp->total_memory()) / 1048576.0));
            if (mp->num_planned() > 0) {
                line(fmt::format("    arena: {:.2f} MB hosting {} intermediate(s) ({:.2f} MB of buffers)",
                                 static_cast<double>(mp->planned_arena_bytes()) / 1048576.0, mp->num_planned(),
                                 static_cast<double>(mp->planned_tensor_bytes()) / 1048576.0));
            }
        }
    }

    if (out.empty()) {
        out = "  (no optimizations applied)\n";
    }
    return out;
}

void PassManager::populate_default() {
    auto &pm = *this;

    // Detect hardware once and share the profile across cost-model passes.
    auto profile = HardwareProfile::detect_default();

    // Graph-transforming passes (reduce node count first).
    // Order matters: PermuteFusion runs before CSE/DNE so duplicate
    // permute→einsum patterns collapse into the same fused node, and
    // before Materialization / GPU placement so those passes don't
    // allocate / place tensors that are about to be removed.
    pm.add<passes::ConstantFolding>();
    pm.add<passes::ScaleAbsorption>();
    pm.add<passes::PermuteFusion>();
    pm.add<passes::CSE>();
    pm.add<passes::DeadNodeElimination>();
    pm.add<passes::ElementWiseFusion>();
    pm.add<passes::LoopInvariantHoisting>();

    // Chain restructuring belongs in the planning phase: it rewrites GEMM
    // chains using the shared cost model and declares DEFERRED intermediates,
    // so it must precede GEMMBatching/Reorder (which schedule the final
    // node set) and DistributionPlanning/Materialization (which size and
    // allocate the intermediates it introduces). It used to run dead-last,
    // where its restructured nodes got no placement or memory management and
    // its eagerly-created intermediates leaked for the graph's lifetime.
    pm.add<passes::ContractionPlanning>(profile);
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
    pm.add<passes::GEMMBatching>(profile);
    pm.add<passes::Reorder>();
    pm.add<passes::IOPrefetch>();

    // Deferred allocation: decide distribution, then materialize.
    // Runs before GPU passes so GPUPlacement sees correct tensor sizes.
    pm.add<passes::DistributionPlanning>();
    pm.add<passes::Materialization>();

    // Symmetry propagation: now that tensors exist, infer descriptors on
    // graph-owned intermediates and push them to the backing tensors so
    // the rank-2 BLAS dispatch (Phase 2) fires at graph.execute(). Runs
    // here (after Materialization, before GPU placement) so downstream
    // passes and executions see the inferred symmetry.
    pm.add<passes::SymmetryPropagation>();

    // GPU passes, only included when a GPU backend (or mock) is available.
    // GPUPlacement uses the shared HardwareProfile for its cost model.
    if constexpr (gpu::has_gpu || gpu::is_mock) {
        pm.add<passes::GPUPlacement>(profile);
        pm.add<passes::TransferInsertion>();
        pm.add<passes::TransferElimination>();
        pm.add<passes::GPUDiagnostics>();
        pm.add<passes::StreamAssignment>();
    }

    // Distributed communication passes (when MPI or mock is available).
    if constexpr (comm::has_mpi || comm::is_mock) {
        pm.add<passes::InputSlicing>();
        pm.add<passes::SUMMAExpansion>();
        pm.add<passes::CommunicationInsertion>();
        pm.add<passes::CommunicationElimination>();
        pm.add<passes::CommunicationScheduling>();
    }

    // Merge elementwise outputs into dying inputs BEFORE the liveness-based
    // passes: each merge removes a buffer, shortening the intervals
    // FreeInsertion and MemoryPlanning then work with.
    pm.add<passes::InplaceOptimization>();

    // Free intermediates after their last consumer to reduce peak memory.
    pm.add<passes::FreeInsertion>();

    // Analysis and planning passes (examine final graph).
    pm.add<passes::MemoryPlanning>();
}

} // namespace einsums::compute_graph
