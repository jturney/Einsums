//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Options/Get.hpp>

/*
 * The graph optimizer's and executor's options, declared where they are read.
 *
 * Per-pass flags built from a pass name at run time stay on the dynamic API;
 * everything a reader names literally lives here.
 */

EINSUMS_NAMESPACE_BEGIN(option)

/// Optimization passes to skip, by name, comma separated.
inline constinit cl::ConfigOption<std::string> PassDisable =
    cl::config_opt<std::string>("einsums:pass:disable", "Comma-separated list of optimization pass names to skip (e.g. CSE,Reorder)",
                                "ComputeGraph Passes", "", "PASSES");

/// Report what every pass would do without letting it do anything.
inline constinit cl::ConfigOption<bool> PassAnalyze =
    cl::config_flag("einsums:pass:analyze", "Run all passes in analysis-only mode (report findings but do not modify the graph)",
                    "ComputeGraph Passes", false);

/// Log node count and timing on either side of each pass.
inline constinit cl::ConfigOption<bool> PassVerbose = cl::config_flag(
    "einsums:pass:verbose", "Log node count and timing before and after each optimization pass", "ComputeGraph Passes", false);

/// How much the optimization passes narrate about what they did and declined.
///
/// 0 silent, 1 a summary line per pass, 2 each modification and each declined
/// candidate, 3 the per-candidate detail behind a decline. Only a level nobody
/// set through `PassManager::set_verbosity` takes this value, so a program that
/// chooses its own level is unaffected.
///
/// This is the only way to see a thread plan. `Graph::plan_threads()` builds
/// its planner directly rather than through a pass manager, so without a level
/// from here every report the planner makes is unreachable, and whether it
/// widened anything, or declined and why, cannot be observed from a run.
inline constinit cl::ConfigOption<std::int64_t> PassVerbosity =
    cl::config_opt<std::int64_t>("einsums:pass:verbosity",
                                 "How much the optimization passes report about what they changed and what they declined: 0 silent, 1 a "
                                 "summary per pass, 2 each modification and decline, 3 the detail behind each decline",
                                 "ComputeGraph Passes", 0, "LEVEL", cl::Range{0, 3});

/// Dump the algebra a region rewrite raised, before and after it rewrote.
///
/// An optimizer that rewrites mathematics will eventually produce a wrong
/// number, and the bug history in this module is the evidence rather than a
/// worry: the baked-lambda redirect, the full-cover alias miss, the Kahn FIFO
/// hoist. Each produced a plausible graph and a wrong result. A diffable
/// before-and-after of the ALGEBRA is what turns such a case into a readable bug
/// report; a diff of two node-list dumps is not the same thing, because the
/// node list is the encoding and the algebra is the claim.
///
/// Goes to stderr at pass verbosity 2 and up, and is also kept on the pass, so a
/// test asserts on it rather than scraping a stream.
inline constinit cl::ConfigOption<bool> GraphDumpRegions =
    cl::config_flag("einsums:graph:dump-regions",
                    "Dump each region rewrite's expression before and after the rewrite, in the algebraic form rather than as a node "
                    "list. Costs a rendering per region and is the first thing to turn on when a rewrite produces a wrong number",
                    "ComputeGraph Passes", false);

/// Break a grouped batched GEMM into one profiler zone per shape class.
/// Deliberately CHANGES how the GEMM runs, because a per-group breakdown of one
/// parallel loop is not obtainable any other way. See Detail/GroupedBatchedGemm.hpp.
inline constinit cl::ConfigOption<bool> GraphProfileGroups =
    cl::config_flag("einsums:graph:profile-groups",
                    "Break a grouped batched GEMM into one profiler zone per shape class. This runs the SLOWER unfused form, so read it "
                    "for where the arithmetic is, not for what the node costs",
                    "ComputeGraph Passes", false);

/// Check, before each level-scheduled replay, that no level holds two nodes
/// touching overlapping storage. On in debug builds; this is how a release
/// build turns it on, which is what a nondeterministic result wants.
inline constinit cl::ConfigOption<bool> GraphVerifyLevels =
    cl::config_flag("einsums:graph:verify-levels",
                    "Before each level-scheduled replay, check that no execution level holds two nodes touching overlapping storage. "
                    "Costs a second pass over every operand and can report a conflict it cannot disprove",
                    "ComputeGraph Passes",
#if defined(EINSUMS_DEBUG)
                    true
#else
                    false
#endif
    );

/// A calibrated hardware profile to build the cost model from.
///
/// Read once per `CostModel::detect_default()`, which every cost-model pass
/// calls while a pass manager is being populated - long after the option
/// system has parsed. An empty value, or a file that will not load, falls back
/// to the built-in device table with a warning: the profile shapes
/// optimization choices, never correctness.
inline constinit cl::ConfigOption<std::string> HardwareProfile =
    cl::config_opt<std::string>("einsums:hardware:profile",
                                "Path to a calibrated hardware profile JSON to build the cost model from, in place of the built-in "
                                "device table (see the calibrate_hardware tool)",
                                "Hardware", "", "PATH");

/// Keep every node on the host, making GPUPlacement a no-op.
inline constinit cl::ConfigOption<bool> GpuDisable =
    cl::config_flag("einsums:gpu:disable", "Keep every node on the host (GPUPlacement becomes a no-op)", "GPU", false);

EINSUMS_NAMESPACE_END(option)

EINSUMS_NAMESPACE_BEGIN()

/**
 * @brief Give the compute graph's options their command-line presence. Idempotent.
 */
EINSUMS_EXPORT int register_Einsums_ComputeGraph_options();

namespace detail {
[[maybe_unused]] static int const register_options_Einsums_ComputeGraph = register_Einsums_ComputeGraph_options();
}

EINSUMS_NAMESPACE_END()
