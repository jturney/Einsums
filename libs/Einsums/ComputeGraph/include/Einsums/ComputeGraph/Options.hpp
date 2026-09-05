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

/// Let the structural-algebraic phase run its SEARCH passes.
///
/// Off by default, and the default is the point rather than caution. Every other pass in that
/// phase is a recognizer: it walks the graph once, and its runtime is a function of the node
/// count. A search pass's runtime is a function of how many candidates the graph offers, which
/// nobody can predict from the outside, and the whole reason the saved IR exists is so that
/// search happens ONCE, offline or at capture, rather than every time a script is re-run.
///
/// So the interactive default is off and the saved-graph workflow turns it on deliberately. A
/// caller who wants it for one pipeline sets it there rather than here; the option is the
/// process-wide statement.
inline constinit cl::ConfigOption<bool> GraphStructuralSearch =
    cl::config_flag("einsums:graph:structural-search",
                    "Let the structural-algebraic phase run its search passes. Off by default because a search pass's runtime is a "
                    "function of how many candidates a graph offers rather than of how large it is",
                    "ComputeGraph Passes", false);

/// Keep a search pass's result and replay it for a structurally identical graph.
///
/// On by default, because the cost is a content hash of a graph a search was about to be run over
/// and the saving is the search. It is an option rather than a constant for the reason every
/// rewrite knob is one: a wrong number under an optimizer is bisected by turning things off, and a
/// cache is the first thing to suspect when two supposedly identical graphs get different plans.
inline constinit cl::ConfigOption<bool> GraphFactorizationCache =
    cl::config_flag("einsums:graph:factorization-cache",
                    "Keep each search pass's chosen plan in memory and replay it when a structurally identical graph comes back, so a "
                    "pipeline whose stages present the same program searches once",
                    "ComputeGraph Passes", true);

/// The largest number of factors a term may have before a search pass declines it.
///
/// A property of the PROGRAM a caller brings rather than of the pass, which is why it is an
/// option and not a constant: the density-fitted opposite-spin energy is nine leaves, the grid
/// fitted one is fourteen, and an amplitude residual term is more. A cap that fitted the first
/// of those declines the other two silently, and a caller has no way to lift it without editing
/// the pass.
///
/// The cost is stated rather than hidden. The per-term program is `3^N` in the factor count, so
/// fourteen leaves is a few million subset splits and runs in seconds, and twenty is over three
/// billion and does not finish unaided. A cap above the mid-teens is therefore a request to run
/// under `einsums:graph:optimizer-budget` and take the best tree found when it expires, which is
/// what that budget was built to do. Nothing here changes the search's complexity.
inline constinit cl::ConfigOption<std::int64_t> GraphFactorizationMaxFactors = cl::config_opt<std::int64_t>(
    "einsums:graph:factorization-max-factors",
    "The largest number of factors a term may have before MultiTermFactorization declines it. The per-term search is 3^N in this number, "
    "so a value above the mid-teens is a request to run under the optimizer budget and keep the best tree found",
    "ComputeGraph Passes", 14, "COUNT", cl::RangeBetween<std::int64_t>(2, 32));

/// The relative accuracy a Laplace-transform quadrature is built to.
///
/// A TOLERANCE and not a point count, because a tolerance is what composes with the accuracy
/// budget and what an approximation record can state, while a count is a means: the same count
/// over a wider spectral range is a different approximation and says nothing about what the
/// result is worth. The count is derived from this and from the range the bound energies have,
/// and it reaches the setup node's descriptor so a saved graph means the same thing wherever
/// it is loaded.
inline constinit cl::ConfigOption<double> GraphLaplaceEpsilon = cl::config_opt<double>(
    "einsums:graph:laplace-epsilon",
    "Target relative accuracy for the quadrature LaplaceTransform substitutes for a tagged energy denominator. Tightening it grows the "
    "point count and the emitted arithmetic with it",
    "ComputeGraph Passes", 1.0e-6, "EPS");

/// The relative accuracy a tensor-hypercontraction fit is asserted to have.
///
/// A TOLERANCE, spelled the way Part 8.3 asks every per-lossy-pass knob to be spelled, and it
/// reaches the provider rather than living in a pass, so a graph saved after a grid fit means
/// one thing wherever it is loaded. What it is NOT is a measurement: the error of a grid fit
/// against the exact four-index tensor cannot be computed without the tensor the fit exists to
/// avoid forming. What is measured instead, per bind, is the least-squares residual against the
/// three-index tensor the fit was fitted from; see ThcFactorization::residual_param_name.
inline constinit cl::ConfigOption<double> GraphThcEpsilon = cl::config_opt<double>(
    "einsums:graph:thc-epsilon",
    "Relative error a ThcFactorization asserts for the grid fit it substitutes for a tagged four-index tensor, when the caller states "
    "none of its own",
    "ComputeGraph Passes", 1.0e-4, "EPS");

/// How long a search pass may run, in milliseconds. Zero means unlimited.
///
/// The number is a starting point rather than a measurement, which is what an option is for. What
/// is NOT arbitrary is that a search pass which exhausts it keeps the best candidate it had found
/// and reports being cut off, so exhausting the budget costs optimization rather than
/// correctness.
inline constinit cl::ConfigOption<std::int64_t> GraphOptimizerBudget = cl::config_opt<std::int64_t>(
    "einsums:graph:optimizer-budget",
    "Wall-clock allowance in milliseconds for each search pass. A pass that exhausts it keeps the best candidate it found and reports "
    "that it was cut off. Zero means unlimited",
    "ComputeGraph Passes", 1000);

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
