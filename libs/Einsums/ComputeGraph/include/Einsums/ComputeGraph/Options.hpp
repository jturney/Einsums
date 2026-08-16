//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/CommandLine/Get.hpp>
#include <Einsums/Config/Namespace.hpp>

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
