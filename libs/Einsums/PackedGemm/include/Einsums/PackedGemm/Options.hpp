//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Options/Get.hpp>

#include <cstdint>

/*
 * The packed engine's options, declared where the engine reads them.
 */

EINSUMS_NAMESPACE_BEGIN(option)

/// Cap, in MiB, on the temporary buffers the multi-K flatten route allocates.
///
/// The route transposes whole operands (448 MiB on ccsd's `ab-cad-dcb`) into buffers that only grow.
/// A cap transposes a chunk of K at a time instead, costing 2-12% on that shape depending on the
/// cap and precision. Zero, the default, means no cap.
inline constinit cl::ConfigOption<std::int64_t> PackedGemmFlattenBudget = cl::config_opt<std::int64_t>(
    "einsums:packed-gemm:flatten-budget", "Cap the multi-K flatten route's temporary buffers, in MiB (0 = transpose whole operands)",
    "PackedGemm", 0, "MIB", cl::RangeBetween<std::int64_t>(0, std::int64_t{1} << 20));

/// Run complex contractions on x86 through the rung's REAL tile kernel, by the 1m method.
///
/// Changes the engine, not which contractions reach the packed loops. SME always uses 1m; rungs
/// below x86-64-v2 and non-x86 rungs have no 1m route and ignore it.
inline constinit cl::ConfigOption<bool> PackedGemmComplex1m = cl::config_flag(
    "einsums:packed-gemm:complex-1m", "Run complex contractions on x86 through the real tile kernel (1m method)", "PackedGemm", false);

/// Physical cores the engine assumes share one L3, in place of the count read from sysfs.
///
/// Zero detects it (from sysfs on Linux, else 1). It sizes the per-L3 teams that share a packed B
/// panel, never the result. Tests pin it.
inline constinit cl::ConfigOption<std::int64_t> PackedGemmCoresPerL3 =
    cl::config_opt<std::int64_t>("einsums:packed-gemm:cores-per-l3", "Physical cores sharing one L3 cache (0 = detect)", "PackedGemm", 0,
                                 "N", cl::RangeBetween<std::int64_t>(0, 4096));

/// Budget, in KiB, for the block-GEMM strategy's M by N temporary of C.
///
/// Zero derives it: four times the L1 (the M4 optimum), floored at 512 KiB (for Zen+'s 32 KiB L1).
/// Speed and memory only. For sweeps.
inline constinit cl::ConfigOption<std::int64_t> PackedGemmCTempBudget = cl::config_opt<std::int64_t>(
    "einsums:packed-gemm:c-temp-budget", "Budget for the block-GEMM strategy's C temporary, in KiB (0 = derive from the L1)", "PackedGemm",
    0, "KIB", cl::RangeBetween<std::int64_t>(0, std::int64_t{1} << 30));

/// Print each packed contraction's resolved plan and blocking to stderr.
///
/// Group dims and strides, register and cache blocks, and the write-back taken. For diagnosis.
inline constinit cl::ConfigOption<bool> PackedGemmDumpPlan =
    cl::config_flag("einsums:packed-gemm:dump-plan", "Print each packed contraction's plan and blocking to stderr", "PackedGemm", false);

EINSUMS_NAMESPACE_END(option)

EINSUMS_NAMESPACE_BEGIN()

/**
 * @brief Give the packed engine's options their command-line presence. Idempotent.
 */
EINSUMS_EXPORT int register_Einsums_PackedGemm_options();

namespace detail {
[[maybe_unused]] static int const register_options_Einsums_PackedGemm = register_Einsums_PackedGemm_options();
}

EINSUMS_NAMESPACE_END()
