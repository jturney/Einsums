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
/// That route makes one index of each operand contiguous so a single GEMM can
/// span the whole K, and it does so by transposing WHOLE operands: 448 MiB on
/// ccsd's `ab-cad-dcb`, both sides, into thread-local buffers that only ever
/// grow. Setting a cap makes it transpose a chunk of K at a time instead - the
/// chunk is a sub-block of each operand, which HPTT can read - and the
/// contraction becomes a short chain of large GEMMs.
///
/// Zero, the default, means no cap. Chunking is not free, and how much it costs
/// depends on the operands' layouts: it slices one K index, and an operand
/// whose slowest axis that is reads its chunk contiguously while one whose
/// FASTEST axis it is reads a fraction of every cache line it touches. Both
/// happen at once on `ab-cad-dcb`, where the measured cost against the
/// whole-operand flatten, single then double, is 4.2% and 1.9% at 64 MiB, 9.3%
/// and 1.8% at 32, and 11.7% and 3.1% at 16.
///
/// So this is for a workload that would rather have the memory than the last
/// few percent. Contractions that do not reach the flatten route - which is
/// most of them, the packed loops being the usual answer - are unaffected
/// whatever it is set to.
inline constinit cl::ConfigOption<std::int64_t> PackedGemmFlattenBudget = cl::config_opt<std::int64_t>(
    "einsums:packed-gemm:flatten-budget", "Cap the multi-K flatten route's temporary buffers, in MiB (0 = transpose whole operands)",
    "PackedGemm", 0, "MIB", cl::RangeBetween<std::int64_t>(0, std::int64_t{1} << 20));

/// Run complex contractions on x86 through the rung's REAL tile kernel, by the 1m method.
///
/// Off by default. Without it a complex contraction on an x86 rung takes the
/// block strategy on the scatter path (one vendor GEMM per cache block, then a
/// scatter) and the portable complex tile on the tile path, because the vector
/// tile only exists for float and double. With it, both take Van Zee's 1m
/// method instead: A packs in the expanded 1e form, B in the 1r form, and the
/// real vector tile of the underlying type computes interleaved complex output
/// that is scattered straight into C. It changes the engine only, never the
/// route: which contractions reach the packed loops is decided exactly as
/// before.
///
/// The SME rung always runs complex this way and ignores the flag, and so does
/// every rung below x86-64-v2 and every non-x86 rung, which have no 1m route to
/// switch to.
inline constinit cl::ConfigOption<bool> PackedGemmComplex1m = cl::config_flag(
    "einsums:packed-gemm:complex-1m", "Run complex contractions on x86 through the real tile kernel (1m method)", "PackedGemm", false);

/// Physical cores the engine assumes share one L3, in place of the count read from sysfs.
///
/// Zero, the default, detects it: on Linux from CPU 0's L3 `shared_cpu_list`
/// divided by its SMT siblings, and 1 wherever that cannot be read, macOS
/// included. The count decides whether a threaded contraction splits its
/// threads into per-L3 teams that each share one packed B panel, so a wrong
/// value changes how the work is divided, never the result.
///
/// Setting it pins the machine model, which is how a test asserts the teams a
/// particular part forms on a runner whose own caches would form others.
inline constinit cl::ConfigOption<std::int64_t> PackedGemmCoresPerL3 =
    cl::config_opt<std::int64_t>("einsums:packed-gemm:cores-per-l3", "Physical cores sharing one L3 cache (0 = detect)", "PackedGemm", 0,
                                 "N", cl::RangeBetween<std::int64_t>(0, 4096));

/// Budget, in KiB, for the block-GEMM strategy's M by N temporary of C.
///
/// Zero, the default, derives it: four times the L1, floored at 512 KiB. The
/// multiplier is where the M4 optimum sat, and the floor is there because on a
/// Zen+ with a 32 KiB L1 the same multiplier measured worst of every value
/// swept on a rank-6 CCSD(T) shape. It bounds the strategy's N block and sizes
/// its M block, so it moves speed and memory, never the result. For sweeps.
inline constinit cl::ConfigOption<std::int64_t> PackedGemmCTempBudget = cl::config_opt<std::int64_t>(
    "einsums:packed-gemm:c-temp-budget", "Budget for the block-GEMM strategy's C temporary, in KiB (0 = derive from the L1)", "PackedGemm",
    0, "KIB", cl::RangeBetween<std::int64_t>(0, std::int64_t{1} << 30));

/// Print each packed contraction's resolved plan and blocking to stderr.
///
/// The dimensions and strides of every operand group, the register and cache
/// blocks, and which scatter the write-back took. Off by default, and for
/// diagnosis: a plan read off the dump settles questions that re-deriving it
/// from the construction rules has repeatedly got wrong.
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
