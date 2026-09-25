//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/**
 * @file RungLadder.hpp
 * @brief The per-rung entry points of a kernel compiled once per rung, as @ref einsums::simd::select takes them.
 *
 * A module whose kernel translation units go through ``einsums_add_simd_dispatch_sources``
 * compiles one copy per rung, each in its own ``arch_<rung>`` namespace, and the target is told
 * which rungs it built through ``EINSUMS_SIMD_HAS_RUNG_<RUNG>``. These macros turn that set into
 * the two things every dispatching module needs:
 *
 * @code
 * // Declare the entry in each namespace that was built.
 * #define DECLARE(ns) namespace ns { void kernel(float const *, float *, std::size_t); }
 * EINSUMS_SIMD_FOR_EACH_BUILT_RUNG(DECLARE)
 * #undef DECLARE
 *
 * // Pick the best one for this machine.
 * static KernelFn const kernel = einsums::simd::select<KernelFn>(EINSUMS_SIMD_LADDER(kernel));
 * @endcode
 *
 * A rung that was not built fills its slot with nullptr, which @ref einsums::simd::select falls
 * through. ``arch_native`` (single-TU aarch64 builds and pinned targets) occupies the baseline
 * slot: on aarch64 it IS the toolchain baseline, so an opt-in SME rung can still out-rank it
 * through the ordinary walk.
 */

#include <Einsums/SIMD/RuntimeFeatures.hpp>

#if defined(EINSUMS_SIMD_HAS_RUNG_NATIVE)
#    define EINSUMS_SIMD_BASELINE_SLOT(fn) &arch_native::fn
#    define EINSUMS_SIMD_EACH_NATIVE(X)    X(arch_native)
#elif defined(EINSUMS_SIMD_HAS_RUNG_BASELINE)
#    define EINSUMS_SIMD_BASELINE_SLOT(fn) &arch_baseline::fn
#    define EINSUMS_SIMD_EACH_NATIVE(X)
#else
#    define EINSUMS_SIMD_BASELINE_SLOT(fn) nullptr
#    define EINSUMS_SIMD_EACH_NATIVE(X)
#endif

#if defined(EINSUMS_SIMD_HAS_RUNG_BASELINE)
#    define EINSUMS_SIMD_EACH_BASELINE(X) X(arch_baseline)
#else
#    define EINSUMS_SIMD_EACH_BASELINE(X)
#endif

#if defined(EINSUMS_SIMD_HAS_RUNG_V2)
#    define EINSUMS_SIMD_V2_SLOT(fn) &arch_v2::fn
#    define EINSUMS_SIMD_EACH_V2(X)  X(arch_v2)
#else
#    define EINSUMS_SIMD_V2_SLOT(fn) nullptr
#    define EINSUMS_SIMD_EACH_V2(X)
#endif

#if defined(EINSUMS_SIMD_HAS_RUNG_V3)
#    define EINSUMS_SIMD_V3_SLOT(fn) &arch_v3::fn
#    define EINSUMS_SIMD_EACH_V3(X)  X(arch_v3)
#else
#    define EINSUMS_SIMD_V3_SLOT(fn) nullptr
#    define EINSUMS_SIMD_EACH_V3(X)
#endif

#if defined(EINSUMS_SIMD_HAS_RUNG_V4)
#    define EINSUMS_SIMD_V4_SLOT(fn) &arch_v4::fn
#    define EINSUMS_SIMD_EACH_V4(X)  X(arch_v4)
#else
#    define EINSUMS_SIMD_V4_SLOT(fn) nullptr
#    define EINSUMS_SIMD_EACH_V4(X)
#endif

#if defined(EINSUMS_SIMD_HAS_RUNG_SME)
#    define EINSUMS_SIMD_SME_SLOT(fn) &arch_sme::fn
#    define EINSUMS_SIMD_EACH_SME(X)  X(arch_sme)
#else
#    define EINSUMS_SIMD_SME_SLOT(fn) nullptr
#    define EINSUMS_SIMD_EACH_SME(X)
#endif

/// Invoke @p X with the name of every ``arch_<rung>`` namespace this target built.
#define EINSUMS_SIMD_FOR_EACH_BUILT_RUNG(X)                                                                                                \
    EINSUMS_SIMD_EACH_NATIVE(X)                                                                                                            \
    EINSUMS_SIMD_EACH_BASELINE(X) EINSUMS_SIMD_EACH_V2(X) EINSUMS_SIMD_EACH_V3(X) EINSUMS_SIMD_EACH_V4(X) EINSUMS_SIMD_EACH_SME(X)

/// The five ladder slots of @p fn, in the order @ref einsums::simd::select takes them.
#define EINSUMS_SIMD_LADDER(fn)                                                                                                            \
    EINSUMS_SIMD_BASELINE_SLOT(fn), EINSUMS_SIMD_V2_SLOT(fn), EINSUMS_SIMD_V3_SLOT(fn), EINSUMS_SIMD_V4_SLOT(fn), EINSUMS_SIMD_SME_SLOT(fn)
