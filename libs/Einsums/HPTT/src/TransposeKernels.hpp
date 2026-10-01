//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// The boundary between HPTT's planner, compiled once, and its kernels, compiled once per SIMD
// dispatch rung (TransposeKernels.cpp, through stripes_add_dispatch_sources).
//
// Everything a rung's copy defines must be its own: the linker keeps one definition per name, so a
// std::vector member or a logging template instantiated in the kernels would be emitted by every rung
// under the same name, at that rung's flags, and the program would call whichever copy it kept, AVX-512
// on a CPU without it included. So the kernels see only this header's flat argument block and
// nothing from the planner's side: no containers, no logging, no assertions. The planner reaches them
// through the table one rung's copy returns.

#pragma once

#include <Einsums/Config/Namespace.hpp>
#include <Einsums/HPTT/HPTTTypes.hpp>

#include <Stripes/RungLadder.hpp>
#include <cstddef>

EINSUMS_NAMESPACE_BEGIN(hptt)

class Plan;

/// What one execution of a plan reads. The arrays hold dim entries; threadIds holds numThreads.
template <typename floatType>
struct KernelArgs {
    floatType const *A;
    floatType       *B;
    floatType        alpha;
    floatType        beta;
    int              dim;
    int              perm0; ///< perm[0]: zero when the fastest index stays in place.
    bool             conjA;
    size_t           innerStrideA;
    size_t           innerStrideB;
    size_t const    *sizeA;
    size_t const    *offsetA;
    size_t const    *offsetB;
    size_t const    *lda;
    size_t const    *ldb;
    Plan const      *plan; ///< The plan to run; nullptr only for the one- and two-dimensional paths, which need none.
    int              numThreads;
    int const       *threadIds;
    bool             callerManagedThreads;
};

/// One rung's kernels for one element type, and the geometry the planner builds plans around.
template <typename floatType>
struct TransposeKernels {
    int vector_bits; ///< Register width of the rung, which the plan file records.
    int rung;        ///< stripes::InstructionSet of the rung, for the plan file's messages.
    int blocking;    ///< Edge of the macro-kernel tile, in elements: the plan's loop increment.

    /// Run the plan, spawning threads when there are several and streaming B when beta is zero.
    void (*execute)(KernelArgs<floatType> const &args) noexcept;

    /// Run the plan once, unthreaded within each task and without streaming, to time it.
    void (*execute_estimate)(KernelArgs<floatType> const &args) noexcept;
};

#define EINSUMS_HPTT_DECLARE_RUNG_KERNELS(ns)                                                                                              \
    namespace ns {                                                                                                                         \
    template <typename floatType>                                                                                                          \
    TransposeKernels<floatType> const &transpose_kernels() noexcept;                                                                       \
    }

STRIPES_FOR_EACH_BUILT_RUNG(EINSUMS_HPTT_DECLARE_RUNG_KERNELS)

#undef EINSUMS_HPTT_DECLARE_RUNG_KERNELS

/// The kernels of the best rung this machine supports, chosen once per element type.
template <typename floatType>
TransposeKernels<floatType> const &selected_transpose_kernels();

EINSUMS_NAMESPACE_END(hptt)
