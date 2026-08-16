//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Config/Namespace.hpp>

EINSUMS_NAMESPACE_BEGIN(blas)

/**
 * @brief Limit the calling thread to @p nthreads BLAS threads, where the vendor allows it.
 *
 * einsums parallelizes across work items and then calls BLAS from each of
 * them, so a BLAS that opens its own thread team per call multiplies the two
 * counts. Threads that dispatch work items therefore ask for a single-threaded
 * BLAS, which is what @ref einsums::TaskPool does for its workers.
 *
 * Whether that needs saying at all depends on how the vendor threads:
 *
 * - Vendors that thread with OpenMP (conda-forge's ``openblas`` is built this
 *   way) read the calling thread's ``nthreads-var`` ICV, so ``omp_set_num_threads``
 *   already governs them and this function has nothing to add.
 * - MKL runs its own runtime, which does not share ICVs with ours when the two
 *   load separate OpenMP libraries - the usual case on Windows, where MKL
 *   brings ``libiomp5md`` and we link ``libomp``. It does expose a thread-local
 *   setter, which is what this function calls.
 * - Accelerate schedules through GCD with no per-thread control, and a
 *   reference BLAS is serial. Both are no-ops here.
 *
 * The vendor is decided at build time, from the one FindTargetLAPACK resolved.
 * An earlier version looked the symbol up at run time so no build would carry
 * a vendor-specific dependency; it silently found nothing under MKL's layered
 * Windows DLLs and left this a no-op on the only platform that needs it. A
 * declaration the linker has to satisfy fails loudly instead.
 *
 * Only ever affects the calling thread. There is deliberately no process-wide
 * form: OpenBLAS's ``openblas_set_num_threads`` is global, so calling it from a
 * worker would also strip the threads from BLAS calls made on the main thread.
 *
 * @param[in] nthreads Thread count to request. Values below 1 are ignored.
 *
 * @versionadded{2.0.0}
 */
EINSUMS_EXPORT void set_num_threads_this_thread(int nthreads);

/**
 * @brief Whether the linked BLAS exposes the per-thread control above.
 *
 * False means @ref set_num_threads_this_thread is a no-op, either because the
 * vendor threads through OpenMP (and so is already governed by the OpenMP ICVs)
 * or because it offers no per-thread knob at all. Intended for tests and for
 * reporting the runtime configuration, not for branching in hot code.
 *
 * @versionadded{2.0.0}
 */
EINSUMS_EXPORT bool has_per_thread_control();

/**
 * @brief Threads the linked BLAS would currently use on the calling thread.
 *
 * Exists so that @ref set_num_threads_this_thread can be shown to have done
 * something rather than merely not crashed; a caller has no other way to see
 * the vendor's state. Returns 0 when the vendor cannot be asked, which is
 * every vendor for which @ref has_per_thread_control is false.
 *
 * @return The vendor's thread count for this thread, or 0 if unknown.
 *
 * @versionadded{2.0.0}
 */
EINSUMS_EXPORT int get_num_threads_this_thread();

/**
 * @brief Whether the linked BLAS threads through our own OpenMP runtime.
 *
 * True means ``omp_set_num_threads`` on a thread governs the BLAS calls that
 * thread makes, which is what lets a caller give one BLAS call a thread count
 * of its own without a per-thread vendor setter. It is the second half of the
 * question @ref has_per_thread_control answers: a vendor for which both are
 * false - Accelerate, or a reference BLAS - runs on a schedule no caller can
 * govern, and callers that budget threads have to treat it as fixed.
 *
 * Only OpenBLAS answers true, and only when it was built with OpenMP rather
 * than with its own pthread pool, which is asked of the library itself at
 * first call rather than assumed: the OpenMP and pthread builds carry the same
 * name and the same symbols. Whether OpenBLAS is the linked vendor at all is a
 * build-time question, and one that cannot be answered from the library name
 * because a conda environment resolves every vendor through the same
 * ``libblas`` shim; the build asks the resolved libraries for a symbol only
 * OpenBLAS defines.
 *
 * @versionadded{2.0.0}
 */
EINSUMS_EXPORT bool threads_with_openmp();

/**
 * @brief Mark the calling thread as holding a node-scoped OpenMP width.
 *
 * An executor that gives one task a thread team of its own - the moldable
 * scheduler's `WidthGuard` - raises the calling thread's OpenMP ICV for the
 * task and restores it after. Kernels einsums threads itself may fork from
 * that ICV freely. Vendor BLAS calls may NOT: an OpenMP-built OpenBLAS obeys
 * each caller's ICV but does not support concurrent callers whose ICVs
 * differ. Its server syncs a process-global thread count to the caller's
 * value on every call, frees shared pack buffers when that global shrinks,
 * and sizes its work queue from the global while forking the team from the
 * caller - so two callers disagreeing about the width corrupt results or
 * deadlock. This holds for every OpenBLAS release through at least 0.3.34.
 *
 * While the flag is set, the BLAS module's vendor wrappers therefore clamp
 * the ICV to one thread around the vendor call and restore it after, no
 * matter what any width plan says. The flag is thread-local; the guard that
 * raises the ICV sets it and clears it with the same lifetime.
 *
 * The invariant this buys is not "every caller presents width 1" - the
 * thread that called execute() helps run tasks and is not pinned, so it
 * presents its own baseline ICV. It is that AT MOST ONE width above 1 is
 * ever presented to the vendor: node-scoped widths are clamped here, worker
 * pins are 1, and a width-1 caller never reaches the vendor's global sync at
 * all (its num_cpu_avail early-returns). One stable >1 value is the regime
 * every executor ran in before widths existed and the one the vendor
 * supports.
 *
 * @param[in] active True on entering a node-scoped width, false on leaving.
 *
 * @versionadded{2.0.0}
 */
EINSUMS_EXPORT void set_moldable_width_scope(bool active);

/**
 * @brief Whether the calling thread currently holds a node-scoped OpenMP width.
 *
 * See @ref set_moldable_width_scope for what the answer obligates a vendor
 * BLAS call to do.
 *
 * @versionadded{2.0.0}
 */
EINSUMS_EXPORT bool moldable_width_scope();

/**
 * @brief Whether a vendor call made from this thread right now would be clamped
 *        to one thread.
 *
 * The exact condition the wrappers' fence applies, asked before the call rather
 * than during it: a node-scoped width is held, the vendor threads through our
 * OpenMP runtime, and the width being held is above one. It answers the
 * question a kernel with two implementations has to answer - hand the work to
 * the vendor, or run it here - because under this condition the vendor gets one
 * thread while anything einsums forks from the ICV gets the node's full width.
 *
 * Read per call and never stored. The answer is a property of the calling
 * thread at this instant, so a cache keyed on the contraction rather than on
 * the caller must not hold it.
 *
 * @versionadded{2.0.0}
 */
EINSUMS_EXPORT bool vendor_call_is_fenced();

EINSUMS_NAMESPACE_END(blas)
