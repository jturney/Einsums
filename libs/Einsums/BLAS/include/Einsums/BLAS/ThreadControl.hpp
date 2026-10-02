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
 * For threads that already parallelize across work items (@ref einsums::TaskPool's workers), so
 * BLAS does not multiply the thread count. OpenMP-built OpenBLAS already follows
 * ``omp_set_num_threads``; MKL, on its own runtime, gets its thread-local setter; Accelerate and
 * reference BLAS are no-ops. The vendor is fixed at build time, so a missing symbol fails to link.
 * Calling thread only: OpenBLAS's process-wide setter would also throttle the main thread.
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
 * So @ref set_num_threads_this_thread can be verified. 0 when the vendor cannot be asked.
 *
 * @return The vendor's thread count for this thread, or 0 if unknown.
 *
 * @versionadded{2.0.0}
 */
EINSUMS_EXPORT int get_num_threads_this_thread();

/**
 * @brief Whether the linked BLAS threads through our own OpenMP runtime.
 *
 * True when ``omp_set_num_threads`` governs this thread's BLAS calls: an OpenMP-built OpenBLAS,
 * asked of the library at first call since the pthread build has the same name and symbols. If
 * this and @ref has_per_thread_control are both false, the vendor's threading is ungovernable.
 *
 * @versionadded{2.0.0}
 */
EINSUMS_EXPORT bool threads_with_openmp();

/**
 * @brief Mark the calling thread as holding a node-scoped OpenMP width.
 *
 * Set by the moldable scheduler's `WidthGuard` while it raises the thread's OpenMP ICV for a task.
 * einsums' own kernels may fork from that ICV, but an OpenMP OpenBLAS (through at least 0.3.34)
 * corrupts results or deadlocks when concurrent callers present different widths. So while set,
 * the vendor wrappers clamp the ICV to one thread, which keeps at most one width above 1 ever
 * reaching the vendor (a width-1 caller skips its global sync).
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
 * The wrappers' fence condition, for kernels choosing between the vendor (one thread when fenced)
 * and their own loops (the full width). Per call; never cache it with a contraction.
 *
 * @versionadded{2.0.0}
 */
EINSUMS_EXPORT bool vendor_call_is_fenced();

/**
 * @brief Present ONE thread to the vendor, and to anything einsums forks from the OpenMP ICV, for as long as it exists.
 *
 * Makes reductions deterministic (a threaded @c dot's last bits depend on its thread count) and
 * speeds small level-1 calls, where a fork costs more than the arithmetic. Width one is always safe
 * to present (see @ref set_moldable_width_scope). Calling thread only; nests; restores on destruction.
 *
 * @versionadded{2.0.0}
 */
class EINSUMS_EXPORT SerialVendorScope {
  public:
    SerialVendorScope();
    ~SerialVendorScope();

    SerialVendorScope(SerialVendorScope const &)            = delete;
    SerialVendorScope &operator=(SerialVendorScope const &) = delete;
    SerialVendorScope(SerialVendorScope &&)                 = delete;
    SerialVendorScope &operator=(SerialVendorScope &&)      = delete;

  private:
    int _prior_vendor{0};
    int _prior_omp{1};
};

EINSUMS_NAMESPACE_END(blas)
