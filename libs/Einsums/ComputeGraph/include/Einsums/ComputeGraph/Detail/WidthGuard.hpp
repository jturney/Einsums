//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/// @file WidthGuard.hpp
/// @brief The one RAII that hands a thread a temporary thread width.

#include <Einsums/Config.hpp>

#include <Einsums/BLAS/ThreadControl.hpp>
#include <Einsums/Config/Namespace.hpp>

#include <algorithm>

#ifdef _OPENMP
#    include <omp.h>
#endif

EINSUMS_NAMESPACE_BEGIN(compute_graph::detail)

/**
 * @brief Gives the calling thread @p width threads for as long as it lives, and
 *        hands back exactly what it found.
 *
 * Restoring the SAVED values rather than the literal 1 is the whole point.
 * Pool workers pin themselves to one thread at startup and every unplanned
 * task depends on that pin, so restoring 1 is right for them - but
 * `help_until` also runs tasks on the thread that called `execute()`, which is
 * not pinned, and leaving that thread narrowed would follow the caller into
 * whatever it does after the replay returns. The calibration sweep has the same
 * obligation for the same reason: it may run on a thread somebody else already
 * configured.
 *
 * The executor only constructs one for a planned width, so an unplanned graph
 * reads no ICV and writes none.
 */
class WidthGuard {
  public:
    /**
     * @param[in] width           Threads to request. Values below 1 are read as 1.
     * @param[in] blas_follows_width False gives the vendor BLAS a single thread
     *            while OpenMP gets @p width, which is what a batched kernel
     *            wants: the width goes to the loop over entries, not inside
     *            them.
     * @param[in] moldable_scope  Whether to mark the thread as holding a
     *            node-scoped width, which makes the BLAS wrappers clamp any
     *            vendor call under this guard back to one thread. An executor
     *            wants that; a calibration cell that is measuring the vendor's
     *            own scaling must not have it, or it would measure a clamp.
     */
    explicit WidthGuard(int width, bool blas_follows_width = true, bool moldable_scope = true) : _moldable_scope(moldable_scope) {
        int const requested = std::max(1, width);

        // Both counts are READ before either is written, and the order is not
        // cosmetic. A thread that never set a vendor count of its own has one
        // derived from its OpenMP ICV - MKL_Get_Max_Threads() reports 4 on a
        // thread that just called omp_set_num_threads(4) - so reading the
        // vendor after raising the ICV reads back the width instead of the
        // baseline, and the restore below would then pin the thread to the
        // node's width for good. Pool workers hid this: they set an explicit
        // vendor count of 1 at startup, and an explicit count outranks the ICV.
        // The thread that called execute() sets none, and `help_until` runs
        // nodes there.
        //
        // A vendor that cannot be read cannot be set either (both are the same
        // build-time switch), so a zero here means there is no vendor state to
        // save, and restoring the 0 would ask for a nonsense thread count.
        _prev_blas = blas::get_num_threads_this_thread();
#ifdef _OPENMP
        _prev_omp = omp_get_max_threads();
        omp_set_num_threads(requested);
#endif
        // The raised ICV is for the kernels einsums threads itself; an
        // OpenMP-threaded vendor BLAS must not fork from it (concurrent
        // callers with differing ICVs are outside what OpenBLAS supports, see
        // blas::set_moldable_width_scope). The flag makes the BLAS wrappers
        // clamp any vendor call made under this guard back to one thread.
        if (_moldable_scope) {
            _prev_scope = blas::moldable_width_scope();
            blas::set_moldable_width_scope(true);
        }
        if (_prev_blas > 0) {
            blas::set_num_threads_this_thread(blas_follows_width ? requested : 1);
        }
    }

    ~WidthGuard() {
        if (_moldable_scope) {
            blas::set_moldable_width_scope(_prev_scope);
        }
#ifdef _OPENMP
        omp_set_num_threads(_prev_omp);
#endif
        // Last, and after the ICV is back: on a vendor whose count tracks the
        // ICV this write is what makes the restored baseline stick rather than
        // being read back through an ICV that is still the node's width.
        if (_prev_blas > 0) {
            blas::set_num_threads_this_thread(_prev_blas);
        }
    }

    WidthGuard(WidthGuard const &)            = delete;
    WidthGuard &operator=(WidthGuard const &) = delete;
    WidthGuard(WidthGuard &&)                 = delete;
    WidthGuard &operator=(WidthGuard &&)      = delete;

  private:
#ifdef _OPENMP
    int _prev_omp{1};
#endif
    int  _prev_blas{0};
    bool _prev_scope{false};
    bool _moldable_scope{true};
};

EINSUMS_NAMESPACE_END(compute_graph::detail)
