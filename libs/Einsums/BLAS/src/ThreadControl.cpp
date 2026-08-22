//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/BLAS/Defines.hpp>
#include <Einsums/BLAS/ThreadControl.hpp>
#include <Einsums/BLASVendor/Defines.hpp>
#include <Einsums/Config/Namespace.hpp>

#include <string_view>

#ifdef _OPENMP
#    include <omp.h>
#endif

#if defined(EINSUMS_HAVE_MKL)
// Declared rather than pulled from mkl.h so this stays independent of which
// MKL headers the build has, and of the LP64/ILP64 spelling: both take and
// return a plain int. BLASVendor's InitializeFinalize.cpp declares
// mkl_set_xerbla the same way.
extern "C" {
int MKL_Set_Num_Threads_Local(int nthreads);
int MKL_Get_Max_Threads(void);
}
#endif

#if defined(EINSUMS_HAVE_OPENBLAS)
// Declared rather than included for the same reason as MKL's: openblas's own
// header is not on the include path of a build that resolved BLAS through a
// vendor-neutral shim. The build proved this symbol links before defining
// EINSUMS_HAVE_OPENBLAS.
extern "C" {
char *openblas_get_config(void);
}
#endif

EINSUMS_NAMESPACE_BEGIN(blas)

void set_num_threads_this_thread(int nthreads) {
    if (nthreads < 1) {
        return;
    }
#if defined(EINSUMS_HAVE_MKL)
    MKL_Set_Num_Threads_Local(nthreads);
#else
    (void)nthreads;
#endif
}

bool has_per_thread_control() {
#if defined(EINSUMS_HAVE_MKL)
    return true;
#else
    return false;
#endif
}

int get_num_threads_this_thread() {
#if defined(EINSUMS_HAVE_MKL)
    return MKL_Get_Max_Threads();
#else
    return 0;
#endif
}

bool threads_with_openmp() {
#if defined(EINSUMS_HAVE_OPENBLAS)
    // openblas_get_config() returns a space-separated list of the flags the
    // library was built with - "OpenBLAS 0.3.32 NO_AFFINITY USE_OPENMP VORTEX"
    // on the conda build this is developed against. The OpenMP flag is a BARE
    // token, not an assignment: only the numeric settings (MAX_THREADS=N) carry
    // a value, so looking for "USE_OPENMP=1" finds nothing on a library that is
    // in fact built with OpenMP. A pthread-built OpenBLAS carries the same
    // symbols and the same name as the OpenMP one but keeps its own pool, so
    // the ICV does not reach it and the answer is no.
    static bool const built_with_openmp = []() {
        char const *config = openblas_get_config();
        return config != nullptr && std::string_view(config).find("USE_OPENMP") != std::string_view::npos;
    }();
    return built_with_openmp;
#else
    return false;
#endif
}

namespace {
thread_local bool moldable_width_scope_active = false;
}

void set_moldable_width_scope(bool active) {
    moldable_width_scope_active = active;
}

bool moldable_width_scope() {
    return moldable_width_scope_active;
}

bool vendor_call_is_fenced() {
#ifdef _OPENMP
    // Mirrors VendorWidthFence in BLAS.cpp term for term. Kept here rather than
    // shared with it because that fence is on the hot path of 48 wrappers and
    // constructs its clamp from the same reads.
    return moldable_width_scope_active && threads_with_openmp() && omp_get_max_threads() > 1;
#else
    return false;
#endif
}

SerialVendorScope::SerialVendorScope() {
    // Read both counts before writing either: a vendor count that tracks the OpenMP ICV reads back the value just written otherwise, and
    // the restore would then pin the thread to it.
    _prior_vendor = get_num_threads_this_thread();
#ifdef _OPENMP
    _prior_omp = omp_get_max_threads();
    if (_prior_omp > 1) {
        omp_set_num_threads(1);
    }
#endif
    if (_prior_vendor > 1) {
        set_num_threads_this_thread(1);
    }
}

SerialVendorScope::~SerialVendorScope() {
#ifdef _OPENMP
    if (_prior_omp > 1) {
        omp_set_num_threads(_prior_omp);
    }
#endif
    if (_prior_vendor > 1) {
        set_num_threads_this_thread(_prior_vendor);
    }
}

EINSUMS_NAMESPACE_END(blas)
