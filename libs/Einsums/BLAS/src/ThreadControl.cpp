//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/BLAS/ThreadControl.hpp>
#include <Einsums/BLASVendor/Defines.hpp>

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

namespace einsums::blas {

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

} // namespace einsums::blas
