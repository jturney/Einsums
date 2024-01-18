//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#if defined(_OPENMP)
#    include <omp.h>

#    if defined(__INTEL_LLVM_COMPILER) || defined(__INTEL_COMPILER)
#        define EINSUMS_OMP_PARALLEL_FOR _Pragma("omp parallel for simd")
#        define EINSUMS_OMP_SIMD         _Pragma("omp simd")
#        define EINSUMS_OMP_PARALLEL     _Pragma("omp parallel")
#    else
#        define EINSUMS_OMP_PARALLEL_FOR _Pragma("omp parallel for")
#        define EINSUMS_OMP_SIMD
#        define EINSUMS_OMP_PARALLEL _Pragma("omp parallel")
#    endif

#else

#    include <einsums/config/export_definitions.hpp>

#    if defined(__cplusplus)
extern "C" {
#    endif

int EINSUMS_EXPORT  omp_get_max_threads();
int EINSUMS_EXPORT  omp_get_num_threads();
void EINSUMS_EXPORT omp_set_num_threads(int);
int EINSUMS_EXPORT  omp_get_thread_num();
int EINSUMS_EXPORT  omp_in_parallel();

/**
 * @brief A nonzero value enables nested parallelism, while zero disables nested parallelism.
 *
 * @param val
 */
void EINSUMS_EXPORT omp_set_nested(int val);

/**
 * @brief A nonzero value means nested parallelism is enabled.
 *
 * @return int
 */
int EINSUMS_EXPORT omp_get_nested();

void EINSUMS_EXPORT omp_set_max_active_levels(int max_levels);

#    if defined(__cplusplus)
}
#    endif

#    define EINSUMS_OMP_PARALLEL_FOR
#    define EINSUMS_OMP_SIMD
#    define EINSUMS_OMP_PARALLEL
#endif

#if defined(__cplusplus)
namespace einsums {

struct DisableOMPNestedScope {
    DisableOMPNestedScope() {
        _old_nested = omp_get_nested();
        omp_set_nested(0);
    }

    ~DisableOMPNestedScope() { omp_set_nested(_old_nested); }

  private:
    int _old_nested;
};

struct DisableOMPThreads {
    DisableOMPThreads() {
        _old_max_threads = omp_get_max_threads();
        omp_set_num_threads(1);
    }

    ~DisableOMPThreads() { omp_set_num_threads(_old_max_threads); }

  private:
    int _old_max_threads;
};

} // namespace einsums
#endif
