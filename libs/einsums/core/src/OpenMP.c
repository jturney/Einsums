//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include "einsums/OpenMP.hpp"

int omp_get_max_threads() {
    return 1;
}

int omp_get_thread_num() {
    return 0;
}

int omp_get_num_threads() {
    return 1;
}

void omp_set_num_threads(int nthread) {
    (void)nthread;
}

int omp_in_parallel() {
    return 0;
}

void omp_set_nested(int val) {
    (void)val;
}

int omp_get_nested() {
    return 0;
}

void omp_set_max_active_levels(int max_levels) {
    (void)max_levels;
}
