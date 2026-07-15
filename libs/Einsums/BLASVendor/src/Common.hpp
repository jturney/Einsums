//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/Assert.hpp>
#include <Einsums/BLAS/Types.hpp>

#ifndef EINSUMS_WINDOWS
#    include <Einsums/HPTT/HPTT.hpp>
#    include <Einsums/HPTT/HPTTTypes.hpp>
#endif

#include <omp.h>

#ifndef FC_SYMBOL
#    define FC_SYMBOL 2
#endif

#if FC_SYMBOL == 1
/* Mangling for Fortran global symbols without underscores. */
#    define FC_GLOBAL(name, NAME) name
#elif FC_SYMBOL == 2
/* Mangling for Fortran global symbols with underscores. */
#    define FC_GLOBAL(name, NAME) name##_
#elif FC_SYMBOL == 3
/* Mangling for Fortran global symbols without underscores. */
#    define FC_GLOBAL(name, NAME) NAME
#elif FC_SYMBOL == 4
/* Mangling for Fortran global symbols with underscores. */
#    define FC_GLOBAL(name, NAME) NAME##_
#endif

namespace einsums::blas::vendor {

inline bool lsame(char ca, char cb) {
    return std::tolower(ca) == std::tolower(cb);
}

enum class OrderMajor { Column, Row, C = Row, Fortran = Column };

// OrderMajor indicates the order of the input matrix. C is Row, Fortran is Column
template <OrderMajor Order, typename T>
void transpose(int_t m, int_t n, T const *in, int_t ldin, T *out, int_t ldout) {
    if (in == nullptr || out == nullptr) {
        return;
    }
    EINSUMS_ASSERT(m >= 0);
    EINSUMS_ASSERT(n >= 0);

#ifndef EINSUMS_WINDOWS
    int    perm[]    = {1, 0};
    size_t size_in[] = {static_cast<unsigned long>(m), static_cast<unsigned long>(n)}, outer_size_in[2], outer_size_out[2];
    if constexpr (Order == OrderMajor::Row) {
        outer_size_in[0]  = static_cast<unsigned long>(m);
        outer_size_in[1]  = static_cast<unsigned long>(ldin);
        outer_size_out[0] = static_cast<unsigned long>(n);
        outer_size_out[1] = static_cast<unsigned long>(ldout);
    } else {
        outer_size_in[0]  = static_cast<unsigned long>(ldin);
        outer_size_in[1]  = static_cast<unsigned long>(n);
        outer_size_out[0] = static_cast<unsigned long>(ldout);
        outer_size_out[1] = static_cast<unsigned long>(m);
    }

    auto plan = hptt::create_plan(perm, 2, T{1.0}, in, size_in, outer_size_in, T{0.0}, out, outer_size_out, hptt::ESTIMATE,
                                  omp_get_max_threads(), {}, Order == OrderMajor::Row);

    plan->execute();
#else
    // HPTT is unavailable on Windows; plain blocked loop transpose. Slow but
    // correct - revisit with a Windows-capable HPTT or librett.
    int_t x, y;
    if constexpr (Order == OrderMajor::Column) {
        x = n;
        y = m;
    } else {
        x = m;
        y = n;
    }
    for (int_t i = 0; i < std::min(y, ldin); i++) {
        for (int_t j = 0; j < std::min(x, ldout); j++) {
            out[(size_t)i * ldout + j] = in[(size_t)j * ldin + i];
        }
    }
#endif
}

template <OrderMajor Order, typename T, typename Alloc1>
void transpose(int_t m, int_t n, std::vector<T, Alloc1> const &in, int_t ldin, T *out, int_t ldout) {
    transpose<Order>(m, n, in.data(), ldin, out, ldout);
}

template <OrderMajor Order, typename T, typename Alloc2>
void transpose(int_t m, int_t n, T const *in, int_t ldin, std::vector<T, Alloc2> &out, int_t ldout) {
    transpose<Order>(m, n, in, ldin, out.data(), ldout);
}

template <OrderMajor Order, typename T, typename Alloc1, typename Alloc2>
void transpose(int_t m, int_t n, std::vector<T, Alloc1> const &in, int_t ldin, std::vector<T, Alloc2> &out, int_t ldout) {
    transpose<Order>(m, n, in.data(), ldin, out.data(), ldout);
}
} // namespace einsums::blas::vendor