//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// The one place the stream kernel's rung is chosen. Compiled once, without arch flags; picks the
// best built rung, cached per element type. No `sme` rung.

#include <Einsums/Config/Namespace.hpp>
#include <Einsums/PackedGemm/StreamKernel.hpp>

#include <Stripes/RungLadder.hpp>
#include <Stripes/RuntimeFeatures.hpp>
#include <complex>
#include <cstdint>

EINSUMS_NAMESPACE_BEGIN(packed_gemm)

#define EINSUMS_STREAM_DECLARE_RUNG_ENTRY(ns)                                                                                              \
    namespace ns {                                                                                                                         \
    template <typename T>                                                                                                                  \
    void stream_inner(T *cb, T const *sp, T const *w, T alpha, int64_t n, int64_t co, int64_t si, int64_t wo, int64_t ds, int64_t dc,      \
                      int64_t dw);                                                                                                         \
    template <typename T>                                                                                                                  \
    void stream_tile(T *cb, T const *sp, T const *w, T alpha, int64_t m, int64_t n, int64_t co, int64_t si, int64_t wo, int64_t ds,        \
                     int64_t dc, int64_t dw, int64_t ds2, int64_t dc2, int64_t dw2);                                                       \
    }

STRIPES_FOR_EACH_BUILT_RUNG(EINSUMS_STREAM_DECLARE_RUNG_ENTRY)

#undef EINSUMS_STREAM_DECLARE_RUNG_ENTRY

#define EINSUMS_STREAM_DEFINE_ENTRY(T)                                                                                                     \
    template <>                                                                                                                            \
    EINSUMS_EXPORT StreamInnerFn<T> stream_inner_entry<T>() {                                                                              \
        static StreamInnerFn<T> const fn = stripes::select<StreamInnerFn<T>>(STRIPES_LADDER(stream_inner<T>));                             \
        return fn;                                                                                                                         \
    }                                                                                                                                      \
    template <>                                                                                                                            \
    EINSUMS_EXPORT StreamTileFn<T> stream_tile_entry<T>() {                                                                                \
        static StreamTileFn<T> const fn = stripes::select<StreamTileFn<T>>(STRIPES_LADDER(stream_tile<T>));                                \
        return fn;                                                                                                                         \
    }

EINSUMS_STREAM_DEFINE_ENTRY(float)
EINSUMS_STREAM_DEFINE_ENTRY(double)
EINSUMS_STREAM_DEFINE_ENTRY(std::complex<float>)
EINSUMS_STREAM_DEFINE_ENTRY(std::complex<double>)

#undef EINSUMS_STREAM_DEFINE_ENTRY

EINSUMS_NAMESPACE_END(packed_gemm)
