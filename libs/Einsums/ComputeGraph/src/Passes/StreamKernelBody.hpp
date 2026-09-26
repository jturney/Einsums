//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// Kernel body for the stream-fusion inner loop. Private implementation header
// (src/Passes, not installed): it is included only by StreamKernelImpl.cpp,
// which first defines EINSUMS_STREAM_KERNEL_NS to that rung's namespace
// (arch_<rung>) and is itself compiled once per rung by
// einsums_add_simd_dispatch_sources(). Each copy compiles at its rung's ISA,
// so VecTraits<T>::lanes and the Vec ops resolve to that rung's vector width.

#ifndef EINSUMS_STREAM_KERNEL_NS
#    error "StreamKernelBody.hpp requires EINSUMS_STREAM_KERNEL_NS to be defined before inclusion"
#endif

#include <Einsums/Config/Namespace.hpp>
#include <Einsums/SIMD/ComplexVec.hpp>
#include <Einsums/SIMD/Operations.hpp>
#include <Einsums/SIMD/Vec.hpp>

#include <complex>
#include <cstdint>
#include <type_traits>

EINSUMS_NAMESPACE_BEGIN(compute_graph::passes)
namespace EINSUMS_STREAM_KERNEL_NS {

/// Innermost stream loop for one member. See StreamKernel.hpp for the stride
/// triple contract. Real and complex types take the vectorized fast paths;
/// non-unit stride patterns and exotic element types take the scalar fallback.
template <typename T>
void stream_inner(T *cb, T const *sp, T const *w, T const alpha, int64_t const n, int64_t const co, int64_t const si, int64_t const wo,
                  int64_t const ds, int64_t const dc, int64_t const dw) {
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
        using namespace einsums::simd;
        constexpr int64_t L = VecTraits<T>::lanes;

        // (1,1,0) scaled AXPY: W is constant across the walk -> C[i] += coeff*S[i].
        if (ds == 1 && dc == 1 && dw == 0) {
            T const      coeff = alpha * w[wo];
            Vec<T> const vc    = broadcast(coeff);
            int64_t      i     = 0;
            for (; i + L <= n; i += L) {
                storeu(cb + co + i, fmadd(vc, loadu(sp + si + i), loadu(cb + co + i)));
            }
            for (; i < n; ++i) {
                cb[co + i] += coeff * sp[si + i];
            }
            return;
        }

        // (1,1,1) Hadamard FMA: three contiguous streams -> C[i] += alpha*S[i]*W[i].
        if (ds == 1 && dc == 1 && dw == 1) {
            Vec<T> const va = broadcast(alpha);
            int64_t      i  = 0;
            for (; i + L <= n; i += L) {
                Vec<T> const prod = mul(loadu(sp + si + i), loadu(w + wo + i));
                storeu(cb + co + i, fmadd(va, prod, loadu(cb + co + i)));
            }
            for (; i < n; ++i) {
                cb[co + i] += alpha * sp[si + i] * w[wo + i];
            }
            return;
        }

        // (1,0,1) dot reduction: C is fixed -> C[co] += alpha * sum_i S[i]*W[i].
        // Four independent accumulators keep four FMA chains in flight; one
        // chain would wait out the FMA latency on every step. The reordered
        // summation is the same class as BLAS; results match the scalar
        // oracle to tolerance.
        if (ds == 1 && dc == 0 && dw == 1) {
            T const *s    = sp + si;
            T const *x    = w + wo;
            Vec<T>   acc0 = broadcast(T{0});
            Vec<T>   acc1 = acc0;
            Vec<T>   acc2 = acc0;
            Vec<T>   acc3 = acc0;
            int64_t  i    = 0;
            for (; i + 4 * L <= n; i += 4 * L) {
                acc0 = fmadd(loadu(s + i), loadu(x + i), acc0);
                acc1 = fmadd(loadu(s + i + L), loadu(x + i + L), acc1);
                acc2 = fmadd(loadu(s + i + 2 * L), loadu(x + i + 2 * L), acc2);
                acc3 = fmadd(loadu(s + i + 3 * L), loadu(x + i + 3 * L), acc3);
            }
            for (; i + L <= n; i += L) {
                acc0 = fmadd(loadu(s + i), loadu(x + i), acc0);
            }
            Vec<T> const acc = add(add(acc0, acc1), add(acc2, acc3));
            T            lane_buf[L];
            storeu(lane_buf, acc);
            T sum = T{0};
            for (int64_t l = 0; l < L; ++l) {
                sum += lane_buf[l];
            }
            for (; i < n; ++i) {
                sum += s[i] * x[i];
            }
            cb[co] += alpha * sum;
            return;
        }
    } else if constexpr (std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>) {
        // Complex: the same three stride triples through the interleaved
        // CVec<U> ops (U = float/double). L is the COMPLEX lane count, half the
        // underlying real width. The pass rejects conjugated members, so plain
        // complex products suffice. On NEON<double> L collapses to 1 (2 real
        // lanes = 1 complex), so the win here is on the wider x86 rungs.
        using U = typename T::value_type;
        using namespace einsums::simd;
        constexpr int64_t L = VecTraits<U>::lanes / 2;

        // (1,1,0) scaled AXPY: C[i] += (alpha*W) * S[i].
        if (ds == 1 && dc == 1 && dw == 0) {
            T const       coeff = alpha * w[wo];
            CVec<U> const vc    = complex_broadcast(coeff);
            int64_t       i     = 0;
            for (; i + L <= n; i += L) {
                complex_storeu(cb + co + i, complex_fmadd(vc, complex_loadu(sp + si + i), complex_loadu(cb + co + i)));
            }
            for (; i < n; ++i) {
                cb[co + i] += coeff * sp[si + i];
            }
            return;
        }

        // (1,1,1) Hadamard FMA: C[i] += alpha * S[i] * W[i].
        if (ds == 1 && dc == 1 && dw == 1) {
            CVec<U> const va = complex_broadcast(alpha);
            int64_t       i  = 0;
            for (; i + L <= n; i += L) {
                CVec<U> const prod = complex_mul(complex_loadu(sp + si + i), complex_loadu(w + wo + i));
                complex_storeu(cb + co + i, complex_fmadd(va, prod, complex_loadu(cb + co + i)));
            }
            for (; i < n; ++i) {
                cb[co + i] += alpha * sp[si + i] * w[wo + i];
            }
            return;
        }

        // (1,0,1) dot reduction: C[co] += alpha * sum_i S[i]*W[i], with four
        // accumulators for the same reason as the real case.
        if (ds == 1 && dc == 0 && dw == 1) {
            T const *s    = sp + si;
            T const *x    = w + wo;
            CVec<U>  acc0 = complex_broadcast(T{0});
            CVec<U>  acc1 = acc0;
            CVec<U>  acc2 = acc0;
            CVec<U>  acc3 = acc0;
            int64_t  i    = 0;
            for (; i + 4 * L <= n; i += 4 * L) {
                acc0 = complex_fmadd(complex_loadu(s + i), complex_loadu(x + i), acc0);
                acc1 = complex_fmadd(complex_loadu(s + i + L), complex_loadu(x + i + L), acc1);
                acc2 = complex_fmadd(complex_loadu(s + i + 2 * L), complex_loadu(x + i + 2 * L), acc2);
                acc3 = complex_fmadd(complex_loadu(s + i + 3 * L), complex_loadu(x + i + 3 * L), acc3);
            }
            for (; i + L <= n; i += L) {
                acc0 = complex_fmadd(complex_loadu(s + i), complex_loadu(x + i), acc0);
            }
            CVec<U> const acc = complex_add(complex_add(acc0, acc1), complex_add(acc2, acc3));
            T             lane_buf[L];
            complex_storeu(lane_buf, acc);
            T sum{0};
            for (int64_t l = 0; l < L; ++l) {
                sum += lane_buf[l];
            }
            for (; i < n; ++i) {
                sum += s[i] * x[i];
            }
            cb[co] += alpha * sum;
            return;
        }
    }

    // Scalar strided-odometer fallback: exotic element types and every
    // non-unit / both-broadcast stride pattern.
    int64_t c = co, s = si, x = wo;
    for (int64_t i = 0; i < n; ++i) {
        cb[c] += alpha * sp[s] * w[x];
        s += ds;
        c += dc;
        x += dw;
    }
}

} // namespace EINSUMS_STREAM_KERNEL_NS
EINSUMS_NAMESPACE_END(compute_graph::passes)
