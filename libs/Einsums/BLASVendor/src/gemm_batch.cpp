//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/Config.hpp>

#include <Einsums/BLASVendor/Vendor.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Profile.hpp>

#include <algorithm>
#include <array>
#include <vector>

#include "Common.hpp"

#ifdef _OPENMP
#    include <omp.h>
#endif

EINSUMS_NAMESPACE_BEGIN(blas::vendor)

// Batch GEMM: batch_count GEMMs sharing every parameter but the pointers, as the vendor GEMM in a
// parallel loop. A native vendor batch could be added behind a config define.

extern "C" {
extern void FC_GLOBAL(sgemm, SGEMM)(char *, char *, int_t *, int_t *, int_t *, float *, float const *, int_t *, float const *, int_t *,
                                    float *, float *, int_t *);
extern void FC_GLOBAL(dgemm, DGEMM)(char *, char *, int_t *, int_t *, int_t *, double *, double const *, int_t *, double const *, int_t *,
                                    double *, double *, int_t *);
extern void FC_GLOBAL(cgemm, CGEMM)(char *, char *, int_t *, int_t *, int_t *, std::complex<float> *, std::complex<float> const *, int_t *,
                                    std::complex<float> const *, int_t *, std::complex<float> *, std::complex<float> *, int_t *);
extern void FC_GLOBAL(zgemm, ZGEMM)(char *, char *, int_t *, int_t *, int_t *, std::complex<double> *, std::complex<double> const *,
                                    int_t *, std::complex<double> const *, int_t *, std::complex<double> *, std::complex<double> *,
                                    int_t *);
}

namespace {

/// Largest dimension that the inline kernel below will take from the vendor GEMM.
constexpr int_t small_gemm_dim = 16;

template <typename T>
constexpr bool is_complex_v = false;
template <typename U>
constexpr bool is_complex_v<std::complex<U>> = true;

template <typename T>
inline T conj_if(T v, bool conj) {
    if constexpr (is_complex_v<T>) {
        return conj ? std::conj(v) : v;
    } else {
        return v;
    }
}

/// One tiny column-major GEMM, written so the innermost loop walks the unit-stride index.
template <typename T>
void small_gemm(char transa, char transb, int_t m, int_t n, int_t k, T alpha, T const *a, int_t lda, T const *b, int_t ldb, T beta, T *c,
                int_t ldc) {
    bool const ta = transa != 'N' && transa != 'n';
    bool const tb = transb != 'N' && transb != 'n';
    bool const ca = transa == 'C' || transa == 'c';
    bool const cb = transb == 'C' || transb == 'c';

    std::array<T, small_gemm_dim * small_gemm_dim> acc;
    for (int_t j = 0; j < n; j++) {
        T *cj = acc.data() + j * m;
        for (int_t i = 0; i < m; i++) {
            cj[i] = T{};
        }
        for (int_t p = 0; p < k; p++) {
            T const bp = conj_if(tb ? b[j + p * ldb] : b[p + j * ldb], cb);
            if (ta) {
                for (int_t i = 0; i < m; i++) {
                    cj[i] += conj_if(a[p + i * lda], ca) * bp;
                }
            } else {
                T const *ap = a + p * lda;
                for (int_t i = 0; i < m; i++) {
                    cj[i] += ap[i] * bp;
                }
            }
        }
    }
    // beta == 0 must not read C at all: BLAS lets the caller pass uninitialized memory.
    if (beta == T{}) {
        for (int_t j = 0; j < n; j++) {
            for (int_t i = 0; i < m; i++) {
                c[i + j * ldc] = alpha * acc[i + j * m];
            }
        }
    } else {
        for (int_t j = 0; j < n; j++) {
            for (int_t i = 0; i < m; i++) {
                c[i + j * ldc] = alpha * acc[i + j * m] + beta * c[i + j * ldc];
            }
        }
    }
}

/// Should this batch run on the inline kernel rather than on the vendor GEMM?
///
/// The vendor GEMM is faster alone but serializes when called concurrently (a 9x9 batch: 0.44 ms on
/// one thread, 1.04 ms on ten; the inline kernel goes 0.85 -> 0.21). So the cutoff rises with the
/// team: dim 10 at four threads, 16 at ten.
inline int_t small_gemm_cutoff(int_t threads) {
    return std::min<int_t>(small_gemm_dim, 6 + threads);
}

inline bool use_small_gemm(int_t m, int_t n, int_t k) {
#ifdef _OPENMP
    int_t const threads = omp_get_max_threads();
    if (threads <= 1) {
        return false;
    }
    int_t const cutoff = small_gemm_cutoff(threads);
    return m <= cutoff && n <= cutoff && k <= cutoff;
#else
    return false;
#endif
}

/// One vendor GEMM, with the Fortran-ABI pointer plumbing hidden.
///
/// Scalars by value, so workers never share the caller's stack slots (which TSan flags).
/// @{
inline void vendor_gemm(char transa, char transb, int_t m, int_t n, int_t k, float alpha, float const *a, int_t lda, float const *b,
                        int_t ldb, float beta, float *c, int_t ldc) {
    FC_GLOBAL(sgemm, SGEMM)(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
}
inline void vendor_gemm(char transa, char transb, int_t m, int_t n, int_t k, double alpha, double const *a, int_t lda, double const *b,
                        int_t ldb, double beta, double *c, int_t ldc) {
    FC_GLOBAL(dgemm, DGEMM)(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
}
inline void vendor_gemm(char transa, char transb, int_t m, int_t n, int_t k, std::complex<float> alpha, std::complex<float> const *a,
                        int_t lda, std::complex<float> const *b, int_t ldb, std::complex<float> beta, std::complex<float> *c, int_t ldc) {
    FC_GLOBAL(cgemm, CGEMM)(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
}
inline void vendor_gemm(char transa, char transb, int_t m, int_t n, int_t k, std::complex<double> alpha, std::complex<double> const *a,
                        int_t lda, std::complex<double> const *b, int_t ldb, std::complex<double> beta, std::complex<double> *c,
                        int_t ldc) {
    FC_GLOBAL(zgemm, ZGEMM)(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc);
}
/// @}

/// The uniform batch, shared by all four element types.
template <typename T>
void gemm_batch_impl(char transa, char transb, int_t m, int_t n, int_t k, T alpha, T const **a_array, int_t lda, T const **b_array,
                     int_t ldb, T beta, T **c_array, int_t ldc, int_t batch_count) {
    if (batch_count <= 0 || m == 0 || n == 0) {
        return;
    }

    // Static schedule for the uniform small kernel, dynamic for the vendor GEMM. Inside another team,
    // open no region of our own: a nested team around an OpenMP OpenBLAS miscomputes (the extractor
    // in ComputeGraph/Detail/GroupedBatchedGemm.hpp guards the same way).
    if (use_small_gemm(m, n, k)) {
#ifdef _OPENMP
#    pragma omp parallel for schedule(static) if (!omp_in_parallel())
#endif
        for (int_t i = 0; i < batch_count; i++) {
            small_gemm<T>(transa, transb, m, n, k, alpha, a_array[i], lda, b_array[i], ldb, beta, c_array[i], ldc);
        }
        return;
    }

#ifdef _OPENMP
#    pragma omp parallel for schedule(dynamic) if (!omp_in_parallel())
#endif
    for (int_t i = 0; i < batch_count; i++) {
        vendor_gemm(transa, transb, m, n, k, alpha, a_array[i], lda, b_array[i], ldb, beta, c_array[i], ldc);
    }
}

/// The grouped batch: every group's members flattened into one parallel loop.
///
/// One OpenMP region for the whole call. Per-group values go into small tables, and each item finds
/// its group by binary search.
template <typename T>
void gemm_batch_grouped_impl(char const *transa_array, char const *transb_array, int_t const *m_array, int_t const *n_array,
                             int_t const *k_array, T const *alpha_array, T const **a_array, int_t const *lda_array, T const **b_array,
                             int_t const *ldb_array, T const *beta_array, T **c_array, int_t const *ldc_array, int_t group_count,
                             int_t const *group_size) {
    if (group_count <= 0) {
        return;
    }

    // ``live`` holds the groups with work (m, n and members nonzero) and running member totals. A zero
    // k is not empty: it still scales C by beta.
    std::vector<int_t> live;
    std::vector<int_t> live_start;
    std::vector<int_t> live_offset;
    live.reserve(static_cast<size_t>(group_count));
    live_start.reserve(static_cast<size_t>(group_count) + 1);
    live_offset.reserve(static_cast<size_t>(group_count));

    int_t offset = 0;
    int_t total  = 0;
    for (int_t g = 0; g < group_count; ++g) {
        int_t const size = group_size[g];
        if (size > 0 && m_array[g] != 0 && n_array[g] != 0) {
            live.push_back(g);
            live_start.push_back(total);
            live_offset.push_back(offset);
            total += size;
        }
        offset += size > 0 ? size : 0;
    }
    if (total == 0) {
        return;
    }
    live_start.push_back(total);

    // Each group picks its kernel by its own dims.
    std::vector<char> small(live.size());
    for (size_t s = 0; s < live.size(); ++s) {
        int_t const g = live[s];
        small[s]      = static_cast<char>(use_small_gemm(m_array[g], n_array[g], k_array[g]));
    }

    auto const *starts = live_start.data();
    auto const  n_live = static_cast<int_t>(live.size());

    // Same guard, same reason as the uniform batch above.
#ifdef _OPENMP
#    pragma omp parallel for schedule(dynamic) if (!omp_in_parallel())
#endif
    for (int_t t = 0; t < total; ++t) {
        // The last group whose start is <= t. Groups are non-empty, so this
        // lands on the group holding item t.
        auto const  s = static_cast<int_t>(std::upper_bound(starts + 1, starts + n_live + 1, t) - (starts + 1));
        int_t const g = live[s];
        int_t const i = live_offset[s] + (t - starts[s]);

        if (small[s]) {
            small_gemm<T>(transa_array[g], transb_array[g], m_array[g], n_array[g], k_array[g], alpha_array[g], a_array[i], lda_array[g],
                          b_array[i], ldb_array[g], beta_array[g], c_array[i], ldc_array[g]);
        } else {
            vendor_gemm(transa_array[g], transb_array[g], m_array[g], n_array[g], k_array[g], alpha_array[g], a_array[i], lda_array[g],
                        b_array[i], ldb_array[g], beta_array[g], c_array[i], ldc_array[g]);
        }
    }
}

} // namespace

void sgemm_batch(char transa, char transb, int_t m, int_t n, int_t k, float alpha, float const **a_array, int_t lda, float const **b_array,
                 int_t ldb, float beta, float **c_array, int_t ldc, int_t batch_count) {
    LabeledSection0();
    gemm_batch_impl<float>(transa, transb, m, n, k, alpha, a_array, lda, b_array, ldb, beta, c_array, ldc, batch_count);
}

void dgemm_batch(char transa, char transb, int_t m, int_t n, int_t k, double alpha, double const **a_array, int_t lda,
                 double const **b_array, int_t ldb, double beta, double **c_array, int_t ldc, int_t batch_count) {
    LabeledSection0();
    gemm_batch_impl<double>(transa, transb, m, n, k, alpha, a_array, lda, b_array, ldb, beta, c_array, ldc, batch_count);
}

void cgemm_batch(char transa, char transb, int_t m, int_t n, int_t k, std::complex<float> alpha, std::complex<float> const **a_array,
                 int_t lda, std::complex<float> const **b_array, int_t ldb, std::complex<float> beta, std::complex<float> **c_array,
                 int_t ldc, int_t batch_count) {
    LabeledSection0();
    gemm_batch_impl<std::complex<float>>(transa, transb, m, n, k, alpha, a_array, lda, b_array, ldb, beta, c_array, ldc, batch_count);
}

void zgemm_batch(char transa, char transb, int_t m, int_t n, int_t k, std::complex<double> alpha, std::complex<double> const **a_array,
                 int_t lda, std::complex<double> const **b_array, int_t ldb, std::complex<double> beta, std::complex<double> **c_array,
                 int_t ldc, int_t batch_count) {
    LabeledSection0();
    gemm_batch_impl<std::complex<double>>(transa, transb, m, n, k, alpha, a_array, lda, b_array, ldb, beta, c_array, ldc, batch_count);
}

void sgemm_batch_grouped(char const *transa_array, char const *transb_array, int_t const *m_array, int_t const *n_array,
                         int_t const *k_array, float const *alpha_array, float const **a_array, int_t const *lda_array,
                         float const **b_array, int_t const *ldb_array, float const *beta_array, float **c_array, int_t const *ldc_array,
                         int_t group_count, int_t const *group_size) {
    LabeledSection0();
    gemm_batch_grouped_impl<float>(transa_array, transb_array, m_array, n_array, k_array, alpha_array, a_array, lda_array, b_array,
                                   ldb_array, beta_array, c_array, ldc_array, group_count, group_size);
}

void dgemm_batch_grouped(char const *transa_array, char const *transb_array, int_t const *m_array, int_t const *n_array,
                         int_t const *k_array, double const *alpha_array, double const **a_array, int_t const *lda_array,
                         double const **b_array, int_t const *ldb_array, double const *beta_array, double **c_array, int_t const *ldc_array,
                         int_t group_count, int_t const *group_size) {
    LabeledSection0();
    gemm_batch_grouped_impl<double>(transa_array, transb_array, m_array, n_array, k_array, alpha_array, a_array, lda_array, b_array,
                                    ldb_array, beta_array, c_array, ldc_array, group_count, group_size);
}

void cgemm_batch_grouped(char const *transa_array, char const *transb_array, int_t const *m_array, int_t const *n_array,
                         int_t const *k_array, std::complex<float> const *alpha_array, std::complex<float> const **a_array,
                         int_t const *lda_array, std::complex<float> const **b_array, int_t const *ldb_array,
                         std::complex<float> const *beta_array, std::complex<float> **c_array, int_t const *ldc_array, int_t group_count,
                         int_t const *group_size) {
    LabeledSection0();
    gemm_batch_grouped_impl<std::complex<float>>(transa_array, transb_array, m_array, n_array, k_array, alpha_array, a_array, lda_array,
                                                 b_array, ldb_array, beta_array, c_array, ldc_array, group_count, group_size);
}

void zgemm_batch_grouped(char const *transa_array, char const *transb_array, int_t const *m_array, int_t const *n_array,
                         int_t const *k_array, std::complex<double> const *alpha_array, std::complex<double> const **a_array,
                         int_t const *lda_array, std::complex<double> const **b_array, int_t const *ldb_array,
                         std::complex<double> const *beta_array, std::complex<double> **c_array, int_t const *ldc_array, int_t group_count,
                         int_t const *group_size) {
    LabeledSection0();
    gemm_batch_grouped_impl<std::complex<double>>(transa_array, transb_array, m_array, n_array, k_array, alpha_array, a_array, lda_array,
                                                  b_array, ldb_array, beta_array, c_array, ldc_array, group_count, group_size);
}

EINSUMS_NAMESPACE_END(blas::vendor)
