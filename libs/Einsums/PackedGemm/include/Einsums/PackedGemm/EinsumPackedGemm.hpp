//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

// This header is included from Dispatch.hpp.

#include <Einsums/BLAS.hpp>
#include <Einsums/Concepts/TensorConcepts.hpp>
#include <Einsums/Logging.hpp>
#include <Einsums/PackedGemm/ContractionKey.hpp>
#include <Einsums/PackedGemm/Packing.hpp>
#include <Einsums/Profile/Profile.hpp>

#include <algorithm>
#include <string>
#include <unordered_set>
#include <vector>

#ifdef _OPENMP
#    include <omp.h>
#endif

namespace einsums::packed_gemm {

// ---------------------------------------------------------------------------
// Compile-time helpers
// ---------------------------------------------------------------------------

/// Extract the static letter string from each index type in a tuple.
template <typename... Indices>
std::vector<std::string> index_letters_from_tuple(std::tuple<Indices...> const & /*unused*/) {
    return {std::string(Indices::letter)...};
}

/// De-duplicate while preserving first-occurrence order.
inline std::vector<std::string> unique_ordered(std::vector<std::string> const &v) {
    std::vector<std::string>        result;
    std::unordered_set<std::string> seen;
    for (auto const &s : v) {
        if (seen.insert(s).second) {
            result.push_back(s);
        }
    }
    return result;
}

/// Compute the unique link indices: A intersection B minus C, preserving order from A.
inline std::vector<std::string> compute_link_indices(std::vector<std::string> const &a_raw, std::vector<std::string> const &b_raw,
                                                     std::vector<std::string> const &c_unique) {
    std::unordered_set<std::string> const b_set(b_raw.begin(), b_raw.end());
    std::unordered_set<std::string> const c_set(c_unique.begin(), c_unique.end());
    std::vector<std::string>              link;
    std::unordered_set<std::string>       seen;
    for (auto const &idx : a_raw) {
        if (b_set.count(idx) && !c_set.count(idx) && seen.insert(idx).second) {
            link.push_back(idx);
        }
    }
    return link;
}

// ---------------------------------------------------------------------------
// Runtime tensor info extraction (works for any BasicTensor rank)
// ---------------------------------------------------------------------------

template <einsums::BasicTensorConcept TensorType>
TensorDescriptor tensor_descriptor(TensorType const &t) {
    TensorDescriptor td;
    if constexpr (requires { std::remove_cvref_t<TensorType>::Rank; }) {
        td.rank = std::remove_cvref_t<TensorType>::Rank;
    } else {
        td.rank = t.rank();
    }
    td.dtype = get_scalar_type<typename TensorType::ValueType>();
    return td;
}

// ---------------------------------------------------------------------------
// BLIS-style packed contraction with BLAS GEMM tiles
// ---------------------------------------------------------------------------

/// @brief Execute a tensor contraction via Pack-A / Pack-B + BLAS GEMM tiles (BLIS-style).
///
/// For multi-K contractions (rank-3+), flattens A and B into contiguous M*K / K*N buffers
/// and calls BLAS GEMM directly.  For single-K, uses BLIS-style tiled packing with BLAS
/// GEMM per tile.
template <typename ValueType, einsums::BasicTensorConcept CType, einsums::BasicTensorConcept AType, einsums::BasicTensorConcept BType>
void blis_contraction(PackingPlan const &plan, CType &C, AType const &A, BType const &B, ValueType alpha, ValueType beta,
                      bool conj_a = false, bool conj_b = false) {
    LabeledSection0();
    auto const &cfg = cpu_config();
    int const   MR  = cfg.MR;
    int const   NR  = cfg.NR;

    // Cache-aware blocking: tile sizes adapt to sizeof(ValueType) and CPU cache hierarchy.
    auto const blk = compute_blocking(static_cast<int64_t>(sizeof(ValueType)));

    int64_t const M          = plan.M_total;
    int64_t const N          = plan.N_total;
    int64_t const K          = plan.K_total;
    bool const    multi_m    = (plan.c_m_dims.size() > 1);
    bool const    multi_n    = (plan.c_n_dims.size() > 1);
    int64_t const C_m_stride = plan.c_m_dims[0].tensor_stride;
    int64_t const C_n_stride = plan.c_n_dims[0].tensor_stride;

    // For multi-M/N, col_major detection uses the first C_m dim stride.
    // The flat-to-offset conversion handles the rest.
    bool const C_col_major = (!multi_m && C_m_stride == 1);

    constexpr bool is_complex =
        (get_scalar_type<ValueType>() == ScalarType::Complex64 || get_scalar_type<ValueType>() == ScalarType::Complex128);

    // -------------------------------------------------------------------------
    // Batch loop: iterate over all batch slices.
    // For non-batched contractions, batch_total=1 and batch_dims is empty,
    // so this is a single iteration with zero offsets.
    // -------------------------------------------------------------------------
    auto const  &batch_dims = plan.batch_dims;
    size_t const nb         = batch_dims.size();

    // -------------------------------------------------------------------------
    // Batch GEMM fast path: if single-K, single-M, single-N with compatible
    // strides, precompute pointer arrays and call gemm_batch() for all batches
    // at once. This is much faster than looping over batches individually.
    // -------------------------------------------------------------------------
    if (plan.batch_total > 1 && plan.k_dims_in_a.size() == 1 && !multi_m && !multi_n) {
        // NOLINTNEXTLINE(readability-identifier-naming)
        using blas_int = einsums::blas::int_t;

        int64_t const m_stride   = plan.m_dims[0].tensor_stride;
        int64_t const n_stride   = plan.n_dims[0].tensor_stride;
        int64_t const k_stride_a = plan.k_dims_in_a[0].tensor_stride;
        int64_t const k_stride_b = plan.k_dims_in_b[0].tensor_stride;

        // Check if strides are compatible with a simple GEMM call
        // (same logic as the single-K fast path inside the batch loop)
        char     transA = 'N', transB = 'N';
        blas_int lda_val = 0, ldb_val = 0, ldc_val = 0;
        bool     can_batch = false;

        if (C_col_major && m_stride == 1) {
            transA  = 'N';
            lda_val = static_cast<blas_int>(k_stride_a);
            ldc_val = static_cast<blas_int>(C_n_stride);
            if (k_stride_b == 1) {
                transB    = 'N';
                ldb_val   = static_cast<blas_int>(n_stride);
                can_batch = true;
            } else if (n_stride == 1) {
                transB    = 'T';
                ldb_val   = static_cast<blas_int>(k_stride_b);
                can_batch = true;
            }
        } else if (C_col_major && k_stride_a == 1) {
            transA  = 'T';
            lda_val = static_cast<blas_int>(m_stride);
            ldc_val = static_cast<blas_int>(C_n_stride);
            if (k_stride_b == 1) {
                transB    = 'N';
                ldb_val   = static_cast<blas_int>(n_stride);
                can_batch = true;
            } else if (n_stride == 1) {
                transB    = 'T';
                ldb_val   = static_cast<blas_int>(k_stride_b);
                can_batch = true;
            }
        }

        if (can_batch && !conj_a && !conj_b) {
            LabeledSection("gemm_batch fast path");

            // Precompute pointer arrays
            int64_t const                  bt = plan.batch_total;
            std::vector<ValueType const *> a_ptrs(static_cast<size_t>(bt));
            std::vector<ValueType const *> b_ptrs(static_cast<size_t>(bt));
            std::vector<ValueType *>       c_ptrs(static_cast<size_t>(bt));

            for (int64_t batch = 0; batch < bt; ++batch) {
                int64_t a_off = 0, b_off = 0, c_off = 0;
                int64_t rem = batch;
                for (int d = static_cast<int>(nb) - 1; d >= 0; --d) {
                    int64_t const bi = rem % batch_dims[static_cast<size_t>(d)].size;
                    rem /= batch_dims[static_cast<size_t>(d)].size;
                    a_off += bi * batch_dims[static_cast<size_t>(d)].a_stride;
                    b_off += bi * batch_dims[static_cast<size_t>(d)].b_stride;
                    c_off += bi * batch_dims[static_cast<size_t>(d)].c_stride;
                }
                a_ptrs[static_cast<size_t>(batch)] = A.data() + a_off;
                b_ptrs[static_cast<size_t>(batch)] = B.data() + b_off;
                c_ptrs[static_cast<size_t>(batch)] = C.data() + c_off;
            }

            einsums::blas::gemm_batch<ValueType>(transA, transB, static_cast<blas_int>(M), static_cast<blas_int>(N),
                                                 static_cast<blas_int>(K), alpha, a_ptrs.data(), lda_val, b_ptrs.data(), ldb_val, beta,
                                                 c_ptrs.data(), ldc_val, static_cast<blas_int>(bt));
            return;
        }
    }

    // -------------------------------------------------------------------------
    // Per-batch loop (fallback when gemm_batch can't be used)
    // -------------------------------------------------------------------------
    bool const parallel_batch = (plan.batch_total >= 4) && (M * N < 10000);

#ifdef _OPENMP
#    pragma omp parallel for schedule(dynamic) if (parallel_batch)
#endif
    for (int64_t batch = 0; batch < plan.batch_total; ++batch) {
        // Compute batch offsets for A, B, C.
        int64_t a_batch_off = 0, b_batch_off = 0, c_batch_off = 0;
        if (nb > 0) {
            int64_t rem = batch;
            for (int d = static_cast<int>(nb) - 1; d >= 0; --d) {
                int64_t const bi = rem % batch_dims[static_cast<size_t>(d)].size;
                rem /= batch_dims[static_cast<size_t>(d)].size;
                a_batch_off += bi * batch_dims[static_cast<size_t>(d)].a_stride;
                b_batch_off += bi * batch_dims[static_cast<size_t>(d)].b_stride;
                c_batch_off += bi * batch_dims[static_cast<size_t>(d)].c_stride;
            }
        }

        ValueType       *C_data = C.data() + c_batch_off;
        ValueType const *A_data = A.data() + a_batch_off;
        ValueType const *B_data = B.data() + b_batch_off;

        // -------------------------------------------------------------------------
        // Multi-K fast path: flatten A and B into contiguous M*K / K*N buffers,
        // then call BLAS GEMM directly.
        // -------------------------------------------------------------------------
        if (plan.k_dims_in_a.size() > 1 && !multi_m && !multi_n) {
            // Multi-K fast path: only for single-M, single-N (can map to flat BLAS GEMM).
            LabeledSection("flatten + GEMM");
            // NOLINTNEXTLINE(readability-identifier-naming)
            using blas_int = einsums::blas::int_t;

            auto const   &k_dims_a = plan.k_dims_in_a;
            auto const   &k_dims_b = plan.k_dims_in_b;
            int64_t const m_stride = plan.m_dims[0].tensor_stride;
            int64_t const n_stride = plan.n_dims[0].tensor_stride;
            size_t const  nk       = k_dims_a.size();

            std::vector<int64_t> k_cum(nk);
            k_cum[nk - 1] = 1;
            for (int d = static_cast<int>(nk) - 2; d >= 0; --d) {
                // NOLINTNEXTLINE(bugprone-misplaced-widening-cast)
                k_cum[static_cast<size_t>(d)] = k_cum[static_cast<size_t>(d + 1)] * k_dims_a[static_cast<size_t>(d + 1)].size;
            }

            // Zero-copy detection
            bool a_zero_copy = (m_stride == 1) && !conj_a;
            bool b_zero_copy = (n_stride == 1) && !conj_b;
            for (size_t d = 0; d < nk && (a_zero_copy || b_zero_copy); ++d) {
                if (a_zero_copy && k_dims_a[d].tensor_stride != k_cum[d] * M)
                    a_zero_copy = false;
                if (b_zero_copy && k_dims_b[d].tensor_stride != k_cum[d] * N)
                    b_zero_copy = false;
            }

            // Both zero-copy: single GEMM, no copy at all
            if (a_zero_copy && b_zero_copy) {
                if (C_col_major) {
                    einsums::blas::gemm<ValueType>('N', 'T', static_cast<blas_int>(M), static_cast<blas_int>(N), static_cast<blas_int>(K),
                                                   alpha, A_data, static_cast<blas_int>(M), B_data, static_cast<blas_int>(N), beta, C_data,
                                                   static_cast<blas_int>(C_n_stride));
                } else {
                    einsums::blas::gemm<ValueType>('N', 'T', static_cast<blas_int>(N), static_cast<blas_int>(M), static_cast<blas_int>(K),
                                                   alpha, B_data, static_cast<blas_int>(N), A_data, static_cast<blas_int>(M), beta, C_data,
                                                   static_cast<blas_int>(C_m_stride));
                }
                continue; // next batch slice
            }

            // At least one side needs copying.
            //
            // Strategy: HPTT-transpose the full tensor into a flat M*K / K*N buffer
            // (cache-blocked, SIMD-optimized), then call KC-tiled BLAS GEMM over the
            // already-contiguous flat buffer.  Falls back to scalar gather loops only
            // on Windows (no HPTT) or when the source tensor is non-contiguous.

            // Allocate full-size flat buffers (M*K and K*N) for sides that need copying.
            static thread_local std::vector<ValueType> tls_A_flat, tls_B_flat;
            ValueType                                 *A_flat = nullptr;
            ValueType                                 *B_flat = nullptr;
            if (!a_zero_copy) {
                tls_A_flat.resize(static_cast<size_t>(M * K));
                A_flat = tls_A_flat.data();
            }
            if (!b_zero_copy) {
                tls_B_flat.resize(static_cast<size_t>(K * N));
                B_flat = tls_B_flat.data();
            }

#if !defined(EINSUMS_WINDOWS)
            // Read ranks at runtime so the path works for both compile-time-rank
            // (Tensor<T, K>) and runtime-rank (RuntimeTensor<T, Alloc>) operands.
            auto rank_of = [](auto const &t) -> int {
                using TT = std::remove_cvref_t<decltype(t)>;
                // TT::Rank exists for BOTH compile-time tensors (Rank = K >= 0) and
                // runtime-rank tensors (Rank = dynamic_rank = -1, a sentinel). Only
                // trust it when it is a real rank; otherwise read the live rank.
                if constexpr (requires { TT::Rank; }) {
                    if constexpr (TT::Rank >= 0) {
                        return static_cast<int>(TT::Rank);
                    } else {
                        return static_cast<int>(t.rank());
                    }
                } else {
                    return static_cast<int>(t.rank());
                }
            };
            int const rank_a_rt = rank_of(A);
            int const rank_b_rt = rank_of(B);

            // Check contiguity for HPTT (only for sides that need copying).
            bool use_hptt = true;
            if (!a_zero_copy) {
                int64_t expected = 1;
                for (int i = 0; i < rank_a_rt; ++i) {
                    if (static_cast<int64_t>(A.stride(i)) != expected) {
                        use_hptt = false;
                        break;
                    }
                    expected *= static_cast<int64_t>(A.dim(i));
                }
            }
            if (use_hptt && !b_zero_copy) {
                int64_t expected = 1;
                for (int i = 0; i < rank_b_rt; ++i) {
                    if (static_cast<int64_t>(B.stride(i)) != expected) {
                        use_hptt = false;
                        break;
                    }
                    expected *= static_cast<int64_t>(B.dim(i));
                }
            }

            if (use_hptt) {
                // HPTT-transpose the full tensor(s) into flat M*K / K*N layout once,
                // then do KC-tiled GEMM over the contiguous flat buffers.
                int num_threads = 1;
#    ifdef _OPENMP
                num_threads = omp_get_max_threads();
#    endif

                if (!a_zero_copy) {
                    std::vector<int>    perm_a(rank_a_rt);
                    std::vector<size_t> sizes_a(rank_a_rt);
                    for (int i = 0; i < rank_a_rt; ++i) {
                        sizes_a[i] = A.dim(static_cast<size_t>(i));
                    }
                    perm_a[0] = static_cast<int>(plan.m_dims[0].tensor_pos);
                    for (size_t i = 0; i < nk; ++i) {
                        perm_a[nk - i] = static_cast<int>(k_dims_a[i].tensor_pos);
                    }
                    hptt_transpose(perm_a.data(), rank_a_rt, A_data, sizes_a.data(), A_flat, num_threads, conj_a);
                }
                if (!b_zero_copy) {
                    std::vector<int>    perm_b(rank_b_rt);
                    std::vector<size_t> sizes_b(rank_b_rt);
                    for (int i = 0; i < rank_b_rt; ++i) {
                        sizes_b[i] = B.dim(static_cast<size_t>(i));
                    }
                    perm_b[0] = static_cast<int>(plan.n_dims[0].tensor_pos);
                    for (size_t i = 0; i < nk; ++i) {
                        perm_b[nk - i] = static_cast<int>(k_dims_b[i].tensor_pos);
                    }
                    hptt_transpose(perm_b.data(), rank_b_rt, B_data, sizes_b.data(), B_flat, num_threads, conj_b);
                }

                // A_flat is now col-major M*K; B_flat is row-major K*N.
                // KC-tiled GEMM over the flat buffers.
                ValueType const *A_base = a_zero_copy ? A_data : A_flat;
                ValueType const *B_base = b_zero_copy ? B_data : B_flat;
                int64_t const    KC     = std::min(K, blk.KC);
                for (int64_t kc = 0; kc < K; kc += KC) {
                    int64_t const   kc_len = std::min(KC, K - kc);
                    ValueType const beta_k = (kc == 0) ? beta : ValueType{1};

                    // A_base is M*K col-major: column kc starts at A_base + kc*M
                    // B_base is K*N row-major: row kc starts at B_base + kc*N
                    ValueType const *A_ptr = A_base + kc * M;
                    ValueType const *B_ptr = B_base + kc * N;

                    if (C_col_major) {
                        einsums::blas::gemm<ValueType>('N', 'T', static_cast<blas_int>(M), static_cast<blas_int>(N),
                                                       static_cast<blas_int>(kc_len), alpha, A_ptr, static_cast<blas_int>(M), B_ptr,
                                                       static_cast<blas_int>(N), beta_k, C_data, static_cast<blas_int>(C_n_stride));
                    } else {
                        einsums::blas::gemm<ValueType>('N', 'T', static_cast<blas_int>(N), static_cast<blas_int>(M),
                                                       static_cast<blas_int>(kc_len), alpha, B_ptr, static_cast<blas_int>(N), A_ptr,
                                                       static_cast<blas_int>(M), beta_k, C_data, static_cast<blas_int>(C_m_stride));
                    }
                }
            } else
#endif // !EINSUMS_WINDOWS
            {
                // Scalar gather fallback (Windows or non-contiguous tensors).
                int64_t const KC = std::min(K, blk.KC);
                for (int64_t kc = 0; kc < K; kc += KC) {
                    int64_t const   kc_len = std::min(KC, K - kc);
                    ValueType const beta_k = (kc == 0) ? beta : ValueType{1};

                    ValueType const *A_ptr;
                    if (a_zero_copy) {
                        A_ptr = A_data + kc * M;
                    } else {
                        ValueType *A_tile = A_flat; // reuse start of buffer for each tile
                        for (int64_t kf = 0; kf < kc_len; ++kf) {
                            int64_t off_a = 0, rem = kc + kf;
                            for (size_t d = 0; d < nk; ++d) {
                                off_a += (rem / k_cum[d]) * k_dims_a[d].tensor_stride;
                                rem = rem % k_cum[d];
                            }
                            if (m_stride == 1 && !conj_a) {
                                std::memcpy(A_tile + kf * M, A_data + off_a, static_cast<size_t>(M) * sizeof(ValueType));
                            } else {
                                for (int64_t m = 0; m < M; ++m) {
                                    ValueType val = A_data[m * m_stride + off_a];
                                    if constexpr (is_complex) {
                                        if (conj_a)
                                            val = std::conj(val);
                                    }
                                    A_tile[m + kf * M] = val;
                                }
                            }
                        }
                        A_ptr = A_tile;
                    }

                    ValueType const *B_ptr;
                    if (b_zero_copy) {
                        B_ptr = B_data + kc * N;
                    } else {
                        ValueType *B_tile = B_flat; // reuse start of buffer for each tile
                        for (int64_t kf = 0; kf < kc_len; ++kf) {
                            int64_t off_b = 0, rem = kc + kf;
                            for (size_t d = 0; d < nk; ++d) {
                                off_b += (rem / k_cum[d]) * k_dims_b[d].tensor_stride;
                                rem = rem % k_cum[d];
                            }
                            if (n_stride == 1 && !conj_b) {
                                std::memcpy(B_tile + kf * N, B_data + off_b, static_cast<size_t>(N) * sizeof(ValueType));
                            } else {
                                for (int64_t n = 0; n < N; ++n) {
                                    ValueType val = B_data[off_b + n * n_stride];
                                    if constexpr (is_complex) {
                                        if (conj_b)
                                            val = std::conj(val);
                                    }
                                    B_tile[kf * N + n] = val;
                                }
                            }
                        }
                        B_ptr = B_tile;
                    }

                    if (C_col_major) {
                        einsums::blas::gemm<ValueType>('N', 'T', static_cast<blas_int>(M), static_cast<blas_int>(N),
                                                       static_cast<blas_int>(kc_len), alpha, A_ptr, static_cast<blas_int>(M), B_ptr,
                                                       static_cast<blas_int>(N), beta_k, C_data, static_cast<blas_int>(C_n_stride));
                    } else {
                        einsums::blas::gemm<ValueType>('N', 'T', static_cast<blas_int>(N), static_cast<blas_int>(M),
                                                       static_cast<blas_int>(kc_len), alpha, B_ptr, static_cast<blas_int>(N), A_ptr,
                                                       static_cast<blas_int>(M), beta_k, C_data, static_cast<blas_int>(C_m_stride));
                    }
                }
            }

            // Reclaim excess thread-local buffer memory to avoid bloat across
            // contractions of varying sizes.
            auto shrink = [](auto &v) {
                if (v.capacity() > 2 * v.size() && v.capacity() > 4096) {
                    v.shrink_to_fit();
                }
            };
            shrink(tls_A_flat);
            shrink(tls_B_flat);

            continue; // next batch slice
        }

        // -------------------------------------------------------------------------
        // Single-K fast path: call BLAS GEMM directly without packing.
        //
        // When K has a single dimension, the contraction is a standard GEMM with
        // strides.  BLAS can handle this directly via lda/ldb/ldc parameters,
        // avoiding the expensive pack_A/pack_B + tiled micro-GEMM.
        // -------------------------------------------------------------------------
        if (plan.k_dims_in_a.size() == 1 && !multi_m && !multi_n) {
            // Single-K fast path: only for single-M, single-N (direct BLAS GEMM dispatch).
            // NOLINTNEXTLINE(readability-identifier-naming)
            using blas_int = einsums::blas::int_t;

            int64_t const m_stride   = plan.m_dims[0].tensor_stride;
            int64_t const n_stride   = plan.n_dims[0].tensor_stride;
            int64_t const k_stride_a = plan.k_dims_in_a[0].tensor_stride;
            int64_t const k_stride_b = plan.k_dims_in_b[0].tensor_stride;

            // Try to map the strides to a BLAS gemm call.
            // BLAS gemm(transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc)
            // expects column-major storage: for transA='N', A is lda×K with lda≥M.
            //
            // Our tensor layout:
            //   A[m,k]: element at m*m_stride + k*k_stride_a
            //   B[k,n]: element at k*k_stride_b + n*n_stride
            //   C[m,n]: element at m*C_m_stride + n*C_n_stride
            //
            // For complex types with conjugation, BLAS uses 'C' (conjugate transpose)
            // instead of 'T'.  When 'N' (no transpose) with conjugation is needed,
            // BLAS has no flag — fall through to the BLIS tiled path which conjugates
            // during packing.
            bool dispatched = false;

            // Helper: upgrade 'T' to 'C' when conjugation is requested for complex types.
            auto trans_flag = [](char base, bool conj) -> char {
                if constexpr (is_complex) {
                    if (conj && base == 'T')
                        return 'C';
                }
                return base;
            };
            // Check: BLAS cannot apply conjugation without transpose ('N' + conj).
            auto can_dispatch_n = [](bool conj) -> bool {
                if constexpr (is_complex) {
                    return !conj;
                }
                return true;
            };

            if (C_col_major) {
                // C is column-major (m_stride_c = 1, ldc = n_stride_c)
                if (m_stride == 1) {
                    // A col-major in M → transA='N'
                    if (!can_dispatch_n(conj_a)) {
                        // conj(A) without transpose: can't dispatch
                    } else if (k_stride_b == 1) {
                        // B col-major in K → transB='N'
                        if (can_dispatch_n(conj_b)) {
                            einsums::blas::gemm<ValueType>(trans_flag('N', conj_a), trans_flag('N', conj_b), static_cast<blas_int>(M),
                                                           static_cast<blas_int>(N), static_cast<blas_int>(K), alpha, A_data,
                                                           static_cast<blas_int>(k_stride_a), B_data, static_cast<blas_int>(n_stride), beta,
                                                           C_data, static_cast<blas_int>(C_n_stride));
                            dispatched = true;
                        }
                    } else if (n_stride == 1) {
                        // B col-major in N → transB='T'
                        einsums::blas::gemm<ValueType>(trans_flag('N', conj_a), trans_flag('T', conj_b), static_cast<blas_int>(M),
                                                       static_cast<blas_int>(N), static_cast<blas_int>(K), alpha, A_data,
                                                       static_cast<blas_int>(k_stride_a), B_data, static_cast<blas_int>(k_stride_b), beta,
                                                       C_data, static_cast<blas_int>(C_n_stride));
                        dispatched = true;
                    }
                } else if (k_stride_a == 1) {
                    // A col-major in K → transA='T'
                    if (k_stride_b == 1) {
                        // B col-major in K → transB='N'
                        if (can_dispatch_n(conj_b)) {
                            einsums::blas::gemm<ValueType>(trans_flag('T', conj_a), trans_flag('N', conj_b), static_cast<blas_int>(M),
                                                           static_cast<blas_int>(N), static_cast<blas_int>(K), alpha, A_data,
                                                           static_cast<blas_int>(m_stride), B_data, static_cast<blas_int>(n_stride), beta,
                                                           C_data, static_cast<blas_int>(C_n_stride));
                            dispatched = true;
                        }
                    } else if (n_stride == 1) {
                        einsums::blas::gemm<ValueType>(trans_flag('T', conj_a), trans_flag('T', conj_b), static_cast<blas_int>(M),
                                                       static_cast<blas_int>(N), static_cast<blas_int>(K), alpha, A_data,
                                                       static_cast<blas_int>(m_stride), B_data, static_cast<blas_int>(k_stride_b), beta,
                                                       C_data, static_cast<blas_int>(C_n_stride));
                        dispatched = true;
                    }
                }
            } else if (C_n_stride == 1) {
                // C is row-major (n_stride_c = 1, ldc = m_stride_c)
                // Use identity: C^T = (alpha*A*B + beta*C)^T = alpha*B^T*A^T + beta*C^T
                // Note: A and B are swapped in the BLAS call, so conj flags swap too.
                if (n_stride == 1) {
                    // B is the BLAS "A" arg → transA_blas='N', conj_b applies
                    if (!can_dispatch_n(conj_b)) {
                        // conj(B) without transpose: can't dispatch
                    } else if (m_stride == 1) {
                        // A is the BLAS "B" arg → transB_blas='T', conj_a applies
                        einsums::blas::gemm<ValueType>(trans_flag('N', conj_b), trans_flag('T', conj_a), static_cast<blas_int>(N),
                                                       static_cast<blas_int>(M), static_cast<blas_int>(K), alpha, B_data,
                                                       static_cast<blas_int>(k_stride_b), A_data, static_cast<blas_int>(k_stride_a), beta,
                                                       C_data, static_cast<blas_int>(C_m_stride));
                        dispatched = true;
                    } else if (k_stride_a == 1) {
                        // A is the BLAS "B" arg → transB_blas='N', conj_a applies
                        if (can_dispatch_n(conj_a)) {
                            einsums::blas::gemm<ValueType>(trans_flag('N', conj_b), trans_flag('N', conj_a), static_cast<blas_int>(N),
                                                           static_cast<blas_int>(M), static_cast<blas_int>(K), alpha, B_data,
                                                           static_cast<blas_int>(k_stride_b), A_data, static_cast<blas_int>(m_stride), beta,
                                                           C_data, static_cast<blas_int>(C_m_stride));
                            dispatched = true;
                        }
                    }
                } else if (k_stride_b == 1) {
                    // B is the BLAS "A" arg → transA_blas='T', conj_b applies
                    if (m_stride == 1) {
                        // A is the BLAS "B" arg → transB_blas='T', conj_a applies
                        einsums::blas::gemm<ValueType>(trans_flag('T', conj_b), trans_flag('T', conj_a), static_cast<blas_int>(N),
                                                       static_cast<blas_int>(M), static_cast<blas_int>(K), alpha, B_data,
                                                       static_cast<blas_int>(n_stride), A_data, static_cast<blas_int>(k_stride_a), beta,
                                                       C_data, static_cast<blas_int>(C_m_stride));
                        dispatched = true;
                    } else if (k_stride_a == 1) {
                        // A is the BLAS "B" arg → transB_blas='N', conj_a applies
                        if (can_dispatch_n(conj_a)) {
                            einsums::blas::gemm<ValueType>(trans_flag('T', conj_b), trans_flag('N', conj_a), static_cast<blas_int>(N),
                                                           static_cast<blas_int>(M), static_cast<blas_int>(K), alpha, B_data,
                                                           static_cast<blas_int>(n_stride), A_data, static_cast<blas_int>(m_stride), beta,
                                                           C_data, static_cast<blas_int>(C_m_stride));
                            dispatched = true;
                        }
                    }
                }
            }

            if (dispatched) {
                continue; // next batch slice
            }
        }

        // -------------------------------------------------------------------------
        // Fallback: BLIS-style tiled packing with BLAS GEMM per tile.
        // -------------------------------------------------------------------------
        // NOLINTNEXTLINE(readability-identifier-naming)
        using blas_int = einsums::blas::int_t;

        int64_t const mc_panels_max = (blk.MC + MR - 1) / MR;
        int64_t const nc_panels_max = (blk.NC + NR - 1) / NR;
        auto const    ap_buf_elems  = static_cast<size_t>(mc_panels_max * MR * blk.KC);
        auto const    bp_buf_elems  = static_cast<size_t>(nc_panels_max * NR * blk.KC);

        // For multi-M/N: we need a temporary contiguous C tile buffer because
        // the multi-dim C elements are non-contiguous in memory.
        bool const needs_c_scatter = (multi_m || multi_n);

        {
            LabeledSection("C++ packing and kernel");
#ifdef _OPENMP
            // Only parallelize the NC loop if the batch loop is NOT parallel
            // (to avoid nested parallelism / oversubscription).
#    pragma omp parallel for schedule(static) if (!parallel_batch)
#endif
            for (int64_t nc = 0; nc < N; nc += blk.NC) {
                static thread_local std::vector<ValueType> tls_Ap, tls_Bp, tls_Ct;
                tls_Ap.resize(ap_buf_elems);
                tls_Bp.resize(bp_buf_elems);
                ValueType    *Ap     = tls_Ap.data();
                ValueType    *Bp     = tls_Bp.data();
                int64_t const nc_len = std::min(blk.NC, N - nc);

                for (int64_t kc = 0; kc < K; kc += blk.KC) {
                    int64_t const kc_len = std::min(blk.KC, K - kc);

                    bool bp_packed = false;

                    for (int64_t mc = 0; mc < M; mc += blk.MC) {
                        int64_t const mc_len = std::min(blk.MC, M - mc);

                        // Beta prescale: apply once per (mc, nc) block on first kc tile.
                        if (kc == 0 && beta != ValueType{1}) {
                            if (needs_c_scatter) {
                                // Multi-M/N: element-by-element prescale via flat-to-offset
                                for (int64_t mi = mc; mi < mc + mc_len; ++mi) {
                                    int64_t const m_off = flat_to_offset(mi, plan.c_m_dims);
                                    for (int64_t ni = nc; ni < nc + nc_len; ++ni) {
                                        int64_t const n_off = flat_to_offset(ni, plan.c_n_dims);
                                        C_data[m_off + n_off] *= beta;
                                    }
                                }
                            } else if (C_col_major) {
                                for (int64_t ni = nc; ni < nc + nc_len; ++ni) {
                                    ValueType *col = C_data + mc + ni * C_n_stride;
                                    for (int64_t i = 0; i < mc_len; ++i) {
                                        col[i] *= beta;
                                    }
                                }
                            } else {
                                for (int64_t mi = mc; mi < mc + mc_len; ++mi) {
                                    ValueType *row = C_data + mi * C_m_stride + nc;
                                    for (int64_t j = 0; j < nc_len; ++j) {
                                        row[j] *= beta;
                                    }
                                }
                            }
                        }

                        // BLAS fallback: pack + per-tile GEMM.
                        if (!bp_packed) {
                            pack_B(Bp, B_data, plan, kc, kc_len, nc, nc_len, NR, conj_b);
                            bp_packed = true;
                        }
                        pack_A(Ap, A_data, plan, mc, mc_len, kc, kc_len, MR, conj_a);

                        int64_t const num_jr = (nc_len + NR - 1) / NR;
                        int64_t const num_ir = (mc_len + MR - 1) / MR;

                        if (needs_c_scatter) {
                            // Multi-M/N: GEMM into a contiguous temp tile, then scatter to C.
                            for (int64_t jr = 0; jr < num_jr; ++jr) {
                                int64_t const nr_actual = std::min(static_cast<int64_t>(NR), nc_len - jr * NR);

                                for (int64_t ir = 0; ir < num_ir; ++ir) {
                                    int64_t const mr_actual = std::min(static_cast<int64_t>(MR), mc_len - ir * MR);

                                    // Use a contiguous MR*NR temp buffer for the GEMM output.
                                    tls_Ct.resize(static_cast<size_t>(MR) * NR);
                                    std::fill(tls_Ct.begin(), tls_Ct.end(), ValueType{0});
                                    ValueType *Ct = tls_Ct.data();

                                    ValueType *Ap_panel = Ap + ir * MR * kc_len;
                                    ValueType *Bp_panel = Bp + jr * NR * kc_len;

                                    // GEMM into contiguous Ct (col-major: ldc = MR)
                                    einsums::blas::gemm<ValueType>('N', 'T', static_cast<blas_int>(mr_actual),
                                                                   static_cast<blas_int>(nr_actual), static_cast<blas_int>(kc_len), alpha,
                                                                   Ap_panel, static_cast<blas_int>(MR), Bp_panel, static_cast<blas_int>(NR),
                                                                   ValueType{0}, Ct, static_cast<blas_int>(MR));

                                    // Scatter Ct back to C using multi-dim offsets
                                    for (int64_t jj = 0; jj < nr_actual; ++jj) {
                                        int64_t const n_off = flat_to_offset(nc + jr * NR + jj, plan.c_n_dims);
                                        for (int64_t ii = 0; ii < mr_actual; ++ii) {
                                            int64_t const m_off = flat_to_offset(mc + ir * MR + ii, plan.c_m_dims);
                                            C_data[m_off + n_off] += Ct[jj * MR + ii];
                                        }
                                    }
                                }
                            }
                        } else {
                            // Single-M, single-N: direct GEMM into C (original fast path).
                            for (int64_t jr = 0; jr < num_jr; ++jr) {
                                int64_t const nr_actual = std::min(static_cast<int64_t>(NR), nc_len - jr * NR);

                                for (int64_t ir = 0; ir < num_ir; ++ir) {
                                    int64_t const mr_actual = std::min(static_cast<int64_t>(MR), mc_len - ir * MR);

                                    ValueType *Ap_panel = Ap + ir * MR * kc_len;
                                    ValueType *Bp_panel = Bp + jr * NR * kc_len;
                                    ValueType *C_tile   = C_data + (mc + ir * MR) * C_m_stride + (nc + jr * NR) * C_n_stride;

                                    if (C_col_major) {
                                        einsums::blas::gemm<ValueType>(
                                            'N', 'T', static_cast<blas_int>(mr_actual), static_cast<blas_int>(nr_actual),
                                            static_cast<blas_int>(kc_len), alpha, Ap_panel, static_cast<blas_int>(MR), Bp_panel,
                                            static_cast<blas_int>(NR), ValueType{1}, C_tile, static_cast<blas_int>(C_n_stride));
                                    } else {
                                        einsums::blas::gemm<ValueType>(
                                            'N', 'T', static_cast<blas_int>(nr_actual), static_cast<blas_int>(mr_actual),
                                            static_cast<blas_int>(kc_len), alpha, Bp_panel, static_cast<blas_int>(NR), Ap_panel,
                                            static_cast<blas_int>(MR), ValueType{1}, C_tile, static_cast<blas_int>(C_m_stride));
                                    }
                                }
                            }
                        }
                    }
                }

                // Reclaim excess thread-local buffer memory.
                auto shrink_tls = [](auto &v) {
                    if (v.capacity() > 2 * v.size() && v.capacity() > 4096) {
                        v.shrink_to_fit();
                    }
                };
                shrink_tls(tls_Ap);
                shrink_tls(tls_Bp);
                if (needs_c_scatter)
                    shrink_tls(tls_Ct);
            }
        }

    } // end batch loop
}

// ---------------------------------------------------------------------------
// Main template bridge
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Runtime entry point — accepts a pre-built ContractionSpec.
// ---------------------------------------------------------------------------

/// @brief Attempt to execute an einsum contraction via the packed GEMM backend
///        from a runtime-built ContractionSpec.
///
/// This is the runtime-form entry point. It accepts a ContractionSpec that the
/// caller has already populated from string indices (or from a compile-time
/// index pack via the convenience overload below). Works uniformly for typed
/// `Tensor<T, K>`, `RuntimeTensor<T, Alloc>`, or any `BasicTensorConcept`
/// operand — all dispatch decisions (rank classification, batch handling,
/// kernel selection) happen at runtime against the spec.
///
/// Returns `true` if the contraction was handled; `false` if the caller should
/// fall back (to a direct BLAS GEMM, generic loop, etc.).
template <einsums::BasicTensorConcept AType, einsums::BasicTensorConcept BType, einsums::BasicTensorConcept CType>
bool try_packed_gemm(ContractionSpec const &spec_in, einsums::ValueTypeT<CType> C_prefactor, CType *C,
                     einsums::BiggestTypeT<typename AType::ValueType, typename BType::ValueType> AB_prefactor, AType const &A,
                     BType const &B) {
    LabeledSection("packed_gemm: {} <- {} ; {}", fmt::join(spec_in.c_indices, ","), fmt::join(spec_in.a_indices, ","),
                   fmt::join(spec_in.b_indices, ","));

    using ValueType  = typename AType::ValueType;
    using ValueTypeB = typename BType::ValueType;

    constexpr ScalarType st   = get_scalar_type<ValueType>();
    constexpr ScalarType st_b = get_scalar_type<ValueTypeB>();
    if constexpr (st == ScalarType::Unknown) {
        ProfileAnnotate("packed_gemm_skip", "unknown_scalar_type");
        return false;
    }
    if constexpr (st != st_b) {
        ProfileAnnotate("packed_gemm_skip", "mixed_dtype");
        return false;
    }

    // Snapshot the spec — we may need to refresh derived fields (target/link/all)
    // if the caller didn't fill them, and we want to set scalar_type from T.
    ContractionSpec spec = spec_in;
    if (spec.target_indices.empty()) {
        spec.target_indices = unique_ordered(spec.c_indices);
    }
    if (spec.link_indices.empty()) {
        spec.link_indices = compute_link_indices(spec.a_indices, spec.b_indices, spec.target_indices);
    }
    if (spec.all_indices.empty()) {
        spec.all_indices = spec.target_indices;
        for (auto const &l : spec.link_indices)
            spec.all_indices.push_back(l);
    }
    spec.scalar_type   = st;
    spec.scalar_output = spec.c_indices.empty();

    auto const &c_raw  = spec.c_indices;
    auto const &a_raw  = spec.a_indices;
    auto const &b_raw  = spec.b_indices;
    auto const &target = spec.target_indices;
    auto const &link   = spec.link_indices;

    // --- Build ContractionKey ---
    ContractionKey key;
    key.spec   = spec;
    key.a_desc = tensor_descriptor(A);
    key.b_desc = tensor_descriptor(B);
    key.c_desc = tensor_descriptor(*C);
    key.target_dims.resize(target.size());
    for (size_t ti = 0; ti < target.size(); ++ti) {
        for (size_t ci = 0; ci < c_raw.size(); ++ci) {
            if (c_raw[ci] == target[ti]) {
                key.target_dims[ti] = static_cast<int64_t>(C->dim(ci));
                break;
            }
        }
    }

    key.link_dims.resize(link.size());
    for (size_t li = 0; li < link.size(); ++li) {
        for (size_t ai = 0; ai < a_raw.size(); ++ai) {
            if (a_raw[ai] == link[li]) {
                key.link_dims[li] = static_cast<int64_t>(A.dim(ai));
                break;
            }
        }
    }

    {
        size_t const kernel_hash = std::hash<ContractionKey>{}(key);
        ProfileAnnotate("packed_gemm_hash", std::to_string(kernel_hash));
    }

    // -------------------------------------------------------------------------
    // Classify target indices to check viability.
    // -------------------------------------------------------------------------
    {
        std::unordered_set<std::string> const a_set(a_raw.begin(), a_raw.end());
        std::unordered_set<std::string> const b_set(b_raw.begin(), b_raw.end());
        size_t                                m_count = 0, n_count = 0;
        for (auto const &ci : target) {
            bool const in_a = a_set.count(ci) > 0;
            bool const in_b = b_set.count(ci) > 0;
            if (in_a && !in_b)
                ++m_count;
            else if (in_b && !in_a)
                ++n_count;
            // in_a && in_b → batch dim (handled by packing plan)
        }
        // Skip contractions that BLAS GEMM can handle directly (no batch, single M/N/K).
        if (m_count == 1 && n_count == 1 && link.size() == 1 && !spec.conj_a && !spec.conj_b && m_count + n_count == target.size()) {
            ProfileAnnotate("packed_gemm_skip", "defer_to_direct_gemm");
            return false; // Deferred to direct BLAS GEMM — not a rejection.
        }
        if (m_count == 0) {
            ProfileAnnotate("packed_gemm_skip", "no_m_dims");
            EINSUMS_LOG_INFO("PackedGemm: skipping — no M-dims (all C indices come from B). "
                             "Consider rewriting as GEMV or transposing.");
            return false;
        }
        if (n_count == 0) {
            ProfileAnnotate("packed_gemm_skip", "no_n_dims");
            EINSUMS_LOG_INFO("PackedGemm: skipping — no N-dims (all C indices come from A). "
                             "Consider rewriting as GEMV or transposing.");
            return false;
        }
        if (link.empty()) {
            ProfileAnnotate("packed_gemm_skip", "no_link_indices");
            EINSUMS_LOG_INFO("PackedGemm: skipping — no link (contraction) indices. "
                             "This is an outer/direct product, not a contraction.");
            return false;
        }
    }

    // -------------------------------------------------------------------------
    // Pack-A / Pack-B path (BLIS-style, with optional batch dims).
    // -------------------------------------------------------------------------
    PackingPlan const *cached_topo = PackingPlanCache::instance().lookup(key);
    PackingPlan        plan;
    if (cached_topo) {
        plan = *cached_topo;
        ProfileAnnotate("packed_gemm_plan", "cached");
    } else {
        plan = compute_packing_topology(key);
        if (plan.valid) {
            PackingPlanCache::instance().insert(key, plan);
            ProfileAnnotate("packed_gemm_plan", "computed");
        }
    }
    if (plan.valid) {
        fill_strides(plan, A, B, *C);
        sort_k_dims_for_packing(plan);

        bool const multi_m = (plan.c_m_dims.size() > 1);
        bool const multi_n = (plan.c_n_dims.size() > 1);

        // For single-M/N: require stride-1 in C's M or N dim (col or row major).
        // For multi-M/N: the scatter path handles arbitrary strides, so always proceed.
        if (multi_m || multi_n || plan.c_m_dims[0].tensor_stride == 1 || plan.c_n_dims[0].tensor_stride == 1) {
            ProfileAnnotate("packed_gemm_path", multi_m || multi_n ? "scatter" : "single_mn");
            blis_contraction<ValueType>(plan, *C, A, B, static_cast<ValueType>(AB_prefactor), static_cast<ValueType>(C_prefactor),
                                        spec.conj_a, spec.conj_b);
            return true;
        }
        ProfileAnnotate("packed_gemm_skip", "non_stride1_c");
        EINSUMS_LOG_INFO("PackedGemm: skipping — single-M/N with non-stride-1 C layout "
                         "(M stride={}, N stride={}). Consider permuting C first.",
                         plan.c_m_dims[0].tensor_stride, plan.c_n_dims[0].tensor_stride);
    } else {
        ProfileAnnotate("packed_gemm_skip", "invalid_topology");
        EINSUMS_LOG_INFO("PackedGemm: skipping — packing topology invalid for this contraction pattern.");
    }

    // Contraction doesn't fit packed GEMM — fall back to generic algorithm.
    return false;
}

// ---------------------------------------------------------------------------
// Compile-time index-pack overload — thin shim that builds the ContractionSpec
// from `Indices...` packs and forwards to the runtime entry point.
// ---------------------------------------------------------------------------

/// @brief Attempt to execute the einsum contraction via the packed GEMM backend.
///
/// Compile-time-indices form, used by `tensor_algebra::einsum<CIndices...,
/// AIndices..., BIndices...>` callers. Internally builds a ContractionSpec
/// from the index packs and forwards to the runtime overload above.
///
/// Returns `true` if the contraction was handled; `false` if the backend
/// should fall back to `einsum_generic_algorithm`.
template <bool ConjA, bool ConjB, einsums::BasicTensorConcept AType, einsums::BasicTensorConcept BType, typename CType,
          typename... CIndices, typename... AIndices, typename... BIndices>
    requires(einsums::BasicTensorConcept<CType> || (einsums::ScalarConcept<CType> && sizeof...(CIndices) == 0))
bool try_packed_gemm(einsums::ValueTypeT<CType> C_prefactor, std::tuple<CIndices...> const & /*C_indices_tup*/, CType *C,
                     einsums::BiggestTypeT<typename AType::ValueType, typename BType::ValueType> AB_prefactor,
                     std::tuple<AIndices...> const & /*A_indices_tup*/, AType const &A, std::tuple<BIndices...> const & /*B_indices_tup*/,
                     BType const &B) {
    LabeledSection0();

    // Scalar-output (CType is `T`, not a tensor) is not yet routed through the
    // runtime entry point — keep the original handling for that one shape.
    if constexpr (!einsums::TensorConcept<CType>) {
        return false; // The original implementation built a degenerate key here;
                      // current packed-GEMM kernels require a tensor C anyway.
    } else {
        ContractionSpec spec;
        spec.c_indices = index_letters_from_tuple(std::tuple<CIndices...>{});
        spec.a_indices = index_letters_from_tuple(std::tuple<AIndices...>{});
        spec.b_indices = index_letters_from_tuple(std::tuple<BIndices...>{});
        spec.conj_a    = ConjA;
        spec.conj_b    = ConjB;
        // Derived fields (target/link/all_indices, scalar_type) are filled in by
        // the runtime entry point.
        return try_packed_gemm<AType, BType, CType>(spec, C_prefactor, C, AB_prefactor, A, B);
    }
}

} // namespace einsums::packed_gemm
