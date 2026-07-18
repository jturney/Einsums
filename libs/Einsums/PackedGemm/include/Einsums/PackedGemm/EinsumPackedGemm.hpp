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
#include <Einsums/PackedGemm/MicroKernel.hpp>
#include <Einsums/PackedGemm/Packing.hpp>
#include <Einsums/Profile/Profile.hpp>

#include <algorithm>
#include <cstring>
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

    // Resolve the SIMD-dispatch rung's tile kernel and its register-block
    // shape once per contraction; the per-tile call below is through this
    // pointer, keeping rung resolution out of the hot loop. The shape comes
    // from the same rung as the kernel (NEON/AVX: cpu_config vector
    // blocking; SME: ZA-tile blocking), so the panels are packed in the
    // geometry the kernel expects.
    MicroKernelFn<ValueType> const micro_tile = micro_kernel_entry<ValueType>();
    MicroKernelShape const         shape      = micro_kernel_shape<ValueType>();
    int const                      MR         = shape.mr;
    int const                      NR         = shape.nr;

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

    // Scatter is needed for multi-M/N outputs and for single-M/N layouts
    // where neither output dim is unit-stride (batched C with a stride-1
    // batch index, strided views, synthetic unit dims with stride 0).
    bool const scatter_c = multi_m || multi_n || (C_m_stride != 1 && C_n_stride != 1);

    // BLAS requires the output leading dimension to be at least the number of
    // rows of the stored result: M for a column-major result (ldc = C_n_stride),
    // N for the swapped form (ldc = C_m_stride). For a transposed or degenerate
    // (size-1) output axis the natural stride can collapse below that minimum
    // (e.g. "nm <- mkq ; kqn" with n=1 gives C_n_stride=1 < M), so clamp up. This
    // is a no-op for non-degenerate outputs (the real stride already meets the
    // bound) and safe for a size-1 axis whose stride spans one element BLAS never
    // indexes. Use these as the ldc argument to every gemm call below.
    int64_t const ldc_col = std::max<int64_t>(C_n_stride, M);
    int64_t const ldc_row = std::max<int64_t>(C_m_stride, N);

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
    if (plan.batch_total > 1 && plan.k_dims_in_a.size() == 1 && !multi_m && !multi_n && !plan.synthetic) {
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

            // BLAS validates the leading dimensions against the stored-operand
            // row counts (lda >= rows of op-form: M for transA='N', K for 'T';
            // ldb >= K for transB='N', N for 'T'; ldc >= M). When any of M/N/K
            // is 1 the corresponding axis stride is meaningless and can collapse
            // below that minimum (e.g. K=1 makes both m_stride and k_stride_a == 1,
            // so lda_val=k_stride_a=1 < M). Clamp up to the BLAS minimum: a no-op
            // for non-degenerate operands (the real stride already meets it), and
            // safe for a size-1 axis since that stride is never used to index.
            lda_val = std::max<blas_int>(lda_val, (transA == 'N') ? static_cast<blas_int>(M) : static_cast<blas_int>(K));
            ldb_val = std::max<blas_int>(ldb_val, (transB == 'N') ? static_cast<blas_int>(K) : static_cast<blas_int>(N));
            ldc_val = std::max<blas_int>(ldc_val, static_cast<blas_int>(M));

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
        // The flatten+GEMM path writes C directly and supports only stride-1
        // column- or row-major outputs; scatter-layout C goes to the tiled or
        // block-GEMM paths below.
        if (plan.k_dims_in_a.size() > 1 && !scatter_c && !plan.synthetic) {
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
                                                   static_cast<blas_int>(ldc_col));
                } else {
                    einsums::blas::gemm<ValueType>('N', 'T', static_cast<blas_int>(N), static_cast<blas_int>(M), static_cast<blas_int>(K),
                                                   alpha, B_data, static_cast<blas_int>(N), A_data, static_cast<blas_int>(M), beta, C_data,
                                                   static_cast<blas_int>(ldc_row));
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
            //
            // Batched contractions (nb > 0) must not use the HPTT flatten path:
            // it builds the transpose plan from the operand's full rank and
            // sizes (including the batch dims) while A_flat/B_flat are sized for a
            // single batch slice (M*K / K*N) and A_data/B_data are already offset
            // to the current slice. HPTT then transposes the whole batched tensor
            // into the inner-sized buffer -> heap-buffer-overflow. Fall through to
            // the batch-aware scalar gather below, which honors the slice offset
            // and the inner strides. (TODO: a proper per-slice batched HPTT path
            // would recover the transpose perf for batched multi-K contractions.)
            bool use_hptt = (nb == 0) && !plan.coalesced;
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
#ifdef _OPENMP
                num_threads = omp_get_max_threads();
#endif

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
                                                       static_cast<blas_int>(N), beta_k, C_data, static_cast<blas_int>(ldc_col));
                    } else {
                        einsums::blas::gemm<ValueType>('N', 'T', static_cast<blas_int>(N), static_cast<blas_int>(M),
                                                       static_cast<blas_int>(kc_len), alpha, B_ptr, static_cast<blas_int>(N), A_ptr,
                                                       static_cast<blas_int>(M), beta_k, C_data, static_cast<blas_int>(ldc_row));
                    }
                }
            } else {
                // Scalar gather fallback (non-contiguous tensors).
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
                                                       static_cast<blas_int>(N), beta_k, C_data, static_cast<blas_int>(ldc_col));
                    } else {
                        einsums::blas::gemm<ValueType>('N', 'T', static_cast<blas_int>(N), static_cast<blas_int>(M),
                                                       static_cast<blas_int>(kc_len), alpha, B_ptr, static_cast<blas_int>(N), A_ptr,
                                                       static_cast<blas_int>(M), beta_k, C_data, static_cast<blas_int>(ldc_row));
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
        if (plan.k_dims_in_a.size() == 1 && !multi_m && !multi_n && !plan.synthetic) {
            // Single-K fast path: only for single-M, single-N (direct BLAS GEMM dispatch).
            // NOLINTNEXTLINE(readability-identifier-naming)
            using blas_int = einsums::blas::int_t;

            int64_t const m_stride   = plan.m_dims[0].tensor_stride;
            int64_t const n_stride   = plan.n_dims[0].tensor_stride;
            int64_t const k_stride_a = plan.k_dims_in_a[0].tensor_stride;
            int64_t const k_stride_b = plan.k_dims_in_b[0].tensor_stride;

            // Clamp a stride-derived leading dimension up to the BLAS minimum (the
            // row count of the stored operand for that call). A degenerate
            // (size-1) axis can collapse the natural stride below the minimum
            // (e.g. "snm <- mkn ; ksm"-style specs); the clamp is a no-op
            // otherwise and is safe because the stride is unused when its axis is
            // size 1. Each call below passes the BLAS m-dimension the leading dim
            // must cover.
            auto ld = [](int64_t stride, int64_t min_rows) { return static_cast<blas_int>(std::max<int64_t>(stride, min_rows)); };

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
            // BLAS has no flag, so fall through to the BLIS tiled path which conjugates
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
                                                           ld(k_stride_a, M), B_data, ld(n_stride, K), beta, C_data,
                                                           static_cast<blas_int>(ldc_col));
                            dispatched = true;
                        }
                    } else if (n_stride == 1) {
                        // B col-major in N → transB='T'
                        einsums::blas::gemm<ValueType>(trans_flag('N', conj_a), trans_flag('T', conj_b), static_cast<blas_int>(M),
                                                       static_cast<blas_int>(N), static_cast<blas_int>(K), alpha, A_data, ld(k_stride_a, M),
                                                       B_data, ld(k_stride_b, N), beta, C_data, static_cast<blas_int>(ldc_col));
                        dispatched = true;
                    }
                } else if (k_stride_a == 1) {
                    // A col-major in K → transA='T'
                    if (k_stride_b == 1) {
                        // B col-major in K → transB='N'
                        if (can_dispatch_n(conj_b)) {
                            einsums::blas::gemm<ValueType>(trans_flag('T', conj_a), trans_flag('N', conj_b), static_cast<blas_int>(M),
                                                           static_cast<blas_int>(N), static_cast<blas_int>(K), alpha, A_data,
                                                           ld(m_stride, K), B_data, ld(n_stride, K), beta, C_data,
                                                           static_cast<blas_int>(ldc_col));
                            dispatched = true;
                        }
                    } else if (n_stride == 1) {
                        einsums::blas::gemm<ValueType>(trans_flag('T', conj_a), trans_flag('T', conj_b), static_cast<blas_int>(M),
                                                       static_cast<blas_int>(N), static_cast<blas_int>(K), alpha, A_data, ld(m_stride, K),
                                                       B_data, ld(k_stride_b, N), beta, C_data, static_cast<blas_int>(ldc_col));
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
                                                       static_cast<blas_int>(M), static_cast<blas_int>(K), alpha, B_data, ld(k_stride_b, N),
                                                       A_data, ld(k_stride_a, M), beta, C_data, static_cast<blas_int>(ldc_row));
                        dispatched = true;
                    } else if (k_stride_a == 1) {
                        // A is the BLAS "B" arg → transB_blas='N', conj_a applies
                        if (can_dispatch_n(conj_a)) {
                            einsums::blas::gemm<ValueType>(trans_flag('N', conj_b), trans_flag('N', conj_a), static_cast<blas_int>(N),
                                                           static_cast<blas_int>(M), static_cast<blas_int>(K), alpha, B_data,
                                                           ld(k_stride_b, N), A_data, ld(m_stride, K), beta, C_data,
                                                           static_cast<blas_int>(ldc_row));
                            dispatched = true;
                        }
                    }
                } else if (k_stride_b == 1) {
                    // B is the BLAS "A" arg → transA_blas='T', conj_b applies
                    if (m_stride == 1) {
                        // A is the BLAS "B" arg → transB_blas='T', conj_a applies
                        einsums::blas::gemm<ValueType>(trans_flag('T', conj_b), trans_flag('T', conj_a), static_cast<blas_int>(N),
                                                       static_cast<blas_int>(M), static_cast<blas_int>(K), alpha, B_data, ld(n_stride, K),
                                                       A_data, ld(k_stride_a, M), beta, C_data, static_cast<blas_int>(ldc_row));
                        dispatched = true;
                    } else if (k_stride_a == 1) {
                        // A is the BLAS "B" arg → transB_blas='N', conj_a applies
                        if (can_dispatch_n(conj_a)) {
                            einsums::blas::gemm<ValueType>(trans_flag('T', conj_b), trans_flag('N', conj_a), static_cast<blas_int>(N),
                                                           static_cast<blas_int>(M), static_cast<blas_int>(K), alpha, B_data,
                                                           ld(n_stride, K), A_data, ld(m_stride, K), beta, C_data,
                                                           static_cast<blas_int>(ldc_row));
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

        // K blocking: the kernel rung may deepen the cache-derived KC (the SME
        // rung's ZA accumulators need no C cache blocking), which cuts the
        // number of beta/scatter read-modify-write passes over C and ZA
        // extractions to one per tile. The block-GEMM scatter strategy gets
        // the same deep default: the vendor GEMM blocks K internally, so the
        // only KC role left is bounding the packed panels and the number of
        // scatter passes. The M block shrinks in compensation so the packed
        // A panel (MC_blk * KC_blk) stays within ~4 MiB.
        int64_t kc_hint = shape.kc;
        if (kc_hint == 0 && shape.block_gemm && scatter_c) {
            kc_hint = 4096;
        }
        int64_t const KC_blk = (kc_hint > 0) ? std::min<int64_t>(std::max<int64_t>(kc_hint, blk.KC), K) : blk.KC;
        int64_t       MC_blk = blk.MC;
        if (KC_blk > blk.KC) {
            int64_t const mc_cap = (int64_t{4} << 20) / (KC_blk * static_cast<int64_t>(sizeof(ValueType)));
            MC_blk               = std::clamp((mc_cap / MR) * MR, static_cast<int64_t>(MR), blk.MC);
        }

        // For multi-M/N: we need a temporary contiguous C tile buffer because
        // the multi-dim C elements are non-contiguous in memory.
        bool const needs_c_scatter = scatter_c;

        // The NC loop is the parallel loop: shrink the NC block below the
        // cache-derived blk.NC when needed so every thread gets at least one
        // block. The cost is re-packing A once per extra NC block, which is a
        // bandwidth-trivial price next to leaving all but one core idle on
        // tall-N contractions (N <= blk.NC previously ran fully serial).
        int64_t NC_blk = blk.NC;
#ifdef _OPENMP
        if (!parallel_batch) {
            int const nthreads = omp_get_max_threads();
            if (nthreads > 1) {
                int64_t const per_thread = (N + nthreads - 1) / nthreads;
                int64_t const rounded    = ((per_thread + NR - 1) / NR) * NR;
                NC_blk                   = std::clamp(rounded, static_cast<int64_t>(NR), blk.NC);
            }
        }
#endif

        // Size the packing buffers from the blocks actually used, not the
        // cache-derived maxima - with a deep KC_blk, sizing from blk.NC would
        // allocate NC/NC_blk times more B-panel memory than any iteration
        // touches.
        int64_t const mc_panels_max = (MC_blk + MR - 1) / MR;
        int64_t const nc_panels_max = (std::min(NC_blk, N) + NR - 1) / NR;
        auto const    ap_buf_elems  = static_cast<size_t>(mc_panels_max * MR * KC_blk);
        auto const    bp_buf_elems  = static_cast<size_t>(nc_panels_max * NR * KC_blk);

        {
            LabeledSection("C++ packing and kernel");
#ifdef _OPENMP
            // Only parallelize the NC loop if the batch loop is NOT parallel
            // (to avoid nested parallelism / oversubscription).
#    pragma omp parallel for schedule(static) if (!parallel_batch)
#endif
            for (int64_t nc = 0; nc < N; nc += NC_blk) {
                static thread_local std::vector<ValueType> tls_Ap, tls_Bp, tls_Ct;
                tls_Ap.resize(ap_buf_elems);
                tls_Bp.resize(bp_buf_elems);
                ValueType    *Ap     = tls_Ap.data();
                ValueType    *Bp     = tls_Bp.data();
                int64_t const nc_len = std::min(NC_blk, N - nc);

                // Scatter path: precompute the C offset tables (one entry per
                // flat index) instead of paying a div/mod chain per element in
                // the beta prescale and tile scatter loops below. n-offsets are
                // invariant for the whole nc block; m-offsets are refreshed per
                // mc block inside the kc loop.
                static thread_local std::vector<int64_t> c_n_offsets, c_m_offsets;
                if (needs_c_scatter) {
                    precompute_offsets(nc, nc_len, plan.c_n_dims, c_n_offsets);
                }

                // ---- Block-GEMM scatter strategy ----
                // One vendor GEMM per (mc, kc) block: pack A to a plain
                // column-major mc_len x kc_len matrix and B to k-major
                // kc_len x nc_len, GEMM into a contiguous C block, then
                // scatter-accumulate through the offset tables. Vendor
                // libraries run cache-blocked GEMMs of this size at full
                // speed (including matrix units the tile kernels cannot
                // reach, e.g. Accelerate's AMX/SME), while the packed blocks
                // and C temp stay cache-sized and thread-local - no
                // operand-sized temporaries, unlike Sort+GEMM.
                if (needs_c_scatter && shape.block_gemm) {
                    // NOLINTNEXTLINE(readability-identifier-naming)
                    using blas_int = einsums::blas::int_t;
                    static thread_local std::vector<ValueType> tls_Af, tls_Bf, tls_Cb;
                    tls_Af.resize(static_cast<size_t>(MC_blk * KC_blk));
                    tls_Bf.resize(static_cast<size_t>(nc_len) * static_cast<size_t>(KC_blk));
                    tls_Cb.resize(static_cast<size_t>(MC_blk) * static_cast<size_t>(nc_len));

                    for (int64_t kc = 0; kc < K; kc += KC_blk) {
                        int64_t const kc_len = std::min(KC_blk, K - kc);
                        pack_B_flat(tls_Bf.data(), B_data, plan, kc, kc_len, nc, nc_len, conj_b);

                        for (int64_t mc = 0; mc < M; mc += MC_blk) {
                            int64_t const mc_len = std::min(MC_blk, M - mc);
                            precompute_offsets(mc, mc_len, plan.c_m_dims, c_m_offsets);

                            // Beta prescale once per (mc, nc) block on the first kc slice.
                            if (kc == 0 && beta != ValueType{1}) {
                                for (int64_t mi = 0; mi < mc_len; ++mi) {
                                    int64_t const m_off = c_m_offsets[static_cast<size_t>(mi)];
                                    for (int64_t ni = 0; ni < nc_len; ++ni) {
                                        C_data[m_off + c_n_offsets[static_cast<size_t>(ni)]] *= beta;
                                    }
                                }
                            }

                            pack_A_flat(tls_Af.data(), A_data, plan, mc, mc_len, kc, kc_len, conj_a);

                            einsums::blas::gemm<ValueType>('N', 'T', static_cast<blas_int>(mc_len), static_cast<blas_int>(nc_len),
                                                           static_cast<blas_int>(kc_len), alpha, tls_Af.data(),
                                                           static_cast<blas_int>(mc_len), tls_Bf.data(), static_cast<blas_int>(nc_len),
                                                           ValueType{0}, tls_Cb.data(), static_cast<blas_int>(mc_len));

                            // Scatter-accumulate the contiguous block into C. When C's
                            // stride along the fastest flat m coordinate is 1, the
                            // destination decomposes into contiguous runs and the
                            // accumulation vectorizes.
                            bool const    c_m_unit = plan.c_m_dims.back().tensor_stride == 1;
                            int64_t const c_m_fast = plan.c_m_dims.back().size;
                            for (int64_t j = 0; j < nc_len; ++j) {
                                int64_t const    n_off = c_n_offsets[static_cast<size_t>(j)];
                                ValueType const *src   = tls_Cb.data() + j * mc_len;
                                if (c_m_unit) {
                                    int64_t pos = 0;
                                    while (pos < mc_len) {
                                        int64_t const    run = std::min(c_m_fast - ((mc + pos) % c_m_fast), mc_len - pos);
                                        ValueType       *dst = C_data + c_m_offsets[static_cast<size_t>(pos)] + n_off;
                                        ValueType const *s   = src + pos;
                                        for (int64_t r = 0; r < run; ++r) {
                                            dst[r] += s[r];
                                        }
                                        pos += run;
                                    }
                                    continue;
                                }
                                for (int64_t i2 = 0; i2 < mc_len; ++i2) {
                                    C_data[c_m_offsets[static_cast<size_t>(i2)] + n_off] += src[i2];
                                }
                            }
                        }
                    }
                    continue; // next nc block
                }

                for (int64_t kc = 0; kc < K; kc += KC_blk) {
                    int64_t const kc_len = std::min(KC_blk, K - kc);

                    bool bp_packed = false;

                    for (int64_t mc = 0; mc < M; mc += MC_blk) {
                        int64_t const mc_len = std::min(MC_blk, M - mc);

                        if (needs_c_scatter) {
                            precompute_offsets(mc, mc_len, plan.c_m_dims, c_m_offsets);
                        }

                        // Beta prescale: apply once per (mc, nc) block on first kc tile.
                        if (kc == 0 && beta != ValueType{1}) {
                            if (needs_c_scatter) {
                                // Multi-M/N: element-by-element prescale via the offset tables
                                for (int64_t mi = 0; mi < mc_len; ++mi) {
                                    int64_t const m_off = c_m_offsets[static_cast<size_t>(mi)];
                                    for (int64_t ni = 0; ni < nc_len; ++ni) {
                                        C_data[m_off + c_n_offsets[static_cast<size_t>(ni)]] *= beta;
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

                                    // Micro-kernel into contiguous Ct (col-major: rs=1, cs=MR)
                                    micro_tile(static_cast<int>(MR), static_cast<int>(NR), kc_len, alpha, Ap_panel, Bp_panel, mr_actual,
                                               nr_actual, Ct, 1, MR);

                                    // Scatter Ct back to C using the precomputed offset tables
                                    for (int64_t jj = 0; jj < nr_actual; ++jj) {
                                        int64_t const n_off = c_n_offsets[static_cast<size_t>(jr * NR + jj)];
                                        for (int64_t ii = 0; ii < mr_actual; ++ii) {
                                            C_data[c_m_offsets[static_cast<size_t>(ir * MR + ii)] + n_off] += Ct[jj * MR + ii];
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

                                    // Micro-kernel accumulates directly into strided C; the
                                    // (rs, cs) pair covers both col-major (1, C_n_stride) and
                                    // row-major (C_m_stride, 1) layouts without a branch.
                                    micro_tile(static_cast<int>(MR), static_cast<int>(NR), kc_len, alpha, Ap_panel, Bp_panel, mr_actual,
                                               nr_actual, C_tile, C_m_stride, C_n_stride);
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
// Runtime entry point: accepts a pre-built ContractionSpec.
// ---------------------------------------------------------------------------

/// @brief Attempt to execute an einsum contraction via the packed GEMM backend
///        from a runtime-built ContractionSpec.
///
/// This is the runtime-form entry point. It accepts a ContractionSpec that the
/// caller has already populated from string indices (or from a compile-time
/// index pack via the convenience overload below). Works uniformly for typed
/// `Tensor<T, K>`, `RuntimeTensor<T, Alloc>`, or any `BasicTensorConcept`
/// operand. All dispatch decisions, including rank classification, batch
/// handling, and kernel selection, happen at runtime against the spec.
///
/// Returns `true` if the contraction was handled; `false` if the caller should
/// fall back (to a direct BLAS GEMM, generic loop, etc.).
///
/// @param allow_scatter When false, contractions that remain multi-M/N after
///        dim coalescing are declined instead of taking the slow per-tile
///        scatter path. Pass false from callers that have a faster fallback
///        (the compile-time einsum dispatch falls back to Sort+GEMM); leave
///        true for callers whose only alternative is a generic loop (the
///        ComputeGraph runtime string dispatch).
template <einsums::BasicTensorConcept AType, einsums::BasicTensorConcept BType, einsums::BasicTensorConcept CType>
bool try_packed_gemm(ContractionSpec const &spec_in, einsums::ValueTypeT<CType> C_prefactor, CType *C,
                     einsums::BiggestTypeT<typename AType::ValueType, typename BType::ValueType> AB_prefactor, AType const &A,
                     BType const &B, bool allow_scatter = true) {
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

    // Snapshot the spec, since we may need to refresh derived fields (target/link/all)
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
    // Classify target indices. Empty M/N/link groups are no longer
    // rejections: compute_packing_topology synthesizes a unit dim so GEMV-
    // and outer-product-shaped contractions run through the same block/tile
    // machinery. What the classification still decides is (a) the direct-GEMM
    // deferral and (b) whether Sort+GEMM exists as a fallback for the policy
    // check below (it requires all three groups non-empty).
    // -------------------------------------------------------------------------
    bool ttgt_exists = false;
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
            return false; // Deferred to direct BLAS GEMM, not a rejection.
        }
        ttgt_exists = m_count > 0 && n_count > 0 && !link.empty();
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
        coalesce_plan(plan);

        bool const multi_m = (plan.c_m_dims.size() > 1);
        bool const multi_n = (plan.c_n_dims.size() > 1);
        // Scatter is needed for multi-M/N and for single-M/N layouts where
        // neither output dim is unit-stride (e.g. a batched C whose batch
        // index owns stride 1). The block/tile scatter paths handle all of
        // these; the only question is policy.
        bool const needs_scatter = multi_m || multi_n || (plan.c_m_dims[0].tensor_stride != 1 && plan.c_n_dims[0].tensor_stride != 1);

        // Decline only when a TTGT fallback actually exists for this shape
        // and neither the caller nor the rung wants the scatter path. GEMV-
        // and outer-product-shaped contractions (synthetic plans) have no
        // Sort+GEMM fallback - their alternative is the generic loop, which
        // the scatter path beats on every architecture.
        if (needs_scatter && ttgt_exists && !allow_scatter && !micro_kernel_shape<ValueType>().fast_scatter) {
            ProfileAnnotate("packed_gemm_skip", "scatter_defer_to_ttgt");
            EINSUMS_LOG_INFO("PackedGemm: declining — scatter-path shape, the caller has a TTGT fallback, "
                             "and this rung's kernel does not beat it on the scatter path.");
            return false;
        }

        ProfileAnnotate("packed_gemm_path", needs_scatter ? "scatter" : "single_mn");
        blis_contraction<ValueType>(plan, *C, A, B, static_cast<ValueType>(AB_prefactor), static_cast<ValueType>(C_prefactor), spec.conj_a,
                                    spec.conj_b);
        return true;
    } else {
        ProfileAnnotate("packed_gemm_skip", "invalid_topology");
        EINSUMS_LOG_INFO("PackedGemm: skipping — packing topology invalid for this contraction pattern.");
    }

    // Contraction doesn't fit packed GEMM, so fall back to generic algorithm.
    return false;
}

// ---------------------------------------------------------------------------
// Compile-time index-pack overload: thin shim that builds the ContractionSpec
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
                     BType const &B, bool allow_scatter = true) {
    LabeledSection0();

    // Scalar-output (CType is `T`, not a tensor) is not yet routed through the
    // runtime entry point, so keep the original handling for that one shape.
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
        return try_packed_gemm<AType, BType, CType>(spec, C_prefactor, C, AB_prefactor, A, B, allow_scatter);
    }
}

} // namespace einsums::packed_gemm
