//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

// This header is included from Dispatch.hpp.

#include <Einsums/BLAS.hpp>
#include <Einsums/Concepts/TensorConcepts.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Logging.hpp>
#include <Einsums/PackedGemm/ContractionKey.hpp>
#include <Einsums/PackedGemm/MicroKernel.hpp>
#include <Einsums/PackedGemm/Packing.hpp>
#include <Einsums/Profile/Profile.hpp>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

#ifdef _OPENMP
#    include <omp.h>
#endif

EINSUMS_NAMESPACE_BEGIN(packed_gemm)

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

/// Outer-product destination size at which the k=1 GEMM stops paying.
///
/// ger has no beta, so a destination prefactor costs a separate pass over C -
/// and C traffic IS the cost of an outer product. A k=1 GEMM folds beta in and
/// makes one pass, which wins while C is small; past that, GEMM's blocking and
/// threading overhead outgrows the extra pass and plain ger wins.
///
/// Measured graph-vs-eager on the same shapes (lower is better), three runs,
/// A/B interleaved in both orders:
///
///   C elements   256    2.4k    10k    65k    332k     1M    5.3M
///   k=1 gemm    2.1-2.8 2.5-2.6 0.36-0.43 0.56-0.60 1.05-1.10 0.93-0.95 1.46-1.50
///   ger         1.9-3.0 2.5-2.9 1.07-1.10 0.91-0.94 0.58-0.61 0.52-0.55 0.84-0.87
///
/// The crossover sits between 65k and 332k; this threshold is in that gap.
/// Below ~2.4k the two agree within noise - the call overhead dominates - so
/// only the middle of the range actually decides it.
constexpr size_t kOuterGemmMaxElems = 1u << 17;

/// Outer-product destination size below which no BLAS call is worth making.
///
/// Matches the existing small-outer deferral further down, which measured the
/// packed path losing to the generic loop under ~4k output elements. At that
/// size a BLAS call's fixed cost is the entire measurement, so the direct paths
/// leave the decision to the deferral rather than pre-empting a tuned choice
/// with a noisier one.
constexpr size_t kOuterMinElems = 1u << 12;

/// @brief Collapse axes [@p begin, @p end) of @p t into one (extent, stride).
///
/// Succeeds only when that run of axes tiles memory exactly **in axis order**:
/// taken first to last, each axis must begin where the previous one ended
/// (s_next == s * d). A permuted or gappy view fails, and the caller must not
/// flatten it. Extent-1 axes are ignored - their stride is arbitrary and a
/// permuted view can leave one at a boundary with an inflated value.
///
/// Axis order is the whole point, and this used to sort the axes by stride
/// before checking. That answers "do these axes tile memory?", which is not the
/// question: the caller flattens two operands independently and then walks both
/// flat runs in lockstep, so they have to agree on which index varies fastest.
/// Sorting made a reversed run look flattenable, and a C[i,j] <- A[k] B[k,i,j]
/// with a permuted B then came out transposed - correct memory, wrong order.
/// A run that is not already in increasing-stride order now declines, and the
/// caller falls through to the generic loop.
///
/// This is the runtime twin of the compile-time `contiguous_positions` check
/// the eager dispatcher uses to decide the same question.
template <einsums::BasicTensorConcept TensorType>
bool flatten_run(TensorType const &t, size_t begin, size_t end, size_t &extent, size_t &stride_out) {
    extent = 1;
    std::vector<std::pair<size_t, size_t>> ds; // (stride, dim), extent > 1 only
    ds.reserve(end - begin);
    for (size_t i = begin; i < end; ++i) {
        size_t const d = t.dim(i);
        if (d == 0) {
            return false;
        }
        extent *= d;
        if (d > 1) {
            ds.emplace_back(t.stride(i), d);
        }
    }
    if (ds.empty()) { // every axis is a singleton: one element, stride irrelevant
        stride_out = 1;
        return true;
    }
    // Deliberately NOT sorted: see above. The axes must already tile in the
    // order the caller will walk them.
    stride_out    = ds.front().first;
    size_t expect = stride_out;
    for (auto const &[s, d] : ds) {
        if (s != expect) {
            return false;
        }
        expect = s * d;
    }
    return true;
}

/// @brief True when @p prefix ++ @p suffix == @p whole, element for element.
inline bool is_concatenation(std::vector<std::string> const &whole, std::vector<std::string> const &prefix,
                             std::vector<std::string> const &suffix) {
    if (whole.size() != prefix.size() + suffix.size()) {
        return false;
    }
    return std::equal(prefix.begin(), prefix.end(), whole.begin()) &&
           std::equal(suffix.begin(), suffix.end(), whole.begin() + prefix.size());
}

// ---------------------------------------------------------------------------
// Runtime tensor info extraction (works for any BasicTensor rank)
// ---------------------------------------------------------------------------

template <einsums::BasicTensorConcept TensorType>
TensorDescriptor tensor_descriptor(TensorType const &t) {
    TensorDescriptor td;
    // TensorType::Rank exists for BOTH compile-time tensors (Rank = K >= 0) and
    // runtime-rank ones (Rank = dynamic_rank = -1, a sentinel), so it is only a
    // real rank when non-negative. Taking the sentinel at face value gave every
    // runtime-rank operand a descriptor rank of (size_t)-1 - which went unnoticed
    // while rank was merely hashed and compared, since they all shared the same
    // wrong value and the dims vectors did the real disambiguating.
    using TT = std::remove_cvref_t<TensorType>;
    if constexpr (requires { TT::Rank; }) {
        if constexpr (TT::Rank >= 0) {
            td.rank = static_cast<size_t>(TT::Rank);
        } else {
            td.rank = t.rank();
        }
    } else {
        td.rank = t.rank();
    }
    td.dtype = get_scalar_type<typename TensorType::ValueType>();
    td.strides.resize(td.rank);
    for (size_t i = 0; i < td.rank; ++i) {
        td.strides[i] = static_cast<int64_t>(t.stride(i));
    }
    return td;
}

/// @brief Whether @p td still describes @p t, without building a descriptor.
///
/// The comparison @ref tensor_descriptor + operator== would do, done in place:
/// a ContractionSite checks three of these per call, and allocating three
/// stride vectors to throw them away is the cost it exists to avoid.
template <einsums::BasicTensorConcept TensorType>
bool descriptor_matches(TensorDescriptor const &td, TensorType const &t) {
    using TT    = std::remove_cvref_t<TensorType>;
    size_t rank = 0;
    if constexpr (requires { TT::Rank; }) {
        if constexpr (TT::Rank >= 0) {
            rank = static_cast<size_t>(TT::Rank);
        } else {
            rank = t.rank();
        }
    } else {
        rank = t.rank();
    }
    if (td.rank != rank || td.dtype != get_scalar_type<typename TensorType::ValueType>() || td.strides.size() != rank) {
        return false;
    }
    for (size_t i = 0; i < rank; ++i) {
        if (td.strides[i] != static_cast<int64_t>(t.stride(i))) {
            return false;
        }
    }
    return true;
}

/// @brief Whether a memoized @p key still describes this contraction.
///
/// Checks exactly what the ContractionKey encodes - topology, operand layout,
/// and the runtime sizes of the target and link dimensions - because that is
/// the plan cache's own soundness contract: equal key, valid plan. Everything
/// is compared in place, so a match allocates nothing.
///
/// @p spec_in's derived fields (target/all/link) are checked only when the
/// caller filled them; they are functions of the raw c/a/b lists, which are
/// compared unconditionally.
template <einsums::BasicTensorConcept AType, einsums::BasicTensorConcept BType, einsums::BasicTensorConcept CType>
bool site_key_matches(ContractionKey const &key, ContractionSpec const &spec_in, ScalarType st, AType const &A, BType const &B,
                      CType const &C) {
    if (key.spec.scalar_type != st || key.spec.conj_a != spec_in.conj_a || key.spec.conj_b != spec_in.conj_b) {
        return false;
    }
    if (key.spec.c_indices != spec_in.c_indices || key.spec.a_indices != spec_in.a_indices || key.spec.b_indices != spec_in.b_indices) {
        return false;
    }
    if (!spec_in.link_indices.empty() && key.spec.link_indices != spec_in.link_indices) {
        return false;
    }
    if (!descriptor_matches(key.a_desc, A) || !descriptor_matches(key.b_desc, B) || !descriptor_matches(key.c_desc, C)) {
        return false;
    }

    auto const &target = key.spec.target_indices;
    auto const &c_raw  = key.spec.c_indices;
    if (key.target_dims.size() != target.size()) {
        return false;
    }
    for (size_t ti = 0; ti < target.size(); ++ti) {
        int64_t dim = 0;
        for (size_t ci = 0; ci < c_raw.size(); ++ci) {
            if (c_raw[ci] == target[ti]) {
                dim = static_cast<int64_t>(C.dim(ci));
                break;
            }
        }
        if (key.target_dims[ti] != dim) {
            return false;
        }
    }

    auto const &link  = key.spec.link_indices;
    auto const &a_raw = key.spec.a_indices;
    if (key.link_dims.size() != link.size()) {
        return false;
    }
    for (size_t li = 0; li < link.size(); ++li) {
        int64_t dim = 0;
        for (size_t ai = 0; ai < a_raw.size(); ++ai) {
            if (a_raw[ai] == link[li]) {
                dim = static_cast<int64_t>(A.dim(ai));
                break;
            }
        }
        if (key.link_dims[li] != dim) {
            return false;
        }
    }
    return true;
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
            // the batch-aware gather below, which honors the slice offset and the
            // inner strides.
            //
            // A per-slice batched HPTT path (each slice described to HPTT as an
            // embedded subtensor via outer sizes + inner stride, one cached plan
            // rebound per slice) was implemented and benchmarked on an Apple M4
            // (2026-07-19) and does NOT pay: the gather below copies one KC x M
            // tile at a time immediately before the GEMM consumes it, so the
            // copy stays fused in cache, while an up-front HPTT transpose of the
            // whole M*K slice round-trips a multi-MB flat buffer through DRAM.
            // Measured wash at small K to 15% SLOWER at large K, on both the
            // memcpy (lead stride 1) and strided-lead gather variants. Revisit
            // only with a cache-resident (KC-tiled) transpose scheme, or on
            // hardware whose strided-load throughput is much worse relative to
            // its cache bandwidth. The BatchedMultiK tests pin the layouts that
            // experiment covered.
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
        // Clamped to K on BOTH branches: a K-block larger than K is never useful,
        // and the packing buffers below are sized from KC_blk, so an unclamped
        // cache-derived blk.KC inflates the panels for a small contraction. That
        // stayed hidden while the L2 was being under-detected; correcting the L2
        // made blk.KC bigger and the small shapes paid for it.
        int64_t const KC_blk = std::min<int64_t>((kc_hint > 0) ? std::max<int64_t>(kc_hint, blk.KC) : blk.KC, K);
        int64_t       MC_blk = blk.MC;
        if (KC_blk > blk.KC) {
            int64_t const mc_cap = (int64_t{4} << 20) / (KC_blk * static_cast<int64_t>(sizeof(ValueType)));
            MC_blk               = std::clamp((mc_cap / MR) * MR, static_cast<int64_t>(MR), blk.MC);
        }

        // For multi-M/N: we need a temporary contiguous C tile buffer because
        // the multi-dim C elements are non-contiguous in memory.
        bool const needs_c_scatter = scatter_c;

        // The NC loop is the parallel loop, but only when there is enough work to
        // pay for the region. Entering and leaving one costs a fork/join barrier
        // -- measured at init into cpu_config().min_parallel_flops -- and for a
        // small contraction that is orders of magnitude more than the arithmetic
        // it distributes. A tiled CCSD contraction is ~2 KFLOP against a ~20 us
        // region; expanding a tiled einsum into thousands of such nodes made the
        // replay several times SLOWER with more threads.
        double const work_flops    = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);
        bool const   worth_threads = work_flops >= static_cast<double>(cpu_config().min_parallel_flops);
        bool const   parallel_nc   = !parallel_batch && worth_threads;

        // Shrink the NC block below the cache-derived blk.NC so every thread gets
        // at least one block. The cost is re-packing A once per extra NC block, a
        // bandwidth-trivial price next to leaving all but one core idle on tall-N
        // contractions (N <= blk.NC previously ran fully serial). Only worth doing
        // when the loop is actually going to run in parallel: otherwise it buys
        // extra re-packing for nothing.
        int64_t NC_blk = blk.NC;
#ifdef _OPENMP
        if (parallel_nc) {
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
        // Panel counts follow the work actually done, not the cache-derived maxima:
        // a contraction narrower than its block gets a buffer its own size.
        int64_t const mc_panels_max = (std::min(MC_blk, M) + MR - 1) / MR;
        int64_t const nc_panels_max = (std::min(NC_blk, N) + NR - 1) / NR;
        auto const    ap_buf_elems  = static_cast<size_t>(mc_panels_max * MR * KC_blk);
        auto const    bp_buf_elems  = static_cast<size_t>(nc_panels_max * NR * KC_blk);

        {
            LabeledSection("C++ packing and kernel");
#ifdef _OPENMP
            // Only parallelize the NC loop if the batch loop is NOT parallel (to
            // avoid nested parallelism / oversubscription) AND the contraction is
            // big enough to pay for the region.
#    pragma omp parallel for schedule(static) if (parallel_nc)
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
                if (needs_c_scatter || (is_complex && shape.use_1m)) {
                    precompute_offsets(nc, nc_len, plan.c_n_dims, c_n_offsets);
                }

                // ---- 1m complex strategy (rungs with a real matrix kernel) ----
                // Complex tile work runs on the REAL kernel via Van Zee's 1m
                // method: A packs 1e, B packs 1r, and the real (2M x N) output
                // is interleaved complex, scattered directly. Working extents
                // double (Mh = 2M, Kh = 2K); MR/NR here are the real kernel's
                // geometry (see MicroKernelShape::use_1m). Conjugation folds
                // into the packing signs; the complex alpha applies at the
                // scatter. Measured 1.74x over Sort+GEMM for complex<double>
                // on the M4 SME rung, with no operand-sized temporaries.
                if constexpr (is_complex) {
                    if (shape.use_1m) {
                        using RealT                           = RemoveComplexT<ValueType>;
                        MicroKernelFn<RealT> const micro_real = micro_kernel_entry<RealT>();
                        int64_t const              Mh         = 2 * M;
                        int64_t const              Kh         = 2 * K;
                        int64_t const              KHC        = std::min<int64_t>(int64_t{4096}, Kh); // even: Kh even, 4096 even
                        int64_t const              MHC        = 256;                                  // even
                        int64_t const              num_ir_max = (MHC + MR - 1) / MR;
                        int64_t const              num_jr_max = (nc_len + NR - 1) / NR;

                        static thread_local std::vector<RealT> tls_Ap1, tls_Bp1, tls_Cb1;
                        tls_Ap1.resize(static_cast<size_t>(num_ir_max * MR * KHC));
                        tls_Bp1.resize(static_cast<size_t>(num_jr_max * NR * KHC));
                        tls_Cb1.resize(static_cast<size_t>(MHC) * static_cast<size_t>(nc_len));

                        for (int64_t kh = 0; kh < Kh; kh += KHC) {
                            int64_t const kh_len = std::min(KHC, Kh - kh);
                            pack_B_1m_panels<RealT>(tls_Bp1.data(), B_data, plan, kh, kh_len, nc, nc_len, NR, conj_b);

                            for (int64_t mh = 0; mh < Mh; mh += MHC) {
                                int64_t const mh_len = std::min(MHC, Mh - mh);
                                precompute_offsets(mh / 2, mh_len / 2, plan.c_m_dims, c_m_offsets);

                                // Beta prescale once per (mh, nc) block on the first kh slice.
                                if (kh == 0 && beta != ValueType{1}) {
                                    for (int64_t mi = 0; mi < mh_len / 2; ++mi) {
                                        int64_t const m_off = c_m_offsets[static_cast<size_t>(mi)];
                                        for (int64_t ni = 0; ni < nc_len; ++ni) {
                                            C_data[m_off + c_n_offsets[static_cast<size_t>(ni)]] *= beta;
                                        }
                                    }
                                }

                                pack_A_1m_panels<RealT>(tls_Ap1.data(), A_data, plan, mh, mh_len, kh, kh_len, MR, conj_a);

                                std::fill(tls_Cb1.begin(), tls_Cb1.begin() + static_cast<size_t>(mh_len) * nc_len, RealT{0});
                                int64_t const num_jr = (nc_len + NR - 1) / NR;
                                int64_t const num_ir = (mh_len + MR - 1) / MR;
                                for (int64_t jr = 0; jr < num_jr; ++jr) {
                                    int64_t const nr_eff = std::min<int64_t>(NR, nc_len - jr * NR);
                                    for (int64_t ir = 0; ir < num_ir; ++ir) {
                                        int64_t const mr_eff = std::min<int64_t>(MR, mh_len - ir * MR);
                                        micro_real(MR, NR, kh_len, RealT{1}, tls_Ap1.data() + ir * MR * kh_len,
                                                   tls_Bp1.data() + jr * NR * kh_len, mr_eff, nr_eff,
                                                   tls_Cb1.data() + (jr * NR) * mh_len + ir * MR, 1, mh_len);
                                    }
                                }

                                // Complex scatter: even/odd real row pairs are re/im.
                                for (int64_t j = 0; j < nc_len; ++j) {
                                    int64_t const n_off = c_n_offsets[static_cast<size_t>(j)];
                                    RealT const  *src   = tls_Cb1.data() + j * mh_len;
                                    for (int64_t ii = 0; ii < mh_len; ii += 2) {
                                        C_data[c_m_offsets[static_cast<size_t>(ii / 2)] + n_off] += alpha * ValueType{src[ii], src[ii + 1]};
                                    }
                                }
                            }
                        }
                        continue; // next nc block
                    }
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
                    bool const                                 use_3m = is_complex && shape.use_3m;
                    if (!use_3m) {
                        tls_Af.resize(static_cast<size_t>(MC_blk * KC_blk));
                        tls_Bf.resize(static_cast<size_t>(nc_len) * static_cast<size_t>(KC_blk));
                        tls_Cb.resize(static_cast<size_t>(MC_blk) * static_cast<size_t>(nc_len));
                    }

                    // 3m buffers: three real splits of A, B, and the block
                    // product, laid out as consecutive segments.
                    using Real3m = RemoveComplexT<ValueType>;
                    static thread_local std::vector<Real3m> tls_A3, tls_B3, tls_T3;
                    if (use_3m) {
                        tls_A3.resize(3 * static_cast<size_t>(MC_blk * KC_blk));
                        tls_B3.resize(3 * static_cast<size_t>(nc_len) * static_cast<size_t>(KC_blk));
                        tls_T3.resize(3 * static_cast<size_t>(MC_blk) * static_cast<size_t>(nc_len));
                    }

                    for (int64_t kc = 0; kc < K; kc += KC_blk) {
                        int64_t const kc_len = std::min(KC_blk, K - kc);
                        if constexpr (is_complex) {
                            if (use_3m) {
                                size_t const bseg = static_cast<size_t>(nc_len) * static_cast<size_t>(kc_len);
                                pack_B_3m_flat<Real3m>(tls_B3.data(), tls_B3.data() + bseg, tls_B3.data() + 2 * bseg, B_data, plan, kc,
                                                       kc_len, nc, nc_len, conj_b);
                            }
                        }
                        if (!use_3m) {
                            pack_B_flat(tls_Bf.data(), B_data, plan, kc, kc_len, nc, nc_len, conj_b);
                        }

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

                            if constexpr (is_complex) {
                                if (use_3m) {
                                    // ---- 3m: three real GEMMs, combined at the scatter ----
                                    size_t const aseg = static_cast<size_t>(mc_len) * static_cast<size_t>(kc_len);
                                    size_t const bseg = static_cast<size_t>(nc_len) * static_cast<size_t>(kc_len);
                                    size_t const tseg = static_cast<size_t>(mc_len) * static_cast<size_t>(nc_len);
                                    pack_A_3m_flat<Real3m>(tls_A3.data(), tls_A3.data() + aseg, tls_A3.data() + 2 * aseg, A_data, plan, mc,
                                                           mc_len, kc, kc_len, conj_a);
                                    for (int t = 0; t < 3; ++t) {
                                        einsums::blas::gemm<Real3m>('N', 'T', static_cast<blas_int>(mc_len), static_cast<blas_int>(nc_len),
                                                                    static_cast<blas_int>(kc_len), Real3m{1}, tls_A3.data() + t * aseg,
                                                                    static_cast<blas_int>(mc_len), tls_B3.data() + t * bseg,
                                                                    static_cast<blas_int>(nc_len), Real3m{0}, tls_T3.data() + t * tseg,
                                                                    static_cast<blas_int>(mc_len));
                                    }
                                    Real3m const *t1 = tls_T3.data();
                                    Real3m const *t2 = tls_T3.data() + tseg;
                                    Real3m const *t3 = tls_T3.data() + 2 * tseg;
                                    for (int64_t j = 0; j < nc_len; ++j) {
                                        int64_t const n_off = c_n_offsets[static_cast<size_t>(j)];
                                        for (int64_t i2 = 0; i2 < mc_len; ++i2) {
                                            size_t const idx = static_cast<size_t>(j * mc_len + i2);
                                            Real3m const re  = t1[idx] - t2[idx];
                                            Real3m const im  = t3[idx] - t1[idx] - t2[idx];
                                            C_data[c_m_offsets[static_cast<size_t>(i2)] + n_off] += alpha * ValueType{re, im};
                                        }
                                    }
                                    continue; // next mc block
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
/// @param site Optional memo owned by a caller that repeats this exact
///        contraction (a graph node). See @ref ContractionSite: a hit skips
///        assembling the spec, the key and its stride vectors, hashing them,
///        and the plan-cache lookup, none of which can change between two
///        calls that the key compares equal for.
template <einsums::BasicTensorConcept AType, einsums::BasicTensorConcept BType, einsums::BasicTensorConcept CType>
bool try_packed_gemm(ContractionSpec const &spec_in, einsums::ValueTypeT<CType> C_prefactor, CType *C,
                     einsums::BiggestTypeT<typename AType::ValueType, typename BType::ValueType> AB_prefactor, AType const &A,
                     BType const &B, bool allow_scatter = true, ContractionSite *site = nullptr) {
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

    // Memo hit: this caller already resolved this exact contraction, under the
    // same policy, against operands with this layout and these sizes.
    if (site != nullptr && site->resolved && site->allow_scatter == allow_scatter && site_key_matches(site->key, spec_in, st, A, B, *C)) {
        if (site->plan == nullptr) {
            ProfileAnnotate("packed_gemm_skip", "site_declined");
            return false;
        }
        ProfileAnnotate("packed_gemm_plan", "site");
        blis_contraction<ValueType>(*site->plan, *C, A, B, static_cast<ValueType>(AB_prefactor), static_cast<ValueType>(C_prefactor),
                                    spec_in.conj_a, spec_in.conj_b);
        return true;
    }

    // Records what this call resolved to, so the next one can skip straight to
    // it. A null plan means "declined": that is a property of the key too, and
    // re-deriving it costs the same as finding a plan would.
    auto remember = [site, allow_scatter](ContractionKey const &k, PackingPlan const *p) {
        if (site != nullptr) {
            site->key           = k;
            site->plan          = p;
            site->allow_scatter = allow_scatter;
            site->resolved      = true;
        }
    };

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

    // Hashing the whole key - every index string in the spec, plus three
    // stride vectors - purely to label a profiler annotation, on a path the
    // tiled expansion drives thousands of times per replay. It buys a
    // debugging aid, so it is worth exactly what a recording run will pay for
    // it and nothing on a run that records nothing.
    if (profile::Profiler::instance().enabled()) {
        profile::annotate("packed_gemm_hash", static_cast<int64_t>(std::hash<ContractionKey>{}(key)));
    }

    // -------------------------------------------------------------------------
    // Classify target indices. Empty M/N/link groups are no longer
    // rejections: compute_packing_topology synthesizes a unit dim so GEMV-
    // and outer-product-shaped contractions run through the same block/tile
    // machinery. What the classification still decides is (a) the direct-GEMM
    // deferral and (b) whether Sort+GEMM exists as a fallback for the policy
    // check below (it requires all three groups non-empty).
    // -------------------------------------------------------------------------
    bool ttgt_exists  = false;
    bool outer_shaped = false;
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
            remember(key, nullptr);
            return false; // Deferred to direct BLAS GEMM, not a rejection.
        }
        // ── Direct BLAS fast paths ───────────────────────────────────────
        // The packed structure needs a K dimension to amortize its packing
        // copy. Two shape classes have none, and pay 2-3 memory passes for
        // work one BLAS call does in a single pass:
        //
        //   - outer product (no link indices): a K=1 GEMM, where the
        //     micro-kernel's per-tile setup IS the cost, while ger writes C at
        //     bandwidth.
        //   - GEMV-shaped (no N, or no M): packing copies the largest operand
        //     (read + write) and the kernel then reads it again - three passes
        //     where gemv makes one. This is the "no K reuse to amortize the
        //     packing copy" case noted below.
        //
        // Measured 4-5x on both against the packed path. They apply when the
        // operands' axis groups flatten to BLAS shapes, which is the common
        // case (dense tensors, index groups already adjacent); anything that
        // does not flatten falls through to packing below, unchanged.
        //
        // Conjugated operands are excluded: ger/gerc and the gemv transposes
        // differ for complex, and the packing path already handles conjugation
        // natively.
        if (!spec.conj_a && !spec.conj_b) {
            auto const alpha = static_cast<ValueType>(AB_prefactor);
            auto const beta  = static_cast<ValueType>(C_prefactor);

            // ── Outer product -> ger / k=1 gemm ──
            // C's axes must be A's group followed by B's (or the reverse), so
            // one flat 2-D view of C exists. C must be wholly contiguous, which
            // also makes the destination prefactor a single pass.
            auto try_outer = [&]() -> bool {
                if (!link.empty() || !(is_concatenation(c_raw, a_raw, b_raw) || is_concatenation(c_raw, b_raw, a_raw))) {
                    return false;
                }
                bool const   a_first = is_concatenation(c_raw, a_raw, b_raw);
                size_t const split   = a_first ? a_raw.size() : b_raw.size();

                size_t m = 0, n = 0, s_first = 0, s_second = 0, c_all = 0, s_c = 0;
                size_t a_ext = 0, inc_a = 0, b_ext = 0, inc_b = 0;
                if (!flatten_run(*C, 0, split, m, s_first) || !flatten_run(*C, split, c_raw.size(), n, s_second) ||
                    !flatten_run(*C, 0, c_raw.size(), c_all, s_c) || !flatten_run(A, 0, a_raw.size(), a_ext, inc_a) ||
                    !flatten_run(B, 0, b_raw.size(), b_ext, inc_b) || s_c != 1 || c_all != m * n) {
                    return false;
                }
                // m/n are C's two groups in C's own order; map them back to the
                // operands, which may be swapped relative to that.
                size_t const m_a = a_first ? m : n;
                size_t const n_b = a_first ? n : m;
                if (a_ext != m_a || b_ext != n_b || (s_first != 1 && s_second != 1)) {
                    return false;
                }
                // Below this the fixed cost of a BLAS call is the whole
                // measurement, and the existing small-outer deferral already
                // picks the runtime loop; leave that decision where it is.
                if (m * n < kOuterMinElems) {
                    return false;
                }

                using int_t            = einsums::blas::int_t;
                auto       *cp         = C->data();
                auto const *ap         = A.data();
                auto const *bp         = B.data();
                bool const  rows_first = s_first == 1;

                // Rows are whichever of C's groups carries unit stride; the
                // other is the column axis and supplies ldc.
                auto const rows = static_cast<int_t>(rows_first ? m : n);
                auto const cols = static_cast<int_t>(rows_first ? n : m);
                auto       ldc  = static_cast<int_t>(rows_first ? s_second : s_first);

                // With one column there is no second column to address, so the
                // column stride is unconstrained - and an all-extent-1 group
                // reports a placeholder anyway. BLAS still validates
                // ldc >= rows, so pin it. Anything still short of that is a
                // layout no leading dimension can describe: decline rather than
                // hand the vendor an ldc it will reject.
                if (cols == 1) {
                    ldc = std::max(ldc, rows);
                }
                if (ldc < rows) {
                    return false;
                }

                // Which operand feeds the row axis: C's first group is A's when
                // a_first, so rows come from A iff those agree.
                bool const  rows_from_a = rows_first == a_first;
                auto const *rp          = rows_from_a ? ap : bp;
                auto const *cq          = rows_from_a ? bp : ap;
                auto const  inc_r       = static_cast<int_t>(rows_from_a ? inc_a : inc_b);
                auto const  inc_c2      = static_cast<int_t>(rows_from_a ? inc_b : inc_a);

                // A k=1 GEMM rather than scal-then-ger: ger has no beta, so a
                // destination prefactor would cost a separate pass over C - and
                // C traffic IS the cost of an outer product, so that pass is the
                // whole overhead. gemm folds beta in. Past kOuterGemmMaxElems
                // its blocking overhead outgrows the saved pass. A strided
                // operand has no ldb to express, so it takes ger either way.
                if (inc_r == 1 && inc_c2 == 1 && (static_cast<size_t>(rows) * cols) < kOuterGemmMaxElems) {
                    ProfileAnnotate("packed_gemm_path", "direct_outer_gemm");
                    einsums::blas::gemm<ValueType>('n', 'n', rows, cols, 1, alpha, rp, rows, cq, 1, beta, cp, ldc);
                    return true;
                }
                ProfileAnnotate("packed_gemm_path", "direct_ger");
                if (beta == ValueType{0}) {
                    std::fill(cp, cp + (m * n), ValueType{0});
                } else if (beta != ValueType{1}) {
                    einsums::blas::scal<ValueType>(static_cast<int_t>(m * n), beta, cp, 1);
                }
                einsums::blas::ger<ValueType>(rows, cols, alpha, rp, inc_r, cq, inc_c2, cp, ldc);
                return true;
            };
            if (try_outer()) {
                return true;
            }

            // ── GEMV-shaped -> gemv ──
            // Exactly one operand supplies all of C; the other is entirely
            // contracted. The supplying operand's axes must be C's group and
            // the link group, adjacent in either order.
            if (!link.empty() && (m_count == 0) != (n_count == 0) && m_count + n_count == target.size()) {
                // A and B are distinct types, so the supplying operand cannot be
                // selected into one reference - the shared body is a template
                // instead, instantiated for whichever side supplies C.
                auto try_gemv = [&]<einsums::BasicTensorConcept SupT, einsums::BasicTensorConcept VecT>(
                                    SupT const &S, VecT const &V, std::vector<std::string> const &sup,
                                    std::vector<std::string> const &other) -> bool {
                    // The contracted operand must be exactly the link group, so
                    // it flattens to the gemv vector.
                    if (other != link || !(is_concatenation(sup, c_raw, link) || is_concatenation(sup, link, c_raw))) {
                        return false;
                    }
                    bool const   c_first = is_concatenation(sup, c_raw, link);
                    size_t const split   = c_first ? c_raw.size() : link.size();

                    size_t d0 = 0, s0 = 0, d1 = 0, s1 = 0, v_ext = 0, inc_v = 0, c_ext = 0, inc_c = 0;
                    if (!flatten_run(S, 0, split, d0, s0) || !flatten_run(S, split, sup.size(), d1, s1) ||
                        !flatten_run(V, 0, other.size(), v_ext, inc_v) || !flatten_run(*C, 0, c_raw.size(), c_ext, inc_c)) {
                        return false;
                    }
                    size_t const m   = c_first ? d0 : d1; // C extent
                    size_t const k   = c_first ? d1 : d0; // link extent
                    size_t const s_m = c_first ? s0 : s1;
                    size_t const s_k = c_first ? s1 : s0;
                    if (v_ext != k || c_ext != m) {
                        return false;
                    }
                    using int_t    = einsums::blas::int_t;
                    auto const *sp = S.data();
                    auto const *vp = V.data();
                    auto       *cp = C->data();
                    // Same leading-dimension rule as the outer path: with a
                    // single column the other stride is unconstrained (and an
                    // all-extent-1 group reports a placeholder), but BLAS still
                    // validates lda against the leading extent.
                    if (s_m == 1) {
                        // S is m x k column-major: y = alpha*S*v + beta*y
                        auto lda = static_cast<int_t>(s_k);
                        if (k == 1) {
                            lda = std::max(lda, static_cast<int_t>(m));
                        }
                        if (lda < static_cast<int_t>(m)) {
                            return false;
                        }
                        ProfileAnnotate("packed_gemm_path", "direct_gemv");
                        einsums::blas::gemv<ValueType>('n', static_cast<int_t>(m), static_cast<int_t>(k), alpha, sp, lda, vp,
                                                       static_cast<int_t>(inc_v), beta, cp, static_cast<int_t>(inc_c));
                        return true;
                    }
                    if (s_k == 1) {
                        // S is k x m column-major: y = alpha*S^T*v + beta*y
                        auto lda = static_cast<int_t>(s_m);
                        if (m == 1) {
                            lda = std::max(lda, static_cast<int_t>(k));
                        }
                        if (lda < static_cast<int_t>(k)) {
                            return false;
                        }
                        ProfileAnnotate("packed_gemm_path", "direct_gemv");
                        einsums::blas::gemv<ValueType>('t', static_cast<int_t>(k), static_cast<int_t>(m), alpha, sp, lda, vp,
                                                       static_cast<int_t>(inc_v), beta, cp, static_cast<int_t>(inc_c));
                        return true;
                    }
                    return false;
                };

                if (n_count == 0) {
                    if (try_gemv(A, B, a_raw, b_raw)) {
                        return true;
                    }
                } else if (try_gemv(B, A, b_raw, a_raw)) {
                    return true;
                }
            }
        }

        // Bandwidth-bound shape classes where the packed pass structure
        // (pack + kernel + scatter = 2-3 memory passes) measurably loses to
        // the COMPILE-TIME generic loop's single fused pass, at every size
        // tested:
        //   - batch-dot: no M and no N indices (per-batch dot products)
        //   - GEMV-shaped: exactly one of M/N empty (no K reuse to amortize
        //     the packing copy)
        // The decline is gated on !allow_scatter (the eager einsum dispatch,
        // whose generic fallback is the good one). Runtime callers
        // (allow_scatter=true, e.g. ComputeGraph's string dispatch) fall back
        // to the runtime nested loop instead, which measures ~1000x slower
        // than packed for streamed GEMV shapes - they keep the packed path.
        if (!allow_scatter && m_count == 0 && n_count == 0) {
            ProfileAnnotate("packed_gemm_skip", "defer_to_generic_batch_dot");
            remember(key, nullptr);
            return false;
        }
        if (!allow_scatter && (m_count == 0 || n_count == 0)) {
            ProfileAnnotate("packed_gemm_skip", "defer_to_generic_gemv_shaped");
            remember(key, nullptr);
            return false;
        }
        outer_shaped = link.empty();
        ttgt_exists  = m_count > 0 && n_count > 0 && !link.empty();
    }

    // -------------------------------------------------------------------------
    // Pack-A / Pack-B path (BLIS-style, with optional batch dims).
    // -------------------------------------------------------------------------
    // The cache stores plans that are already filled, k-sorted and coalesced, so
    // a hit is a lookup and nothing else. That is sound because the key pins the
    // strides (TensorDescriptor::strides) and those three steps read nothing
    // else - they never look at an operand pointer. It matters because tiled
    // expansion drives thousands of same-shape contractions through here, where
    // redoing the preparation dominated the arithmetic.
    //
    // The pointer outlives the shared lock deliberately: entries are never
    // erased, and unordered_map is node-based, so rehashing does not invalidate
    // references to mapped values. Nothing is copied on the hot path.
    PackingPlan const *cached = PackingPlanCache::instance().lookup(key);
    PackingPlan        computed;
    if (cached == nullptr) {
        computed = compute_packing_topology(key);
        if (computed.valid) {
            fill_strides(computed, A, B, *C);
            sort_k_dims_for_packing(computed);
            coalesce_plan(computed);
            PackingPlanCache::instance().insert(key, computed);
            // Re-look-up so `cached` names the CACHE's copy, not this frame's.
            // A site remembers the pointer, and only the cache's entries live
            // long enough to be remembered (they are never erased, and the map
            // is node-based, so the address is stable).
            cached = PackingPlanCache::instance().lookup(key);
            ProfileAnnotate("packed_gemm_plan", "computed");
        }
    } else {
        ProfileAnnotate("packed_gemm_plan", "cached");
    }
    PackingPlan const &plan = (cached != nullptr) ? *cached : computed;
    if (plan.valid) {

        bool const multi_m = (plan.c_m_dims.size() > 1);
        bool const multi_n = (plan.c_n_dims.size() > 1);
        // Scatter is needed for multi-M/N and for single-M/N layouts where
        // neither output dim is unit-stride (e.g. a batched C whose batch
        // index owns stride 1). The block/tile scatter paths handle all of
        // these; the only question is policy.
        bool const needs_scatter = multi_m || multi_n || (plan.c_m_dims[0].tensor_stride != 1 && plan.c_n_dims[0].tensor_stride != 1);

        // Outer products (no link indices, K synthesized to 1) pay a fixed ~3.4 us
        // to set the packed passes up, and beat the generic nested loop above
        // roughly 4k output elements. Measured on both CCSD t1*t1 shapes
        // (BenchmarkOuterProduct, Apple M4 Pro, packed vs generic):
        //
        //   elements     256    2.3k    9.2k     230k     922k    14.7M
        //   speedup     0.42x   0.99x   1.51x    4.05x    6.51x    5.25x
        //
        // The two shapes agree to within noise, so one threshold covers both.
        //
        // Note for anyone re-deriving this: what the loser is matters. Below
        // rank-3 output an outer product never reaches here - StringDispatch's
        // GER path takes it first - so the alternative being measured against is
        // always the generic loop, which runs at a flat ~2.2 ns an element.
        if (outer_shaped && plan.M_total * plan.N_total < (int64_t{1} << 12)) {
            ProfileAnnotate("packed_gemm_skip", "defer_small_outer_to_generic");
            remember(key, nullptr);
            return false;
        }

        // Decline when a TTGT fallback exists and either the rung's kernel
        // does not beat it on the scatter path or the shape is batched -
        // Sort+GEMM's per-batch canonical GEMMs measure faster than the
        // scatter engines for batched shapes at every size tested.
        if (needs_scatter && ttgt_exists && !allow_scatter && (!micro_kernel_shape<ValueType>().fast_scatter || plan.batch_total > 1)) {
            ProfileAnnotate("packed_gemm_skip", "scatter_defer_to_ttgt");
            EINSUMS_LOG_INFO("PackedGemm: declining — scatter-path shape, the caller has a TTGT fallback, "
                             "and this rung's kernel does not beat it for this shape.");
            remember(key, nullptr);
            return false;
        }

        ProfileAnnotate("packed_gemm_path", needs_scatter ? "scatter" : "single_mn");
        // Only a cache-owned plan is stable enough to remember. `computed` is
        // this frame's; if the re-lookup above somehow missed, skip the memo
        // rather than record a null plan, which would read as "declined" and
        // wrongly turn every later call away.
        if (cached != nullptr) {
            remember(key, cached);
        }
        blis_contraction<ValueType>(plan, *C, A, B, static_cast<ValueType>(AB_prefactor), static_cast<ValueType>(C_prefactor), spec.conj_a,
                                    spec.conj_b);
        return true;
    } else {
        ProfileAnnotate("packed_gemm_skip", "invalid_topology");
        EINSUMS_LOG_INFO("PackedGemm: skipping — packing topology invalid for this contraction pattern.");
        remember(key, nullptr);
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

EINSUMS_NAMESPACE_END(packed_gemm)
