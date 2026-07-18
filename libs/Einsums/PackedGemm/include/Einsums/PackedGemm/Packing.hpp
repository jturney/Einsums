//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

// This header is included from EinsumPackedGemm.hpp.

#include <Einsums/Concepts/TensorConcepts.hpp>
#include <Einsums/PackedGemm/ContractionKey.hpp>
#include <Einsums/Profile/Profile.hpp>

#include <algorithm>
#include <complex>
#include <cstdint>
#include <cstring>
#include <memory>
#include <numeric>
#include <type_traits>
#include <vector>

namespace einsums::packed_gemm {

/// @brief Describes one dimension of a tensor as seen during packing.
struct DimSpec {
    size_t  tensor_pos{0};    ///< Position of this dim in the tensor's raw index list
    int64_t size{0};          ///< Runtime extent of this dimension
    int64_t tensor_stride{0}; ///< Stride (in elements) in the source/destination tensor
};

/// @brief tensor_pos of a synthesized unit dimension (see compute_packing_topology).
///
/// GEMV-shaped (no M or no N indices) and outer-product-shaped (no link
/// indices) contractions get a unit dim (size 1, stride 0) in the empty
/// group so the whole GEMM machinery applies unchanged. The sentinel pos
/// marks entries that must never be used to index a real tensor axis.
inline constexpr size_t kSyntheticDimPos = static_cast<size_t>(-1);

/// @brief Describes how a contraction can be packed as a flat BLIS-style GEMM.
///
/// For a contraction C[m,n] = beta*C[m,n] + alpha * Sum_k A[m,k] * B[k,n]
/// (generalised to arbitrary tensor rank with multiple M, N, and K dims):
///
///   m_dims      : target dims in A (not B)                 -> M_total = product(sizes)
///   n_dims      : target dims in B (not A)                 -> N_total = product(sizes)
///   k_dims_in_a : link dims ordered by link_indices order  -> K_total = product(sizes) (in A)
///   k_dims_in_b : link dims ordered by link_indices order  -> K_total = product(sizes) (in B)
///   c_m_dims    : m_dims as seen in C (same logical index, C position/stride)
///   c_n_dims    : n_dims as seen in C
///
/// `valid` is true only when all validity conditions are met.
/// @brief Describes a batch dimension with positions and strides in all three tensors.
struct BatchDimSpec {
    int64_t size{0};
    size_t  a_pos{0}; ///< Position of this dim in A's raw index list
    size_t  b_pos{0}; ///< Position of this dim in B's raw index list
    size_t  c_pos{0}; ///< Position of this dim in C's raw index list
    int64_t a_stride{0};
    int64_t b_stride{0};
    int64_t c_stride{0};
};

struct PackingPlan {
    bool                 valid{false};
    std::vector<DimSpec> m_dims;      ///< Target dims in A only (M group)
    std::vector<DimSpec> n_dims;      ///< Target dims in B only (N group)
    std::vector<DimSpec> k_dims_in_a; ///< Link dims, A positions/strides (link_indices order)
    std::vector<DimSpec> k_dims_in_b; ///< Link dims, B positions/strides (link_indices order)
    std::vector<DimSpec> c_m_dims;    ///< m_dims as seen in C
    std::vector<DimSpec> c_n_dims;    ///< n_dims as seen in C
    int64_t              M_total{0};
    int64_t              N_total{0};
    int64_t              K_total{0};

    /// Batch dimensions (target indices in both A and B).
    /// Empty for non-batched contractions.
    std::vector<BatchDimSpec> batch_dims;
    int64_t                   batch_total{1}; ///< Product of all batch dim sizes

    /// True when coalesce_plan() merged at least one pair of dims. Merged
    /// DimSpecs have meaningless tensor_pos values, so the HPTT flatten path
    /// (which rebuilds axis permutations from tensor_pos) must be skipped.
    bool coalesced{false};

    /// True when compute_packing_topology() synthesized a unit dim for an
    /// empty M/N/K group (GEMV- or outer-product-shaped contraction). The
    /// direct BLAS fast paths (gemm_batch, single-K, multi-K flatten) must
    /// be skipped: a synthetic stride of 0 maps to invalid leading
    /// dimensions there. The tiled/block scatter paths handle it naturally.
    bool synthetic{false};
};

// ---------------------------------------------------------------------------
// CPU vector configuration
// ---------------------------------------------------------------------------

/// @brief CPU vector and cache configuration detected at init time.
struct CpuConfig {
    int VL; ///< SIMD vector length in doubles: SSE=2, AVX=4, AVX-512=8
    int MR; ///< M register-block: 2*VL (two vector registers per j-column)
    int NR; ///< N register-block: 6 (LLVM unroll threshold; fixed across CPUs)

    int64_t l1_cache_size; ///< L1 data cache size in bytes (per core)
    int64_t l2_cache_size; ///< L2 cache size in bytes (per core)
    int64_t l3_cache_size; ///< L3 cache size in bytes (shared)
};

// ---------------------------------------------------------------------------
// BLIS cache-blocking parameters
// ---------------------------------------------------------------------------

/// @brief Cache-blocking parameters for BLIS-style GEMM, tuned per element size.
struct BlockingParams {
    int64_t KC; ///< K tile size (L2 blocking)
    int64_t MC; ///< M tile size (L2 blocking)
    int64_t NC; ///< N tile size (L3 blocking)
    int64_t NR; ///< N register-block
};

/// @brief Compute blocking parameters for the given element size in bytes.
///
/// Uses the CPU cache hierarchy to choose KC, MC, NC such that:
///   - One A panel (MC * KC * elem_size) fits in ~half the L2 cache.
///   - One B panel (KC * NC * elem_size) fits in ~half the L3 cache.
/// This automatically handles complex types (16 bytes) vs real (8 bytes).
EINSUMS_EXPORT BlockingParams compute_blocking(int64_t elem_size);

// Convenience: default blocking for 8-byte elements (double / complex<float>).
inline constexpr int64_t BLIS_NR = 6; ///< N register-block (fully unrolled by LLVM)

/// @brief Return the native CPU vector configuration (computed once, cached).
EINSUMS_EXPORT CpuConfig const &cpu_config();

// ---------------------------------------------------------------------------
// PackingPlanCache
// ---------------------------------------------------------------------------

/// @brief Thread-safe cache of filled PackingPlans, keyed by ContractionKey.
///
/// Skips recomputation of compute_packing_topology on cache hits.
class EINSUMS_EXPORT PackingPlanCache {
  public:
    static PackingPlanCache &instance();

    /// @brief Look up a cached PackingPlan.  Returns nullptr if not found.
    [[nodiscard]] PackingPlan const *lookup(ContractionKey const &key) const;

    /// @brief Insert a filled PackingPlan into the cache.
    void insert(ContractionKey const &key, PackingPlan plan);

    // Non-copyable singleton.
    PackingPlanCache(PackingPlanCache const &)            = delete;
    PackingPlanCache &operator=(PackingPlanCache const &) = delete;

  private:
    PackingPlanCache();
    ~PackingPlanCache();

    struct Impl;
    std::unique_ptr<Impl> _impl;
};

// ---------------------------------------------------------------------------
// Topology analysis (no strides -- call fill_strides afterwards)
// ---------------------------------------------------------------------------

/// @brief Compute the BLIS packing topology for a contraction.
///
/// Returns `plan.valid == true` only when:
///   - No scalar output
///   - No repeated a_indices or b_indices (Hadamard)
///   - At least one M dim, at least one N dim, at least one K dim
///   - Multi-M, multi-N, and multi-K are all supported
///
/// `plan.m_dims[i].tensor_stride` etc. are left as 0; call `fill_strides` to fill them.
PackingPlan EINSUMS_EXPORT compute_packing_topology(ContractionKey const &key);

/// @brief Fill tensor strides in a PackingPlan from the actual runtime tensors.
///
/// Must be called after `compute_packing_topology` returns a valid plan.
template <einsums::BasicTensorConcept AType, einsums::BasicTensorConcept BType, einsums::BasicTensorConcept CType>
void fill_strides(PackingPlan &plan, AType const &A, BType const &B, CType const &C) {
    // Synthetic unit dims (kSyntheticDimPos) keep their stride of 0; they do
    // not correspond to any real tensor axis.
    for (auto &ds : plan.m_dims) {
        if (ds.tensor_pos != kSyntheticDimPos)
            ds.tensor_stride = static_cast<int64_t>(A.stride(ds.tensor_pos));
    }
    for (auto &ds : plan.k_dims_in_a) {
        if (ds.tensor_pos != kSyntheticDimPos)
            ds.tensor_stride = static_cast<int64_t>(A.stride(ds.tensor_pos));
    }
    for (auto &ds : plan.n_dims) {
        if (ds.tensor_pos != kSyntheticDimPos)
            ds.tensor_stride = static_cast<int64_t>(B.stride(ds.tensor_pos));
    }
    for (auto &ds : plan.k_dims_in_b) {
        if (ds.tensor_pos != kSyntheticDimPos)
            ds.tensor_stride = static_cast<int64_t>(B.stride(ds.tensor_pos));
    }
    for (auto &ds : plan.c_m_dims) {
        if (ds.tensor_pos != kSyntheticDimPos)
            ds.tensor_stride = static_cast<int64_t>(C.stride(ds.tensor_pos));
    }
    for (auto &ds : plan.c_n_dims) {
        if (ds.tensor_pos != kSyntheticDimPos)
            ds.tensor_stride = static_cast<int64_t>(C.stride(ds.tensor_pos));
    }
    for (auto &bds : plan.batch_dims) {
        bds.a_stride = static_cast<int64_t>(A.stride(bds.a_pos));
        bds.b_stride = static_cast<int64_t>(B.stride(bds.b_pos));
        bds.c_stride = static_cast<int64_t>(C.stride(bds.c_pos));
    }
}

/// @brief Reorder k_dims_in_a and k_dims_in_b (same permutation) for cache-friendly packing.
///
/// Must be called after fill_strides() so tensor_stride values are populated.
inline void sort_k_dims_for_packing(PackingPlan &plan) {
    size_t const n = plan.k_dims_in_a.size();
    if (n <= 1)
        return;

    // Build index permutation sorted descending by max(stride_A, stride_B).
    std::vector<size_t> perm(n);
    std::iota(perm.begin(), perm.end(), size_t{0});
    std::ranges::sort(perm, [&](size_t a, size_t b) {
        int64_t const max_a = std::max(plan.k_dims_in_a[a].tensor_stride, plan.k_dims_in_b[a].tensor_stride);
        int64_t const max_b = std::max(plan.k_dims_in_a[b].tensor_stride, plan.k_dims_in_b[b].tensor_stride);
        return max_a > max_b; // descending: largest stride first (slowest varying)
    });

    std::vector<DimSpec> new_ka(n), new_kb(n);
    for (size_t i = 0; i < n; ++i) {
        new_ka[i] = plan.k_dims_in_a[perm[i]];
        new_kb[i] = plan.k_dims_in_b[perm[i]];
    }
    plan.k_dims_in_a = std::move(new_ka);
    plan.k_dims_in_b = std::move(new_kb);
}

/// @brief Coalesce adjacent dims within each group when they tile contiguously
///        in every tensor that sees them.
///
/// Each group is a pair of parallel DimSpec vectors describing the same logical
/// indices in two tensors (M: A and C; N: B and C; K: A and B). The group is
/// first reordered by descending primary-tensor stride (the same permutation is
/// applied to the mirror so flat-index enumeration stays consistent), then
/// neighbouring dims are merged whenever slow.stride == fast.stride * fast.size
/// holds in BOTH tensors. This converts e.g. col-major
/// C(a,b,i,j) = A(a,b,e,f) * B(e,f,i,j) from a multi-M/N/K plan (scatter path)
/// into a single-M/N/K plan that maps to one strided BLAS GEMM with no copies.
///
/// Must be called after fill_strides() (and after sort_k_dims_for_packing(),
/// which it may re-permute). Sets plan.coalesced when any merge happened.
inline void coalesce_plan(PackingPlan &plan) {
    auto coalesce_group = [&plan](std::vector<DimSpec> &prim, std::vector<DimSpec> &mirr) {
        size_t const n = prim.size();
        if (n <= 1) {
            return;
        }

        // Order both vectors by descending primary stride so the last entry is
        // the fastest-varying flat coordinate (the flat_to_offset convention).
        std::vector<size_t> perm(n);
        std::iota(perm.begin(), perm.end(), size_t{0});
        std::stable_sort(perm.begin(), perm.end(), [&](size_t x, size_t y) { return prim[x].tensor_stride > prim[y].tensor_stride; });

        std::vector<DimSpec> ps(n), ms(n);
        for (size_t d = 0; d < n; ++d) {
            ps[d] = prim[perm[d]];
            ms[d] = mirr[perm[d]];
        }

        std::vector<DimSpec> pout{ps[0]}, mout{ms[0]};
        for (size_t d = 1; d < n; ++d) {
            DimSpec   &pl     = pout.back();
            DimSpec   &ml     = mout.back();
            bool const contig = pl.tensor_stride == ps[d].tensor_stride * ps[d].size && //
                                ml.tensor_stride == ms[d].tensor_stride * ms[d].size;
            if (contig) {
                pl.size *= ps[d].size;
                pl.tensor_stride = ps[d].tensor_stride;
                ml.size *= ms[d].size;
                ml.tensor_stride = ms[d].tensor_stride;
                plan.coalesced   = true;
            } else {
                pout.push_back(ps[d]);
                mout.push_back(ms[d]);
            }
        }
        prim = std::move(pout);
        mirr = std::move(mout);
    };

    coalesce_group(plan.m_dims, plan.c_m_dims);
    coalesce_group(plan.n_dims, plan.c_n_dims);
    coalesce_group(plan.k_dims_in_a, plan.k_dims_in_b);
}

// ---------------------------------------------------------------------------
// Packing kernels (micro-panel layout for register blocking)
// ---------------------------------------------------------------------------

/// @brief Compute the byte offset for a flat index into a multi-dim group.
///
/// Converts flat index `idx` into multi-dimensional coordinates for `dims`
/// and returns the sum of coordinate * stride products.
inline int64_t flat_to_offset(int64_t idx, std::vector<DimSpec> const &dims) {
    int64_t offset = 0;
    for (int d = static_cast<int>(dims.size()) - 1; d >= 0; --d) {
        int64_t const coord = idx % dims[static_cast<size_t>(d)].size;
        idx /= dims[static_cast<size_t>(d)].size;
        offset += coord * dims[static_cast<size_t>(d)].tensor_stride;
    }
    return offset;
}

/// @brief Precompute offsets for a range of flat indices into a multi-dim group.
///
/// For 2-dim and 3-dim groups (rank-3 and rank-4 tensors), uses nested loops
/// with direct stride multiplication instead of the generic flat_to_offset
/// which requires expensive division/modulo per element.
inline void precompute_offsets(int64_t start, int64_t len, std::vector<DimSpec> const &dims, std::vector<int64_t> &out) {
    out.resize(static_cast<size_t>(len));

    if (dims.size() == 2) {
        // Rank-3 specialization: 2 dimensions, no division needed per element.
        // flat_index = i * dim1_size + j
        int64_t const dim1    = dims[1].size;
        int64_t const stride0 = dims[0].tensor_stride;
        int64_t const stride1 = dims[1].tensor_stride;

        // Decompose start into (i0, j0)
        int64_t i = start / dim1;
        int64_t j = start % dim1;

        for (int64_t idx = 0; idx < len; ++idx) {
            out[static_cast<size_t>(idx)] = i * stride0 + j * stride1;
            // Increment (i, j): just increment j and carry to i
            if (++j >= dim1) {
                j = 0;
                ++i;
            }
        }
        return;
    }

    if (dims.size() == 3) {
        // Rank-4 specialization: 3 dimensions, no division needed per element.
        // flat_index = i * dim1_size * dim2_size + j * dim2_size + k
        int64_t const dim1    = dims[1].size;
        int64_t const dim2    = dims[2].size;
        int64_t const stride0 = dims[0].tensor_stride;
        int64_t const stride1 = dims[1].tensor_stride;
        int64_t const stride2 = dims[2].tensor_stride;
        int64_t const dim12   = dim1 * dim2;

        // Decompose start into (i0, j0, k0)
        int64_t       i   = start / dim12;
        int64_t const rem = start % dim12;
        int64_t       j   = rem / dim2;
        int64_t       k   = rem % dim2;

        for (int64_t idx = 0; idx < len; ++idx) {
            out[static_cast<size_t>(idx)] = i * stride0 + j * stride1 + k * stride2;
            // Increment (i, j, k) with carry
            if (++k >= dim2) {
                k = 0;
                if (++j >= dim1) {
                    j = 0;
                    ++i;
                }
            }
        }
        return;
    }

    // Generic fallback for 4+ dimensions
    for (int64_t i = 0; i < len; ++i) {
        out[static_cast<size_t>(i)] = flat_to_offset(start + i, dims);
    }
}

/// @brief Gather-pack A[mc:mc+mc_len, kc:kc+kc_len] into column-major MR*KC panels.
///
/// Supports single-M (fast paths with memcpy) and multi-M (flat-index-to-offset conversion).
template <typename T>
// NOLINTNEXTLINE(readability-identifier-naming)
void pack_A(T *Ap, T const *A_data, PackingPlan const &plan, int64_t mc_start, int64_t mc_len, int64_t kc_start, int64_t kc_len, int MR,
            bool conj = false) {
    LabeledSectionInternal0();
    auto const &m_dims  = plan.m_dims;
    auto const &k_dims  = plan.k_dims_in_a;
    bool const  multi_m = (m_dims.size() > 1);

    int64_t const num_panels  = (mc_len + MR - 1) / MR;
    int64_t const full_panels = mc_len / MR;
    int64_t const tail        = mc_len % MR;

    if (tail > 0) {
        T *last_panel = Ap + full_panels * MR * kc_len;
        std::fill(last_panel, last_panel + MR * kc_len, T{});
    }

    // --- Single-M fast paths (unchanged from before) ---
    if (!multi_m && k_dims.size() == 1) {
        int64_t const m_stride = m_dims[0].tensor_stride;
        int64_t const k_stride = k_dims[0].tensor_stride;

        if (m_stride == 1 && !conj) {
            for (int64_t k_local = 0; k_local < kc_len; ++k_local) {
                T const *src = A_data + mc_start + (kc_start + k_local) * k_stride;
                for (int64_t p = 0; p < full_panels; ++p) {
                    std::memcpy(Ap + p * MR * kc_len + k_local * MR, src + p * MR, static_cast<size_t>(MR) * sizeof(T));
                }
                if (tail > 0) {
                    std::memcpy(Ap + full_panels * MR * kc_len + k_local * MR, src + full_panels * MR,
                                static_cast<size_t>(tail) * sizeof(T));
                }
            }
            return;
        }

        for (int64_t k_local = 0; k_local < kc_len; ++k_local) {
            int64_t const k_offset = (kc_start + k_local) * k_stride;
            T const      *src      = A_data + k_offset + mc_start * m_stride;
            for (int64_t p = 0; p < num_panels; ++p) {
                int64_t const panel_len = (p < full_panels) ? MR : tail;
                T            *dst       = Ap + p * MR * kc_len + k_local * MR;
                for (int64_t i = 0; i < panel_len; ++i) {
                    T val = src[(p * MR + i) * m_stride];
                    if constexpr (std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>) {
                        if (conj)
                            val = std::conj(val);
                    }
                    dst[i] = val;
                }
            }
        }
        return;
    }

    // --- General path: precompute K offsets (multi-K or single-K) ---
    static thread_local std::vector<int64_t> k_offsets;
    if (k_dims.size() == 1) {
        k_offsets.resize(static_cast<size_t>(kc_len));
        int64_t const k_stride = k_dims[0].tensor_stride;
        for (int64_t k_local = 0; k_local < kc_len; ++k_local) {
            k_offsets[static_cast<size_t>(k_local)] = (kc_start + k_local) * k_stride;
        }
    } else {
        precompute_offsets(kc_start, kc_len, k_dims, k_offsets);
    }

    // --- General path: precompute M offsets (multi-M or single-M) ---
    static thread_local std::vector<int64_t> m_offsets;
    if (multi_m) {
        precompute_offsets(mc_start, mc_len, m_dims, m_offsets);
    } else {
        m_offsets.resize(static_cast<size_t>(mc_len));
        int64_t const m_stride = m_dims[0].tensor_stride;
        for (int64_t i = 0; i < mc_len; ++i) {
            m_offsets[static_cast<size_t>(i)] = (mc_start + i) * m_stride;
        }
    }

    // --- Contiguous-run detection ---
    // After coalesce_plan() the group is ordered by descending stride, so the
    // last dim is the fastest flat coordinate. When it has unit stride, a
    // panel row whose flat range stays inside one segment of that dim is a
    // contiguous MR-element run in A - copy it with memcpy (which the
    // compiler vectorizes) instead of the scalar gather. Panels that straddle
    // a segment boundary, non-unit fast strides, and conjugation fall back to
    // the scalar path. (Fully mergeable neighbours were already coalesced, so
    // any straddle is a genuine discontinuity.)
    bool const    m_fast_unit = !conj && m_dims.back().tensor_stride == 1;
    int64_t const m_fast_size = m_dims.back().size;

    // --- Pack with precomputed offsets ---
    for (int64_t k_local = 0; k_local < kc_len; ++k_local) {
        int64_t const k_offset = k_offsets[static_cast<size_t>(k_local)];
        for (int64_t p = 0; p < num_panels; ++p) {
            int64_t const panel_len = (p < full_panels) ? MR : tail;
            T            *dst       = Ap + p * MR * kc_len + k_local * MR;
            if (m_fast_unit && ((mc_start + p * MR) % m_fast_size) + panel_len <= m_fast_size) {
                std::memcpy(dst, A_data + m_offsets[static_cast<size_t>(p * MR)] + k_offset, static_cast<size_t>(panel_len) * sizeof(T));
                continue;
            }
            for (int64_t i = 0; i < panel_len; ++i) {
                T val = A_data[m_offsets[static_cast<size_t>(p * MR + i)] + k_offset];
                if constexpr (std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>) {
                    if (conj)
                        val = std::conj(val);
                }
                dst[i] = val;
            }
        }
    }
}

/// @brief Gather-pack B[kc:kc+kc_len, nc:nc+nc_len] into row-major KC*NR panels.
///
/// Supports single-N (fast paths with memcpy) and multi-N (flat-index-to-offset conversion).
template <typename T>
// NOLINTNEXTLINE(readability-identifier-naming)
void pack_B(T *Bp, T const *B_data, PackingPlan const &plan, int64_t kc_start, int64_t kc_len, int64_t nc_start, int64_t nc_len, int NR,
            bool conj = false) {
    LabeledSectionInternal0();
    auto const &n_dims  = plan.n_dims;
    auto const &k_dims  = plan.k_dims_in_b;
    bool const  multi_n = (n_dims.size() > 1);

    int64_t const num_panels  = (nc_len + NR - 1) / NR;
    int64_t const full_panels = nc_len / NR;
    int64_t const tail        = nc_len % NR;

    if (tail > 0) {
        T *last_panel = Bp + full_panels * kc_len * NR;
        std::fill(last_panel, last_panel + kc_len * NR, T{});
    }

    // --- Single-N fast paths (unchanged from before) ---
    if (!multi_n && k_dims.size() == 1) {
        int64_t const n_stride = n_dims[0].tensor_stride;
        int64_t const k_stride = k_dims[0].tensor_stride;

        if (n_stride == 1 && !conj) {
            for (int64_t p = 0; p < full_panels; ++p) {
                int64_t n_base = nc_start + p * NR;
                T      *panel  = Bp + p * kc_len * NR;
                for (int64_t k_local = 0; k_local < kc_len; ++k_local) {
                    T const *src = B_data + (kc_start + k_local) * k_stride + n_base;
                    std::memcpy(panel + k_local * NR, src, static_cast<size_t>(NR) * sizeof(T));
                }
            }
            if (tail > 0) {
                int64_t n_base = nc_start + full_panels * NR;
                T      *panel  = Bp + full_panels * kc_len * NR;
                for (int64_t k_local = 0; k_local < kc_len; ++k_local) {
                    T const *src = B_data + (kc_start + k_local) * k_stride + n_base;
                    std::memcpy(panel + k_local * NR, src, static_cast<size_t>(tail) * sizeof(T));
                }
            }
            return;
        }

        for (int64_t k_local = 0; k_local < kc_len; ++k_local) {
            int64_t const k_offset = (kc_start + k_local) * k_stride;
            for (int64_t p = 0; p < num_panels; ++p) {
                int64_t const panel_len = (p < full_panels) ? static_cast<int64_t>(NR) : tail;
                T            *dst       = Bp + p * kc_len * NR + k_local * NR;
                for (int64_t j = 0; j < panel_len; ++j) {
                    T val = B_data[k_offset + (nc_start + p * NR + j) * n_stride];
                    if constexpr (std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>) {
                        if (conj)
                            val = std::conj(val);
                    }
                    dst[j] = val;
                }
            }
        }
        return;
    }

    // --- General path: precompute K offsets ---
    static thread_local std::vector<int64_t> k_offsets;
    if (k_dims.size() == 1) {
        k_offsets.resize(static_cast<size_t>(kc_len));
        int64_t const k_stride = k_dims[0].tensor_stride;
        for (int64_t k_local = 0; k_local < kc_len; ++k_local) {
            k_offsets[static_cast<size_t>(k_local)] = (kc_start + k_local) * k_stride;
        }
    } else {
        precompute_offsets(kc_start, kc_len, k_dims, k_offsets);
    }

    // --- General path: precompute N offsets ---
    static thread_local std::vector<int64_t> n_offsets;
    if (multi_n) {
        precompute_offsets(nc_start, nc_len, n_dims, n_offsets);
    } else {
        n_offsets.resize(static_cast<size_t>(nc_len));
        int64_t const n_stride = n_dims[0].tensor_stride;
        for (int64_t j = 0; j < nc_len; ++j) {
            n_offsets[static_cast<size_t>(j)] = (nc_start + j) * n_stride;
        }
    }

    // --- Contiguous-run detection (see pack_A) ---
    bool const    n_fast_unit = !conj && n_dims.back().tensor_stride == 1;
    int64_t const n_fast_size = n_dims.back().size;

    // --- Pack with precomputed offsets ---
    for (int64_t k_local = 0; k_local < kc_len; ++k_local) {
        int64_t const k_offset = k_offsets[static_cast<size_t>(k_local)];
        for (int64_t p = 0; p < num_panels; ++p) {
            int64_t const panel_len = (p < full_panels) ? static_cast<int64_t>(NR) : tail;
            T            *dst       = Bp + p * kc_len * NR + k_local * NR;
            if (n_fast_unit && ((nc_start + p * NR) % n_fast_size) + panel_len <= n_fast_size) {
                std::memcpy(dst, B_data + k_offset + n_offsets[static_cast<size_t>(p * NR)], static_cast<size_t>(panel_len) * sizeof(T));
                continue;
            }
            for (int64_t j = 0; j < panel_len; ++j) {
                T val = B_data[k_offset + n_offsets[static_cast<size_t>(p * NR + j)]];
                if constexpr (std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>) {
                    if (conj)
                        val = std::conj(val);
                }
                dst[j] = val;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Flat block packing (vendor-GEMM block strategy)
// ---------------------------------------------------------------------------

/// @brief Pack A[mc:mc+mc_len, kc:kc+kc_len] into a plain column-major
///        mc_len x kc_len matrix (element (i, k) at dst[i + k*mc_len]).
///
/// Feeds the block-GEMM scatter strategy: one vendor GEMM per cache block
/// instead of MRxNR micro-tiles. Like the panel packers, unit-stride runs of
/// the fastest (post-coalescing) M dim are copied with memcpy; everything
/// else falls back to the scalar gather. Conjugation is applied here so the
/// vendor GEMM can run without transpose flags.
template <typename T>
// NOLINTNEXTLINE(readability-identifier-naming)
void pack_A_flat(T *dst, T const *A_data, PackingPlan const &plan, int64_t mc_start, int64_t mc_len, int64_t kc_start, int64_t kc_len,
                 bool conj = false) {
    LabeledSectionInternal0();
    auto const &m_dims = plan.m_dims;
    auto const &k_dims = plan.k_dims_in_a;

    static thread_local std::vector<int64_t> k_offsets, m_offsets;
    precompute_offsets(kc_start, kc_len, k_dims, k_offsets);
    precompute_offsets(mc_start, mc_len, m_dims, m_offsets);

    bool const    m_fast_unit = !conj && m_dims.back().tensor_stride == 1;
    int64_t const m_fast_size = m_dims.back().size;

    for (int64_t k_local = 0; k_local < kc_len; ++k_local) {
        int64_t const k_offset = k_offsets[static_cast<size_t>(k_local)];
        T            *col      = dst + k_local * mc_len;
        if (m_fast_unit) {
            int64_t pos = 0;
            while (pos < mc_len) {
                int64_t const run = std::min(m_fast_size - ((mc_start + pos) % m_fast_size), mc_len - pos);
                std::memcpy(col + pos, A_data + m_offsets[static_cast<size_t>(pos)] + k_offset, static_cast<size_t>(run) * sizeof(T));
                pos += run;
            }
            continue;
        }
        for (int64_t i = 0; i < mc_len; ++i) {
            T val = A_data[m_offsets[static_cast<size_t>(i)] + k_offset];
            if constexpr (std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>) {
                if (conj)
                    val = std::conj(val);
            }
            col[i] = val;
        }
    }
}

/// @brief Pack B[kc:kc+kc_len, nc:nc+nc_len] into a plain k-major
///        kc_len x nc_len matrix (element (k, j) at dst[k*nc_len + j]) -
///        i.e. the column-major nc_len x kc_len matrix a vendor GEMM
///        consumes with transB='T'.
template <typename T>
// NOLINTNEXTLINE(readability-identifier-naming)
void pack_B_flat(T *dst, T const *B_data, PackingPlan const &plan, int64_t kc_start, int64_t kc_len, int64_t nc_start, int64_t nc_len,
                 bool conj = false) {
    LabeledSectionInternal0();
    auto const &n_dims = plan.n_dims;
    auto const &k_dims = plan.k_dims_in_b;

    static thread_local std::vector<int64_t> k_offsets, n_offsets;
    precompute_offsets(kc_start, kc_len, k_dims, k_offsets);
    precompute_offsets(nc_start, nc_len, n_dims, n_offsets);

    bool const    n_fast_unit = !conj && n_dims.back().tensor_stride == 1;
    int64_t const n_fast_size = n_dims.back().size;

    for (int64_t k_local = 0; k_local < kc_len; ++k_local) {
        int64_t const k_offset = k_offsets[static_cast<size_t>(k_local)];
        T            *row      = dst + k_local * nc_len;
        if (n_fast_unit) {
            int64_t pos = 0;
            while (pos < nc_len) {
                int64_t const run = std::min(n_fast_size - ((nc_start + pos) % n_fast_size), nc_len - pos);
                std::memcpy(row + pos, B_data + k_offset + n_offsets[static_cast<size_t>(pos)], static_cast<size_t>(run) * sizeof(T));
                pos += run;
            }
            continue;
        }
        for (int64_t j = 0; j < nc_len; ++j) {
            T val = B_data[k_offset + n_offsets[static_cast<size_t>(j)]];
            if constexpr (std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>) {
                if (conj)
                    val = std::conj(val);
            }
            row[j] = val;
        }
    }
}

// ---------------------------------------------------------------------------
// HPTT-accelerated transpose (cache-blocked, SIMD-optimized)
// ---------------------------------------------------------------------------

EINSUMS_EXPORT void hptt_transpose(int const *perm, int rank, float const *src, size_t const *sizes, float *dst, int num_threads,
                                   bool conj = false);
EINSUMS_EXPORT void hptt_transpose(int const *perm, int rank, double const *src, size_t const *sizes, double *dst, int num_threads,
                                   bool conj = false);
EINSUMS_EXPORT void hptt_transpose(int const *perm, int rank, std::complex<float> const *src, size_t const *sizes, std::complex<float> *dst,
                                   int num_threads, bool conj = false);
EINSUMS_EXPORT void hptt_transpose(int const *perm, int rank, std::complex<double> const *src, size_t const *sizes,
                                   std::complex<double> *dst, int num_threads, bool conj = false);

} // namespace einsums::packed_gemm
