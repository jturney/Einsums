//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/BLAS.hpp>
#include <Einsums/BLAS/ThreadControl.hpp>
#include <Einsums/Concepts/TensorConcepts.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Logging.hpp>
#include <Einsums/PackedGemm/ContractionKey.hpp>
#include <Einsums/PackedGemm/MicroKernel.hpp>
#include <Einsums/PackedGemm/Options.hpp>
#include <Einsums/PackedGemm/Packing.hpp>
#include <Einsums/PackedGemm/Stream.hpp>
#include <Einsums/Profile/Profile.hpp>
#include <Einsums/SIMD/Prefetch.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_set>
#include <vector>

#ifdef _OPENMP
#    include <omp.h>
#endif

EINSUMS_NAMESPACE_BEGIN(packed_gemm)

/// Output elements below which an outer product is left to the generic loop.
/// Measured, not chosen; see the decline site for the data and its caveats.
inline constexpr int64_t kOuterProductFloor = 768;

/// Elements of the supplying operand below which a GEMV-shaped contraction whose axes do not form a
/// matrix is not streamed: where the stream starts to beat the compiled generic loop. Measured, not
/// chosen; see the stream route for the data.
inline constexpr int64_t kStreamMinElems = 4096;

// Thread-local buffers: bound to a local reference where declared, as in Packing.hpp, whose
// "Thread-local buffers" note gives the reason and the OpenMP rule.

// ---------------------------------------------------------------------------
// Compile-time helpers
// ---------------------------------------------------------------------------

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
/// ger has no beta, so a destination prefactor costs a separate pass over C. A k=1 GEMM folds beta
/// in, which wins while C is small, until GEMM's blocking overhead outgrows the extra pass. The
/// measured crossover lies between 65k and 332k elements.
constexpr size_t kOuterGemmMaxElems = 1u << 17;

/// Outer-product destination size below which no BLAS call is worth making: the call's fixed cost
/// dominates, so the direct paths leave these to the small-outer deferral.
constexpr size_t kOuterMinElems = 1u << 12;

/// @brief Collapse axes [@p begin, @p end) of @p t into one (extent, stride).
///
/// Succeeds only when the axes tile memory exactly in axis order, each beginning where the previous
/// one ended (s_next == s * d). The caller walks two operands' flat runs in lockstep, so the order
/// must agree: sorting by stride would accept a reversed run and transpose the result. Extent-1
/// axes are ignored, since their stride is arbitrary. This is the runtime twin of the compile-time
/// `contiguous_positions` check.
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
    // TensorType::Rank is dynamic_rank (-1) for runtime-rank tensors, so it is a real rank only
    // when non-negative.
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

/// @brief Whether @p td still describes @p t, compared in place so a ContractionSite's per-call
/// check allocates nothing.
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
/// Checks exactly what the ContractionKey encodes (topology, operand layout, and the target and
/// link extents), which is the plan cache's soundness contract, without allocating. @p spec_in's
/// derived fields are checked only when the caller filled them; they follow from the raw index
/// lists, which are always compared.
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

/// Name of the kernel route the most recent @ref blis_contraction call on this thread took:
/// "gemm_batch", "flatten_gemm", "flatten_gemm_hptt", "flatten_gemm_hptt_chunked",
/// "flatten_gemm_gather", "single_k_gemm" or "packed". The flatten suffixes say how the operands
/// were made flat: already flat, transposed by HPTT, or gathered.
///
/// Test introspection only, not an API for steering execution. Thread-local; for a batched
/// contraction it names the last slice this thread ran. Exported and defined out of line so the
/// whole process shares one slot.
[[nodiscard]] EINSUMS_EXPORT char const *&last_contraction_route();

/// Which engine the most recent "packed" contraction on this thread ran: "tile", "block_gemm" (one
/// vendor GEMM per cache block, then a scatter), "3m" (the block strategy on three real GEMMs), or
/// "1m" (complex on the real tile kernel). Written only when the route is "packed". Test
/// introspection only.
[[nodiscard]] EINSUMS_EXPORT char const *&last_packed_engine();

/// Threads per team in the most recent packed contraction on this thread: 1 when every thread
/// packs its own B panel, the cores sharing an L3 when they share one (see blis_contraction).
/// Test introspection only, like @ref last_packed_engine.
[[nodiscard]] EINSUMS_EXPORT int &last_team_size();

/// The route pin the most recent route decision on this thread read. Adaptive means the thread
/// regime decided (@ref einsums::blas::vendor_call_is_fenced). Test introspection only.
[[nodiscard]] EINSUMS_EXPORT KernelRoute &last_route_pin();

/// @brief Whether this contraction is to be packed rather than handed to one vendor GEMM.
///
/// A pinned site answers from its pin. Otherwise the thread regime decides: under a node-scoped
/// width the vendor calls are clamped to one thread (@ref einsums::blas::vendor_call_is_fenced),
/// while the packed loops fork from the widened ICV and get all of it.
[[nodiscard]] inline bool prefer_packed_route(ContractionSite const *site) {
    KernelRoute const pin = site != nullptr ? site->route : KernelRoute::Adaptive;
    last_route_pin()      = pin;
    switch (pin) {
    case KernelRoute::Packed:
        return true;
    case KernelRoute::Vendor:
        return false;
    case KernelRoute::Adaptive:
    default:
        return einsums::blas::vendor_call_is_fenced();
    }
}

/// Bytes a C run must cover before the write-back streams it: two cache lines, the peak of a sweep
/// over 1 to 16.
inline constexpr int64_t kStreamRunBytes = 2 * 64;

/// @brief Whether this contraction should be packed with the roles of A and B exchanged, as C^T =
/// B^T A^T.
///
/// The tile kernel holds a tile as MR-tall vectors, so it stores C with vector stores only when
/// consecutive m coordinates are adjacent in C, and the C scatter wants the same direction. When
/// C's unit stride comes through B, no order inside the M group supplies it and every tile is
/// written with MR*NR scalar stores. Exchanging the roles makes the kernel's m direction C's
/// unit-stride one.
///
/// Only when the N group has a unit stride and the M group has none. When neither is contiguous
/// both orders cost a cache line per element, and the exchange still cuts the register tile along
/// the other group, which wastes lanes when that group is short; hence also the floor of two tiles.
/// Only the tile-scatter path benefits: the direct-C branches, the block-GEMM strategy and the 1m
/// path handle the transpose their own way.
inline bool mn_roles_should_swap(PackingPlan const &plan, MicroKernelShape const &shape, bool is_complex) {
    if (plan.c_m_dims.empty() || plan.c_n_dims.empty()) {
        return false;
    }
    bool const scatter = plan.c_m_dims.size() > 1 || plan.c_n_dims.size() > 1 ||
                         (plan.c_m_dims[0].tensor_stride != 1 && plan.c_n_dims[0].tensor_stride != 1);
    return scatter && !shape.block_gemm && !(is_complex && shape.use_1m) && plan.c_n_dims.back().tensor_stride == 1 &&
           plan.c_m_dims.back().tensor_stride != 1 && plan.N_total >= 2 * static_cast<int64_t>(shape.mr);
}

/// @brief How blis_contraction's threads divide the output: @c tn groups along N, each split
///        @c tm ways along M, with N cut into blocks of @c nc_blk columns.
struct ThreadGrid {
    int     tn     = 1;
    int     tm     = 1;
    int64_t nc_blk = 0;
};

/// @brief The thread grid for an M x N contraction with @p threads threads, M groups a whole number
/// of @p m_unit rows, and N blocks no wider than @p nc_cap.
///
/// Splitting M as well as N keeps a contraction with few N columns from idling threads, and packs A
/// once per N block instead of once per thread. The grid minimizes the busiest thread's share of
/// the output and, among grids within 3% of that, the elements packed (M * n_blocks + N * tm per
/// unit of K). N blocks are cut equal rather than at the cap with a runt, which would idle the
/// threads that draw it.
///
/// N blocks are whole multiples of NR and M groups whole multiples of @p m_unit, so every thread
/// owns its buffers and a region of C no other thread touches.
inline ThreadGrid choose_thread_grid(int threads, int64_t M, int64_t N, int64_t m_unit, int NR, int64_t nc_cap) {
    nc_cap                   = std::max<int64_t>(nc_cap, NR);
    int64_t const n_cap      = std::max<int64_t>(1, (N + NR - 1) / NR);
    int64_t const n_mc       = std::max<int64_t>(1, (M + m_unit - 1) / m_unit);
    int64_t const min_blocks = std::max<int64_t>(1, (N + nc_cap - 1) / nc_cap);

    ThreadGrid best;
    double     best_path = 0;
    double     best_pack = 0;
    for (int tn = 1; tn <= threads && tn <= n_cap; ++tn) {
        // Equal N blocks, a whole number of them per N group, none wider than the cap.
        int64_t const per_group = (min_blocks + tn - 1) / tn;
        int64_t const want      = (N + tn * per_group - 1) / (tn * per_group);
        int64_t const nc_blk    = std::min(nc_cap, std::max<int64_t>(NR, ((want + NR - 1) / NR) * NR));
        int64_t const n_blocks  = (N + nc_blk - 1) / nc_blk;
        int const     tm        = static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(threads / tn, n_mc)));
        // The busiest thread: its share of the work items, each an N block by the largest M group.
        int64_t const items   = n_blocks * tm;
        int64_t const per_thr = (items + threads - 1) / threads;
        int64_t const m_group = std::min(M, ((n_mc + tm - 1) / tm) * m_unit);
        double const  path    = static_cast<double>(per_thr) * static_cast<double>(std::min(nc_blk, N)) * static_cast<double>(m_group);
        double const  pack    = static_cast<double>(M) * static_cast<double>(n_blocks) + static_cast<double>(N) * tm;
        if (best_path == 0 || path < best_path / 1.03 || (path <= best_path * 1.03 && pack < best_pack)) {
            best      = {tn, tm, nc_blk};
            best_path = path;
            best_pack = pack;
        }
    }
    return best;
}

/// @brief A spin barrier for the few threads of one team.
///
/// OpenMP has no barrier for a subset of a parallel region's threads. Members meet twice per K
/// block, after packing the shared panel and before overwriting it. The arriving thread's release
/// and the waiters' acquire order the panel's writes before any member reads it.
struct alignas(64) TeamBarrier {
    std::atomic<int> arrived{0};
    std::atomic<int> generation{0};
    int              size = 1;

    void wait() {
        int const gen = generation.load(std::memory_order_acquire);
        if (arrived.fetch_add(1, std::memory_order_acq_rel) + 1 == size) {
            arrived.store(0, std::memory_order_relaxed);
            generation.store(gen + 1, std::memory_order_release);
            return;
        }
        for (int spins = 0; generation.load(std::memory_order_acquire) == gen; ++spins) {
            if (spins < 4096) {
#if defined(__x86_64__) || defined(__i386__)
                __builtin_ia32_pause();
#elif defined(__aarch64__)
                asm volatile("yield");
#endif
            } else {
                std::this_thread::yield();
            }
        }
    }
};

/// @brief What one team shares besides its panel: the barrier its members meet at, and the counter
/// they claim M blocks from.
///
/// Members claim blocks rather than taking fixed shares, because member speeds vary from one K
/// block to the next and fixed shares make the others wait at every barrier. The counter only
/// rises, and every member steps its view of the current K block's start identically, so it never
/// needs a reset.
struct TeamState {
    TeamBarrier barrier;
    alignas(64) std::atomic<int64_t> next_block{0};
};

/// @brief One team member's view of the B panel its team shares (see blis_contraction).
template <typename T>
struct TeamPanel {
    T         *bp;     ///< the team's packed B panel; null for a team of one, which packs its own
    TeamState *state;  ///< the team's barrier and block counter; null for a team of one
    int        member; ///< this thread's index within the team
    int        size;   ///< threads in the team
    int64_t   *base;   ///< where this member's view of the current K block's blocks begins
};

/// @brief Copy @p n elements to @p dst without first fetching its cache lines.
///
/// An ordinary store to a line the core does not own reads that line from memory first, even when
/// every byte is overwritten; for a C written once and never read, that is an extra pass over C.
/// Streaming stores go through the write-combining buffers, and a line assembled there whole is
/// written with no read, so the run must be contiguous and reasonably long.
///
/// @p dst must be vector-aligned and @p n a whole number of lanes (@ref stream_run_ok tests both).
/// A misaligned head is not handled here: its line would keep the fetch, and staging to realign
/// costs a pass of C through L1.
///
/// @warning Streaming stores are weakly ordered. The caller must @ref einsums::simd::stream_fence()
/// before anything reads @p dst.
template <typename T>
void stream_copy(T *dst, T const *src, int64_t n) {
    // Only float and double have a vector register here. @ref stream_run_ok refuses complex at run
    // time, but the template is instantiated for it.
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
        constexpr int64_t L = einsums::simd::Vec<T>::lanes;
        for (int64_t i = 0; i < n; i += L) {
            einsums::simd::stream_store<T>(dst + i, einsums::simd::loadu(src + i));
        }
    } else {
        std::copy(src, src + n, dst);
    }
}

/// @brief Whether a run of @p n elements at @p dst can be streamed whole.
///
/// Both conditions are about the write-combining buffer: a partial line at
/// either end is an ordinary store, which fetches the line and undoes the
/// point of streaming it.
template <typename T>
bool stream_run_ok(T const *dst, int64_t n) {
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
        constexpr int64_t L = einsums::simd::Vec<T>::lanes;
        return (n % L) == 0 && (reinterpret_cast<uintptr_t>(dst) % static_cast<uintptr_t>(L * sizeof(T))) == 0;
    } else {
        return false;
    }
}

/// @brief Print the resolved plan and blocking for one contraction, behind
/// option::PackedGemmDumpPlan. Out of line so @ref blis_contraction carries only the flag test.
inline void dump_packed_plan(PackingPlan const &plan, int64_t M, int64_t N, int64_t K, int MR, int NR, int64_t MC, int64_t NC, int64_t KC,
                             bool a_order, AOrderFlush const &af, bool n_inner, bool compose, bool runs_stream) {
    auto const dims = [](std::vector<DimSpec> const &d) {
        std::string s;
        for (auto const &x : d) {
            s += fmt::format("{}{}/{}", s.empty() ? "" : ",", x.size, x.tensor_stride);
        }
        return s.empty() ? std::string{"-"} : s;
    };
    fmt::print(stderr,
               "[packed plan] M={} N={} K={} MR={} NR={} MC={} NC={} KC={}"
               " m_dims(A)=[{}] c_m_dims(C)=[{}] n_dims(B)=[{}] c_n_dims(C)=[{}]"
               " a_order={} xa={} xc={} n_inner={} compose={} runs_stream={} swap_ab={}\n",
               M, N, K, MR, NR, MC, NC, KC, dims(plan.m_dims), dims(plan.c_m_dims), dims(plan.n_dims), dims(plan.c_n_dims), a_order, af.xa,
               af.xc, n_inner, compose, runs_stream, plan.swap_ab);
    std::fflush(stderr);
}

/// @brief Map one strided M x N x K slice onto a single BLAS gemm.
///
/// Returns false and calls nothing when the strides admit no gemm form, which
/// is what lets a caller probe a candidate slicing without computing anything.
///
/// BLAS gemm(transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc)
/// expects column-major storage: for transA='N', A is lda x K with lda >= M.
/// The operands here are
///   A[m,k]: m * m_stride + k * ksa
///   B[k,n]: k * ksb     + n * n_stride
///   C[m,n]: m * (c_col_major ? 1 : ...) + n * c_n_stride
/// For complex types with conjugation BLAS uses 'C' rather than 'T', and it has
/// no way to conjugate WITHOUT transposing, so those combinations report false
/// and leave the caller to conjugate during packing.
template <typename T>
bool gemm_from_strides(T *c_data, T const *a_data, T const *b_data, T alpha, T beta, int64_t M, int64_t N, int64_t k_len, int64_t m_stride,
                       int64_t n_stride, int64_t ksa, int64_t ksb, bool c_col_major, int64_t c_n_stride, int64_t ldc_col, int64_t ldc_row,
                       bool conj_a, bool conj_b) {
    // NOLINTNEXTLINE(readability-identifier-naming)
    using blas_int            = einsums::blas::int_t;
    constexpr bool is_complex = (get_scalar_type<T>() == ScalarType::Complex64 || get_scalar_type<T>() == ScalarType::Complex128);

    // Clamp a stride-derived leading dimension up to the BLAS minimum. A size-1 axis can collapse
    // its stride below it, and that stride is never used.
    auto ld = [](int64_t stride, int64_t min_rows) { return static_cast<blas_int>(std::max<int64_t>(stride, min_rows)); };

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

    if (c_col_major) {
        // C is column-major (m_stride_c = 1, ldc = n_stride_c)
        if (m_stride == 1) {
            // A col-major in M → transA='N'
            if (!can_dispatch_n(conj_a)) {
                // conj(A) without transpose: can't dispatch
            } else if (ksb == 1) {
                // B col-major in K → transB='N'
                if (can_dispatch_n(conj_b)) {
                    einsums::blas::gemm<T>(trans_flag('N', conj_a), trans_flag('N', conj_b), static_cast<blas_int>(M),
                                           static_cast<blas_int>(N), static_cast<blas_int>(k_len), alpha, a_data, ld(ksa, M), b_data,
                                           ld(n_stride, k_len), beta, c_data, static_cast<blas_int>(ldc_col));
                    dispatched = true;
                }
            } else if (n_stride == 1) {
                // B col-major in N → transB='T'
                einsums::blas::gemm<T>(trans_flag('N', conj_a), trans_flag('T', conj_b), static_cast<blas_int>(M), static_cast<blas_int>(N),
                                       static_cast<blas_int>(k_len), alpha, a_data, ld(ksa, M), b_data, ld(ksb, N), beta, c_data,
                                       static_cast<blas_int>(ldc_col));
                dispatched = true;
            }
        } else if (ksa == 1) {
            // A col-major in K → transA='T'
            if (ksb == 1) {
                // B col-major in K → transB='N'
                if (can_dispatch_n(conj_b)) {
                    einsums::blas::gemm<T>(trans_flag('T', conj_a), trans_flag('N', conj_b), static_cast<blas_int>(M),
                                           static_cast<blas_int>(N), static_cast<blas_int>(k_len), alpha, a_data, ld(m_stride, k_len),
                                           b_data, ld(n_stride, k_len), beta, c_data, static_cast<blas_int>(ldc_col));
                    dispatched = true;
                }
            } else if (n_stride == 1) {
                einsums::blas::gemm<T>(trans_flag('T', conj_a), trans_flag('T', conj_b), static_cast<blas_int>(M), static_cast<blas_int>(N),
                                       static_cast<blas_int>(k_len), alpha, a_data, ld(m_stride, k_len), b_data, ld(ksb, N), beta, c_data,
                                       static_cast<blas_int>(ldc_col));
                dispatched = true;
            }
        }
    } else if (c_n_stride == 1) {
        // C is row-major (n_stride_c = 1, ldc = m_stride_c)
        // Use identity: C^T = (alpha*A*B + beta*C)^T = alpha*B^T*A^T + beta*C^T
        // Note: A and B are swapped in the BLAS call, so conj flags swap too.
        if (n_stride == 1) {
            // B is the BLAS "A" arg → transA_blas='N', conj_b applies
            if (!can_dispatch_n(conj_b)) {
                // conj(B) without transpose: can't dispatch
            } else if (m_stride == 1) {
                // A is the BLAS "B" arg → transB_blas='T', conj_a applies
                einsums::blas::gemm<T>(trans_flag('N', conj_b), trans_flag('T', conj_a), static_cast<blas_int>(N), static_cast<blas_int>(M),
                                       static_cast<blas_int>(k_len), alpha, b_data, ld(ksb, N), a_data, ld(ksa, M), beta, c_data,
                                       static_cast<blas_int>(ldc_row));
                dispatched = true;
            } else if (ksa == 1) {
                // A is the BLAS "B" arg → transB_blas='N', conj_a applies
                if (can_dispatch_n(conj_a)) {
                    einsums::blas::gemm<T>(trans_flag('N', conj_b), trans_flag('N', conj_a), static_cast<blas_int>(N),
                                           static_cast<blas_int>(M), static_cast<blas_int>(k_len), alpha, b_data, ld(ksb, N), a_data,
                                           ld(m_stride, k_len), beta, c_data, static_cast<blas_int>(ldc_row));
                    dispatched = true;
                }
            }
        } else if (ksb == 1) {
            // B is the BLAS "A" arg → transA_blas='T', conj_b applies
            if (m_stride == 1) {
                // A is the BLAS "B" arg → transB_blas='T', conj_a applies
                einsums::blas::gemm<T>(trans_flag('T', conj_b), trans_flag('T', conj_a), static_cast<blas_int>(N), static_cast<blas_int>(M),
                                       static_cast<blas_int>(k_len), alpha, b_data, ld(n_stride, k_len), a_data, ld(ksa, M), beta, c_data,
                                       static_cast<blas_int>(ldc_row));
                dispatched = true;
            } else if (ksa == 1) {
                // A is the BLAS "B" arg → transB_blas='N', conj_a applies
                if (can_dispatch_n(conj_a)) {
                    einsums::blas::gemm<T>(trans_flag('T', conj_b), trans_flag('N', conj_a), static_cast<blas_int>(N),
                                           static_cast<blas_int>(M), static_cast<blas_int>(k_len), alpha, b_data, ld(n_stride, k_len),
                                           a_data, ld(m_stride, k_len), beta, c_data, static_cast<blas_int>(ldc_row));
                    dispatched = true;
                }
            }
        }
    }

    return dispatched;
}

/// @brief Write an A-order C block back to C, transposing its runs out.
///
/// The flat M coordinate was ordered for the packing operand, so block row i holds coordinate i %
/// @p xa of the fastest index and i / @p xa of C's: a run of C is a column of the block at stride
/// @p xa. It is read back a strip at a time, one cache line wide in the fastest coordinate, so each
/// block line is consumed whole and the staged transpose stays in L1. What reaches memory is the
/// same sequential run of C the composed write-back sends.
///
/// Out of line deliberately: inlined into @ref blis_contraction's hottest loop nest, it slowed
/// contractions that never take this path.
///
/// @p Cb is the mc_len x nb_cur block in column-major order, @p c_m_offsets is indexed by
/// block-local flat M and @p c_n_offsets by N block position.
template <typename T>
#if defined(__GNUC__) || defined(__clang__)
__attribute__((noinline))
#endif
void flush_c_block_transposed(T *C_data, T const *Cb, int64_t mc, int64_t mc_len, int64_t nb, int64_t nb_cur,
                              std::vector<int64_t> const &c_m_offsets, std::vector<int64_t> const &c_n_offsets, int64_t xa, int64_t xc,
                              bool store_c, bool may_stream_c, bool &streamed_c) {
    int64_t const rows = mc_len / xa;
    int64_t const c0   = (mc / xa) % xc;

    // One cache line of the fastest coordinate, and as many rows as keep the
    // staged strip and the block lines it reads both inside L1.
    int64_t const lanes = std::max<int64_t>(int64_t{64} / static_cast<int64_t>(sizeof(T)), 1);
    int64_t const aw    = std::min(xa, lanes);
    int64_t const rt    = std::clamp<int64_t>(int64_t{2048} / aw, 1, rows);

    static thread_local std::vector<T> tls_flush_slot;
    auto                              &tls_flush = bind_thread_local(tls_flush_slot);
    tls_flush.resize(static_cast<size_t>(aw) * static_cast<size_t>(rt));
    T *buf = tls_flush.data();

    for (int64_t j = 0; j < nb_cur; ++j) {
        T const      *src_j = Cb + j * mc_len;
        int64_t const n_off = c_n_offsets[static_cast<size_t>(nb + j)];
        for (int64_t r0 = 0; r0 < rows; r0 += rt) {
            int64_t const rt_cur = std::min(rt, rows - r0);
            for (int64_t a0 = 0; a0 < xa; a0 += aw) {
                int64_t const aw_cur = std::min(aw, xa - a0);
                // Read the block along the strip: both orders use each line whole, but only this
                // one's loads are contiguous and vectorise. The transposing stores land in the
                // staging strip, in L1.
                for (int64_t r = 0; r < rt_cur; ++r) {
                    T const *s = src_j + (r0 + r) * xa + a0;
                    for (int64_t a = 0; a < aw_cur; ++a) {
                        buf[a * rt_cur + r] = s[a];
                    }
                }
                for (int64_t a = 0; a < aw_cur; ++a) {
                    T const *s = buf + a * rt_cur;
                    int64_t  r = 0;
                    while (r < rt_cur) {
                        // C's index wraps at xc, where the next row belongs to a
                        // different outer coordinate and the run ends.
                        int64_t const run = std::min(xc - ((c0 + r0 + r) % xc), rt_cur - r);
                        T            *dst = C_data + c_m_offsets[static_cast<size_t>((r0 + r) * xa + a0 + a)] + n_off;
                        if (may_stream_c && store_c && run * static_cast<int64_t>(sizeof(T)) >= kStreamRunBytes &&
                            stream_run_ok(dst, run)) {
                            stream_copy(dst, s + r, run);
                            streamed_c = true;
                        } else if (store_c) {
                            std::copy(s + r, s + r + run, dst);
                        } else {
                            for (int64_t q = 0; q < run; ++q) {
                                dst[q] += s[r + q];
                            }
                        }
                        r += run;
                    }
                }
            }
        }
    }
}

/// @brief Execute a tensor contraction with the packed engine: the vendor fast paths when the plan
/// maps onto one GEMM or a batch of them, and otherwise BLIS-style packed loops over the resolved
/// SIMD rung's tile kernel.
template <typename ValueType, einsums::BasicTensorConcept CType, einsums::BasicTensorConcept AType, einsums::BasicTensorConcept BType>
void blis_contraction(PackingPlan const &plan, CType &C, AType const &A, BType const &B, ValueType alpha, ValueType beta,
                      bool conj_a = false, bool conj_b = false, bool prefer_packed = false) {
    LabeledSection0();

    // Resolve the rung's tile kernel and its register-block shape once per contraction, so the
    // panels are packed in the geometry that kernel expects and rung resolution stays out of the
    // hot loop.
    MicroKernelFn<ValueType> const micro_tile = micro_kernel_entry<ValueType>();
    MicroKernelShape const         shape      = micro_kernel_shape<ValueType>();
    int const                      MR         = shape.mr;
    int const                      NR         = shape.nr;

    // `prefer_packed` keeps the whole contraction in the packed loops rather than handing it to one
    // vendor GEMM; the caller resolved it (@ref prefer_packed_route). The gemm_batch path is
    // exempt: its vendor entry point is einsums' own OpenMP loop over serial GEMMs, which forks
    // from the ICV like the packed loops. The answer is not stored in the plan, because a route
    // belongs to the node's ContractionSite, not to a packing plan.

    int64_t const M = plan.M_total;
    int64_t const N = plan.N_total;
    int64_t const K = plan.K_total;

    // Cache-aware blocking from the resolved kernel's tile and this contraction's extents: whether
    // C survives a sweep decides how large KC should be.
    auto const    blk        = compute_blocking(static_cast<int64_t>(sizeof(ValueType)), MR, NR, M, N, K);
    bool const    multi_m    = (plan.c_m_dims.size() > 1);
    bool const    multi_n    = (plan.c_n_dims.size() > 1);
    int64_t const C_m_stride = plan.c_m_dims[0].tensor_stride;
    int64_t const C_n_stride = plan.c_n_dims[0].tensor_stride;

    // For multi-M/N, col_major detection uses the first C_m dim stride; the flat-to-offset
    // conversion handles the rest.
    bool const C_col_major = (!multi_m && C_m_stride == 1);

    // Scatter is needed for multi-M/N outputs and for single-M/N layouts where neither output dim
    // is unit-stride (batched C with a stride-1 batch index, strided views, synthetic unit dims).
    bool const scatter_c = multi_m || multi_n || (C_m_stride != 1 && C_n_stride != 1);

    // BLAS requires ldc to be at least the stored result's row count. A transposed or size-1 output
    // axis can collapse the natural stride below it (e.g. "nm <- mkq ; kqn" with n=1), so clamp up;
    // the clamped stride is never used to index. Every gemm call below takes ldc from these.
    int64_t const ldc_col = std::max<int64_t>(C_n_stride, M);
    int64_t const ldc_row = std::max<int64_t>(C_m_stride, N);

    constexpr bool is_complex =
        (get_scalar_type<ValueType>() == ScalarType::Complex64 || get_scalar_type<ValueType>() == ScalarType::Complex128);

    // ------------------------------------------------------------------------- Batch loop: iterate
    // over all batch slices. A non-batched contraction is one iteration with zero offsets.
    // -------------------------------------------------------------------------
    auto const  &batch_dims = plan.batch_dims;
    size_t const nb         = batch_dims.size();

    // ------------------------------------------------------------------------- Batch GEMM fast
    // path: single-K, single-M, single-N with compatible strides becomes one gemm_batch() call. It
    // is kept under a node width, unlike the single-GEMM deferrals below, because gemm_batch is
    // einsums' own OpenMP loop over the batch and so consumes the width.
    // -------------------------------------------------------------------------
    //
    // `!plan.swap_ab` is a guard, not a policy: this path reads A and B directly rather than
    // through the role-resolved pointers below. A swap always implies a multi-dim group today, so
    // it never fires, but it keeps a wider swap from silently breaking this path.
    if (plan.batch_total > 1 && plan.k_dims_in_a.size() == 1 && !multi_m && !multi_n && !plan.synthetic && !plan.swap_ab) {
        // NOLINTNEXTLINE(readability-identifier-naming)
        using blas_int = einsums::blas::int_t;

        int64_t const m_stride   = plan.m_dims[0].tensor_stride;
        int64_t const n_stride   = plan.n_dims[0].tensor_stride;
        int64_t const k_stride_a = plan.k_dims_in_a[0].tensor_stride;
        int64_t const k_stride_b = plan.k_dims_in_b[0].tensor_stride;

        // Same stride test as the single-K fast path inside the batch loop.
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
            last_contraction_route() = "gemm_batch";

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

            // BLAS validates each leading dimension against the stored operand's row count. A
            // size-1 axis can collapse its stride below that (K=1 makes k_stride_a 1 < M), so clamp
            // up; that stride is never used to index.
            lda_val = std::max<blas_int>(lda_val, (transA == 'N') ? static_cast<blas_int>(M) : static_cast<blas_int>(K));
            ldb_val = std::max<blas_int>(ldb_val, (transB == 'N') ? static_cast<blas_int>(K) : static_cast<blas_int>(N));
            ldc_val = std::max<blas_int>(ldc_val, static_cast<blas_int>(M));

            einsums::blas::gemm_batch<ValueType>(transA, transB, static_cast<blas_int>(M), static_cast<blas_int>(N),
                                                 static_cast<blas_int>(K), alpha, a_ptrs.data(), lda_val, b_ptrs.data(), ldb_val, beta,
                                                 c_ptrs.data(), ldc_val, static_cast<blas_int>(bt));
            return;
        }
    }

    // ------------------------------------------------------------------------- Per-batch loop
    // (fallback when gemm_batch can't be used)
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

        // With `plan.swap_ab` the plan describes C^T = B^T A^T, so the plan's A role reads B and
        // vice versa. The batch strides and conjugation flags were already swapped to match.
        ValueType       *C_data = C.data() + c_batch_off;
        ValueType const *A_data = (plan.swap_ab ? static_cast<ValueType const *>(B.data()) : A.data()) + a_batch_off;
        ValueType const *B_data = (plan.swap_ab ? static_cast<ValueType const *>(A.data()) : B.data()) + b_batch_off;

        // ------------------------------------------------------------------------- Multi-K fast
        // path: flatten A and B into contiguous M*K / K*N buffers, then call BLAS GEMM directly.
        // ------------------------------------------------------------------------- Writes C
        // directly, so only stride-1 column- or row-major outputs; a scattered C goes to the packed
        // paths below. Every exit is a vendor GEMM, which a fenced width runs on one thread, so
        // under prefer_packed the packed loops take the contraction instead.
        if (plan.k_dims_in_a.size() > 1 && !scatter_c && !plan.synthetic && !prefer_packed) {
            // Multi-K fast path: only for single-M, single-N (can map to flat BLAS GEMM).
            LabeledSection("flatten + GEMM");
            last_contraction_route() = "flatten_gemm";
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

            // ---- Peel the outer K dims into a loop of ordinary GEMMs ----
            //
            // Holding every K dim but one fixed leaves an ordinary strided GEMM, so a multi-K
            // contraction can be a loop of GEMMs accumulating into C with no copy at all. That
            // beats the flatten when one operand would otherwise be transposed whole, as on
            // ab-acd-dbc, whose B has a K index at unit stride; C stays cache-resident across the
            // slices. Only when a copy would otherwise happen: the both zero-copy case above is
            // already one GEMM.
            {
                // Candidates, largest extent first, so each GEMM is as big as the shape allows.
                std::vector<size_t> order(nk);
                std::iota(order.begin(), order.end(), size_t{0});
                std::stable_sort(order.begin(), order.end(), [&](size_t x, size_t y) { return k_dims_a[x].size > k_dims_a[y].size; });

                bool sliced = false;
                for (size_t ki : order) {
                    int64_t const k_len = k_dims_a[ki].size;
                    int64_t const ksa   = k_dims_a[ki].tensor_stride;
                    int64_t const ksb   = k_dims_b[ki].tensor_stride;
                    int64_t       outer = 1;
                    for (size_t d = 0; d < nk; ++d) {
                        if (d != ki) {
                            outer *= k_dims_a[d].size;
                        }
                    }
                    // A slice has to carry enough arithmetic to be worth a vendor call and its
                    // internal re-packing.
                    if (k_len < 2 || outer < 2 || M * N * k_len < (int64_t{1} << 20)) {
                        continue;
                    }

                    // The first slice both tests the strides and does its share of the work:
                    // gemm_from_strides calls nothing when it cannot map them, so a failed
                    // candidate leaves C untouched.
                    if (!gemm_from_strides<ValueType>(C_data, A_data, B_data, alpha, beta, M, N, k_len, m_stride, n_stride, ksa, ksb,
                                                      C_col_major, C_n_stride, ldc_col, ldc_row, conj_a, conj_b)) {
                        continue;
                    }

                    // The rest accumulate. The strides are the same for every slice, so none can
                    // fail to map.
                    std::vector<int64_t> coord(nk, 0);
                    for (int64_t t = 1; t < outer; ++t) {
                        int64_t off_a = 0, off_b = 0;
                        for (size_t d = nk; d-- > 0;) {
                            if (d == ki) {
                                continue;
                            }
                            if (++coord[d] < k_dims_a[d].size) {
                                break;
                            }
                            coord[d] = 0;
                        }
                        for (size_t d = 0; d < nk; ++d) {
                            if (d != ki) {
                                off_a += coord[d] * k_dims_a[d].tensor_stride;
                                off_b += coord[d] * k_dims_b[d].tensor_stride;
                            }
                        }
                        gemm_from_strides<ValueType>(C_data, A_data + off_a, B_data + off_b, alpha, ValueType{1}, M, N, k_len, m_stride,
                                                     n_stride, ksa, ksb, C_col_major, C_n_stride, ldc_col, ldc_row, conj_a, conj_b);
                    }
                    last_contraction_route() = "gemm_k_loop";
                    sliced                   = true;
                    break;
                }
                if (sliced) {
                    continue; // next batch slice
                }
            }

            // At least one side needs copying: HPTT-transpose it into a flat M*K / K*N buffer and
            // run KC-tiled GEMM over that, or, when HPTT cannot describe the operand, gather it a
            // KC tile at a time.

            // Sized once the route is known, below: the HPTT branch needs whole M*K / K*N buffers,
            // while the gather refills one KC tile at a time and never reads past M*KC / KC*N.
            // These buffers only grow, so sizing the gather for the worst case would pin K/KC times
            // the memory it can use.
            static thread_local std::vector<ValueType> tls_A_flat_slot, tls_B_flat_slot;
            auto                                      &tls_A_flat = bind_thread_local(tls_A_flat_slot);
            auto                                      &tls_B_flat = bind_thread_local(tls_B_flat_slot);
            ValueType                                 *A_flat     = nullptr;
            ValueType                                 *B_flat     = nullptr;

            // Ranks read at run time, so compile-time-rank and runtime-rank operands both work.
            auto rank_of = [](auto const &t) -> int {
                using TT = std::remove_cvref_t<decltype(t)>;
                // TT::Rank is dynamic_rank (-1) for runtime-rank tensors; trust it only when
                // non-negative.
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

            // Describe a dense operand to HPTT.
            //
            // HPTT takes no strides: sizes[0] is the fastest axis and the rest follow as a dense
            // product chain. Any dense tensor fits once its axes are relabelled into
            // ascending-stride order, so sort them, describe the operand in that order, and
            // renumber the permutation through the same relabelling. That is what lets row-major
            // operands take the HPTT route. Extent-1 axes carry no layout and are dropped from both
            // sides.
            //
            // Returns false, leaving the caller on the gather, when the operand is dense in no axis
            // order (a padded, strided or broadcast view). @p ord_out comes back as the
            // layout-carrying axes in HPTT's numbering, which a caller needs to read a sub-block
            // along one of them.
            auto describe_for_hptt = [](auto const &tensor, int rank, std::vector<int> const &out_order, std::vector<size_t> &sizes,
                                        std::vector<int> &perm, std::vector<int> &ord_out) -> bool {
                auto extent = [&](int i) { return static_cast<int64_t>(tensor.dim(static_cast<size_t>(i))); };
                auto stride = [&](int i) { return static_cast<int64_t>(tensor.stride(static_cast<size_t>(i))); };

                // Axes that carry layout, in ascending-stride order.
                std::vector<int> ord;
                ord.reserve(static_cast<size_t>(rank));
                for (int i = 0; i < rank; ++i) {
                    if (extent(i) > 1) {
                        ord.push_back(i);
                    }
                }
                std::sort(ord.begin(), ord.end(), [&](int a, int b) { return stride(a) < stride(b); });
                if (ord.empty()) {
                    return false; // a single element: not worth a plan
                }

                // Dense product chain in that order, or HPTT cannot say it.
                int64_t expected = 1;
                for (int const p : ord) {
                    if (stride(p) != expected) {
                        return false;
                    }
                    expected *= extent(p);
                }

                std::vector<int> hptt_pos(static_cast<size_t>(rank), -1);
                sizes.resize(ord.size());
                for (size_t p = 0; p < ord.size(); ++p) {
                    hptt_pos[static_cast<size_t>(ord[p])] = static_cast<int>(p);
                    sizes[p]                              = static_cast<size_t>(extent(ord[p]));
                }

                // perm[j] = the source axis, in HPTT's numbering, that becomes destination axis j.
                perm.clear();
                perm.reserve(ord.size());
                for (int const pos : out_order) {
                    if (pos < 0 || pos >= rank) {
                        return false;
                    }
                    if (int const h = hptt_pos[static_cast<size_t>(pos)]; h >= 0) {
                        perm.push_back(h);
                    }
                }
                // Anything other than a permutation of the layout-carrying axes means the plan's
                // dims do not account for this operand.
                ord_out = ord;
                return perm.size() == ord.size();
            };

            // Batched contractions (nb > 0) must not use the HPTT flatten: it describes the
            // operand's full rank, batch dims included, while the flat buffers hold one slice and
            // A_data/B_data already point at it, so the transpose would overflow them. The gather
            // below honours the slice offset.
            //
            // A per-slice HPTT path does not pay either: the gather copies one KC tile just before
            // the GEMM consumes it, so the copy stays in cache, where a whole-slice transpose
            // round-trips the buffer through DRAM. The BatchedMultiK tests pin the layouts that
            // comparison covered.
            bool use_hptt = (nb == 0) && !plan.coalesced;

            std::vector<int>    perm_a, perm_b, ord_a, ord_b;
            std::vector<size_t> sizes_a, sizes_b;

            // Destination axis order: A_flat is col-major M x K and B_flat col-major N x K, with
            // the K axes in reverse plan order so the flat K index matches k_cum.
            if (use_hptt && !a_zero_copy) {
                std::vector<int> out_a;
                out_a.reserve(nk + 1);
                out_a.push_back(static_cast<int>(plan.m_dims[0].tensor_pos));
                for (size_t i = 0; i < nk; ++i) {
                    out_a.push_back(static_cast<int>(k_dims_a[nk - 1 - i].tensor_pos));
                }
                use_hptt = describe_for_hptt(A, rank_a_rt, out_a, sizes_a, perm_a, ord_a);
            }
            if (use_hptt && !b_zero_copy) {
                std::vector<int> out_b;
                out_b.reserve(nk + 1);
                out_b.push_back(static_cast<int>(plan.n_dims[0].tensor_pos));
                for (size_t i = 0; i < nk; ++i) {
                    out_b.push_back(static_cast<int>(k_dims_b[nk - 1 - i].tensor_pos));
                }
                use_hptt = describe_for_hptt(B, rank_b_rt, out_b, sizes_b, perm_b, ord_b);
            }

            // How much K to hold at once.
            //
            // By default the flatten transposes whole operands so one GEMM spans all of K. A chunk
            // of K is a sub-block of each operand, which HPTT can read (@ref hptt_transpose), so
            // option::PackedGemmFlattenBudget can cap the buffers and turn the contraction into a
            // short chain of large GEMMs. It is off by default because chunking is not free; the
            // option's documentation says what it costs.
            //
            // Chunks run along the outermost plan K dim, where the flat K index varies slowest, so
            // a chunk is a contiguous range of flat K, as both the destination layout and a
            // zero-copy operand's offset assume.
            int64_t const flat_budget_bytes = config::get(option::PackedGemmFlattenBudget) << 20;
            int64_t const elem_bytes        = static_cast<int64_t>(sizeof(ValueType));
            int64_t const per_k             = (a_zero_copy ? 0 : M) + (b_zero_copy ? 0 : N);
            int64_t const k_outer_extent    = k_dims_a[0].size;
            int64_t const k_inner           = K / k_outer_extent;

            int64_t k_len = K;
            if (use_hptt && flat_budget_bytes > 0 && per_k > 0 && k_outer_extent > 1) {
                int64_t const by_budget = flat_budget_bytes / (elem_bytes * per_k * k_inner);
                k_len                   = std::clamp<int64_t>(by_budget, 1, k_outer_extent) * k_inner;
            }

            if (!a_zero_copy) {
                tls_A_flat.resize(static_cast<size_t>(M) * static_cast<size_t>(use_hptt ? k_len : std::min(K, blk.KC)));
                A_flat = tls_A_flat.data();
            }
            if (!b_zero_copy) {
                tls_B_flat.resize(static_cast<size_t>(use_hptt ? k_len : std::min(K, blk.KC)) * static_cast<size_t>(N));
                B_flat = tls_B_flat.data();
            }

            if (use_hptt) {
                last_contraction_route() = k_len == K ? "flatten_gemm_hptt" : "flatten_gemm_hptt_chunked";
                int num_threads          = 1;
#ifdef _OPENMP
                num_threads = omp_get_max_threads();
#endif
                // HPTT reads the chunk's extent along the outermost K dim, inside the whole
                // enclosing extents (the outer sizes). A chunked axis that is not the operand's
                // slowest leaves gaps between rows, and the outer sizes are how those are
                // expressed.
                std::vector<size_t> chunk_a = sizes_a, chunk_b = sizes_b;
                int                 axpos_a = -1, axpos_b = -1;
                int64_t             stride_a = 0, stride_b = 0;
                if (k_len != K) {
                    auto locate = [](std::vector<int> const &ord, size_t tensor_pos) {
                        for (size_t i = 0; i < ord.size(); ++i) {
                            if (ord[i] == static_cast<int>(tensor_pos)) {
                                return static_cast<int>(i);
                            }
                        }
                        return -1;
                    };
                    if (!a_zero_copy) {
                        axpos_a  = locate(ord_a, k_dims_a[0].tensor_pos);
                        stride_a = static_cast<int64_t>(A.stride(k_dims_a[0].tensor_pos));
                    }
                    if (!b_zero_copy) {
                        axpos_b  = locate(ord_b, k_dims_b[0].tensor_pos);
                        stride_b = static_cast<int64_t>(B.stride(k_dims_b[0].tensor_pos));
                    }
                    // An operand that does not carry the chunked axis (extent 1, dropped from the
                    // description) cannot be read in pieces.
                    if ((!a_zero_copy && axpos_a < 0) || (!b_zero_copy && axpos_b < 0)) {
                        k_len = K;
                        if (!a_zero_copy) {
                            tls_A_flat.resize(static_cast<size_t>(M) * static_cast<size_t>(K));
                            A_flat = tls_A_flat.data();
                        }
                        if (!b_zero_copy) {
                            tls_B_flat.resize(static_cast<size_t>(K) * static_cast<size_t>(N));
                            B_flat = tls_B_flat.data();
                        }
                        last_contraction_route() = "flatten_gemm_hptt";
                    }
                }

                for (int64_t kc = 0; kc < K; kc += k_len) {
                    int64_t const   kc_len = std::min(k_len, K - kc);
                    ValueType const beta_k = (kc == 0) ? beta : ValueType{1};
                    int64_t const   k_out  = kc / k_inner; // coordinate of the outermost K dim

                    if (!a_zero_copy) {
                        if (axpos_a >= 0) {
                            chunk_a[static_cast<size_t>(axpos_a)] = static_cast<size_t>(kc_len / k_inner);
                        }
                        hptt_transpose(perm_a.data(), static_cast<int>(chunk_a.size()), A_data + k_out * stride_a, chunk_a.data(),
                                       kc_len == K ? nullptr : sizes_a.data(), A_flat, num_threads, conj_a);
                    }
                    if (!b_zero_copy) {
                        if (axpos_b >= 0) {
                            chunk_b[static_cast<size_t>(axpos_b)] = static_cast<size_t>(kc_len / k_inner);
                        }
                        hptt_transpose(perm_b.data(), static_cast<int>(chunk_b.size()), B_data + k_out * stride_b, chunk_b.data(),
                                       kc_len == K ? nullptr : sizes_b.data(), B_flat, num_threads, conj_b);
                    }

                    // A_flat is col-major M x kc_len and B_flat row-major kc_len x N. A zero-copy
                    // side is already that over all of K, so it is indexed at the chunk's offset.
                    ValueType const *A_base = a_zero_copy ? A_data + kc * M : A_flat;
                    ValueType const *B_base = b_zero_copy ? B_data + kc * N : B_flat;

                    if (C_col_major) {
                        einsums::blas::gemm<ValueType>('N', 'T', static_cast<blas_int>(M), static_cast<blas_int>(N),
                                                       static_cast<blas_int>(kc_len), alpha, A_base, static_cast<blas_int>(M), B_base,
                                                       static_cast<blas_int>(N), beta_k, C_data, static_cast<blas_int>(ldc_col));
                    } else {
                        einsums::blas::gemm<ValueType>('N', 'T', static_cast<blas_int>(N), static_cast<blas_int>(M),
                                                       static_cast<blas_int>(kc_len), alpha, B_base, static_cast<blas_int>(N), A_base,
                                                       static_cast<blas_int>(M), beta_k, C_data, static_cast<blas_int>(ldc_row));
                    }
                }
            } else {
                // Scalar gather fallback (non-contiguous tensors).
                last_contraction_route() = "flatten_gemm_gather";
                int64_t const KC         = std::min(K, blk.KC);
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

            // Reclaim excess thread-local buffer memory so contractions of varying sizes do not
            // bloat it.
            auto shrink = [](auto &v) {
                if (v.capacity() > 2 * v.size() && v.capacity() > 4096) {
                    v.shrink_to_fit();
                }
            };
            shrink(tls_A_flat);
            shrink(tls_B_flat);

            continue; // next batch slice
        }

        // ------------------------------------------------------------------------- Single-K fast
        // path: a single K dimension is a standard strided GEMM, which BLAS handles through
        // lda/ldb/ldc without packing. Skipped under prefer_packed, since this is the whole
        // contraction in one vendor call.
        // -------------------------------------------------------------------------
        if (plan.k_dims_in_a.size() == 1 && !multi_m && !multi_n && !plan.synthetic && !prefer_packed) {
            // Single-K fast path: only for single-M, single-N (direct BLAS GEMM dispatch).
            // NOLINTNEXTLINE(readability-identifier-naming)
            using blas_int = einsums::blas::int_t;

            int64_t const m_stride   = plan.m_dims[0].tensor_stride;
            int64_t const n_stride   = plan.n_dims[0].tensor_stride;
            int64_t const k_stride_a = plan.k_dims_in_a[0].tensor_stride;
            int64_t const k_stride_b = plan.k_dims_in_b[0].tensor_stride;

            bool const dispatched =
                gemm_from_strides<ValueType>(C_data, A_data, B_data, alpha, beta, M, N, K, m_stride, n_stride, k_stride_a, k_stride_b,
                                             C_col_major, C_n_stride, ldc_col, ldc_row, conj_a, conj_b);

            if (dispatched) {
                last_contraction_route() = "single_k_gemm";
                continue; // next batch slice
            }
        }

        // ------------------------------------------------------------------------- Packed loops:
        // BLIS-style packing around the rung's tile kernel, or one vendor GEMM per cache block for
        // the block strategy.
        // -------------------------------------------------------------------------
        last_contraction_route() = "packed";
        // NOLINTNEXTLINE(readability-identifier-naming)
        using blas_int = einsums::blas::int_t;

        // K blocking: the rung may deepen the cache-derived KC (SME's ZA accumulators need no C
        // cache blocking), which cuts the beta/scatter passes over C to one per tile. The
        // block-GEMM scatter strategy gets the same deep default, since the vendor GEMM blocks K
        // internally. The M block shrinks to keep the A panel (MC_blk * KC_blk) within ~4 MiB.
        int64_t kc_hint = shape.kc;
        if (kc_hint == 0 && shape.block_gemm && scatter_c) {
            kc_hint = 4096;
        }
        // KC is clamped to K on both branches: the packing buffers are sized from KC_blk, so an
        // unclamped blk.KC would inflate the panels of a small contraction.
        //
        // Teams: the threads that share an L3 share one packed B panel in it, packing it together
        // and each computing its own rows against it. Private panels either overflow the shared L3
        // or, cut narrow enough to fit, re-pack A for every narrow N block. Only the tile path
        // shares; the block-GEMM and 1m strategies keep panels of their own. One thread, a nested
        // region, or a thread count that does not divide into whole teams keeps private panels.
        int n_threads = 1;
        int team_size = 1;
#ifdef _OPENMP
        n_threads = omp_get_max_threads();
        if (n_threads > 1 && !omp_in_parallel() && !parallel_batch && !(scatter_c && shape.block_gemm) && !(is_complex && shape.use_1m) &&
            2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K) >=
                static_cast<double>(cpu_config().min_parallel_flops)) {
            int const per_l3 = cpu_config().cores_per_l3;
            if (per_l3 > 1 && n_threads % per_l3 == 0) {
                team_size = per_l3;
            }
        }
#endif
        bool const team_mode = team_size > 1;
        // A quarter of the L3 for the shared panel, not half: the team's A blocks and the C blocks
        // it writes back pass through the same L3.
        int64_t const team_panel_budget = std::max<int64_t>(cpu_config().l3_cache_size / 4, int64_t{256} << 10);

        // A shared panel is KC x NC, and A is re-packed once per N block while C is swept once per
        // K block, so under a fixed panel budget the K block that moves the least is sqrt(2 *
        // budget / element size). It never shrinks below what the cache model gives a single
        // thread.
        int64_t KC_blk = std::min<int64_t>((kc_hint > 0) ? std::max<int64_t>(kc_hint, blk.KC) : blk.KC, K);
        if (team_mode) {
            auto const    es = static_cast<double>(sizeof(ValueType));
            int64_t const kc_team =
                std::max<int64_t>(64, (static_cast<int64_t>(std::sqrt(2.0 * static_cast<double>(team_panel_budget) / es)) / 8) * 8);
            int64_t const kc_base = compute_blocking(static_cast<int64_t>(sizeof(ValueType)), MR, NR).KC;
            KC_blk                = std::min(KC_blk, std::min(K, std::max(kc_team, kc_base)));
        }
        // Bound the A panel at ~4 MiB. The bound is on the panel's size itself; gating it on a
        // comparison of K values let 8 MiB panels through and switched off when the cache model
        // grew.
        int64_t const mc_cap = (int64_t{4} << 20) / (KC_blk * static_cast<int64_t>(sizeof(ValueType)));

        // A scattered C needs a contiguous C block to accumulate into, because its elements are not
        // contiguous in memory.
        bool const needs_c_scatter = scatter_c;
        bool const block_strategy  = needs_c_scatter && shape.block_gemm;
        last_packed_engine()       = (is_complex && shape.use_1m) ? "1m"
                                     : block_strategy             ? ((is_complex && shape.use_3m) ? "3m" : "block_gemm")
                                                                  : "tile";

        // Is this an M group ordered for A, with C's contiguity one coordinate in (see @ref
        // AOrderFlush)? Then pack_A is a memcpy and the C block's write-back transposes C's runs
        // out, which changes what the M block must be a whole number of (xa rows) and which
        // write-back runs. Not const: the M block sizing below turns it off when no streamable run
        // can form.
        AOrderFlush const blk_aorder  = a_order_flush(plan.m_dims, plan.c_m_dims);
        bool              use_a_order = blk_aorder.valid && needs_c_scatter && !block_strategy && !(is_complex && shape.use_1m);

        // Budget for the block-GEMM strategy's MC by NC C temporary; it bounds NC below and sizes
        // the block strategy's M block here. Four times L1, floored at 512 KiB because a 32 KB L1
        // gives a budget too small to pay for the scatter. option::PackedGemmCTempBudget overrides
        // it for sweeps.
        int64_t c_temp_budget = std::max<int64_t>(4 * cpu_config().l1_cache_size, int64_t{512} << 10);
        if (int64_t const kib = config::get(option::PackedGemmCTempBudget); kib > 0) {
            c_temp_budget = kib << 10;
        }

        // The tile loops keep the A panel L2-resident, so blk.MC bounds their MC. The block
        // strategy's A block is consumed by a vendor GEMM that re-packs it, and that GEMM also
        // re-packs the whole B block on every call, so a small MC multiplies B traffic. Its MC
        // comes instead from the A-panel cap and the C temporary: enough rows to use the budget at
        // this N, or a square temporary, whichever is larger.
        int64_t MC_blk = std::clamp((mc_cap / MR) * MR, static_cast<int64_t>(MR), blk.MC);

        // pack_A gathers along A's own contiguous axis when the flat M coordinate was ordered for C
        // (its unit-stride M axis second-fastest, see coalesce_plan): rows i, i + X, i + 2X ... are
        // adjacent in A. Each cache line is read whole only if the block holds a line's worth of
        // those rows per value of the fastest coordinate, so the block is raised to (line / elem) *
        // X rows within the A-panel cap.
        auto const line_rows = [&](std::vector<DimSpec> const &dims) -> int64_t {
            if (dims.size() < 2 || dims.back().tensor_stride == 1 || dims[dims.size() - 2].tensor_stride != 1) {
                return 0;
            }
            return (int64_t{64} / static_cast<int64_t>(sizeof(ValueType))) * dims.back().size;
        };
        if (!block_strategy) {
            // (line / elem) * X is a multiple of MR for the vector tile, and pack_A's fast strip
            // needs the block whole in X, so it is taken exactly when the panel cap allows. Only
            // when packing is a real share of the work (4K >= N): a taller block costs the kernel
            // its L1-resident A panel.
            int64_t const want = line_rows(plan.m_dims);
            if (want > MC_blk && want % MR == 0 && want <= (mc_cap / MR) * MR && 4 * K >= N) {
                MC_blk = want;
            }
        }

        // The A-order block is xa rows of C's contiguous index at a time.
        //
        // It must be a whole number of xa so every flat row carries the same set of C coordinates,
        // which makes the write-back's gather a constant stride, and it wants as much of C's index
        // as it can afford, because that index's run is the write-back's contiguous span. When a
        // whole segment does not fit, a divisor of it keeps every span the same length instead of
        // leaving a short remainder that cannot stream. Capped by the A panel and by the C block at
        // the narrowest N chunk.
        if (use_a_order) {
            int64_t const elem      = static_cast<int64_t>(sizeof(ValueType));
            int64_t const cb_budget = std::max<int64_t>(cpu_config().l2_cache_size / 2, int64_t{64} << 10);
            int64_t const cap_panel = std::max<int64_t>(mc_cap / blk_aorder.xa, 1);
            int64_t const cap_ctemp = std::max<int64_t>(cb_budget / (NR * elem * blk_aorder.xa), 1);
            int64_t const cap_run   = std::max<int64_t>(std::min(cap_panel, cap_ctemp), 1);

            int64_t run = std::min(blk_aorder.xc, cap_run);
            while (run > 1 && blk_aorder.xc % run != 0) {
                --run;
            }
            // A span shorter than a couple of cache lines is worth less than an even division of
            // the segment, so take the length instead.
            if (run * elem < kStreamRunBytes) {
                run = std::min(blk_aorder.xc, cap_run);
            }
            int64_t const want = blk_aorder.xa * run;
            if (run * elem >= kStreamRunBytes && want >= MR && want <= M) {
                MC_blk = want;
            } else {
                // No run worth transposing for fits the caps. The ordering cost model already
                // refuses these, so this is a backstop; keep the ordinary scatter.
                use_a_order = false;
            }
        }
        if (block_strategy) {
            int64_t const elem     = static_cast<int64_t>(sizeof(ValueType));
            int64_t const nc_seen  = std::max<int64_t>(std::min<int64_t>(blk.NC, N), 1);
            int64_t const by_temp  = c_temp_budget / (nc_seen * elem);
            int64_t const balanced = static_cast<int64_t>(std::sqrt(static_cast<double>(c_temp_budget / elem)));
            int64_t const want     = std::min(mc_cap, std::max(by_temp, balanced));
            MC_blk                 = std::max<int64_t>((want / MR) * MR, MR);
        }

        // Keep the M block a whole number of C's fastest segment when the C block flush can compose
        // its destination into contiguous spans. A block that ends mid-segment sends its tail down
        // the fallback walk, which reads the block transposed and scatters. The step is
        // lcm(segment, MR) so the block stays whole in the register tile too; a segment that cannot
        // reach a whole step is left alone.
        //
        // On the team path the A block is first held to twice the L2: members claim M blocks (see
        // TeamState), which balances only if each has several to claim, and the raises can
        // otherwise make MC thousands of rows. The A-ordered write-back sizes its block itself and
        // is exempt. The one raise past that cap is to a whole C segment, whose streamed write-back
        // is worth more than balance only while each thread still has four blocks to claim.
        int64_t const mc_cap_team =
            (team_mode && !use_a_order)
                ? std::max<int64_t>(MR, (((2 * cpu_config().l2_cache_size) / (KC_blk * static_cast<int64_t>(sizeof(ValueType)))) / MR) * MR)
                : std::numeric_limits<int64_t>::max();
        MC_blk = std::min(MC_blk, mc_cap_team);

        // Set when the block is kept a whole number of C segments: its blocks must then also start
        // on a segment, so the thread grid cuts M only at whole blocks.
        bool span_aligned = false;
        if (needs_c_scatter && !block_strategy && plan.c_m_dims.back().tensor_stride == 1) {
            int64_t const fm = plan.c_m_dims.back().size;
            // Whole segments are wanted whenever the write-back intends to stream: composition, or
            // a run long enough on its own. A block shorter than the segment cuts every run
            // partial, nothing streams, and the accumulator is pure overhead.
            if (fm > 1 && (plan.c_n_dims.back().tensor_stride == fm || fm * static_cast<int64_t>(sizeof(ValueType)) >= kStreamRunBytes)) {
                int64_t const step = std::lcm<int64_t, int64_t>(fm, MR);
                span_aligned       = true;
                if (step <= MC_blk) {
                    MC_blk = (MC_blk / step) * step;
                } else if ((step <= mc_cap_team || M >= 4 * step * static_cast<int64_t>(n_threads)) && step <= (mc_cap / MR) * MR &&
                           step <= M) {
                    // The block is smaller than one C segment, so every run it cuts is partial and
                    // the write-back never streams. Raise it to one whole segment. This is
                    // affordable on the small-K shapes where it happens, since the A panel is MC *
                    // KC; the A-panel cap still bounds it.
                    MC_blk = step;
                }
            }
        }

        // beta == 0 says C's prior contents are irrelevant, so the first K block stores its result
        // and later blocks accumulate onto it, as the direct-BLAS paths do through beta_k. A
        // separate `*= 0` pass over a scattered C would touch one element per cache line, and would
        // not honour BLAS's beta == 0 contract either: NaN * 0 is NaN, so an uninitialized C would
        // leak through.
        bool const overwrite_c = (beta == ValueType{0});

        // Whether the C block write-back may stream past the cache: C must be written and never
        // read (overwrite_c), K must fit one block so no later block re-reads lines this one pushed
        // out, and the element type must have a vector register. Run length is decided per span at
        // the write-back.
        bool const may_stream_c = overwrite_c && K <= KC_blk && !is_complex;

        // Which side the C scatter walks innermost: the group holding C's smallest stride, so the
        // inner loop is contiguous when C's unit-stride index came from B rather than A.
        //
        // A synthesized unit dim keeps a stride of 0 (see Packing.cpp), and a group of extent 1
        // carries no locality, so neither argues either way. An A-order M group is the exception:
        // its fastest C stride is large by construction, but its locality is one coordinate in,
        // where the transposing write-back reaches it, so it keeps the inner loop.
        int64_t const c_m_fastest     = plan.c_m_dims.back().tensor_stride;
        int64_t const c_n_fastest     = plan.c_n_dims.back().tensor_stride;
        bool const    scatter_n_inner = !use_a_order && c_n_fastest != 0 && (c_m_fastest == 0 || c_n_fastest < c_m_fastest);

        // Does C's destination compose into whole contiguous spans?
        //
        // When C's fastest m index has unit stride and its fastest n index steps by exactly that
        // index's extent, the two are adjacent halves of one dense run: element (i, j) sits at base
        // + j * Fm + i, so a whole m segment by a whole n segment is Fm * Fn consecutive elements.
        // That holds whenever the two groups hold neighbouring indices of a dense C, which is
        // common. The C block write-back then walks those spans in one sequential sweep, which is
        // what lets it stream past the cache (@ref stream_copy).
        int64_t const blk_m_fast = plan.c_m_dims.back().size;
        int64_t const blk_n_fast = plan.c_n_dims.back().size;
        bool const    blk_compose =
            plan.c_m_dims.back().tensor_stride == 1 && blk_m_fast > 1 && plan.c_n_dims.back().tensor_stride == blk_m_fast;

        // A contraction that does not compose can still be worth the block if its m runs alone are
        // long enough to stream: the write-combining buffers need runs of whole cache lines, not a
        // contiguous rectangle. The partial lines at each end keep their fetch, so a run of one or
        // two lines is not worth the block's L2 round trip.
        bool const blk_runs_stream =
            plan.c_m_dims.back().tensor_stride == 1 && blk_m_fast * static_cast<int64_t>(sizeof(ValueType)) >= kStreamRunBytes;

        // The NC loop is the parallel loop, but only when there is enough work to pay for a
        // fork/join (cpu_config().min_parallel_flops). For a small contraction the region costs
        // orders of magnitude more than the arithmetic it distributes, so a tiled einsum of
        // thousands of such nodes would get slower with more threads.
        double const work_flops    = 2.0 * static_cast<double>(M) * static_cast<double>(N) * static_cast<double>(K);
        bool const   worth_threads = work_flops >= static_cast<double>(cpu_config().min_parallel_flops);
        bool const   parallel_nc   = !parallel_batch && worth_threads;

        // NC starts from the cache model's blk.NC; the thread grid below shrinks it so every thread
        // gets a block, at the price of re-packing A once per extra block.
        int64_t NC_blk = blk.NC;
        if (!block_strategy) {
            // Mirror of the MC raise above for pack_B.
            int64_t const want = line_rows(plan.n_dims);
            if (want > NC_blk && 4 * K >= M) {
                int64_t const nc_cap = (int64_t{4} << 20) / (KC_blk * static_cast<int64_t>(sizeof(ValueType)));
                NC_blk               = std::clamp(((want + NR - 1) / NR) * NR, NC_blk, std::max<int64_t>((nc_cap / NR) * NR, NC_blk));
            }
        }

        // Bound the MC by NC C temporary the block-GEMM strategy allocates. compute_blocking sizes
        // MC and NC from separate panel budgets, so their product is unconstrained and grows with
        // the element size. An oversized temporary evicts the vendor GEMM's own packed buffers; too
        // small a one shrinks NC until A is re-packed many more times. The budget is c_temp_budget
        // above. It lives here rather than in compute_blocking because only this strategy allocates
        // the temporary.
        if (block_strategy) {
            int64_t const max_nc = ((c_temp_budget / (MC_blk * static_cast<int64_t>(sizeof(ValueType)))) / NR) * NR;
            if (max_nc >= NR && max_nc < NC_blk) {
                NC_blk = max_nc;
            }
        }
        // The thread grid (see @ref choose_thread_grid): N cut into equal blocks, a whole number
        // per N group, no wider than the cache model's blk.NC, and each N block's M extent cut into
        // grid.tm groups. M groups start at whole MC blocks when a block's position matters (the
        // A-ordered write-back and the C-segment alignment assume it), and at any MR row otherwise,
        // which lets M be cut evenly where MC blocks do not divide it.
        //
        // Teams share a panel only when each member has four M blocks to claim; below that the
        // imbalance costs more than the shared panel saves, and every thread works alone on N
        // columns of its own.
        bool const    use_teams = team_mode && (M + MC_blk - 1) / MC_blk >= 4 * static_cast<int64_t>(team_size);
        int64_t const m_unit    = (use_a_order || span_aligned) ? MC_blk : static_cast<int64_t>(MR);
        ThreadGrid    grid;
#ifdef _OPENMP
        if (parallel_nc) {
            int const nthreads = omp_get_max_threads();
            if (nthreads > 1 && use_teams) {
                // The grid's units are teams, and a team's N block is its shared panel, sized to
                // the panel budget at this KC. Members split each team's rows among themselves.
                int64_t const panel_nc =
                    std::max<int64_t>(NR, ((team_panel_budget / (KC_blk * static_cast<int64_t>(sizeof(ValueType)))) / NR) * NR);
                grid   = choose_thread_grid(nthreads / team_size, M, N, m_unit * team_size, NR, panel_nc);
                NC_blk = grid.nc_blk;
            } else if (nthreads > 1) {
                grid   = choose_thread_grid(nthreads, M, N, m_unit, NR, blk.NC);
                NC_blk = grid.nc_blk;
            }
        }
#endif
        int64_t const n_nc_blocks = (N + NC_blk - 1) / NC_blk;
        int64_t const n_m_units   = (M + m_unit - 1) / m_unit;
        int64_t const m_groups    = std::min<int64_t>(grid.tm, n_m_units);

        // Size the packing buffers from the blocks actually used, not the cache-derived maxima: a
        // contraction narrower than its block gets a buffer its own size.
        int64_t const mc_panels_max = (std::min(MC_blk, M) + MR - 1) / MR;
        int64_t const nc_panels_max = (std::min(NC_blk, N) + NR - 1) / NR;
        auto const    ap_buf_elems  = static_cast<size_t>(mc_panels_max * MR * KC_blk);
        auto const    bp_buf_elems  = static_cast<size_t>(nc_panels_max * NR * KC_blk);

        // What the plan and the blocking came out as, on request: the strides settle which loop
        // order won, and the blocks settle whether the write-back's runs form. Printed from out of
        // line: this function is hot, and a formatting lambda in its body is not free even when the
        // flag is off.
        if (config::get(option::PackedGemmDumpPlan)) {
            dump_packed_plan(plan, M, N, K, MR, NR, MC_blk, NC_blk, KC_blk, use_a_order, blk_aorder, scatter_n_inner, blk_compose,
                             blk_runs_stream);
        }

        {
            LabeledSection("C++ packing and kernel");
            // The work items are the grid's: N blocks by M groups, each group taken by one team.
            int64_t const n_items    = n_nc_blocks * m_groups;
            auto const    group_rows = [&](int64_t item, int64_t part, int64_t parts, int64_t &m_lo, int64_t &m_hi) {
                int64_t const g  = item % m_groups;
                int64_t const u0 = (g * n_m_units) / m_groups;
                int64_t const u1 = ((g + 1) * n_m_units) / m_groups;
                m_lo             = std::min(M, (u0 + ((u1 - u0) * part) / parts) * m_unit);
                m_hi             = std::min(M, (u0 + ((u1 - u0) * (part + 1)) / parts) * m_unit);
            };

            // Every thread count runs the same loop: teams of team_size consecutive threads, which
            // is one thread per team unless threads sharing an L3 share panels. A team of one packs
            // into the thread-local panel. A larger team's panel lives in a buffer the calling
            // thread owns, one stretch per team, first touched by the members that pack it, on
            // their own node.
            int const    run_threads = parallel_nc ? n_threads : 1;
            int const    members     = (parallel_nc && use_teams) ? team_size : 1;
            int const    n_teams     = run_threads / members;
            size_t const stride      = (bp_buf_elems + 15) & ~size_t{15};
            last_team_size()         = members;
            static thread_local std::vector<ValueType> team_panels;
            std::vector<TeamState>                     states;
            if (members > 1) {
                team_panels.resize(stride * static_cast<size_t>(n_teams));
                states = std::vector<TeamState>(static_cast<size_t>(n_teams));
                for (auto &st : states) {
                    st.barrier.size = members;
                }
            }
            // Taken here, on the calling thread: a thread_local named inside the region below would
            // be each worker's own, empty, instance.
            ValueType *const panels = members > 1 ? team_panels.data() : nullptr;
#ifdef _OPENMP
#    pragma omp parallel num_threads(run_threads) if (run_threads > 1)
#endif
            {
                // Which team this thread is in and which member of it. The runtime can give fewer
                // threads than asked for, and then the teams are not whole: each thread works
                // alone, on a panel of its own.
                int tid = 0;
                int got = 1;
#ifdef _OPENMP
                tid = omp_get_thread_num();
                got = omp_get_num_threads();
#endif
                bool const                 whole   = got == run_threads;
                int const                  team_id = whole ? tid / members : tid;
                int const                  teams   = whole ? n_teams : got;
                int64_t                    base    = 0;
                TeamPanel<ValueType> const ctx =
                    (whole && members > 1) ? TeamPanel<ValueType>{panels + stride * static_cast<size_t>(team_id),
                                                                  &states[static_cast<size_t>(team_id)], tid % members, members, &base}
                                           : TeamPanel<ValueType>{nullptr, nullptr, 0, 1, &base};
                TeamPanel<ValueType> const *const team = &ctx;
                LabeledSectionInternal("team: all items of one thread");

                // One work item: an N block and one M group of it, whose M blocks the team's
                // members claim. Every thread owns the buffer it packs A into and the region of C
                // its blocks cover. The team owns the B panel: a team of one packs it alone into
                // the thread-local panel; a larger team packs it together and meets around it.
                for (int64_t item = team_id; item < n_items; item += teams) {
                    int64_t m_lo = 0, m_hi = 0;
                    group_rows(item, 0, 1, m_lo, m_hi);
                    int64_t const                              nc = (item / m_groups) * NC_blk;
                    static thread_local std::vector<ValueType> tls_Ap_slot, tls_Bp_slot, tls_Ct_slot;
                    auto                                      &tls_Ap     = bind_thread_local(tls_Ap_slot);
                    auto                                      &tls_Bp     = bind_thread_local(tls_Bp_slot);
                    auto                                      &tls_Ct     = bind_thread_local(tls_Ct_slot);
                    bool                                       streamed_c = false;
                    tls_Ap.resize(ap_buf_elems);
                    if (team->bp == nullptr) {
                        tls_Bp.resize(bp_buf_elems);
                    }
                    ValueType    *Ap     = tls_Ap.data();
                    ValueType    *Bp     = team->bp != nullptr ? team->bp : tls_Bp.data();
                    int64_t const nc_len = std::min(NC_blk, N - nc);

                    // Scatter path: precompute the C offset tables, one entry per flat index,
                    // instead of a div/mod chain per element. n-offsets are fixed for the nc block;
                    // m-offsets are refreshed per mc block.
                    static thread_local std::vector<int64_t> c_n_offsets_slot, c_m_offsets_slot;
                    auto                                    &c_n_offsets = bind_thread_local(c_n_offsets_slot);
                    auto                                    &c_m_offsets = bind_thread_local(c_m_offsets_slot);
                    if (needs_c_scatter || (is_complex && shape.use_1m)) {
                        precompute_offsets(nc, nc_len, plan.c_n_dims, c_n_offsets);
                    }

                    // ---- 1m complex strategy (rungs with a real matrix kernel) ---- Complex runs
                    // on the real kernel by Van Zee's 1m method: A packs 1e, B packs 1r, and the
                    // real (2M x N) output is interleaved complex, scattered directly. Working
                    // extents double (Mh = 2M, Kh = 2K); MR/NR are the real kernel's (see
                    // MicroKernelShape::use_1m). Conjugation folds into the packing signs and alpha
                    // applies at the scatter. No operand-sized temporaries.
                    if constexpr (is_complex) {
                        if (shape.use_1m) {
                            using RealT                           = RemoveComplexT<ValueType>;
                            MicroKernelFn<RealT> const micro_real = micro_kernel_entry<RealT>();
                            int64_t const              Mh         = 2 * M;
                            int64_t const              Kh         = 2 * K;
                            // The K and M blocks, in real units. A rung that fixes the K block
                            // (SME, whose ZA tiles hold C across the whole K loop) keeps it and the
                            // M block it was tuned with. A register kernel needs its B micro-panel
                            // (NR * KHC reals) to stay in L1 across the M sweep, so it takes the
                            // cache model's blocking on the real extents. Both blocks must be even
                            // so a block never splits a complex element's (re, im) pair.
                            int64_t KHC = 0;
                            int64_t MHC = 0;
                            if (shape.kc > 0) {
                                KHC = std::min<int64_t>(shape.kc, Kh);
                                MHC = 256;
                            } else {
                                BlockingParams const blk_1m = compute_blocking(static_cast<int64_t>(sizeof(RealT)), MR, NR, Mh, N, Kh);
                                KHC                         = std::min<int64_t>(blk_1m.KC, Kh);
                                MHC                         = blk_1m.MC;
                            }
                            KHC                      = std::max<int64_t>(2, KHC - KHC % 2);
                            MHC                      = std::max<int64_t>(2, MHC - MHC % 2);
                            int64_t const num_ir_max = (MHC + MR - 1) / MR;
                            int64_t const num_jr_max = (nc_len + NR - 1) / NR;

                            static thread_local std::vector<RealT> tls_Ap1_slot, tls_Bp1_slot, tls_Cb1_slot;
                            auto                                  &tls_Ap1 = bind_thread_local(tls_Ap1_slot);
                            auto                                  &tls_Bp1 = bind_thread_local(tls_Bp1_slot);
                            auto                                  &tls_Cb1 = bind_thread_local(tls_Cb1_slot);
                            tls_Ap1.resize(static_cast<size_t>(num_ir_max * MR * KHC));
                            tls_Bp1.resize(static_cast<size_t>(num_jr_max * NR * KHC));
                            tls_Cb1.resize(static_cast<size_t>(MHC) * static_cast<size_t>(nc_len));

                            for (int64_t kh = 0; kh < Kh; kh += KHC) {
                                int64_t const kh_len = std::min(KHC, Kh - kh);
                                pack_B_1m_panels<RealT>(tls_Bp1.data(), B_data, plan, kh, kh_len, nc, nc_len, NR, conj_b);

                                for (int64_t mh = 2 * m_lo; mh < 2 * m_hi; mh += MHC) {
                                    int64_t const mh_len = std::min(MHC, 2 * m_hi - mh);
                                    precompute_offsets(mh / 2, mh_len / 2, plan.c_m_dims, c_m_offsets);

                                    // Beta prescale once per (mh, nc) block on the first kh slice.
                                    // Skipped entirely when the scatter below stores.
                                    bool const store_c = overwrite_c && kh == 0;
                                    if (kh == 0 && beta != ValueType{1} && !overwrite_c) {
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
                                        if (store_c) {
                                            for (int64_t ii = 0; ii < mh_len; ii += 2) {
                                                C_data[c_m_offsets[static_cast<size_t>(ii / 2)] + n_off] =
                                                    alpha * ValueType{src[ii], src[ii + 1]};
                                            }
                                            continue;
                                        }
                                        for (int64_t ii = 0; ii < mh_len; ii += 2) {
                                            C_data[c_m_offsets[static_cast<size_t>(ii / 2)] + n_off] +=
                                                alpha * ValueType{src[ii], src[ii + 1]};
                                        }
                                    }
                                }
                            }
                            continue; // next nc block
                        }
                    }

                    // ---- Block-GEMM scatter strategy ---- One vendor GEMM per (mc, kc) block:
                    // pack A to a column-major mc_len x kc_len matrix and B to k-major kc_len x
                    // nc_len, GEMM into a contiguous C block, then scatter-accumulate through the
                    // offset tables. Vendor GEMMs of this size run at full speed, including on
                    // matrix units the tile kernels cannot reach, while the blocks stay cache-sized
                    // and thread-local.
                    if (needs_c_scatter && shape.block_gemm) {
                        // NOLINTNEXTLINE(readability-identifier-naming)
                        using blas_int = einsums::blas::int_t;
                        static thread_local std::vector<ValueType> tls_Af_slot, tls_Bf_slot, tls_Cb_slot;
                        auto                                      &tls_Af = bind_thread_local(tls_Af_slot);
                        auto                                      &tls_Bf = bind_thread_local(tls_Bf_slot);
                        auto                                      &tls_Cb = bind_thread_local(tls_Cb_slot);
                        bool const                                 use_3m = is_complex && shape.use_3m;
                        if (!use_3m) {
                            tls_Af.resize(static_cast<size_t>(MC_blk * KC_blk));
                            tls_Bf.resize(static_cast<size_t>(nc_len) * static_cast<size_t>(KC_blk));
                            tls_Cb.resize(static_cast<size_t>(MC_blk) * static_cast<size_t>(nc_len));
                        }

                        // 3m buffers: three real splits of A, B, and the block
                        // product, laid out as consecutive segments.
                        using Real3m = RemoveComplexT<ValueType>;
                        static thread_local std::vector<Real3m> tls_A3_slot, tls_B3_slot, tls_T3_slot;
                        auto                                   &tls_A3 = bind_thread_local(tls_A3_slot);
                        auto                                   &tls_B3 = bind_thread_local(tls_B3_slot);
                        auto                                   &tls_T3 = bind_thread_local(tls_T3_slot);
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

                            for (int64_t mc = m_lo; mc < m_hi; mc += MC_blk) {
                                int64_t const mc_len = std::min(MC_blk, m_hi - mc);
                                precompute_offsets(mc, mc_len, plan.c_m_dims, c_m_offsets);

                                // Beta prescale once per (mc, nc) block on the first kc slice.
                                // Skipped entirely when the scatters below store.
                                bool const store_c = overwrite_c && kc == 0;
                                if (kc == 0 && beta != ValueType{1} && !overwrite_c) {
                                    LabeledSectionInternal("C beta prescale");
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
                                        pack_A_3m_flat<Real3m>(tls_A3.data(), tls_A3.data() + aseg, tls_A3.data() + 2 * aseg, A_data, plan,
                                                               mc, mc_len, kc, kc_len, conj_a);
                                        for (int t = 0; t < 3; ++t) {
                                            einsums::blas::gemm<Real3m>('N', 'T', static_cast<blas_int>(mc_len),
                                                                        static_cast<blas_int>(nc_len), static_cast<blas_int>(kc_len),
                                                                        Real3m{1}, tls_A3.data() + t * aseg, static_cast<blas_int>(mc_len),
                                                                        tls_B3.data() + t * bseg, static_cast<blas_int>(nc_len), Real3m{0},
                                                                        tls_T3.data() + t * tseg, static_cast<blas_int>(mc_len));
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
                                                ValueType   *dst = C_data + c_m_offsets[static_cast<size_t>(i2)] + n_off;
                                                if (store_c) {
                                                    *dst = alpha * ValueType{re, im};
                                                } else {
                                                    *dst += alpha * ValueType{re, im};
                                                }
                                            }
                                        }
                                        continue; // next mc block
                                    }
                                }

                                pack_A_flat(tls_Af.data(), A_data, plan, mc, mc_len, kc, kc_len, conj_a);

                                {
                                    LabeledSectionInternal("block GEMM (vendor)");
                                    // Swapping the operands computes Bf * Af^T, so the block
                                    // temporary comes out transposed and the n-inner scatter reads
                                    // it contiguously. Striding the temporary instead would sweep
                                    // an MC by NC buffer once per m index.
                                    if (scatter_n_inner) {
                                        einsums::blas::gemm<ValueType>('N', 'T', static_cast<blas_int>(nc_len),
                                                                       static_cast<blas_int>(mc_len), static_cast<blas_int>(kc_len), alpha,
                                                                       tls_Bf.data(), static_cast<blas_int>(nc_len), tls_Af.data(),
                                                                       static_cast<blas_int>(mc_len), ValueType{0}, tls_Cb.data(),
                                                                       static_cast<blas_int>(nc_len));
                                    } else {
                                        einsums::blas::gemm<ValueType>('N', 'T', static_cast<blas_int>(mc_len),
                                                                       static_cast<blas_int>(nc_len), static_cast<blas_int>(kc_len), alpha,
                                                                       tls_Af.data(), static_cast<blas_int>(mc_len), tls_Bf.data(),
                                                                       static_cast<blas_int>(nc_len), ValueType{0}, tls_Cb.data(),
                                                                       static_cast<blas_int>(mc_len));
                                    }
                                }

                                // Scatter-accumulate the contiguous block into C, in contiguous
                                // runs wherever C's fastest flat coordinate has unit stride.
                                LabeledSectionInternal("C block scatter");
                                if (scatter_n_inner) {
                                    // Mirror of the loop below with m and n exchanged; tls_Cb is
                                    // nc_len x mc_len here.
                                    bool const    c_n_unit = plan.c_n_dims.back().tensor_stride == 1;
                                    int64_t const c_n_fast = plan.c_n_dims.back().size;
                                    for (int64_t i2 = 0; i2 < mc_len; ++i2) {
                                        int64_t const    m_off = c_m_offsets[static_cast<size_t>(i2)];
                                        ValueType const *src   = tls_Cb.data() + i2 * nc_len;
                                        if (c_n_unit) {
                                            int64_t pos = 0;
                                            while (pos < nc_len) {
                                                int64_t const    run = std::min(c_n_fast - ((nc + pos) % c_n_fast), nc_len - pos);
                                                ValueType       *dst = C_data + m_off + c_n_offsets[static_cast<size_t>(pos)];
                                                ValueType const *s   = src + pos;
                                                if (store_c) {
                                                    std::copy(s, s + run, dst);
                                                } else {
                                                    for (int64_t r = 0; r < run; ++r) {
                                                        dst[r] += s[r];
                                                    }
                                                }
                                                pos += run;
                                            }
                                            continue;
                                        }
                                        if (store_c) {
                                            for (int64_t j = 0; j < nc_len; ++j) {
                                                C_data[m_off + c_n_offsets[static_cast<size_t>(j)]] = src[j];
                                            }
                                            continue;
                                        }
                                        for (int64_t j = 0; j < nc_len; ++j) {
                                            C_data[m_off + c_n_offsets[static_cast<size_t>(j)]] += src[j];
                                        }
                                    }
                                    continue; // next mc block
                                }
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
                                            if (store_c) {
                                                std::copy(s, s + run, dst);
                                            } else {
                                                for (int64_t r = 0; r < run; ++r) {
                                                    dst[r] += s[r];
                                                }
                                            }
                                            pos += run;
                                        }
                                        continue;
                                    }
                                    if (store_c) {
                                        for (int64_t i2 = 0; i2 < mc_len; ++i2) {
                                            C_data[c_m_offsets[static_cast<size_t>(i2)] + n_off] = src[i2];
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

                        // A team packs its shared panel together, each member a slice of its NR
                        // panels (pack_B lays panel p at p * kc_len * NR), and meets before any
                        // member reads it.
                        if (team->size > 1) {
                            int64_t const panels = (nc_len + NR - 1) / NR;
                            int64_t const p0     = (panels * team->member) / team->size;
                            int64_t const p1     = (panels * (team->member + 1)) / team->size;
                            if (p1 > p0) {
                                LabeledSectionInternal("team: pack B slice");
                                pack_B(Bp + p0 * kc_len * NR, B_data, plan, kc, kc_len, nc + p0 * NR,
                                       std::min(nc_len - p0 * NR, (p1 - p0) * NR), NR, conj_b);
                            }
                            {
                                LabeledSectionInternal("team: wait for the packed panel");
                                team->state->barrier.wait();
                            }
                            bp_packed = true;
                        }

                        // The M blocks of this item's rows. A team of one walks them in order; a
                        // team's members claim them from the team's counter (see TeamState), so a
                        // member on a faster core takes more.
                        int64_t const n_mc_blk = (m_hi - m_lo + MC_blk - 1) / MC_blk;
                        int64_t const blk_lo   = *team->base;
                        int64_t       walk     = 0;
                        auto const    claim    = [&]() -> int64_t {
                            if (team->size == 1) {
                                return walk < n_mc_blk ? walk++ : -1;
                            }
                            int64_t t = team->state->next_block.load(std::memory_order_relaxed);
                            do {
                                if (t >= blk_lo + n_mc_blk) {
                                    return -1;
                                }
                            } while (!team->state->next_block.compare_exchange_weak(t, t + 1, std::memory_order_relaxed));
                            return t - blk_lo;
                        };
                        for (int64_t blk_i = claim(); blk_i >= 0; blk_i = claim()) {
                            int64_t const mc     = m_lo + blk_i * MC_blk;
                            int64_t const mc_len = std::min(MC_blk, m_hi - mc);

                            if (needs_c_scatter) {
                                precompute_offsets(mc, mc_len, plan.c_m_dims, c_m_offsets);
                            }

                            // Beta prescale, once per (mc, nc) block on the first kc tile. The
                            // scatter branch stores on the first K block when beta == 0 (see
                            // overwrite_c) and needs none. The direct-C branches still do, because
                            // the kernel only accumulates, but for beta == 0 they clear C rather
                            // than scale it.
                            bool const store_c = overwrite_c && kc == 0 && needs_c_scatter;
                            if (kc == 0 && beta != ValueType{1} && !store_c) {
                                LabeledSectionInternal("C beta prescale");
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
                                        if (overwrite_c) {
                                            std::fill(col, col + mc_len, ValueType{0});
                                            continue;
                                        }
                                        for (int64_t i = 0; i < mc_len; ++i) {
                                            col[i] *= beta;
                                        }
                                    }
                                } else {
                                    for (int64_t mi = mc; mi < mc + mc_len; ++mi) {
                                        ValueType *row = C_data + mi * C_m_stride + nc;
                                        if (overwrite_c) {
                                            std::fill(row, row + nc_len, ValueType{0});
                                            continue;
                                        }
                                        for (int64_t j = 0; j < nc_len; ++j) {
                                            row[j] *= beta;
                                        }
                                    }
                                }
                            }

                            // Pack B once per kc (a team already has), then this block's A.
                            if (!bp_packed) {
                                pack_B(Bp, B_data, plan, kc, kc_len, nc, nc_len, NR, conj_b);
                                bp_packed = true;
                            }
                            pack_A(Ap, A_data, plan, mc, mc_len, kc, kc_len, MR, conj_a);

                            int64_t const num_jr = (nc_len + NR - 1) / NR;
                            int64_t const num_ir = (mc_len + MR - 1) / MR;

                            if (needs_c_scatter && !scatter_n_inner && (blk_compose || blk_runs_stream || use_a_order)) {
                                LabeledSectionInternal("micro-kernel loop, C block");
                                // ---- Cache-resident C block ----
                                //
                                // The tiles accumulate into one contiguous mc_len x nc_len block,
                                // written back to C once. The kernel's C operand then has unit row
                                // stride whatever C's layout, which is the whole-vector store path,
                                // and the write-back's runs are as long as C's fastest index allows
                                // rather than capped at MR. Only for the m-inner scatter: @ref
                                // mn_roles_should_swap has already turned every n-inner case with a
                                // contiguous direction into an m-inner one.
                                //
                                // The accumulator is bounded by chunking the N block, not by
                                // shrinking NC: A's DRAM traffic scales with 1/NC, and the packed B
                                // block already covers the whole N block, so a chunk costs no extra
                                // packing. The budget is half the L2, not the block strategy's
                                // c_temp_budget: that budget sizes a vendor GEMM's output, the only
                                // thing competing for L2 there, while this accumulator shares the
                                // L2 with the A panel and B block live across the same loops.
                                int64_t const cb_budget = std::max<int64_t>(cpu_config().l2_cache_size / 2, int64_t{64} << 10);
                                int64_t       nb_len    = cb_budget / (mc_len * static_cast<int64_t>(sizeof(ValueType)));
                                nb_len                  = std::max<int64_t>((nb_len / NR) * NR, NR);
                                nb_len                  = std::min(nb_len, nc_len);

                                for (int64_t nb = 0; nb < nc_len; nb += nb_len) {
                                    int64_t const nb_cur  = std::min(nb_len, nc_len - nb);
                                    int64_t const jr_base = nb / NR;
                                    tls_Ct.assign(static_cast<size_t>(mc_len) * static_cast<size_t>(nb_cur), ValueType{0});
                                    ValueType *Cb = tls_Ct.data();

                                    // Which packed block the tile loops keep resident. The standard
                                    // order streams the A panel and reuses one NR x KC column of B,
                                    // which is right while B's block is the larger. When B's whole
                                    // block fits in half the L1 alongside one A panel (N is small,
                                    // as on the intensli shapes), run the A panel innermost
                                    // instead, so a freshly gathered A panel is consumed while it
                                    // is still in L1 rather than read back through L2 once per N
                                    // tile.
                                    int64_t const num_jr_b = (nb_cur + NR - 1) / NR;
                                    bool const    b_block_resident =
                                        nb_cur * kc_len * static_cast<int64_t>(sizeof(ValueType)) * 2 <= cpu_config().l1_cache_size;
                                    if (b_block_resident) {
                                        for (int64_t ir = 0; ir < num_ir; ++ir) {
                                            int64_t const mr_actual = std::min(static_cast<int64_t>(MR), mc_len - ir * MR);
                                            for (int64_t jr = 0; jr < num_jr_b; ++jr) {
                                                int64_t const nr_actual = std::min(static_cast<int64_t>(NR), nb_cur - jr * NR);
                                                micro_tile(static_cast<int>(MR), static_cast<int>(NR), kc_len, alpha, Ap + ir * MR * kc_len,
                                                           Bp + (jr_base + jr) * NR * kc_len, mr_actual, nr_actual,
                                                           Cb + ir * MR + jr * NR * mc_len, 1, mc_len);
                                            }
                                        }
                                    } else {
                                        for (int64_t jr = 0; jr < num_jr_b; ++jr) {
                                            int64_t const nr_actual = std::min(static_cast<int64_t>(NR), nb_cur - jr * NR);
                                            for (int64_t ir = 0; ir < num_ir; ++ir) {
                                                int64_t const mr_actual = std::min(static_cast<int64_t>(MR), mc_len - ir * MR);
                                                micro_tile(static_cast<int>(MR), static_cast<int>(NR), kc_len, alpha, Ap + ir * MR * kc_len,
                                                           Bp + (jr_base + jr) * NR * kc_len, mr_actual, nr_actual,
                                                           Cb + ir * MR + jr * NR * mc_len, 1, mc_len);
                                            }
                                        }
                                    }

                                    LabeledSectionInternal("C block scatter");

                                    if (use_a_order && (mc % blk_aorder.xa) == 0 && (mc_len % blk_aorder.xa) == 0) {
                                        flush_c_block_transposed<ValueType>(C_data, Cb, mc, mc_len, nb, nb_cur, c_m_offsets, c_n_offsets,
                                                                            blk_aorder.xa, blk_aorder.xc, store_c, may_stream_c,
                                                                            streamed_c);
                                        continue; // next C block chunk
                                    }

                                    if (blk_compose) {
                                        int64_t pos = 0;
                                        while (pos < mc_len) {
                                            int64_t const run_m = std::min(blk_m_fast - ((mc + pos) % blk_m_fast), mc_len - pos);
                                            // A partial m segment breaks the span - its rows
                                            // stop short of the next n step - so those fall
                                            // back to the column walk below.
                                            if (run_m == blk_m_fast) {
                                                int64_t jj = 0;
                                                while (jj < nb_cur) {
                                                    int64_t const run_n = std::min(blk_n_fast - ((nc + nb + jj) % blk_n_fast), nb_cur - jj);
                                                    ValueType    *dst   = C_data + c_m_offsets[static_cast<size_t>(pos)] +
                                                                          c_n_offsets[static_cast<size_t>(nb + jj)];
                                                    int64_t const span  = run_n * blk_m_fast;
                                                    // The span's columns are already adjacent in C,
                                                    // so each streams in place with no staging. A
                                                    // span shorter than a few lines cannot fill a
                                                    // write-combining buffer, and streaming it
                                                    // would pay a partial write and keep the fetch.
                                                    if (may_stream_c && store_c &&
                                                        span * static_cast<int64_t>(sizeof(ValueType)) >= kStreamRunBytes &&
                                                        stream_run_ok(dst, run_m)) {
                                                        for (int64_t q = 0; q < run_n; ++q) {
                                                            stream_copy(dst + q * blk_m_fast, Cb + (jj + q) * mc_len + pos, run_m);
                                                        }
                                                        streamed_c = true;
                                                        jj += run_n;
                                                        continue;
                                                    }
                                                    for (int64_t q = 0; q < run_n; ++q) {
                                                        ValueType const *s = Cb + (jj + q) * mc_len + pos;
                                                        ValueType       *d = dst + q * blk_m_fast;
                                                        if (store_c) {
                                                            std::copy(s, s + run_m, d);
                                                        } else {
                                                            for (int64_t r = 0; r < run_m; ++r) {
                                                                d[r] += s[r];
                                                            }
                                                        }
                                                    }
                                                    jj += run_n;
                                                }
                                                pos += run_m;
                                                continue;
                                            }
                                            for (int64_t j = 0; j < nb_cur; ++j) {
                                                ValueType       *dst = C_data + c_m_offsets[static_cast<size_t>(pos)] +
                                                                       c_n_offsets[static_cast<size_t>(nb + j)];
                                                ValueType const *s   = Cb + j * mc_len + pos;
                                                if (store_c) {
                                                    std::copy(s, s + run_m, dst);
                                                } else {
                                                    for (int64_t r = 0; r < run_m; ++r) {
                                                        dst[r] += s[r];
                                                    }
                                                }
                                            }
                                            pos += run_m;
                                        }
                                        continue; // next C block chunk
                                    }

                                    for (int64_t j = 0; j < nb_cur; ++j) {
                                        int64_t const    n_off = c_n_offsets[static_cast<size_t>(nb + j)];
                                        ValueType const *src   = Cb + j * mc_len;
                                        if (plan.c_m_dims.back().tensor_stride == 1) {
                                            int64_t pos = 0;
                                            while (pos < mc_len) {
                                                int64_t const    run = std::min(blk_m_fast - ((mc + pos) % blk_m_fast), mc_len - pos);
                                                ValueType       *dst = C_data + c_m_offsets[static_cast<size_t>(pos)] + n_off;
                                                ValueType const *s   = src + pos;
                                                if (may_stream_c && store_c &&
                                                    run * static_cast<int64_t>(sizeof(ValueType)) >= kStreamRunBytes &&
                                                    stream_run_ok(dst, run)) {
                                                    stream_copy(dst, s, run);
                                                    streamed_c = true;
                                                    pos += run;
                                                    continue;
                                                }
                                                if (store_c) {
                                                    std::copy(s, s + run, dst);
                                                } else {
                                                    for (int64_t r = 0; r < run; ++r) {
                                                        dst[r] += s[r];
                                                    }
                                                }
                                                pos += run;
                                            }
                                            continue;
                                        }
                                        if (store_c) {
                                            for (int64_t i2 = 0; i2 < mc_len; ++i2) {
                                                C_data[c_m_offsets[static_cast<size_t>(i2)] + n_off] = src[i2];
                                            }
                                            continue;
                                        }
                                        for (int64_t i2 = 0; i2 < mc_len; ++i2) {
                                            C_data[c_m_offsets[static_cast<size_t>(i2)] + n_off] += src[i2];
                                        }
                                    }
                                } // next C block chunk
                            } else if (needs_c_scatter) {
                                LabeledSectionInternal("micro-kernel loop, tile scatter");
                                // Multi-M/N: run the kernel into a contiguous temporary tile, then
                                // scatter it to C. Where the inner group's fastest index has unit
                                // stride, the scatter walks it in runs that stay inside one extent
                                // of that index, so C is contiguous within each run and the update
                                // is a vector copy or add; looking every element up in the offset
                                // table costs more than the tile's arithmetic at small K.
                                bool const    tile_m_unit = plan.c_m_dims.back().tensor_stride == 1;
                                int64_t const tile_m_fast = plan.c_m_dims.back().size;
                                bool const    tile_n_unit = plan.c_n_dims.back().tensor_stride == 1;
                                int64_t const tile_n_fast = plan.c_n_dims.back().size;
                                tls_Ct.resize(static_cast<size_t>(MR) * NR);
                                for (int64_t jr = 0; jr < num_jr; ++jr) {
                                    int64_t const nr_actual = std::min(static_cast<int64_t>(NR), nc_len - jr * NR);

                                    for (int64_t ir = 0; ir < num_ir; ++ir) {
                                        int64_t const mr_actual = std::min(static_cast<int64_t>(MR), mc_len - ir * MR);

                                        // Use a contiguous MR*NR temp buffer for the GEMM output.
                                        std::fill(tls_Ct.begin(), tls_Ct.end(), ValueType{0});
                                        ValueType *Ct = tls_Ct.data();

                                        ValueType *Ap_panel = Ap + ir * MR * kc_len;
                                        ValueType *Bp_panel = Bp + jr * NR * kc_len;

                                        // Micro-kernel into contiguous Ct (col-major: rs=1, cs=MR)
                                        micro_tile(static_cast<int>(MR), static_cast<int>(NR), kc_len, alpha, Ap_panel, Bp_panel, mr_actual,
                                                   nr_actual, Ct, 1, MR);

                                        // Scatter Ct back to C through the offset tables, innermost
                                        // along whichever of C's index groups is packed closer. Ct
                                        // is MR by NR and cache-resident, so reading it with a
                                        // stride costs nothing.
                                        if (scatter_n_inner) {
                                            for (int64_t ii = 0; ii < mr_actual; ++ii) {
                                                int64_t const m_off = c_m_offsets[static_cast<size_t>(ir * MR + ii)];
                                                if (tile_n_unit) {
                                                    int64_t pos = 0;
                                                    while (pos < nr_actual) {
                                                        int64_t const n_global = nc + jr * NR + pos;
                                                        int64_t const run =
                                                            std::min(tile_n_fast - (n_global % tile_n_fast), nr_actual - pos);
                                                        ValueType *dst = C_data + m_off + c_n_offsets[static_cast<size_t>(jr * NR + pos)];
                                                        ValueType const *src = Ct + pos * MR + ii;
                                                        if (store_c) {
                                                            for (int64_t r = 0; r < run; ++r) {
                                                                dst[r] = src[r * MR];
                                                            }
                                                        } else {
                                                            for (int64_t r = 0; r < run; ++r) {
                                                                dst[r] += src[r * MR];
                                                            }
                                                        }
                                                        pos += run;
                                                    }
                                                    continue;
                                                }
                                                if (store_c) {
                                                    for (int64_t jj = 0; jj < nr_actual; ++jj) {
                                                        C_data[m_off + c_n_offsets[static_cast<size_t>(jr * NR + jj)]] = Ct[jj * MR + ii];
                                                    }
                                                    continue;
                                                }
                                                for (int64_t jj = 0; jj < nr_actual; ++jj) {
                                                    C_data[m_off + c_n_offsets[static_cast<size_t>(jr * NR + jj)]] += Ct[jj * MR + ii];
                                                }
                                            }
                                        } else {
                                            for (int64_t jj = 0; jj < nr_actual; ++jj) {
                                                int64_t const    n_off = c_n_offsets[static_cast<size_t>(jr * NR + jj)];
                                                ValueType const *col   = Ct + jj * MR;
                                                if (tile_m_unit) {
                                                    int64_t pos = 0;
                                                    while (pos < mr_actual) {
                                                        int64_t const m_global = mc + ir * MR + pos;
                                                        int64_t const run =
                                                            std::min(tile_m_fast - (m_global % tile_m_fast), mr_actual - pos);
                                                        ValueType *dst = C_data + c_m_offsets[static_cast<size_t>(ir * MR + pos)] + n_off;
                                                        ValueType const *src = col + pos;
                                                        if (store_c) {
                                                            std::copy(src, src + run, dst);
                                                        } else {
                                                            for (int64_t r = 0; r < run; ++r) {
                                                                dst[r] += src[r];
                                                            }
                                                        }
                                                        pos += run;
                                                    }
                                                    continue;
                                                }
                                                if (store_c) {
                                                    for (int64_t ii = 0; ii < mr_actual; ++ii) {
                                                        C_data[c_m_offsets[static_cast<size_t>(ir * MR + ii)] + n_off] = col[ii];
                                                    }
                                                    continue;
                                                }
                                                for (int64_t ii = 0; ii < mr_actual; ++ii) {
                                                    C_data[c_m_offsets[static_cast<size_t>(ir * MR + ii)] + n_off] += col[ii];
                                                }
                                            }
                                        }
                                    }
                                }
                            } else {
                                LabeledSectionInternal("micro-kernel loop, direct C");
                                // Single-M, single-N: the kernel accumulates directly into C.
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
                        *team->base += n_mc_blk;
                        // Every member is done with this K block's panel before any repacks it. The next K block's
                        // blocks may go to other members, which read the C blocks this one wrote, so its streaming
                        // stores drain first.
                        if (team->size > 1) {
                            if (streamed_c) {
                                einsums::simd::stream_fence();
                            }
                            LabeledSectionInternal("team: wait for the panel to be consumed");
                            team->state->barrier.wait();
                        }
                    }

                    // Reclaim excess thread-local buffer memory.
                    auto shrink_tls = [](auto &v) {
                        if (v.capacity() > 2 * v.size() && v.capacity() > 4096) {
                            v.shrink_to_fit();
                        }
                    };
                    // Streaming stores are weakly ordered, so drain them before this thread's share
                    // of C is read: once per N block, the rarest point still inside the loop that
                    // wrote them.
                    if (streamed_c) {
                        einsums::simd::stream_fence();
                    }

                    shrink_tls(tls_Ap);
                    if (team->bp == nullptr) {
                        shrink_tls(tls_Bp);
                    }
                    if (needs_c_scatter)
                        shrink_tls(tls_Ct);
                }
            }
            if (members > 1 && team_panels.capacity() > 2 * team_panels.size() && team_panels.capacity() > 4096) {
                team_panels.shrink_to_fit();
            }
        }

    } // end batch loop
}

// ---------------------------------------------------------------------------
// Runtime entry point: accepts a pre-built ContractionSpec.
// ---------------------------------------------------------------------------

/// @brief Attempt to execute an einsum contraction via the packed GEMM backend from a runtime-built
/// ContractionSpec.
///
/// Works for any BasicTensorConcept operand, compile-time or runtime rank; every decision (rank
/// classification, batching, kernel selection) is made at run time against the spec. Returns true
/// if the contraction was handled, false if the caller should fall back.
///
/// @param spec_in The contraction's index lists and conjugation flags. @param C_prefactor Scale
/// applied to C before the product is accumulated. @param C The output tensor. @param AB_prefactor
/// Scale applied to the contraction of A and B. @param A The first input tensor. @param B The
/// second input tensor. @param allow_scatter When false, decline contractions that remain multi-M/N
/// after coalescing, for callers with a faster fallback (the compile-time dispatch's Sort+GEMM).
/// Leave true for callers whose only alternative is a generic loop (ComputeGraph's string
/// dispatch). @param site Optional memo owned by a caller that repeats this exact contraction (a
/// graph node); see @ref ContractionSite. A hit skips building and hashing the key and the
/// plan-cache lookup. Its @ref KernelRoute pin, when set, decides vendor versus packed.
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

    // Which kernel this contraction is spent through, resolved once and carried to the direct-GEMM
    // deferral and to blis_contraction's fast paths.
    bool const prefer_packed = prefer_packed_route(site);

    // Memo hit: this caller already resolved this exact contraction, under the same policy and
    // route, for operands with this layout and these sizes. The route matters only for a decline: a
    // plan is valid whichever way the contraction is spent, but a decline stands aside for a vendor
    // GEMM, which is right only while the vendor is the route.
    if (site != nullptr && site->resolved && site->allow_scatter == allow_scatter &&
        (site->plan != nullptr || site->declined_packed == prefer_packed) && site_key_matches(site->key, spec_in, st, A, B, *C)) {
        if (site->plan == nullptr) {
            ProfileAnnotate("packed_gemm_skip", "site_declined");
            return false;
        }
        ProfileAnnotate("packed_gemm_plan", "site");
        blis_contraction<ValueType>(*site->plan, *C, A, B, static_cast<ValueType>(AB_prefactor), static_cast<ValueType>(C_prefactor),
                                    site->plan->swap_ab ? spec_in.conj_b : spec_in.conj_a,
                                    site->plan->swap_ab ? spec_in.conj_a : spec_in.conj_b, prefer_packed);
        return true;
    }

    // Record what this call resolved to, so the next can skip straight to it. A null plan means
    // declined, which is a property of the key too; it records the route it was made under.
    auto remember = [site, allow_scatter, prefer_packed](ContractionKey const &k, PackingPlan const *p) {
        if (site != nullptr) {
            site->key             = k;
            site->plan            = p;
            site->allow_scatter   = allow_scatter;
            site->declined_packed = prefer_packed;
            site->resolved        = true;
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

    // Hashing the whole key only labels a profiler annotation, on a path tiled expansions drive
    // thousands of times per replay, so it is paid only when recording.
    if (profile::Profiler::instance().enabled()) {
        profile::annotate("packed_gemm_hash", static_cast<int64_t>(std::hash<ContractionKey>{}(key)));
    }

    // ------------------------------------------------------------------------- Classify target
    // indices. An empty M, N or link group is not a rejection: compute_packing_topology synthesizes
    // a unit dim so GEMV- and outer-product-shaped contractions run through the same machinery. The
    // classification decides the direct-GEMM deferral and whether Sort+GEMM exists as a fallback
    // (it needs all three groups non-empty).
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
        // Skip contractions that BLAS GEMM handles directly (no batch, single M/N/K), unless the
        // packed route is preferred: that GEMM would be clamped to one thread under a node-scoped
        // width. The recorded decline carries the route, so it is never reused in the other regime.
        if (m_count == 1 && n_count == 1 && link.size() == 1 && !spec.conj_a && !spec.conj_b && m_count + n_count == target.size() &&
            !prefer_packed) {
            ProfileAnnotate("packed_gemm_skip", "defer_to_direct_gemm");
            remember(key, nullptr);
            return false; // Deferred to direct BLAS GEMM, not a rejection.
        }
        // ── Direct BLAS fast paths ───────────────────────────────────────
        // The packed structure needs a K dimension to amortize its packing copy. Two shape classes have
        // none, and one BLAS call does their work in a single pass:
        //
        //   - outer product (no link indices): a K=1 GEMM, or ger, which writes C at bandwidth.
        //   - GEMV-shaped (no N, or no M): packing would copy the largest operand only for the kernel to
        //     read it again.
        //
        // Both apply when the operands' axis groups flatten to BLAS shapes; anything else falls through to
        // packing. Conjugated operands are excluded, since the packing path handles conjugation natively.
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
                // Below this a BLAS call's fixed cost dominates, and the small-outer deferral
                // decides.
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

                // With one column the column stride is unconstrained (an all-extent-1 group reports
                // a placeholder), but BLAS still requires ldc >= rows, so pin it. A layout still
                // short of that has no leading dimension to describe it; decline.
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

                // A k=1 GEMM rather than scal-then-ger: ger has no beta, so a prefactor would cost
                // a separate pass over C, which is the whole cost of an outer product. Past
                // kOuterGemmMaxElems GEMM's blocking overhead outgrows that pass. A strided operand
                // has no ldb to express, so it takes ger.
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
                    // Same leading-dimension rule as the outer path: with a single column the other
                    // stride is unconstrained, but BLAS still validates lda against the leading
                    // extent.
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

        // ── GEMV-shaped -> one storage-order stream ── When the supplying operand's C axes and
        // link axes interleave (the exchange K(i,j) = A(i,k,j,l) B(k,l)), no matrix view exists for
        // gemv, and packing would copy the whole operand only to read it again with nothing to
        // reuse it against. stream_contract reads it once in storage order, the least this
        // bandwidth-bound shape can cost. Below kStreamMinElems the call is a few microseconds
        // whichever way it goes.
        if (!spec.conj_a && !spec.conj_b && !link.empty() && (m_count == 0) != (n_count == 0)) {
            auto try_stream = [&]<einsums::BasicTensorConcept SupT, einsums::BasicTensorConcept VecT>(
                                  SupT const &S, VecT const &V, std::vector<std::string> const &sup,
                                  std::vector<std::string> const &other) -> bool {
                auto const distinct = [](std::vector<std::string> const &v) {
                    return std::unordered_set<std::string>(v.begin(), v.end()).size() == v.size();
                };
                auto const within_sup = [&](std::vector<std::string> const &v) {
                    return std::ranges::all_of(v, [&](std::string const &x) { return std::ranges::find(sup, x) != sup.end(); });
                };
                if (!distinct(sup) || !distinct(other) || !distinct(c_raw) || !within_sup(other) || !within_sup(c_raw)) {
                    return false;
                }

                StreamLayout s_layout, w_layout, c_layout;
                for (size_t d = 0; d < sup.size(); d++) {
                    s_layout.dims.push_back(static_cast<int64_t>(S.dim(d)));
                    s_layout.strides.push_back(static_cast<int64_t>(S.stride(d)));
                }
                int64_t s_elems = 1;
                for (int64_t const d : s_layout.dims) {
                    s_elems *= d;
                }
                if (s_elems < kStreamMinElems) {
                    return false;
                }
                for (size_t d = 0; d < other.size(); d++) {
                    w_layout.dims.push_back(static_cast<int64_t>(V.dim(d)));
                    w_layout.strides.push_back(static_cast<int64_t>(V.stride(d)));
                }
                for (size_t d = 0; d < c_raw.size(); d++) {
                    c_layout.dims.push_back(static_cast<int64_t>(C->dim(d)));
                    c_layout.strides.push_back(static_cast<int64_t>(C->stride(d)));
                }

                StreamTerm<ValueType> term{.c        = C->data(),
                                           .c_layout = std::move(c_layout),
                                           .w        = V.data(),
                                           .w_layout = std::move(w_layout),
                                           .c_axis   = std::vector<int>(sup.size(), -1),
                                           .w_axis   = std::vector<int>(sup.size(), -1),
                                           .alpha    = static_cast<ValueType>(AB_prefactor),
                                           .c_pf     = static_cast<ValueType>(C_prefactor)};
                for (size_t d = 0; d < sup.size(); d++) {
                    if (auto it = std::ranges::find(c_raw, sup[d]); it != c_raw.end()) {
                        term.c_axis[d] = static_cast<int>(it - c_raw.begin());
                    }
                    if (auto it = std::ranges::find(other, sup[d]); it != other.end()) {
                        term.w_axis[d] = static_cast<int>(it - other.begin());
                    }
                }

                // One output: offer every axis of S that C carries, so threads own disjoint slices
                // of C and write them in place. Privatizing would copy and reduce C once per
                // thread, which costs as much as the stream when C is large.
                std::vector<int> partition_axes;
                for (size_t d = 0; d < sup.size(); d++) {
                    if (term.c_axis[d] >= 0 && s_layout.dims[d] > 1) {
                        partition_axes.push_back(static_cast<int>(d));
                    }
                }

                ProfileAnnotate("packed_gemm_path", "stream");
                last_contraction_route() = "stream";
                stream_contract<ValueType>(S.data(), s_layout, {std::move(term)}, partition_axes);
                return true;
            };
            if (n_count == 0 ? try_stream(A, B, a_raw, b_raw) : try_stream(B, A, b_raw, a_raw)) {
                return true;
            }
        }

        // Bandwidth-bound shape classes where the packed passes lose to the compile-time generic
        // loop's single fused pass at every size: batch-dot (no M and no N) and GEMV-shaped (one of
        // M/N empty, nothing to amortize the packing copy). Declined only when !allow_scatter, i.e.
        // for the eager dispatch whose generic fallback is the good one; runtime callers' fallback
        // is slower than the packed path, so they keep it. A GEMV-shaped contraction gets here only
        // when the gemv and stream routes above declined.
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

    // ------------------------------------------------------------------------- Pack-A / Pack-B
    // path (BLIS-style, with optional batch dims).
    // ------------------------------------------------------------------------- The cache stores
    // plans already filled, k-sorted and coalesced, so a hit is only a lookup. That is sound
    // because the key pins the strides and those steps read nothing else. The pointer outlives the
    // shared lock deliberately: entries are never erased, and the map is node-based, so references
    // to mapped values stay valid.
    PackingPlan const *cached = PackingPlanCache::instance().lookup(key);
    PackingPlan        computed;
    if (cached == nullptr) {
        computed = compute_packing_topology(key);
        if (computed.valid) {
            fill_strides(computed, A, B, *C);
            sort_k_dims_for_packing(computed);
            coalesce_plan(computed, static_cast<int64_t>(sizeof(ValueType)));
            // Which operand takes the kernel's M role is a property of the contraction and the
            // resolved kernel, so it is settled once here and cached with the plan.
            if (mn_roles_should_swap(computed, micro_kernel_shape<ValueType>(),
                                     get_scalar_type<ValueType>() == ScalarType::Complex64 ||
                                         get_scalar_type<ValueType>() == ScalarType::Complex128)) {
                computed = transposed_plan(computed);
            }
            PackingPlanCache::instance().insert(key, computed);
            // Re-look-up so `cached` names the cache's copy, not this frame's: a site remembers the
            // pointer, and only cache entries live long enough.
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

        // Outer products (no link indices, K synthesized to 1) pay a fixed setup cost for the
        // packed passes, so below kOuterProductFloor output elements the generic loop wins. Rank-2
        // outer products never get here (StringDispatch's GER path takes them first), so the
        // comparison is always against the generic loop: the crossover moves whenever that loop
        // does, and it differs between machines.
        if (outer_shaped && plan.M_total * plan.N_total < kOuterProductFloor) {
            ProfileAnnotate("packed_gemm_skip", "defer_small_outer_to_generic");
            remember(key, nullptr);
            return false;
        }

        // Decline when a TTGT fallback exists and either the rung's kernel does not beat it on the
        // scatter path or the shape is batched: Sort+GEMM's per-batch GEMMs beat the scatter
        // engines on batched shapes at every size tested.
        if (needs_scatter && ttgt_exists && !allow_scatter && (!micro_kernel_shape<ValueType>().fast_scatter || plan.batch_total > 1)) {
            ProfileAnnotate("packed_gemm_skip", "scatter_defer_to_ttgt");
            EINSUMS_LOG_INFO("PackedGemm: declining — scatter-path shape, the caller has a TTGT fallback, "
                             "and this rung's kernel does not beat it for this shape.");
            remember(key, nullptr);
            return false;
        }

        ProfileAnnotate("packed_gemm_path", needs_scatter ? "scatter" : "single_mn");
        // Only a cache-owned plan is stable enough to remember. If the re-lookup above missed, skip
        // the memo rather than record a null plan, which would read as declined.
        if (cached != nullptr) {
            remember(key, cached);
        }
        blis_contraction<ValueType>(plan, *C, A, B, static_cast<ValueType>(AB_prefactor), static_cast<ValueType>(C_prefactor),
                                    plan.swap_ab ? spec.conj_b : spec.conj_a, plan.swap_ab ? spec.conj_a : spec.conj_b, prefer_packed);
        return true;
    } else {
        ProfileAnnotate("packed_gemm_skip", "invalid_topology");
        EINSUMS_LOG_INFO("PackedGemm: skipping — packing topology invalid for this contraction pattern.");
        remember(key, nullptr);
    }

    // Contraction doesn't fit packed GEMM, so fall back to generic algorithm.
    return false;
}

EINSUMS_NAMESPACE_END(packed_gemm)
