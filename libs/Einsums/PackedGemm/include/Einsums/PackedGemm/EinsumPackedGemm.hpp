//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

// This header is included from Dispatch.hpp.

#include <Einsums/Config.hpp>

#include <Einsums/BLAS.hpp>
#include <Einsums/BLAS/ThreadControl.hpp>
#include <Einsums/Concepts/TensorConcepts.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Logging.hpp>
#include <Einsums/PackedGemm/ContractionKey.hpp>
#include <Einsums/PackedGemm/MicroKernel.hpp>
#include <Einsums/PackedGemm/Packing.hpp>
#include <Einsums/Profile/Profile.hpp>
#include <Einsums/SIMD/Prefetch.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_set>
#include <vector>

#ifdef _OPENMP
#    include <omp.h>
#endif

EINSUMS_NAMESPACE_BEGIN(packed_gemm)

/// Output elements below which an outer product is left to the generic loop.
/// Measured, not chosen; see the decline site for the data and its caveats.
inline constexpr int64_t kOuterProductFloor = 768;

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

/// Name of the kernel route the most recent @ref blis_contraction call on this
/// thread took: "gemm_batch", "flatten_gemm", "flatten_gemm_hptt",
/// "flatten_gemm_gather", "single_k_gemm" or "packed", all but the last being
/// the fast paths that hand the whole contraction to the vendor, and the last
/// the engine's own packed loops.
///
/// The three flatten spellings name the same route by how it fed the vendor:
/// plain when both operands were already flat (no copy), @c _hptt when HPTT
/// transposed them into the flat buffers, @c _gather when the scalar gather
/// did. They are distinguished because the difference is worth a measurable
/// factor and nothing else observes which one ran.
///
/// Test introspection ONLY, mirroring @c dispatch::last_dispatch_route one
/// level down - that one names which BACKEND took the contraction, and this one
/// which kernel inside PackedGemm did. It exists so a test can assert that a
/// node-scoped width sends the work to the packed loops rather than to a vendor
/// GEMM the wrappers would clamp to one thread. Thread-local, and for a batched
/// contraction it names the last slice this thread ran; not an API for steering
/// execution.
///
/// Defined OUT OF LINE, and exported, so the whole process shares one slot; see
/// @c compute_graph::dispatch::last_dispatch_route for why an inline
/// thread-local is not enough.
[[nodiscard]] EINSUMS_EXPORT char const *&last_contraction_route();

/// The route pin the most recent route decision on this thread read.
///
/// Test introspection ONLY, alongside @ref last_contraction_route: that one
/// names which kernel ran, this one whether a pin or the thread regime chose
/// it. Adaptive means the decision came from @ref
/// einsums::blas::vendor_call_is_fenced, which is what an eager caller and an
/// unplanned graph get.
///
/// Exported and defined out of line for the same reason as
/// @ref last_contraction_route.
[[nodiscard]] EINSUMS_EXPORT KernelRoute &last_route_pin();

/// @brief Whether this contraction is to be packed rather than handed to one
///        vendor GEMM.
///
/// A pinned site answers from the pin, at every width including 1. Everything
/// else - an eager caller, an unplanned graph, any caller that passes no site -
/// answers from the thread regime exactly as before pinning existed: a caller
/// holding a node-scoped width has its vendor calls clamped to one thread (@ref
/// einsums::blas::vendor_call_is_fenced), so the deferring fast paths would run
/// the node serially while the packed loops, which fork from the same ICV the
/// width raised, get all of it.
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

/// Bytes a C run must cover before the write-back streams it.
///
/// Two cache lines. Swept over 1, 2, 4, 8 and 16 lines on intensli and ccsd_t:
/// ccsd_t is flat throughout, and intensli single is 0.978x / 0.984x / 0.945x /
/// 0.942x / 0.954x of TBLIS, so 2 is the peak and 16 is clearly bad (it also
/// costs intensli double, 1.110x -> 1.039x). The first cut of this shipped 4 on
/// the reasoning that a 192-byte run wastes too much of itself on the partial
/// lines at its ends; the measurement says otherwise, and those rows do want
/// streaming.
inline constexpr int64_t kStreamRunBytes = 2 * 64;

/// @brief Whether this contraction should be packed with the roles of A and B
///        exchanged, i.e. as C^T = B^T A^T.
///
/// The tile kernel is not symmetric in m and n. It holds a tile as MR-tall
/// vectors, so a destination whose consecutive m coordinates are adjacent in
/// memory takes a vector store, and any other destination takes MR*NR scalar
/// stores through a stack tile. The C scatter wants that same direction, for
/// the same reason: it is the one whose consecutive flat coordinates are
/// contiguous in C.
///
/// When C's unit stride reaches C through B, both wants point at the N group,
/// and no ordering inside the M group can supply it. Reading the tile back with
/// a stride - what the scatter used to do - costs MR*NR strided loads and MR*NR
/// scalar stores per tile against the (MR*NR/lanes) * kc vector FMAs that
/// produced it, which at the rank-6 ccsd_t shapes' K of 24 is more than the
/// arithmetic itself. On the Tensor Contraction Benchmark that split the
/// eighteen ccsd_t mirror pairs cleanly in two, at 20 GF/s against 27 for the
/// same kernel on the mirrored shape.
///
/// So exchange the roles instead: C^T = B^T A^T is the same contraction, B
/// packs into the MR panels, A into the NR panels, and the kernel's m direction
/// IS C's unit-stride one. Measured +25% to +35% on the nine ccsd_t rows whose
/// unit index arrives through B, and level with their mirrors afterwards.
///
/// The test is for a UNIT stride in the N group and none in the M group, not
/// simply the smaller of the two. When neither group is contiguous in C both
/// orders spend a cache line per element, `scatter_n_inner` inside
/// @ref blis_contraction already picks the shorter stride for the inner loop,
/// and there is nothing left for an exchange to win - while it still costs the
/// register tile, which is cut MR deep along whichever group takes the M role.
/// intensli's abcd-dbea-ec has an N group of 24 against an MR of 16, so
/// exchanging computes a third of its FMAs into masked-off lanes; measured
/// 0.70x. Hence
/// also the floor: a group that cannot fill a couple of tiles is the wrong one
/// to cut into them.
///
/// Only the tile-scatter path benefits. The direct-C branches already choose a
/// transposed BLAS call from the same fact, the block-GEMM strategy already
/// swaps its vendor operands, and the 1m complex path packs in a geometry of
/// its own.
inline bool mn_roles_should_swap(PackingPlan const &plan, MicroKernelShape const &shape, bool is_complex) {
    if (plan.c_m_dims.empty() || plan.c_n_dims.empty()) {
        return false;
    }
    bool const scatter = plan.c_m_dims.size() > 1 || plan.c_n_dims.size() > 1 ||
                         (plan.c_m_dims[0].tensor_stride != 1 && plan.c_n_dims[0].tensor_stride != 1);
    return scatter && !shape.block_gemm && !(is_complex && shape.use_1m) && plan.c_n_dims.back().tensor_stride == 1 &&
           plan.c_m_dims.back().tensor_stride != 1 && plan.N_total >= 2 * static_cast<int64_t>(shape.mr);
}

/// @brief Copy @p n elements to @p dst without first fetching its cache lines.
///
/// An ordinary store to a line the core does not already own makes the cache
/// read that line from memory before the write can land - a read-for-ownership
/// - even when every byte of it is about to be overwritten. On a contraction
/// whose C is written once and never read (beta == 0), that is an extra pass
/// over the whole of C: the rank-6 ccsd_t shapes write 369 MB and fetch 369 MB
/// they have no use for, against an arithmetic floor of 71 ms out of 131.
///
/// Streaming stores leave through the write-combining buffers instead, and a
/// line assembled there whole is written with no prior read. That is why the
/// caller must hand this a run that is CONTIGUOUS and reasonably long: a
/// buffer flushed half-full pays a partial write and keeps the read.
///
/// @p dst must be vector-aligned and @p n a whole number of lanes - @ref
/// stream_run_ok is the caller's test for both. Handling a misaligned head
/// in here was tried and is a trap: the head and tail are ordinary stores, so
/// their lines keep the fetch, and paying a staging copy to line the run up
/// costs an entire extra pass of C through L1 - measured -4% on the intensli
/// rows, where the fetch it saves is a small share of the run anyway.
///
/// @warning Streaming stores are weakly ordered. The caller must
/// @ref einsums::simd::stream_fence() before anything reads @p dst.
template <typename T>
void stream_copy(T *dst, T const *src, int64_t n) {
    // Only the types with a vector register on this ISA; complex has none, and
    // @ref stream_run_ok already refuses it at run time, but the template is
    // instantiated for it regardless.
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

/// @brief Execute a tensor contraction via Pack-A / Pack-B + BLAS GEMM tiles (BLIS-style).
///
/// For multi-K contractions (rank-3+), flattens A and B into contiguous M*K / K*N buffers
/// and calls BLAS GEMM directly.  For single-K, uses BLIS-style tiled packing with BLAS
/// GEMM per tile.
/// The operands arrive as @ref OperandView rather than as tensor types on
/// purpose: everything this function reads from them - the data pointer, the
/// rank, the dims and strides - is a runtime value the PackingPlan already
/// works in. Keying the template on ValueType alone is what keeps the engine
/// from being re-instantiated and re-optimized per (rank, tensor template) at
/// every call site; see OperandView for the measurement that motivated it.
template <typename ValueType>
void blis_contraction(PackingPlan const &plan, ValueType *C_base, OperandView<ValueType> const &A, OperandView<ValueType> const &B,
                      ValueType alpha, ValueType beta, bool conj_a = false, bool conj_b = false, bool prefer_packed = false) {
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

    // `prefer_packed` says to keep the whole contraction rather than hand it to
    // one vendor GEMM. It is decided by the caller - see @ref
    // prefer_packed_route - and arrives here already resolved, so the two fast
    // paths below and the packed loops are picked from one answer per call. The
    // gemm_batch path is exempt: its vendor entry point is einsums' own OpenMP
    // loop over serial GEMMs, it forks from the ICV too, and its wrapper carries
    // no fence.
    //
    // Still not stored anywhere here. The plan cache is shared across nodes and
    // a route is not a property of a packing plan; where a route IS a settled
    // fact it is a fact about the NODE, and it lives on that node's
    // ContractionSite (@ref KernelRoute), which is what the caller read.

    int64_t const M = plan.M_total;
    int64_t const N = plan.N_total;
    int64_t const K = plan.K_total;

    // Cache-aware blocking: tile sizes adapt to sizeof(ValueType) and CPU cache
    // hierarchy, derived from the tile the resolved kernel actually computes,
    // and from this contraction's own extents - whether C survives a sweep
    // decides how large KC wants to be, and only M and N say that.
    auto const    blk        = compute_blocking(static_cast<int64_t>(sizeof(ValueType)), MR, NR, M, N, K);
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
    //
    // Kept under a node width, unlike the single-GEMM deferrals below: the
    // gemm_batch entry point is einsums' own OpenMP loop over the batch, which
    // forks from the ICV the width raised and runs its inner GEMMs nested-serial,
    // so it consumes the width instead of losing it to the wrapper fence (which
    // the batch wrappers deliberately do not carry).
    // -------------------------------------------------------------------------
    //
    // The `!plan.swap_ab` term is a guard, not a policy: this is the one path
    // that reads A and B directly rather than through the role-resolved pointers
    // below, so an exchanged plan would pair B's batch strides with A's data. A
    // swap always implies multi-M or multi-N (see @ref mn_roles_should_swap: it
    // needs a scatter, and the only scatter reason left once the N group holds
    // C's unit stride is a multi-dim group), so the term never fires today - it
    // is here so that widening the swap cannot silently break this path.
    if (plan.batch_total > 1 && plan.k_dims_in_a.size() == 1 && !multi_m && !multi_n && !plan.synthetic && !plan.swap_ab) {
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
                a_ptrs[static_cast<size_t>(batch)] = A.data + a_off;
                b_ptrs[static_cast<size_t>(batch)] = B.data + b_off;
                c_ptrs[static_cast<size_t>(batch)] = C_base + c_off;
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

        // `plan.swap_ab` says this plan describes the contraction as C^T = B^T
        // A^T, so the tensor the packers must read for the plan's A role is B
        // and vice versa. The batch strides were mirrored with the rest of the
        // plan, so the offsets already belong to the tensors named here. The
        // caller swapped the conjugation flags to match.
        ValueType       *C_data = C_base + c_batch_off;
        ValueType const *A_data = (plan.swap_ab ? B.data : A.data) + a_batch_off;
        ValueType const *B_data = (plan.swap_ab ? A.data : B.data) + b_batch_off;

        // -------------------------------------------------------------------------
        // Multi-K fast path: flatten A and B into contiguous M*K / K*N buffers,
        // then call BLAS GEMM directly.
        // -------------------------------------------------------------------------
        // The flatten+GEMM path writes C directly and supports only stride-1
        // column- or row-major outputs; scatter-layout C goes to the tiled or
        // block-GEMM paths below.
        //
        // Every exit from this block is a vendor GEMM over the flat buffers -
        // one call when both sides are zero-copy, a serial chain of KC slices
        // otherwise - so under a fenced width the whole block runs on one
        // thread. The packed loops below take the contraction instead; they read
        // the same multi-K plan through pack_A/pack_B.
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
            int const rank_a_rt = A.rank;
            int const rank_b_rt = B.rank;

            // Describe a dense operand to HPTT.
            //
            // HPTT takes no stride vector: sizes[0] is the FASTEST-varying axis
            // and every axis after it follows as a dense product chain. That is
            // column-major by argument convention, not an assumption about the
            // caller's layout - any dense tensor (row-major, column-major, or any
            // other axis order) fits the model once its axes are RELABELLED into
            // ascending-stride order. So sort the axes by stride, describe the
            // operand in that order, and renumber the transpose permutation
            // through the same relabelling. This is what lets row-major operands
            // - what the runtime-tensor and string-einsum paths hand us - take
            // the HPTT route instead of falling to the scalar gather below.
            //
            // Extent-1 axes carry no layout information (their stride is
            // arbitrary and never indexed), so they are dropped from both the
            // source description and the permutation; the destination is dense,
            // so dropping them does not change its element order.
            //
            // Returns false - leaving the caller on the scalar gather - when the
            // operand is dense in no axis order at all: a padded, strided or
            // broadcast view. HPTT could express a padded one through outerSizeA,
            // which hptt_transpose does not plumb through today.
            auto describe_for_hptt = [](auto const &tensor, int rank, std::vector<int> const &out_order, std::vector<size_t> &sizes,
                                        std::vector<int> &perm) -> bool {
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

                // perm[j] = the source axis, in HPTT's numbering, that becomes
                // destination axis j.
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
                // Anything other than a permutation of the layout-carrying axes
                // means the plan's dims do not account for this operand.
                return perm.size() == ord.size();
            };

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

            std::vector<int>    perm_a, perm_b;
            std::vector<size_t> sizes_a, sizes_b;

            // Destination axis order: A_flat is col-major M x K (M fastest) and
            // B_flat is row-major K x N, i.e. col-major N x K (N fastest). In
            // both, the K axes run in reverse plan order so that the flat K index
            // matches k_cum.
            if (use_hptt && !a_zero_copy) {
                std::vector<int> out_a;
                out_a.reserve(nk + 1);
                out_a.push_back(static_cast<int>(plan.m_dims[0].tensor_pos));
                for (size_t i = 0; i < nk; ++i) {
                    out_a.push_back(static_cast<int>(k_dims_a[nk - 1 - i].tensor_pos));
                }
                use_hptt = describe_for_hptt(A, rank_a_rt, out_a, sizes_a, perm_a);
            }
            if (use_hptt && !b_zero_copy) {
                std::vector<int> out_b;
                out_b.reserve(nk + 1);
                out_b.push_back(static_cast<int>(plan.n_dims[0].tensor_pos));
                for (size_t i = 0; i < nk; ++i) {
                    out_b.push_back(static_cast<int>(k_dims_b[nk - 1 - i].tensor_pos));
                }
                use_hptt = describe_for_hptt(B, rank_b_rt, out_b, sizes_b, perm_b);
            }

            if (use_hptt) {
                // HPTT-transpose the full tensor(s) into flat M*K / K*N layout once,
                // then do KC-tiled GEMM over the contiguous flat buffers.
                last_contraction_route() = "flatten_gemm_hptt";
                int num_threads          = 1;
#ifdef _OPENMP
                num_threads = omp_get_max_threads();
#endif

                if (!a_zero_copy) {
                    hptt_transpose(perm_a.data(), static_cast<int>(sizes_a.size()), A_data, sizes_a.data(), A_flat, num_threads, conj_a);
                }
                if (!b_zero_copy) {
                    hptt_transpose(perm_b.data(), static_cast<int>(sizes_b.size()), B_data, sizes_b.data(), B_flat, num_threads, conj_b);
                }

                // A_flat is now col-major M*K; B_flat is row-major K*N - and a
                // zero-copy side is, by the test above, already exactly that.
                // So BOTH operands span the whole K here, and the contraction is
                // one GEMM.
                //
                // It is deliberately NOT tiled over K. The gather branch below
                // tiles because its flat buffer holds one KC slice at a time and
                // has to be refilled; nothing here needs refilling, so a KC chain
                // would only re-read and re-write the whole of C once per slice
                // (K/KC times) and pay a vendor call for each - 282 calls and
                // ~326 MB of avoidable C traffic on ccsd's ab-cad-dcb at
                // K=144384, KC=512. Blocking K is the vendor's job once both
                // operands are flat.
                ValueType const *A_base = a_zero_copy ? A_data : A_flat;
                ValueType const *B_base = b_zero_copy ? B_data : B_flat;

                if (C_col_major) {
                    einsums::blas::gemm<ValueType>('N', 'T', static_cast<blas_int>(M), static_cast<blas_int>(N), static_cast<blas_int>(K),
                                                   alpha, A_base, static_cast<blas_int>(M), B_base, static_cast<blas_int>(N), beta, C_data,
                                                   static_cast<blas_int>(ldc_col));
                } else {
                    einsums::blas::gemm<ValueType>('N', 'T', static_cast<blas_int>(N), static_cast<blas_int>(M), static_cast<blas_int>(K),
                                                   alpha, B_base, static_cast<blas_int>(N), A_base, static_cast<blas_int>(M), beta, C_data,
                                                   static_cast<blas_int>(ldc_row));
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
        //
        // Skipped under a fenced width for the reason given at the top: this is
        // the whole contraction in one vendor call, which is the one shape of
        // work a clamped vendor cannot spread.
        // -------------------------------------------------------------------------
        if (plan.k_dims_in_a.size() == 1 && !multi_m && !multi_n && !plan.synthetic && !prefer_packed) {
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
                last_contraction_route() = "single_k_gemm";
                continue; // next batch slice
            }
        }

        // -------------------------------------------------------------------------
        // Fallback: BLIS-style tiled packing with BLAS GEMM per tile.
        // -------------------------------------------------------------------------
        last_contraction_route() = "packed";
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
        // Bound the A panel at ~4 MiB, which is what the paragraph above promises.
        //
        // This used to be gated on KC_blk > blk.KC, a comparison between two K values
        // that has nothing to do with the panel's size, and the gate failed both ways.
        // It let the panel through at 512 * 4096 * 4 bytes, 8 MiB, whenever KC_blk and
        // blk.KC coincided, which is every large-K single-precision contraction on a
        // rung with a kc hint. And it silently switched off if blk.KC grew, so
        // correcting a detected cache size turned the cap off and cost 0.62x on the
        // large-K cases. A constraint on the panel belongs on the panel.
        int64_t const mc_cap = (int64_t{4} << 20) / (KC_blk * static_cast<int64_t>(sizeof(ValueType)));

        // For multi-M/N: we need a temporary contiguous C tile buffer because
        // the multi-dim C elements are non-contiguous in memory.
        bool const needs_c_scatter = scatter_c;
        bool const block_strategy  = needs_c_scatter && shape.block_gemm;

        // Budget for the block-GEMM strategy's MC by NC C temp; the bound on NC
        // below applies it, and the block strategy's M block is sized from it
        // here. Four times L1 is where the M4 optimum sat, 432 to 504 KB against
        // a 128 KB L1. On a Zen+ with a 32 KB L1 the same multiplier gives 128 KB,
        // which measured WORST of every value swept on the rank-6 ccsd_t shape
        // (MC=2048: 15.7 GF/s, against 24.5 at 512 KB and 25.7 at 2 MB), so the
        // budget is floored at 512 KiB, which leaves the M4 value where it was
        // measured. EINSUMS_EXPERIMENT_C_TEMP_KB overrides it in KB for sweeps.
        int64_t c_temp_budget = std::max<int64_t>(4 * cpu_config().l1_cache_size, int64_t{512} << 10);
        if (char const *e = std::getenv("EINSUMS_EXPERIMENT_C_TEMP_KB")) {
            if (int64_t const kb = std::atoll(e); kb > 0) {
                c_temp_budget = kb * 1024;
            }
        }

        // The tile loops keep the A panel L2-resident, so blk.MC bounds their MC.
        // The block strategy has no such stake: its A block is consumed by a vendor
        // GEMM that re-packs it internally, and what that GEMM pays for is a SMALL
        // M - it also re-packs the whole KC x NC B block on every call, so M/MC
        // calls multiply that traffic. Sized from the cache-derived MC this was 32
        // rows on a Zen+, 93 thousand GEMMs of 32 x 24 x 36 on the intensli shape
        // abcde-efcad-bf, and lifting the clamp measured 1.74x there (5.8 to 10.0
        // GF/s, flat from 512 rows up, in both sweep orders). So the block
        // strategy's MC comes from the A-panel cap and the C temp: enough rows to
        // use the whole budget at the N the shape actually has, or a square
        // temp, whichever is larger.
        int64_t MC_blk = std::clamp((mc_cap / MR) * MR, static_cast<int64_t>(MR), blk.MC);

        // pack_A gathers along A's own contiguous axis when the flat M coordinate
        // was ordered for C (its unit-stride M axis sits second-fastest, see
        // coalesce_plan): rows i, i + X, i + 2X ... are adjacent in A. That reads
        // each cache line whole only if the block holds a line's worth of those
        // rows per value of the fastest coordinate, so the block is raised to
        // (line / elem) * X rows, within the A-panel cap. The panel may then
        // outgrow the L2 the tile kernel likes it in; on the shapes that take
        // this path (abcde-efcad-bf: 768 rows of 36) the kernel's A traffic is
        // trivial next to the gather it replaces.
        auto const line_rows = [&](std::vector<DimSpec> const &dims) -> int64_t {
            if (dims.size() < 2 || dims.back().tensor_stride == 1 || dims[dims.size() - 2].tensor_stride != 1) {
                return 0;
            }
            return (int64_t{64} / static_cast<int64_t>(sizeof(ValueType))) * dims.back().size;
        };
        if (!block_strategy) {
            // (line / elem) * X is a multiple of MR for the vector tile (MR is
            // itself 2 * lanes), and pack_A's fast strip needs the block whole in
            // X, so it is taken exactly when the panel cap allows it.
            // Only when the pack is a real share of the work: A carries M*K
            // elements against C's M*N, and a taller block costs the kernel its
            // L1-resident A panel (the rank-6 ccsd_t shapes, K=24 against
            // N=8000, lost 10% to the raise while packing 300x less than they
            // scatter).
            int64_t const want = line_rows(plan.m_dims);
            if (want > MC_blk && want % MR == 0 && want <= (mc_cap / MR) * MR && 4 * K >= N) {
                MC_blk = want;
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

        // Keep the M block a whole number of C's fastest segment when the C block
        // flush below can compose its destination into contiguous spans.
        //
        // A block that ends mid-segment sends its tail down the fallback walk,
        // which reads the block transposed AND scatters - the worst of both
        // orders. The tail is a fixed number of ROWS, so what it costs is set by
        // how many rows the block has: 16 of a 64-row double block is a quarter
        // of the work, against 8 of a 128-row single block. Measured on ccsd_t
        // before this alignment existed, the span flush was worth +5 to +10% on
        // the single rows and -3 to -4% on the double ones, which is that split
        // and nothing else.
        //
        // The step is lcm(segment, MR) so the block stays whole in the register
        // tile too; a segment that cannot reach a whole step inside the block is
        // left alone rather than shrunk to one.
        if (needs_c_scatter && !block_strategy && plan.c_m_dims.back().tensor_stride == 1) {
            int64_t const fm = plan.c_m_dims.back().size;
            // Whole segments are wanted whenever the write-back intends to
            // stream, which is composition OR a run long enough on its own. A
            // block shorter than the segment cuts every run partial and
            // misaligned, nothing streams, and the accumulator is pure overhead:
            // `abcd-ebad-ce` has Fm = 72 against MC = 64 and measured -9.4%,
            // while the same change is worth +13% to +15% on the rows whose MC
            // already spans the segment.
            if (fm > 1 && (plan.c_n_dims.back().tensor_stride == fm || fm * static_cast<int64_t>(sizeof(ValueType)) >= kStreamRunBytes)) {
                int64_t const step = std::lcm<int64_t, int64_t>(fm, MR);
                if (step <= MC_blk) {
                    MC_blk = (MC_blk / step) * step;
                } else if (step <= (mc_cap / MR) * MR && step <= M) {
                    // The block is SMALLER than one C segment, so every m run it
                    // cuts is partial and the span never forms at all - the
                    // write-back falls back to the column walk for the whole
                    // contraction and the streaming store is never reached. Raise
                    // the block to one whole segment.
                    //
                    // This is the case on twelve of the eighteen ccsd_t shapes,
                    // whose C segment is 384 or 480 elements against a cache-derived
                    // MC of 64 or 128, and on ao2mo's `abcd-ec-abed`, whose segment
                    // is 8064. It is affordable precisely because these are the
                    // small-K shapes: the A panel is MC * KC, so at K = 24 a
                    // 480-row block is 46 KB. The A-panel cap is still the bound,
                    // and a segment that cannot fit under it is left alone.
                    MC_blk = step;
                }
            }
        }

        // beta == 0 says C's prior contents are irrelevant, so the first K block
        // STORES its result and later blocks accumulate onto it. The direct-BLAS
        // paths above already do this through beta_k; the scatter paths below did
        // not, and instead made a separate read-modify-write pass over C to
        // multiply it by zero. On a scatter shape that pass is the dominant cost:
        // C's m and n index groups interleave in memory, so it touches one element
        // per cache line, and it is pure waste when the result is about to be
        // overwritten. Folding it into the scatter also makes beta == 0 mean what
        // BLAS says it means - C is never read - which `*= 0` does not, since
        // NaN * 0 is NaN rather than 0 and an uninitialized C would leak through.
        bool const overwrite_c = (beta == ValueType{0});

        // Whether the C block write-back may stream past the cache.
        //
        // Three things have to hold. C must be written and never read, which is
        // what overwrite_c says. K must fit one block, or a later K block would
        // accumulate onto lines this one just pushed out to memory and have to
        // fetch every one of them back. And the element type must have a vector
        // register to store from.
        //
        // Whether the run is long enough is decided per span at the write-back,
        // since that is where the length is known.
        bool const may_stream_c = overwrite_c && K <= KC_blk && !is_complex;

        // Which side the C scatter walks innermost. Both scatters below were
        // hardwired to n outer, m inner, so only the M group's fastest dimension
        // could make the inner loop contiguous. When C's smallest stride sits in
        // the N group instead, which happens whenever C's unit-stride index came
        // from B rather than A, no ordering of either group can help and the inner
        // loop spends a cache line per element. On the Tensor Contraction
        // Benchmark's rank-6 ccsd_t contractions that splits the eighteen mirror
        // pairs cleanly in two: the nine whose unit index reaches C through A run
        // at 76% to 98% of an equally sized GEMM, and the nine whose unit index
        // arrives through B run at 42% to 66%, with no overlap.
        //
        // A synthesized unit dim keeps a stride of 0 for life (see Packing.cpp), and
        // a group of extent 1 carries no locality to compare, so it never argues for
        // itself and never argues against the other side.
        int64_t const c_m_fastest     = plan.c_m_dims.back().tensor_stride;
        int64_t const c_n_fastest     = plan.c_n_dims.back().tensor_stride;
        bool const    scatter_n_inner = c_n_fastest != 0 && (c_m_fastest == 0 || c_n_fastest < c_m_fastest);

        // Does C's destination COMPOSE into whole contiguous spans?
        //
        // The scatter is written as if C's two index groups were independent, and
        // for a general contraction they are. But when C's fastest m index has unit
        // stride and its fastest n index steps by exactly that index's extent, the
        // two are adjacent halves of one dense run: element (i, j) of the rectangle
        // sits at base + j * Fm + i, so a whole m segment by a whole n segment is
        // Fm * Fn consecutive elements of C.
        //
        // It is the common case, not a curiosity - it holds whenever the two groups
        // happen to hold neighbouring indices of a dense C, which is what the rank-6
        // ccsd_t and the intensli shapes both do (24 x 20 = 480 floats and
        // 48 x 24 = 1152). The C block write-back walks those spans, which turns a
        // scatter of Fm-element pieces revisiting each cache line Fn times from Fn
        // different places into one sequential sweep - and is what lets the
        // write-back stream past the cache at all (@ref stream_copy).
        int64_t const blk_m_fast = plan.c_m_dims.back().size;
        int64_t const blk_n_fast = plan.c_n_dims.back().size;
        bool const    blk_compose =
            plan.c_m_dims.back().tensor_stride == 1 && blk_m_fast > 1 && plan.c_n_dims.back().tensor_stride == blk_m_fast;

        // A contraction that does NOT compose can still be worth the block, if
        // its m runs alone are long enough to stream.
        //
        // Composition is what makes a whole RECTANGLE of C contiguous; it is not
        // what the write-combining buffers need. They need a run that covers
        // whole cache lines, and C's own fastest index supplies one whenever its
        // extent is a few lines: 39 lines on abc-bda-dc double, 9 on
        // abcd-dbea-ec double, 3 on abcde-ecbfa-fd single. The lines at the two
        // ends of each run are partial and keep their fetch, which is why a run
        // of one or two lines is not worth the block's L2 round trip.
        bool const blk_runs_stream =
            plan.c_m_dims.back().tensor_stride == 1 && blk_m_fast * static_cast<int64_t>(sizeof(ValueType)) >= kStreamRunBytes;

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
        if (!block_strategy) {
            // Mirror of the MC raise above for pack_B.
            int64_t const want = line_rows(plan.n_dims);
            if (want > NC_blk && 4 * K >= M) {
                int64_t const nc_cap = (int64_t{4} << 20) / (KC_blk * static_cast<int64_t>(sizeof(ValueType)));
                NC_blk               = std::clamp(((want + NR - 1) / NR) * NR, NC_blk, std::max<int64_t>((nc_cap / NR) * NR, NC_blk));
            }
        }

        // Bound the MC by NC C temp that the block-GEMM scatter strategy allocates.
        //
        // Nothing else constrains it. compute_blocking derives MC from an A-panel
        // budget and NC from a B-panel budget, so their product lands wherever those
        // two leave it, and it scales with the element size while neither budget
        // does. On abcdef-gfbc-dega that gives 504 KB for float and 1008 KB for
        // double, and the double figure costs 1.30x: measured on an M4 with the block
        // path forced, bounding it takes the case from 24.0 to 31.3 GF/s.
        //
        // The cost is not only the scatter's own locality. The vendor GEMM that
        // writes this temp keeps its own packed buffers, and an oversized output
        // buffer evicts them: the GEMM's own time falls from 96.7 ms to 71.7 ms when
        // the temp is halved, which is most of what the case gains. That is also why
        // the smallest temp is not the best. Below roughly 400 KB the bound shrinks
        // NC far enough to re-pack A many more times for no further cache benefit,
        // and at a 32 KB bound pack_A goes from 2.4 ms to 39 ms.
        //
        // The budget itself, and where its floor came from, is c_temp_budget above.
        //
        // This lives here rather than in compute_blocking because only this strategy
        // allocates the temp: the tile and direct-BLAS paths would pay the smaller NC
        // and get nothing back.
        if (block_strategy) {
            int64_t const max_nc = ((c_temp_budget / (MC_blk * static_cast<int64_t>(sizeof(ValueType)))) / NR) * NR;
            if (max_nc >= NR && max_nc < NC_blk) {
                NC_blk = max_nc;
            }
        }
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
                bool                                       streamed_c = false;
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
                                // Swapping the operands computes Bf * Af^T, whose (j, i) is the
                                // (i, j) of Af * Bf^T, so the block temp comes out transposed and
                                // the n-inner scatter reads it contiguously. Striding the temp
                                // instead is not an option here: it is MC by NC, far too large to
                                // sweep once per m index, unlike the tile path's MR by NR buffer.
                                if (scatter_n_inner) {
                                    einsums::blas::gemm<ValueType>(
                                        'N', 'T', static_cast<blas_int>(nc_len), static_cast<blas_int>(mc_len),
                                        static_cast<blas_int>(kc_len), alpha, tls_Bf.data(), static_cast<blas_int>(nc_len), tls_Af.data(),
                                        static_cast<blas_int>(mc_len), ValueType{0}, tls_Cb.data(), static_cast<blas_int>(nc_len));
                                } else {
                                    einsums::blas::gemm<ValueType>(
                                        'N', 'T', static_cast<blas_int>(mc_len), static_cast<blas_int>(nc_len),
                                        static_cast<blas_int>(kc_len), alpha, tls_Af.data(), static_cast<blas_int>(mc_len), tls_Bf.data(),
                                        static_cast<blas_int>(nc_len), ValueType{0}, tls_Cb.data(), static_cast<blas_int>(mc_len));
                                }
                            }

                            // Scatter-accumulate the contiguous block into C. When C's
                            // stride along the fastest flat m coordinate is 1, the
                            // destination decomposes into contiguous runs and the
                            // accumulation vectorizes.
                            LabeledSectionInternal("C block scatter");
                            if (scatter_n_inner) {
                                // Mirror of the loop below with the roles of m and n exchanged.
                                // tls_Cb is nc_len x mc_len here (see the swapped GEMM above).
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

                    for (int64_t mc = 0; mc < M; mc += MC_blk) {
                        int64_t const mc_len = std::min(MC_blk, M - mc);

                        if (needs_c_scatter) {
                            precompute_offsets(mc, mc_len, plan.c_m_dims, c_m_offsets);
                        }

                        // Beta prescale: apply once per (mc, nc) block on first kc tile.
                        // The scatter branch stores on the first K block when beta == 0
                        // (see overwrite_c) and so needs no prescale at all. The
                        // direct-C branches still need one, because the micro-kernel
                        // only ever accumulates into C - but clearing C is a write
                        // where `*= 0` was a read-modify-write over the whole block.
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

                        // BLAS fallback: pack + per-tile GEMM.
                        if (!bp_packed) {
                            pack_B(Bp, B_data, plan, kc, kc_len, nc, nc_len, NR, conj_b);
                            bp_packed = true;
                        }
                        pack_A(Ap, A_data, plan, mc, mc_len, kc, kc_len, MR, conj_a);

                        int64_t const num_jr = (nc_len + NR - 1) / NR;
                        int64_t const num_ir = (mc_len + MR - 1) / MR;

                        if (needs_c_scatter && !scatter_n_inner && (blk_compose || blk_runs_stream)) {
                            LabeledSectionInternal("micro-kernel loop, C block");
                            // ---- Cache-resident C block ----
                            //
                            // The tiles accumulate into one contiguous mc_len x
                            // nc_len block and the block is written back to C
                            // once, instead of every tile making its own trip to
                            // a destination that may be hundreds of megabytes
                            // wide. Two things come of it.
                            //
                            // The kernel's C operand is the block, so its row
                            // stride is 1 whatever C's layout is, which is the
                            // whole-vector store path rather than the stack tile
                            // and MR*NR scalar stores.
                            //
                            // And the write-back's runs are as long as C's own
                            // fastest index allows - up to that index's whole
                            // extent - where the per-tile scatter could never
                            // carry a run past MR, and paid an offset-table
                            // lookup and a run computation for each of them.
                            //
                            // Only for the m-inner scatter. The n-inner variant
                            // reads the block along its long stride, which undoes
                            // the point; @ref mn_roles_should_swap has already
                            // turned every n-inner case that HAS a contiguous
                            // direction into an m-inner one, so what is left
                            // there spends a cache line per element either way.
                            // The accumulator is bounded by CHUNKING the N block,
                            // not by shrinking it.
                            //
                            // Bounding NC instead is the obvious move and it is
                            // wrong: A's DRAM traffic scales with 1/NC, so paying
                            // for a cache-sized C block out of NC charges it to the
                            // largest memory term in the contraction. Measured on
                            // ccsd's rank-4 single, where the budget halved NC from
                            // 2046 and A is 228 MB: every one of the twelve rows lost
                            // 0.8 to 1.7 points of %GEMM, while the twelve double
                            // rows - whose NC the budget did not reach - gained 0.5
                            // to 3.0. The packed B block already covers the whole N
                            // block, so a chunk costs no extra packing.
                            // Half the L2, NOT the block strategy's c_temp_budget.
                            //
                            // That budget sizes a vendor GEMM's output buffer, where
                            // the GEMM re-packs its own operands and the temp is the
                            // only thing competing for L2. This accumulator competes
                            // with the A panel and the B block, which are live across
                            // the same loops, so a budget equal to the whole L2
                            // leaves them nothing and the outcome falls to which
                            // sets the block happens to land in. Measured on the full
                            // ccsd_t group: at 512 KB one row of thirty-six
                            // (`abcdef-gfab-degc` d, and only when run after thirty
                            // other cases had fragmented the heap) collapsed to
                            // 11.3 GF/s against its five identical-shape siblings'
                            // 21.7; at 256 KB it is 21.3 and the group's double
                            // median rises from 1.30x to 1.32x of TBLIS with every
                            // row winning. 128 KB is too small - the double median
                            // falls to 1.11x.
                            int64_t const cb_budget = std::max<int64_t>(cpu_config().l2_cache_size / 2, int64_t{64} << 10);
                            int64_t       nb_len    = cb_budget / (mc_len * static_cast<int64_t>(sizeof(ValueType)));
                            nb_len                  = std::max<int64_t>((nb_len / NR) * NR, NR);
                            nb_len                  = std::min(nb_len, nc_len);

                            for (int64_t nb = 0; nb < nc_len; nb += nb_len) {
                                int64_t const nb_cur  = std::min(nb_len, nc_len - nb);
                                int64_t const jr_base = nb / NR;
                                tls_Ct.assign(static_cast<size_t>(mc_len) * static_cast<size_t>(nb_cur), ValueType{0});
                                ValueType *Cb = tls_Ct.data();

                                // Which of the two packed blocks the tile loops keep
                                // resident.
                                //
                                // The standard order streams the A panel and reuses one
                                // NR x KC column of B, which is right while B's block is
                                // the larger of the two - the shape this blocking was
                                // built for, where NC comes from an L3 budget and MC
                                // from an L2 one. It is exactly wrong for the intensli
                                // shapes, whose whole N is 24: there the B block is a
                                // few kilobytes and the A panel is the one that has just
                                // been gathered at a cache line per sixteen elements, so
                                // reading it back once per N tile pushes it through L2
                                // num_jr times over. Run those with the A panel
                                // innermost instead, so the pack's output is consumed
                                // while it is still in L1.
                                //
                                // The test is on the B block, not on a ratio: it earns
                                // its keep only while the whole thing stays resident
                                // alongside one A panel, and half the L1 is the budget
                                // that leaves room for the panel and the C block rows.
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
                                                // The span's columns are already adjacent in C, so
                                                // each one streams in place - no staging. A span
                                                // shorter than a few lines cannot fill a
                                                // write-combining buffer, so streaming it would pay
                                                // a partial write and keep the fetch.
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
                                            ValueType *dst =
                                                C_data + c_m_offsets[static_cast<size_t>(pos)] + c_n_offsets[static_cast<size_t>(nb + j)];
                                            ValueType const *s = Cb + j * mc_len + pos;
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
                            // Multi-M/N: GEMM into a contiguous temp tile, then scatter to C.
                            //
                            // The scatter walks C's inner group in RUNS where that group's
                            // fastest index has unit stride: a run is the stretch of
                            // consecutive flat coordinates that stays inside one extent of
                            // that index, so within it C is contiguous and the update is a
                            // straight vector copy or add. The block strategy's scatter has
                            // done this all along; the tile scatter used to look every
                            // element up in the offset table, which at K=24 (the rank-6
                            // ccsd_t shapes) cost more than the 288 FMAs the tile computes
                            // and held the path to 15 GF/s where the same kernel reaches 60.
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

                                    // Scatter Ct back to C using the precomputed offset tables,
                                    // innermost along whichever of C's index groups is closer
                                    // packed. Ct is MR by NR and cache resident either way, so
                                    // reading it with a stride costs nothing.
                                    if (scatter_n_inner) {
                                        for (int64_t ii = 0; ii < mr_actual; ++ii) {
                                            int64_t const m_off = c_m_offsets[static_cast<size_t>(ir * MR + ii)];
                                            if (tile_n_unit) {
                                                int64_t pos = 0;
                                                while (pos < nr_actual) {
                                                    int64_t const n_global = nc + jr * NR + pos;
                                                    int64_t const run = std::min(tile_n_fast - (n_global % tile_n_fast), nr_actual - pos);
                                                    ValueType    *dst = C_data + m_off + c_n_offsets[static_cast<size_t>(jr * NR + pos)];
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
                                                    int64_t const run = std::min(tile_m_fast - (m_global % tile_m_fast), mr_actual - pos);
                                                    ValueType    *dst = C_data + c_m_offsets[static_cast<size_t>(ir * MR + pos)] + n_off;
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
                // Streaming stores are weakly ordered against everything else, so
                // this thread's share of C is not reliably visible until they have
                // drained. Once per N block, which is as rare as it can be while
                // still being inside the loop that did the writing.
                if (streamed_c) {
                    einsums::simd::stream_fence();
                }

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
///        calls that the key compares equal for. It also carries the site's
///        @ref KernelRoute pin, which decides vendor-versus-packed here instead
///        of the thread regime when it is set.
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

    // Which kernel this contraction is to be spent through. Resolved once, here,
    // and carried to every decision below that used to read the thread regime
    // for itself: the deferral to a direct vendor GEMM, and the two fast paths
    // inside blis_contraction.
    bool const prefer_packed = prefer_packed_route(site);

    // Memo hit: this caller already resolved this exact contraction, under the
    // same policy and the same route, against operands with this layout and
    // these sizes.
    //
    // The route is only part of the test for a DECLINE. A stored plan is a
    // packing topology and stays valid whichever way the contraction is spent,
    // but a decline is the engine standing aside for a vendor GEMM, which is
    // only right while the vendor is the route - so a decline recorded in one
    // regime must never turn a call away in the other.
    if (site != nullptr && site->resolved && site->allow_scatter == allow_scatter &&
        (site->plan != nullptr || site->declined_packed == prefer_packed) && site_key_matches(site->key, spec_in, st, A, B, *C)) {
        if (site->plan == nullptr) {
            ProfileAnnotate("packed_gemm_skip", "site_declined");
            return false;
        }
        ProfileAnnotate("packed_gemm_plan", "site");
        blis_contraction<ValueType>(*site->plan, C->data(), make_operand_view<ValueType>(A), make_operand_view<ValueType>(B),
                                    static_cast<ValueType>(AB_prefactor), static_cast<ValueType>(C_prefactor),
                                    site->plan->swap_ab ? spec_in.conj_b : spec_in.conj_a,
                                    site->plan->swap_ab ? spec_in.conj_a : spec_in.conj_b, prefer_packed);
        return true;
    }

    // Records what this call resolved to, so the next one can skip straight to
    // it. A null plan means "declined": that is a property of the key too, and
    // re-deriving it costs the same as finding a plan would. A decline also
    // records the route it was made under, which is what the reuse test above
    // re-checks.
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
        //
        // Not when the packed route is preferred: the direct GEMM this defers to
        // would be clamped to one thread under a node-scoped width, and on a
        // pinned node it is the side whose last bit moves with the thread count.
        // The decline recorded here carries the route it was made under, so a
        // memo written in one regime is never read in the other.
        if (m_count == 1 && n_count == 1 && link.size() == 1 && !spec.conj_a && !spec.conj_b && m_count + n_count == target.size() &&
            !prefer_packed) {
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
            coalesce_plan(computed, static_cast<int64_t>(sizeof(ValueType)));
            // Which operand takes the kernel's M role is a property of the
            // contraction and the resolved kernel, so it is settled once, here,
            // and cached with the plan. Doing it per call would rebuild the
            // plan's six stride vectors on every replay of a tiled node.
            if (mn_roles_should_swap(computed, micro_kernel_shape<ValueType>(),
                                     get_scalar_type<ValueType>() == ScalarType::Complex64 ||
                                         get_scalar_type<ValueType>() == ScalarType::Complex128)) {
                computed = transposed_plan(computed);
            }
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
        // always the generic loop. That is why this number is not a constant of
        // the engine: it moves whenever the generic loop does.
        //
        // Re-derived 2026-09-15 on the Zen+ box, after the generic loops were
        // ordered for the layout. An outer product has no link index, so all of
        // its target axes coalesce into one flat sweep - the single biggest case
        // that change helps - and the crossover moved by a factor of five:
        //
        //   elements     256    576    625    784    900    1296   2304
        //   speedup     0.40x  0.96x  0.94x  1.18x  1.22x  1.18x  1.28x
        //
        // (three runs, both shapes, membind + pinned). Generic wins at or below
        // 625; packed wins from 784 up. 768 sits in the gap. The old 4096 was
        // declining shapes the packed path wins by about 1.2x.
        //
        // The two measurements are from different machines, and the crossover
        // depends on the ratio of two implementations rather than on either one,
        // so it is genuinely per-target. 4096 may still be right on the M4.
        if (outer_shaped && plan.M_total * plan.N_total < kOuterProductFloor) {
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
        blis_contraction<ValueType>(plan, C->data(), make_operand_view<ValueType>(A), make_operand_view<ValueType>(B),
                                    static_cast<ValueType>(AB_prefactor), static_cast<ValueType>(C_prefactor),
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

// The engine is compiled ONCE per element type, in BlisContraction.cpp, and
// every consumer links to it instead of generating its own copy. Without these
// declarations a single test translation unit instantiated it 724 times - the
// optimizer expanded 207 KB of kernel per file, which was 70% of that file's
// compile time. The four types below are every type the packed path accepts.
//
// These must stay in step with the explicit instantiations in
// BlisContraction.cpp; a type declared here and not defined there is a link
// error, which is the failure mode you want.
extern template void blis_contraction<float>(PackingPlan const &, float *, OperandView<float> const &, OperandView<float> const &, float,
                                             float, bool, bool, bool);
extern template void blis_contraction<double>(PackingPlan const &, double *, OperandView<double> const &, OperandView<double> const &,
                                              double, double, bool, bool, bool);
extern template void blis_contraction<std::complex<float>>(PackingPlan const &, std::complex<float> *,
                                                           OperandView<std::complex<float>> const &,
                                                           OperandView<std::complex<float>> const &, std::complex<float>,
                                                           std::complex<float>, bool, bool, bool);
extern template void blis_contraction<std::complex<double>>(PackingPlan const &, std::complex<double> *,
                                                            OperandView<std::complex<double>> const &,
                                                            OperandView<std::complex<double>> const &, std::complex<double>,
                                                            std::complex<double>, bool, bool, bool);

EINSUMS_NAMESPACE_END(packed_gemm)
