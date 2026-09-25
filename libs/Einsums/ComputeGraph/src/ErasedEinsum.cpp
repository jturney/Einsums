//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Einsums/ComputeGraph/Detail/ErasedEinsum.hpp>
#include <Einsums/ComputeGraph/Detail/MixedPrecision.hpp>
#include <Einsums/ComputeGraph/Operations.hpp>
#include <Einsums/ComputeGraph/StringDispatch.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>

#include <complex>
#include <stdexcept>
#include <type_traits>

EINSUMS_NAMESPACE_BEGIN(compute_graph::dispatch)

namespace {

template <typename T>
using Impl = einsums::detail::TensorImpl<T>;

/// Views re-seated onto each call's operands. Constructing a view copies an operand's TensorImpl,
/// whose dims and strides are heap vectors, and three of those cost about 126 ns, a tenth of a small
/// contraction's whole eager call. Assigning into a view reuses the vectors it already holds, which
/// is how the replay executor has always done it.
template <typename TC, typename TA, typename TB>
struct Seats {
    RuntimeTensorView<TC> c;
    RuntimeTensorView<TA> a;
    RuntimeTensorView<TB> b;
    bool                  in_use = false;
};

/// Call @p fn with views over @p C, @p A and @p B. The views are this thread's reused ones, unless a
/// call on this thread is already using them, in which case fresh ones are built so the outer call's
/// are not re-seated under it.
template <typename TC, typename TA, typename TB, typename Fn>
void with_views(Impl<TC> &C, Impl<TA> const &A, Impl<TB> const &B, Fn &&fn) {
    thread_local Seats<TC, TA, TB> seats;
    if (seats.in_use) {
        RuntimeTensorView<TC> c(C);
        RuntimeTensorView<TA> a(A);
        RuntimeTensorView<TB> b(B);
        fn(c, a, b);
        return;
    }

    struct Release {
        bool &flag;
        ~Release() { flag = false; }
    };
    seats.in_use = true;
    Release const release{seats.in_use};

    seats.c.impl() = C;
    seats.a.impl() = A;
    seats.b.impl() = B;
    fn(seats.c, seats.a, seats.b);
}

} // namespace

template <typename T>
void erased_string_einsum(ParsedEinsumSpec const &parsed, T c_pf, Impl<T> &C, T ab_pf, Impl<T> const &A, Impl<T> const &B, bool conj_a,
                          bool conj_b) {
    with_views(C, A, B, [&](auto &c, auto &a, auto &b) { string_einsum(parsed, c_pf, &c, ab_pf, a, b, conj_a, conj_b); });
}

template <typename TC, typename TA, typename TB>
void erased_mixed_string_einsum(ParsedEinsumSpec const &parsed, TC c_pf, Impl<TC> &C, detail::PromoteT<TA, TB> ab_pf, Impl<TA> const &A,
                                Impl<TB> const &B, bool conj_a, bool conj_b) {
    if constexpr ((std::is_same_v<TC, TA> && std::is_same_v<TA, TB>) || !detail::storable_v<detail::PromoteT<TA, TB>, TC>) {
        // cg::einsum sends a uniform triple to erased_string_einsum and rejects a complex product
        // into a real output before it gets here, so reaching this is a caller's bug.
        (void)parsed, (void)c_pf, (void)C, (void)ab_pf, (void)A, (void)B, (void)conj_a, (void)conj_b;
        EINSUMS_THROW_EXCEPTION(std::logic_error, "erased_mixed_string_einsum: this operand type triple is not a mixed-precision einsum");
    } else {
        with_views(C, A, B, [&](auto &c, auto &a, auto &b) { mixed_string_einsum(parsed, c_pf, &c, ab_pf, a, b, conj_a, conj_b); });
    }
}

// The four element types, and every triple of them for the mixed-precision entry.
#define EINSUMS_ERASED_EINSUM(T)                                                                                                           \
    template EINSUMS_EXPORT void erased_string_einsum<T>(ParsedEinsumSpec const &, T, Impl<T> &, T, Impl<T> const &, Impl<T> const &,      \
                                                         bool, bool);
EINSUMS_ERASED_EINSUM(float)
EINSUMS_ERASED_EINSUM(double)
EINSUMS_ERASED_EINSUM(std::complex<float>)
EINSUMS_ERASED_EINSUM(std::complex<double>)
#undef EINSUMS_ERASED_EINSUM

#define EINSUMS_STRING_PERMUTE(T)                                                                                                          \
    template EINSUMS_EXPORT void string_permute_impl<T>(ParsedPermuteSpec const &, T, Impl<T> *, T, Impl<T> const &);
EINSUMS_STRING_PERMUTE(float)
EINSUMS_STRING_PERMUTE(double)
EINSUMS_STRING_PERMUTE(std::complex<float>)
EINSUMS_STRING_PERMUTE(std::complex<double>)
#undef EINSUMS_STRING_PERMUTE

#define EINSUMS_ERASED_MIXED(TC, TA, TB)                                                                                                   \
    template EINSUMS_EXPORT void erased_mixed_string_einsum<TC, TA, TB>(                                                                   \
        ParsedEinsumSpec const &, TC, Impl<TC> &, detail::PromoteT<TA, TB>, Impl<TA> const &, Impl<TB> const &, bool, bool);
#define EINSUMS_ERASED_MIXED_B(TC, TA)                                                                                                     \
    EINSUMS_ERASED_MIXED(TC, TA, float)                                                                                                    \
    EINSUMS_ERASED_MIXED(TC, TA, double)                                                                                                   \
    EINSUMS_ERASED_MIXED(TC, TA, std::complex<float>)                                                                                      \
    EINSUMS_ERASED_MIXED(TC, TA, std::complex<double>)
#define EINSUMS_ERASED_MIXED_A(TC)                                                                                                         \
    EINSUMS_ERASED_MIXED_B(TC, float)                                                                                                      \
    EINSUMS_ERASED_MIXED_B(TC, double)                                                                                                     \
    EINSUMS_ERASED_MIXED_B(TC, std::complex<float>)                                                                                        \
    EINSUMS_ERASED_MIXED_B(TC, std::complex<double>)
EINSUMS_ERASED_MIXED_A(float)
EINSUMS_ERASED_MIXED_A(double)
EINSUMS_ERASED_MIXED_A(std::complex<float>)
EINSUMS_ERASED_MIXED_A(std::complex<double>)
#undef EINSUMS_ERASED_MIXED_A
#undef EINSUMS_ERASED_MIXED_B
#undef EINSUMS_ERASED_MIXED

EINSUMS_NAMESPACE_END(compute_graph::dispatch)

EINSUMS_NAMESPACE_BEGIN(compute_graph::detail)

template <typename T>
void capture_string_einsum(CaptureContext &ctx, ParsedEinsumSpec const &parsed, T c_pf, T ab_pf, bool conj_a, bool conj_b,
                           CapturedOperand<T> const &a, CapturedOperand<T> const &b, CapturedOperand<T> const &c) {
    TensorId const    a_id   = a.id;
    TensorId const    b_id   = b.id;
    TensorId const    c_id   = c.id;
    TensorSlot *const a_slot = a.slot;
    TensorSlot *const b_slot = b.slot;
    TensorSlot *const c_slot = c.slot;

    // The descriptor's snapshot, then the live params/indices/site blocks the executor shares with it,
    // seeded from that snapshot by the one helper every einsum builder uses.
    auto desc = detail::build_einsum_descriptor(parsed, c_pf, ab_pf, conj_a, conj_b);
    detail::attach_live_state(desc, parsed.raw);

    // Index-space binding (design part 1.3). A space is a property of the SLOT
    // an index occupies, not of the letter globally, so the letters of this one
    // contraction are resolved against the operands' annotations here and the
    // result is stored per node. Costs nothing for a program that annotates
    // nothing: every operand's `spaces` is empty and the map comes back empty.
    // Runs before the GEMM/batched fast paths below so that a node recorded
    // through one of them still leaves the output's inferred annotation behind.
    desc.letter_spaces =
        detail::bind_einsum_spaces(*ctx.graph(), a_id, b_id, c_id, parsed.a_indices, parsed.b_indices, parsed.c_indices, "cg::einsum");

    // BLAS-level batching hint. Derived by @ref derive_gemm_hint, the one
    // function capture and Graph::make_einsum_node share: the gate, the m/n/k
    // arithmetic and the roles clause used to be duplicated here and there, and
    // a hint describing a matrix product the einsum does not perform stays
    // invisible until GEMMBatching forms a batch from it.
    desc.gemm_hint = derive_gemm_hint(packed_gemm::get_scalar_type<T>(), desc.spec, *ctx.graph(), a_id, b_id, c_id);

    // A permute_view keeps the storage-order FLAG of its parent but presents
    // reordered strides, so is_row_major()/is_column_major() alone cannot
    // prove the canonical layout the fast paths below assume (found by the
    // large-rank differential fuzzer: a view with the slice axes swapped
    // passed the flag gates and produced wrong results). Verify the strides
    // are actually monotone in the flag's direction; size-1 axes are never
    // traversed, so their (possibly inflated) strides are ignored.
    auto layout_matches_flag = [](auto const &impl) {
        bool const   rm    = impl.is_row_major();
        size_t const rank  = impl.rank();
        size_t       prev  = 0;
        bool         first = true;
        for (size_t n = 0; n < rank; ++n) {
            size_t const d = rm ? rank - 1 - n : n;
            if (impl.dim(d) <= 1)
                continue;
            size_t const st = impl.stride(d);
            if (!first && st < prev)
                return false;
            prev  = st;
            first = false;
        }
        return true;
    };

    // ────────────────────────────────────────────────────────────────────
    // Strided-batched GEMM fast path for 3D×3D→3D with a batch index.
    // ────────────────────────────────────────────────────────────────────
    //
    // If the einsum expresses a batched matrix multiply, a 3D tensor
    // where one index appears in A, B, AND C (the batch) and the other
    // three indices form a standard 2D GEMM pattern (target-A, link,
    // target-B), we can collapse N per-batch 2D gemms into a single
    // `blas::gemm_batch` call at execute time. This is the same layout
    // `cublasDgemmStridedBatched` expects on GPU, so the descriptor
    // carries enough info to dispatch there too once the GPU backend is
    // wired up.
    //
    // Both conventions are supported so users don't have to transpose
    // their data to match some arbitrary choice:
    //   - Row-major tensors with batch(es) at the FIRST axes
    //     (e.g. "bij;bjk->bik" shape (B, M, K); or "abij;abjk->abik"
    //     shape (A, B, M, K)), the ML/CUDA convention
    //   - Column-major tensors with batch(es) at the LAST axes
    //     (e.g. "ijb;jkb->ikb" shape (M, K, B); or "ijab;jkab->ikab"
    //     shape (M, K, A, B)), Einsums's default layout
    //
    // Multiple batch indices (rank 4+) are flattened: N batch dims with
    // sizes (d1, d2, ..., dN) become a single effective batch of size
    // prod(di) with uniform stride equal to the product of the per-slice
    // 2D dims. This works as long as all batch indices appear in the
    // outermost contiguous region in memory, in the same relative order
    // across A, B, C, typical of how tensors carry "free" axes like
    // (head, layer, sample, fragment) through to the matmul.
    //
    // In either case the 2D slice at each flat batch index is a
    // contiguous block of memory. Arrangements that don't match (batches
    // at the wrong end for the layout, or reordered across operands)
    // interleave batches in memory; those fall through to the generic
    // string_einsum executor.
    if (auto const ar = a.impl->rank(); ar >= 3 && ar == b.impl->rank() && ar == c.impl->rank()) {
        std::size_t const Rank = ar;
        if (parsed.a_indices.size() == Rank && parsed.b_indices.size() == Rank && parsed.c_indices.size() == Rank &&
            desc.spec.link_indices.size() == 1) {

            std::string const &link = desc.spec.link_indices[0];

            auto find_pos = [](std::vector<std::string> const &idx, std::string const &name) -> int {
                for (int i = 0; std::cmp_less(i, idx.size()); ++i)
                    if (idx[i] == name)
                        return i;
                return -1;
            };

            // Collect batch indices: those appearing in A, B, AND C
            // (and not being the link). Preserve A's order so
            // "abij;abjk->abik" gives batch_names = [a, b] and we can
            // enforce matching positions across operands.
            std::vector<std::string> batch_names;
            for (auto const &idx : parsed.a_indices) {
                auto in_b = std::ranges::find(parsed.b_indices, idx) != parsed.b_indices.end();
                auto in_c = std::ranges::find(parsed.c_indices, idx) != parsed.c_indices.end();
                if (in_b && in_c && idx != link)
                    batch_names.push_back(idx);
            }

            // Each tensor has batch indices + 1 link + 1 target (per A or B),
            // or batch indices + 2 targets (for C). So num_batch == Rank - 2.
            bool const shape_ok = batch_names.size() == Rank - 2;

            // Batch indices must appear at the same positions in all three
            // operands: otherwise flattening the batch doesn't produce
            // consistent strides. Collect those positions (same for A, B, C).
            std::vector<int> batch_positions;
            batch_positions.reserve(batch_names.size());
            bool positions_match = shape_ok;
            for (auto const &bname : batch_names) {
                int pa = find_pos(parsed.a_indices, bname);
                int pb = find_pos(parsed.b_indices, bname);
                int pc = find_pos(parsed.c_indices, bname);
                if (pa < 0 || pa != pb || pa != pc) {
                    positions_match = false;
                    break;
                }
                batch_positions.push_back(pa);
            }

            // Mode selection: for stride math to work with a single
            // batch_stride, the batch axes must form a contiguous
            // outermost block. Row-major outermost = [0..num_batch-1];
            // col-major outermost = [rank-num_batch..rank-1].
            bool const all_contig    = a.impl->is_contiguous() && b.impl->is_contiguous() && c.impl->is_contiguous() &&
                                       layout_matches_flag(*a.impl) && layout_matches_flag(*b.impl) && layout_matches_flag(*c.impl);
            bool const all_row_major = a.impl->is_row_major() && b.impl->is_row_major() && c.impl->is_row_major();
            bool const all_col_major = a.impl->is_column_major() && b.impl->is_column_major() && c.impl->is_column_major();

            auto is_prefix_range = [&](std::vector<int> const &positions, size_t count) {
                if (positions.size() != count)
                    return false;
                for (size_t i = 0; i < count; ++i)
                    if (std::cmp_not_equal(positions[i], i))
                        return false;
                return true;
            };
            auto is_suffix_range = [&](std::vector<int> const &positions, size_t count, size_t rank) {
                if (positions.size() != count)
                    return false;
                for (size_t i = 0; i < count; ++i)
                    if (std::cmp_not_equal(positions[i], rank - count + i))
                        return false;
                return true;
            };

            bool const row_mode = positions_match && all_row_major && is_prefix_range(batch_positions, batch_names.size());
            bool const col_mode = positions_match && all_col_major && is_suffix_range(batch_positions, batch_names.size(), Rank);

            // Conjugated batched einsums skip this gemm_batch fast path (it only
            // emits 'N'/'T' trans, never conjugation) and fall through to the
            // conj-aware generic string_einsum executor below.
            if (shape_ok && positions_match && all_contig && (row_mode || col_mode) && !desc.params->conj_a && !desc.params->conj_b) {
                // Non-batch indices in original order: strip the batch positions.
                // For row_mode they're the LAST 2 positions; for col_mode the FIRST 2.
                std::vector<std::string> a_rest, b_rest;
                if (row_mode) {
                    a_rest = {parsed.a_indices[Rank - 2], parsed.a_indices[Rank - 1]};
                    b_rest = {parsed.b_indices[Rank - 2], parsed.b_indices[Rank - 1]};
                } else {
                    a_rest = {parsed.a_indices[0], parsed.a_indices[1]};
                    b_rest = {parsed.b_indices[0], parsed.b_indices[1]};
                }

                // The descriptor below requires C's two non-batch slice axes in
                // canonical (M, N) order -- M (shared with A) first, N (shared with
                // B) second. Both modes assume this: col_mode maps it to BLAS m/n
                // directly; row_mode emits the transposed product (so it swaps m/n
                // and trans_a/trans_b) to honor row-major storage, but still on a
                // canonical (M, N) output. A transposed output -- e.g.
                // "kji <- jli ; lki", whose slice is (N, M) -- would mis-map m/n
                // against the operands and gemm_batch would silently miscompute
                // (often to zero), so detect it and fall through to the generic
                // einsum (string_einsum) below.
                std::vector<std::string> const c_rest =
                    row_mode ? std::vector<std::string>{parsed.c_indices[Rank - 2], parsed.c_indices[Rank - 1]}
                             : std::vector<std::string>{parsed.c_indices[0], parsed.c_indices[1]};
                std::string const m_index      = (a_rest[0] == link) ? a_rest[1] : a_rest[0];
                std::string const n_index      = (b_rest[0] == link) ? b_rest[1] : b_rest[0];
                bool const        canonical_mn = (c_rest[0] == m_index && c_rest[1] == n_index);
                if (canonical_mn) {

                    // 2D slice dim lookups: positions of the non-batch axes in the
                    // original tensor. row_mode: positions (Rank-2, Rank-1);
                    // col_mode: positions (0, 1).
                    auto a_slice_dim = [&](int local_pos) -> int {
                        int orig = row_mode ? static_cast<int>(Rank - 2) + local_pos : local_pos;
                        return static_cast<int>(a.impl->dim(orig));
                    };
                    auto b_slice_dim = [&](int local_pos) -> int {
                        int orig = row_mode ? static_cast<int>(Rank - 2) + local_pos : local_pos;
                        return static_cast<int>(b.impl->dim(orig));
                    };
                    auto c_slice_dim = [&](int local_pos) -> int {
                        int orig = row_mode ? static_cast<int>(Rank - 2) + local_pos : local_pos;
                        return static_cast<int>(c.impl->dim(orig));
                    };

                    // Flat batch count = product of each batch dim's size. Same
                    // answer whether we read from A, B, or C since the sizes
                    // must agree at construction time (shape compatibility).
                    std::int64_t flat_batch = 1;
                    for (int p : batch_positions)
                        flat_batch *= static_cast<std::int64_t>(a.impl->dim(p));

                    BatchedGemmDescriptor d;
                    d.scalar = blas_scalar_of<T>();

                    char natural_trans_a = (a_rest[0] == link) ? 'T' : 'N';
                    char natural_trans_b = (b_rest[1] == link) ? 'T' : 'N';

                    if (col_mode) {
                        d.trans_a = natural_trans_a;
                        d.trans_b = natural_trans_b;
                        d.m       = c_slice_dim(0);
                        d.n       = c_slice_dim(1);
                        d.k       = (natural_trans_a == 'N') ? a_slice_dim(1) : a_slice_dim(0);
                        d.lda     = a_slice_dim(0);
                        d.ldb     = b_slice_dim(0);
                        d.ldc     = c_slice_dim(0);
                    } else {
                        d.trans_a = natural_trans_b;
                        d.trans_b = natural_trans_a;
                        d.m       = c_slice_dim(1);
                        d.n       = c_slice_dim(0);
                        d.k       = (natural_trans_a == 'N') ? a_slice_dim(1) : a_slice_dim(0);
                        d.lda     = b_slice_dim(1);
                        d.ldb     = a_slice_dim(1);
                        d.ldc     = c_slice_dim(1);
                    }

                    d.alpha          = as<std::complex<double>>(desc.params->ab_pf);
                    d.beta           = as<std::complex<double>>(desc.params->c_pf);
                    d.batch_count    = static_cast<int>(flat_batch);
                    d.strided        = true;
                    d.batch_stride_a = static_cast<std::int64_t>(a_slice_dim(0)) * static_cast<std::int64_t>(a_slice_dim(1));
                    d.batch_stride_b = static_cast<std::int64_t>(b_slice_dim(0)) * static_cast<std::int64_t>(b_slice_dim(1));
                    d.batch_stride_c = static_cast<std::int64_t>(c_slice_dim(0)) * static_cast<std::int64_t>(c_slice_dim(1));

                    bool const swap_ab = row_mode;

                    // The per-slice pointer tables are pure functions of the
                    // three base pointers (base + i*stride); rebuild them only
                    // when a rebind moves a base, not on every replay. One
                    // cache per captured node; a node never runs concurrently
                    // with itself, so no synchronization is needed.
                    struct BatchPtrTables {
                        T const               *base_a{nullptr};
                        T const               *base_b{nullptr};
                        T                     *base_c{nullptr};
                        std::vector<T const *> a_arr;
                        std::vector<T const *> b_arr;
                        std::vector<T *>       c_arr;
                    };
                    auto tables = std::make_shared<BatchPtrTables>();

                    // Each operand's live TensorImpl through its slot, as the graph's own executors read it, so
                    // a rebind or the memory arena is honored and nothing here depends on the caller's
                    // tensor type.
                    OperandAccessor const a_access(a_slot, packed_gemm::get_scalar_type<T>());
                    OperandAccessor const b_access(b_slot, packed_gemm::get_scalar_type<T>());
                    OperandAccessor const c_access(c_slot, packed_gemm::get_scalar_type<T>());
                    auto                  executor = [d, swap_ab, a_access, b_access, c_access, tables]() {
                        LabeledSection("einsum batched execute");
                        ProfileAnnotate("m", static_cast<int64_t>(d.m));
                        ProfileAnnotate("n", static_cast<int64_t>(d.n));
                        ProfileAnnotate("k", static_cast<int64_t>(d.k));
                        ProfileAnnotate("batch", static_cast<int64_t>(d.batch_count));
                        auto const *base_a = static_cast<T const *>(a_access.impl<T>()->data());
                        auto const *base_b = static_cast<T const *>(b_access.impl<T>()->data());
                        auto       *base_c = static_cast<T *>(c_access.impl<T>()->data());

                        if (base_a != tables->base_a || base_b != tables->base_b || base_c != tables->base_c) {
                            tables->a_arr.resize(d.batch_count);
                            tables->b_arr.resize(d.batch_count);
                            tables->c_arr.resize(d.batch_count);
                            for (int i = 0; i < d.batch_count; ++i) {
                                tables->a_arr[i] = base_a + i * d.batch_stride_a;
                                tables->b_arr[i] = base_b + i * d.batch_stride_b;
                                tables->c_arr[i] = base_c + i * d.batch_stride_c;
                            }
                            tables->base_a = base_a;
                            tables->base_b = base_b;
                            tables->base_c = base_c;
                        }

                        T const **blas_a = swap_ab ? tables->b_arr.data() : tables->a_arr.data();
                        T const **blas_b = swap_ab ? tables->a_arr.data() : tables->b_arr.data();

                        if constexpr (std::is_same_v<T, std::complex<float>> || std::is_same_v<T, std::complex<double>>) {
                            using R = typename T::value_type;
                            T alpha{static_cast<R>(d.alpha.real()), static_cast<R>(d.alpha.imag())};
                            T beta{static_cast<R>(d.beta.real()), static_cast<R>(d.beta.imag())};
                            blas::gemm_batch<T>(d.trans_a, d.trans_b, d.m, d.n, d.k, alpha, blas_a, d.lda, blas_b, d.ldb, beta,
                                                tables->c_arr.data(), d.ldc, d.batch_count);
                        } else {
                            blas::gemm_batch<T>(d.trans_a, d.trans_b, d.m, d.n, d.k, static_cast<T>(d.alpha.real()), blas_a, d.lda, blas_b,
                                                d.ldb, static_cast<T>(d.beta.real()), tables->c_arr.data(), d.ldc, d.batch_count);
                        }
                    };

                    auto label = fmt::format("gemm_batch_strided x{} ({}-major, batch={}, M={}, K={}, N={})", d.batch_count,
                                             col_mode ? "col" : "row", fmt::join(batch_names, ","), d.m, d.k, d.n);
                    ctx.record(OpKind::BatchedGemm, std::move(label), {a_id, b_id}, {c_id}, std::move(executor), std::move(d));
                    return;
                } // canonical_mn, otherwise fall through to the generic einsum
            }
        }
    }

    auto label = fmt::format("einsum: C[{}] = A[{}] * B[{}]", fmt::join(parsed.c_indices, ","), fmt::join(parsed.a_indices, ","),
                             fmt::join(parsed.b_indices, ","));

    // The node's operand lists, in the order the builder reads them: A, B from
    // the inputs and C from the outputs. Capture records the two inputs only;
    // the RMW repeat of an accumulating destination is Graph::make_einsum_node's
    // convention, and the builder ignores that trailing position either way.
    std::vector<TensorId> const node_inputs{a_id, b_id};
    std::vector<TensorId> const node_outputs{c_id};

    // The executor comes from build_executor, so capture, a pass that rewrites
    // this node, and a future loader all reach one lowering (design part 3.2).
    // The descriptor is handed over as the OpData the node will carry, so the
    // live params/indices/site handles the builder reads are the very ones the
    // node records.
    OpData op_data{std::move(desc)};
    auto   executor = build_executor(OpKind::Einsum, packed_gemm::get_scalar_type<T>(), c.impl->rank(), op_data, *ctx.graph(),
                                     std::span<TensorId const>{node_inputs}, std::span<TensorId const>{node_outputs});

    ctx.record(OpKind::Einsum, std::move(label), node_inputs, node_outputs, std::move(executor), std::move(op_data));
}

template <typename TC, typename TA, typename TB>
void capture_mixed_string_einsum(CaptureContext &ctx, ParsedEinsumSpec const &parsed, TC c_pf, PromoteT<TA, TB> ab_pf, bool conj_a,
                                 bool conj_b, CapturedOperand<TA> const &a, CapturedOperand<TB> const &b, CapturedOperand<TC> const &c) {
    if constexpr ((std::is_same_v<TC, TA> && std::is_same_v<TA, TB>) || !storable_v<PromoteT<TA, TB>, TC>) {
        // As erased_mixed_string_einsum: cg::einsum never sends these here.
        (void)ctx, (void)parsed, (void)c_pf, (void)ab_pf, (void)conj_a, (void)conj_b, (void)a, (void)b, (void)c;
        EINSUMS_THROW_EXCEPTION(std::logic_error, "capture_mixed_string_einsum: this operand type triple is not a mixed-precision einsum");
    } else {
        TensorId const a_id = a.id;
        TensorId const b_id = b.id;
        TensorId const c_id = c.id;

        auto desc = detail::build_einsum_descriptor(parsed, c_pf, ab_pf, conj_a, conj_b);
        detail::attach_live_state(desc, parsed.raw);
        desc.letter_spaces =
            detail::bind_einsum_spaces(*ctx.graph(), a_id, b_id, c_id, parsed.a_indices, parsed.b_indices, parsed.c_indices, "cg::einsum");

        auto label = fmt::format("einsum: C[{}] = A[{}] * B[{}]", fmt::join(parsed.c_indices, ","), fmt::join(parsed.a_indices, ","),
                                 fmt::join(parsed.b_indices, ","));
        std::vector<TensorId> const node_inputs{a_id, b_id};
        std::vector<TensorId> const node_outputs{c_id};

        // The node's dtype is C's, as for every einsum node; the executor does not rely on it for a
        // mixed one.
        OpData op_data{std::move(desc)};
        auto   executor = build_executor(OpKind::Einsum, packed_gemm::get_scalar_type<TC>(), c.impl->rank(), op_data, *ctx.graph(),
                                         std::span<TensorId const>{node_inputs}, std::span<TensorId const>{node_outputs});
        ctx.record(OpKind::Einsum, std::move(label), node_inputs, node_outputs, std::move(executor), std::move(op_data));
    }
}

#define EINSUMS_CAPTURE_EINSUM(T)                                                                                                          \
    template EINSUMS_EXPORT void capture_string_einsum<T>(CaptureContext &, ParsedEinsumSpec const &, T, T, bool, bool,                    \
                                                          CapturedOperand<T> const &, CapturedOperand<T> const &,                          \
                                                          CapturedOperand<T> const &);
EINSUMS_CAPTURE_EINSUM(float)
EINSUMS_CAPTURE_EINSUM(double)
EINSUMS_CAPTURE_EINSUM(std::complex<float>)
EINSUMS_CAPTURE_EINSUM(std::complex<double>)
#undef EINSUMS_CAPTURE_EINSUM

#define EINSUMS_CAPTURE_MIXED(TC, TA, TB)                                                                                                  \
    template EINSUMS_EXPORT void capture_mixed_string_einsum<TC, TA, TB>(CaptureContext &, ParsedEinsumSpec const &, TC, PromoteT<TA, TB>, \
                                                                         bool, bool, CapturedOperand<TA> const &,                          \
                                                                         CapturedOperand<TB> const &, CapturedOperand<TC> const &);
#define EINSUMS_CAPTURE_MIXED_B(TC, TA)                                                                                                    \
    EINSUMS_CAPTURE_MIXED(TC, TA, float)                                                                                                   \
    EINSUMS_CAPTURE_MIXED(TC, TA, double)                                                                                                  \
    EINSUMS_CAPTURE_MIXED(TC, TA, std::complex<float>)                                                                                     \
    EINSUMS_CAPTURE_MIXED(TC, TA, std::complex<double>)
#define EINSUMS_CAPTURE_MIXED_A(TC)                                                                                                        \
    EINSUMS_CAPTURE_MIXED_B(TC, float)                                                                                                     \
    EINSUMS_CAPTURE_MIXED_B(TC, double)                                                                                                    \
    EINSUMS_CAPTURE_MIXED_B(TC, std::complex<float>)                                                                                       \
    EINSUMS_CAPTURE_MIXED_B(TC, std::complex<double>)
EINSUMS_CAPTURE_MIXED_A(float)
EINSUMS_CAPTURE_MIXED_A(double)
EINSUMS_CAPTURE_MIXED_A(std::complex<float>)
EINSUMS_CAPTURE_MIXED_A(std::complex<double>)
#undef EINSUMS_CAPTURE_MIXED_A
#undef EINSUMS_CAPTURE_MIXED_B
#undef EINSUMS_CAPTURE_MIXED

EINSUMS_NAMESPACE_END(compute_graph::detail)
