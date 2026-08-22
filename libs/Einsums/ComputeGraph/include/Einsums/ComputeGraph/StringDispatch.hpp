//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/**
 * @file StringDispatch.hpp
 * @brief Runtime dispatch for string-based einsum contractions.
 *
 * Dispatches based on the parsed string specification to the appropriate
 * BLAS routine: DOT, GER, GEMV, GEMM, direct product, or throws for
 * unsupported patterns.
 */

#include <Einsums/BLAS/ThreadControl.hpp>
#include <Einsums/ComputeGraph/EinsumSpec.hpp>
#include <Einsums/ComputeGraph/TensorRank.hpp>
#include <Einsums/Concepts/TensorConcepts.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/LinearAlgebra.hpp>
#include <Einsums/PackedGemm/EinsumPackedGemm.hpp>
#include <Einsums/Profile.hpp>
#include <Einsums/TensorAlgebra/Permute.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph::dispatch)

/**
 * @brief Execute a tensor contraction described by a string specification.
 *
 * Classifies the contraction pattern at runtime and dispatches to the
 * most efficient BLAS routine.
 *
 * **Supported patterns:**
 * - GEMM: rank-2 × rank-2 → rank-2, one link index
 * - GEMV: rank-2 × rank-1 → rank-1 (or reverse), one link index
 * - GER: rank-1 × rank-1 → rank-2, no link indices (outer product)
 * - DOT: rank-1 × rank-1 → scalar, all indices contracted
 * - Direct product: same indices on A, B, C (element-wise)
 */

// ── GEMM dispatch (rank-2 × rank-2 → rank-2) ───────────────────────────────

template <typename T, MatrixConcept AType, MatrixConcept BType, MatrixConcept CType>
void string_gemm(ParsedEinsumSpec const &parsed, std::string const &k, T c_pf, CType *C, T ab_pf, AType const &A, BType const &B) {
    // `k` is the single link index; string_einsum classified the contraction
    // (links.size() == 1) before dispatching here, so it is not recomputed.

    // A GEMM op(A)·op(B) yields rows = A's free (non-link) index, cols = B's free
    // index. The output may be requested in EITHER order, so honor c_indices:
    //   C = [freeA, freeB] -> op(A)·op(B)
    //   C = [freeB, freeA] -> op(B)·op(A)   (the transposed product)
    // Without this, a transposed-output contraction (e.g. "ia <- ma ; mi") writes
    // the result with swapped dimensions and trips gemm's dimension check.
    std::string const &freeA = (parsed.a_indices[0] == k) ? parsed.a_indices[1] : parsed.a_indices[0];

    if (parsed.c_indices[0] == freeA) {
        // op(A) must be [freeA, k]; op(B) must be [k, freeB].
        char ta = (parsed.a_indices[0] == k) ? 't' : 'n';
        char tb = (parsed.b_indices[1] == k) ? 't' : 'n';
        linear_algebra::gemm(ta, tb, ab_pf, A, B, c_pf, C);
    } else {
        // Transposed output C = [freeB, freeA]: op(B) must be [freeB, k], op(A) [k, freeA].
        char tb = (parsed.b_indices[0] == k) ? 't' : 'n';
        char ta = (parsed.a_indices[1] == k) ? 't' : 'n';
        linear_algebra::gemm(tb, ta, ab_pf, B, A, c_pf, C);
    }
}

// ── GEMV dispatch (rank-2 × rank-1 → rank-1) ───────────────────────────────

template <typename T, MatrixConcept MatType, VectorConcept VecType, VectorConcept OutType>
void string_gemv_mat_vec(ParsedEinsumSpec const & /*parsed*/, T c_pf, OutType *out, T ab_pf, MatType const &mat, VecType const &vec,
                         std::vector<std::string> const &mat_idx, std::string const &link) {
    char trans = (mat_idx[0] == link) ? 't' : 'n';
    linear_algebra::gemv(trans, ab_pf, mat, vec, c_pf, out);
}

// ── Generic nested-loop contraction ─────────────────────────────────────

/// Name of the kernel route the most recent string_einsum call on this
/// thread selected ("packed_gemm", "gemv_mat_vec", "generic_loop",
/// "generic_loop_repeated_indices", "empty_input_scale_only", ...).
///
/// Test introspection ONLY: lets dispatch-coverage tests assert the intended
/// fast path fired instead of a silent generic-loop fallback, mirroring the
/// eager API's AlgorithmChoice out-parameter. Thread-local; not an API for
/// steering execution.
inline char const *&last_dispatch_route() {
    thread_local char const *route = "none";
    return route;
}

/**
 * @brief Generic runtime nested-loop contraction for arbitrary rank/pattern.
 *
 * Handles any contraction pattern by iterating over all target index
 * combinations (outer loops) and link index combinations (inner summation).
 *
 * Performance note: This is O(product(target_dims) * product(link_dims)),
 * with no BLAS optimization. Use for patterns not covered by specialized
 * BLAS dispatch (rank-3+, multi-link, etc.).
 *
 * Everything that does not vary per element is hoisted: each index carries the
 * SUM of its strides in each operand (which is what makes a repeated letter a
 * diagonal walk) and its slot number, and the loops are odometers stepping those
 * strides rather than rebuilding an offset per element. Recomputing offsets
 * inside the loop cost this routine roughly 25 ns an element -- on a
 * ``ijab <- ia ; jb`` outer product over a 26-orbital CCSD residual, 0.63 ms for
 * 51 kflop. The iteration order is unchanged, last index fastest, so the
 * summation order and therefore the result are bit-for-bit what they were.
 */
template <BasicTensorConcept AType, BasicTensorConcept BType, BasicTensorConcept CType>
    requires requires {
        requires std::is_same_v<typename AType::ValueType, typename BType::ValueType>;
        requires std::is_same_v<typename AType::ValueType, typename CType::ValueType>;
    }
void generic_string_einsum(ParsedEinsumSpec const &parsed, std::vector<std::string> const &links, typename AType::ValueType c_pf, CType *C,
                           typename AType::ValueType ab_pf, AType const &A, BType const &B, bool conj_a = false, bool conj_b = false) {
    using T = typename AType::ValueType;

    auto const &c_idx = parsed.c_indices;
    auto const &a_idx = parsed.a_indices;
    auto const &b_idx = parsed.b_indices;

    // Build a master index table: for each unique index name, store its dimension size
    // and its position in A, B, C (or -1 if not present).
    struct IndexInfo {
        std::string name;
        size_t      dim_size{0};
        // EVERY position the index occupies in each tensor's index list.
        // Repeated letters within one operand ('ii <- ...') mean a diagonal
        // access: all occurrences advance together, so offsets accumulate
        // the strides of every occurrence (empty when absent).
        std::vector<int> pos_in_a;
        std::vector<int> pos_in_b;
        std::vector<int> pos_in_c;
        bool             is_link{false}; // True if this is a contraction (link) index
    };

    // Collect all unique indices preserving target-first, link-second order
    auto                   targets = parsed.target_indices();
    std::vector<IndexInfo> all_indices;
    std::set<std::string>  seen;

    auto add_index = [&](std::string const &name, bool link) {
        if (seen.count(name))
            return;
        seen.insert(name);
        IndexInfo info;
        info.name    = name;
        info.is_link = link;
        // Record every occurrence in each tensor's index list
        for (size_t p = 0; p < a_idx.size(); p++) {
            if (a_idx[p] == name) {
                info.pos_in_a.push_back(static_cast<int>(p));
            }
        }
        for (size_t p = 0; p < b_idx.size(); p++) {
            if (b_idx[p] == name) {
                info.pos_in_b.push_back(static_cast<int>(p));
            }
        }
        for (size_t p = 0; p < c_idx.size(); p++) {
            if (c_idx[p] == name) {
                info.pos_in_c.push_back(static_cast<int>(p));
            }
        }
        // Get dimension size from whichever tensor has this index
        if (!info.pos_in_a.empty())
            info.dim_size = A.dim(info.pos_in_a.front());
        else if (!info.pos_in_b.empty())
            info.dim_size = B.dim(info.pos_in_b.front());
        else if (!info.pos_in_c.empty())
            info.dim_size = C->dim(info.pos_in_c.front());
        all_indices.push_back(info);
    };

    for (auto const &t : targets)
        add_index(t, false);
    for (auto const &l : links)
        add_index(l, true);
    // Letters appearing in only ONE input and not in C (trace letters,
    // e.g. the doubled i of "j <- iik ; kj") are in neither `targets` nor
    // `links` (which holds A-and-B letters only). Einsum semantics sum
    // them; add them as link (summed) indices so their axes iterate.
    for (auto const &name : a_idx)
        add_index(name, true);
    for (auto const &name : b_idx)
        add_index(name, true);

    // Separate into target and link index groups
    std::vector<IndexInfo const *> target_infos, link_infos;
    for (auto const &info : all_indices) {
        if (info.is_link)
            link_infos.push_back(&info);
        else
            target_infos.push_back(&info);
    }

    // Compute total iterations
    size_t target_total = 1;
    for (auto *ti : target_infos)
        target_total *= ti->dim_size;
    size_t link_total = 1;
    for (auto *li : link_infos)
        link_total *= li->dim_size;

    // The dispatcher permits exactly one in-place shape: C aliasing an operand
    // whose index list is IDENTICAL to C's, on the grounds that every element is
    // then read immediately before its own overwrite. That holds for the
    // elementwise BLAS routes, but NOT here - this loop clears C before reading
    // anything, so an aliased operand would be read back as zeros and the whole
    // result would be zero ("ab <- ab ; b" with C aliasing A hit exactly this).
    // Snapshot any operand whose storage overlaps C's and read the copy, which
    // leaves the loop below (and so the summation order, and so the exact
    // floating-point result) untouched.
    //
    // Guarded on a non-empty iteration space: the loop reads nothing when
    // either total is zero, and that is also the only way an operand can carry
    // a zero extent, which the span arithmetic below cannot represent.
    std::vector<T> a_snapshot, b_snapshot;
    T const       *a_data = A.data();
    T const       *b_data = B.data();
    if (target_total != 0 && link_total != 0) {
        auto const span_of = [](auto const &t) {
            size_t last = 0;
            for (size_t d = 0; d < detail::tensor_rank(t); d++) {
                last += (t.dim(d) - 1) * t.stride(d);
            }
            return last + 1;
        };
        // Interval intersection, deliberately conservative: unlike the
        // dispatcher's guard this only decides whether to take a copy, so a
        // false positive costs one allocation rather than a spurious throw.
        T const     *c_lo       = C->data();
        size_t const c_span     = span_of(*C);
        auto const   overlaps_c = [&](auto const &t) {
            T const *lo = t.data();
            return lo < c_lo + c_span && c_lo < lo + span_of(t);
        };
        if (overlaps_c(A)) {
            a_snapshot.assign(a_data, a_data + span_of(A));
            a_data = a_snapshot.data();
        }
        if (overlaps_c(B)) {
            b_snapshot.assign(b_data, b_data + span_of(B));
            b_data = b_snapshot.data();
        }
    }

    // Scale C by c_pf
    if (c_pf == T{0}) {
        C->zero();
    } else if (c_pf != T{1}) {
        linear_algebra::scale(c_pf, C);
    }

    // Per-index step in each operand: the sum over every position the index
    // occupies there, so advancing it once walks all of them together. This is
    // the whole of the per-element index arithmetic, computed once.
    size_t const        n_index = all_indices.size();
    std::vector<size_t> a_step(n_index, 0), b_step(n_index, 0), c_step(n_index, 0);
    for (size_t i = 0; i < n_index; i++) {
        for (int pos : all_indices[i].pos_in_a) {
            a_step[i] += A.stride(pos);
        }
        for (int pos : all_indices[i].pos_in_b) {
            b_step[i] += B.stride(pos);
        }
        for (int pos : all_indices[i].pos_in_c) {
            c_step[i] += C->stride(pos);
        }
    }

    // Odometer axes, outermost first. target_infos / link_infos point into
    // all_indices, so their slot is a subtraction rather than a search.
    struct Axis {
        size_t extent{0};
        size_t a_step{0}, b_step{0}, c_step{0};
    };
    auto axes_of = [&](std::vector<IndexInfo const *> const &infos) {
        std::vector<Axis> axes;
        axes.reserve(infos.size());
        for (auto const *info : infos) {
            auto const slot = static_cast<size_t>(info - all_indices.data());
            axes.push_back({.extent = info->dim_size, .a_step = a_step[slot], .b_step = b_step[slot], .c_step = c_step[slot]});
        }
        return axes;
    };
    std::vector<Axis> const target_axes = axes_of(target_infos);
    std::vector<Axis> const link_axes   = axes_of(link_infos);

    // Step an odometer whose LAST axis runs fastest, keeping the running offsets
    // in step with it. Order matters beyond taste: it fixes the order the link
    // sum accumulates in, and therefore the floating-point result.
    auto advance = [](std::vector<Axis> const &axes, std::vector<size_t> &value, size_t &a_off, size_t &b_off, size_t &c_off) {
        for (size_t k = axes.size(); k-- > 0;) {
            a_off += axes[k].a_step;
            b_off += axes[k].b_step;
            c_off += axes[k].c_step;
            if (++value[k] < axes[k].extent) {
                return;
            }
            // Wrapped: undo the whole axis and carry into the next one out.
            a_off -= axes[k].a_step * axes[k].extent;
            b_off -= axes[k].b_step * axes[k].extent;
            c_off -= axes[k].c_step * axes[k].extent;
            value[k] = 0;
        }
    };

    std::vector<size_t> target_value(target_axes.size(), 0), link_value(link_axes.size(), 0);
    size_t              a_target = 0, b_target = 0, c_offset = 0;

    for (size_t target_flat = 0; target_flat < target_total; target_flat++) {
        T      sum    = T{0};
        size_t a_off  = a_target;
        size_t b_off  = b_target;
        size_t unused = 0;
        std::ranges::fill(link_value, 0);

        for (size_t link_flat = 0; link_flat < link_total; link_flat++) {
            T a_val = a_data[a_off];
            T b_val = b_data[b_off];
            if constexpr (IsComplexV<T>) {
                if (conj_a) {
                    a_val = std::conj(a_val);
                }
                if (conj_b) {
                    b_val = std::conj(b_val);
                }
            }
            sum += a_val * b_val;
            advance(link_axes, link_value, a_off, b_off, unused);
        }

        C->data()[c_offset] += ab_pf * sum;
        advance(target_axes, target_value, a_target, b_target, c_offset);
    }
}

// ── Main dispatch function ──────────────────────────────────────────────────

template <BasicTensorConcept AType, BasicTensorConcept BType, BasicTensorConcept CType>
    requires requires {
        requires std::is_same_v<typename AType::ValueType, typename BType::ValueType>;
        requires std::is_same_v<typename AType::ValueType, typename CType::ValueType>;
    }
void string_einsum(ParsedEinsumSpec const &parsed, typename AType::ValueType c_pf, CType *C, typename AType::ValueType ab_pf,
                   AType const &A, BType const &B, bool conj_a = false, bool conj_b = false,
                   std::vector<std::string> const *precomputed_links = nullptr, packed_gemm::ContractionSite *pg_site = nullptr) {
    using T = typename AType::ValueType;

    LabeledSection("cg::einsum: {} <- {} ; {}", fmt::join(parsed.c_indices, ","), fmt::join(parsed.a_indices, ","),
                   fmt::join(parsed.b_indices, ","));
    // As integers, not std::to_string: the string form was BUILT before
    // annotate() could check whether anything was recording, so every
    // contraction paid three of them whether or not they went anywhere.
    ProfileAnnotate("a_rank", static_cast<int64_t>(detail::tensor_rank(A)));
    ProfileAnnotate("b_rank", static_cast<int64_t>(detail::tensor_rank(B)));
    ProfileAnnotate("c_rank", static_cast<int64_t>(detail::tensor_rank(*C)));

    auto const &c_idx = parsed.c_indices;
    auto const &a_idx = parsed.a_indices;
    auto const &b_idx = parsed.b_indices;

    // Contracted indices (in A and B, not in C). Graph executors pass the
    // list computed once at capture (EinsumIndices::link_indices) so replays
    // don't recompute it; the eager path derives it here. Index lists are
    // tiny (rank-bounded), so linear scans beat building three
    // std::set<std::string>s, and the result is sorted to match the
    // set-based order this code historically produced.
    std::vector<std::string> links_storage;
    if (precomputed_links == nullptr) {
        for (auto const &idx : a_idx) {
            bool const in_b = std::find(b_idx.begin(), b_idx.end(), idx) != b_idx.end();
            bool const in_c = std::find(c_idx.begin(), c_idx.end(), idx) != c_idx.end();
            bool const seen = std::find(links_storage.begin(), links_storage.end(), idx) != links_storage.end();
            if (in_b && !in_c && !seen) {
                links_storage.push_back(idx);
            }
        }
        std::sort(links_storage.begin(), links_storage.end());
    }
    std::vector<std::string> const &links = precomputed_links != nullptr ? *precomputed_links : links_storage;

    // Zero-extent operands: nothing to contract, but BLAS-style semantics
    // still apply the output prefactor. An empty C is a pure
    // no-op; an empty input with a non-empty C (zero-extent link or trace
    // letter) means C = c_pf * C, with c_pf == 0 assigning zero rather than
    // multiplying so stale NaNs never survive. Handled here once so no
    // fast path (BLAS wrappers, PackedGemm tiling, generic loop) needs its
    // own empty-tensor bookkeeping.
    auto const total_size = [](auto const &t) {
        size_t total = 1;
        for (size_t d = 0; d < detail::tensor_rank(t); d++) {
            total *= t.dim(d);
        }
        return total;
    };
    if (total_size(*C) == 0) {
        ProfileAnnotate("dispatch", "empty_output_noop");
        last_dispatch_route() = "empty_output_noop";
        return;
    }
    if (total_size(A) == 0 || total_size(B) == 0) {
        ProfileAnnotate("dispatch", "empty_input_scale_only");
        last_dispatch_route() = "empty_input_scale_only";
        if (c_pf == T{0}) {
            C->zero();
        } else if (c_pf != T{1}) {
            linear_algebra::scale(c_pf, C);
        }
        return;
    }

    // Output aliasing an input is rejected: contractions read operands while
    // writing C, so overlap silently corrupts results (the GEMM-shaped case
    // computed garbage before this check existed). The one provably safe
    // shape is carved out: when C's index list is IDENTICAL to the aliased
    // operand's, every element is read exactly once immediately before its
    // own overwrite (pure elementwise update, e.g. "ij <- ij ; ij" with C
    // aliasing A). A and B sharing a buffer is always fine - inputs are
    // read-only. Runs after the zero-size quick-path so the span arithmetic
    // never sees a zero dimension.
    {
        // Interval overlap alone is NOT proof of element overlap: disjoint
        // column-major slices of one parent interleave in memory. Only
        // provable overlap rejects: both regions contiguous, or identical
        // base pointers (see the eager guard in TensorAlgebra's
        // Backends/Dispatch.hpp for the full rationale).
        struct Region {
            T const *lo;
            T const *hi;
            bool     contiguous;
        };
        auto const region_of = [](auto const &t) -> Region {
            T const *lo     = t.data();
            size_t   last   = 0;
            size_t   nelems = 1;
            for (size_t d = 0; d < detail::tensor_rank(t); d++) {
                last += (t.dim(d) - 1) * t.stride(d);
                nelems *= t.dim(d);
            }
            return {lo, lo + last + 1, last + 1 == nelems};
        };
        auto const c_region   = region_of(*C);
        auto const overlaps_c = [&](auto const &x) {
            auto const r = region_of(x);
            if (!(r.lo < c_region.hi && c_region.lo < r.hi)) {
                return false;
            }
            return (r.contiguous && c_region.contiguous) || r.lo == c_region.lo;
        };
        if ((overlaps_c(A) && a_idx != c_idx) || (overlaps_c(B) && b_idx != c_idx)) {
            EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                    "einsum: output tensor overlaps an input operand. In-place einsum is only supported for pure "
                                    "elementwise updates (the aliased operand's index list identical to the output's); for this "
                                    "contraction, pass a separate output tensor or copy the input first.");
        }
    }

    // Repeated letters within one operand ('ij <- ii ; jj') are diagonal
    // accesses. Every fast path below classifies indices assuming each
    // letter appears at most once per operand - the outer-product/GER
    // routes silently computed wrong values for these specs.
    // The generic loop above is the only repeat-aware path; route there.
    auto const has_repeated_letter = [](std::vector<std::string> const &idx) {
        for (size_t p = 1; p < idx.size(); p++) {
            for (size_t q = 0; q < p; q++) {
                if (idx[p] == idx[q]) {
                    return true;
                }
            }
        }
        return false;
    };
    if (has_repeated_letter(a_idx) || has_repeated_letter(b_idx) || has_repeated_letter(c_idx)) {
        ProfileAnnotate("dispatch", "generic_loop_repeated_indices");
        last_dispatch_route() = "generic_loop_repeated_indices";
        generic_string_einsum(parsed, links, c_pf, C, ab_pf, A, B, conj_a, conj_b);
        return;
    }

    // Lone summed index ("weighted trace"): a letter in exactly one operand,
    // absent from C AND from the shared links (which hold A-and-B letters
    // only). It is a single-operand reduction - "ijk <- ijl ; milk" sums B
    // over m - and no BLAS/PackedGemm call can express it: every fast path
    // below classifies on links.size() and builds a spec over target + link
    // indices only, so a lone index is neither iterated nor summed (PackedGemm
    // pins it to 0, silently dropping the reduction). Only the generic loop is
    // correct here; it adds such letters as summed axes (see the trace-letter
    // handling in generic_string_einsum). This runs before every fast path so
    // the empty-link case (P1-style "ij <- ijk ; ij", already correct) and the
    // link+lone case (previously miscomputed) both route here.
    auto const has_lone_summed_index = [&] {
        for (auto const *idx : {&a_idx, &b_idx}) {
            for (auto const &s : *idx) {
                bool const in_c    = std::find(c_idx.begin(), c_idx.end(), s) != c_idx.end();
                bool const in_link = std::find(links.begin(), links.end(), s) != links.end();
                if (!in_c && !in_link) {
                    return true;
                }
            }
        }
        return false;
    };
    if (has_lone_summed_index()) {
        ProfileAnnotate("dispatch", "generic_loop_lone_summed");
        last_dispatch_route() = "generic_loop_lone_summed";
        generic_string_einsum(parsed, links, c_pf, C, ab_pf, A, B, conj_a, conj_b);
        return;
    }

    // ── Scalar-output full contraction, any rank ────────────────────────────
    // Every index of A is contracted against the SAME index of B and nothing
    // survives into C, so this is a plain dot product over the whole element
    // space however the elements are shaped. linear_algebra::dot takes any
    // rank and handles non-unit strides, so no contiguity requirement is
    // needed; matching index ORDER is the real precondition, since dot pairs
    // elements by position in the logical index space.
    //
    // This sits AHEAD of the !conj_a && !conj_b gate below because it is the
    // one BLAS shape with a conjugating form: PackedGemm rejects a rank-0
    // output, so without this a conjugated full contraction had nowhere to go
    // but the serial generic loop.
    // dot() needs SameRank, which for two typed operands is a compile-time
    // property and for two runtime-rank ones is automatic (their static Rank is
    // the same dynamic sentinel), so the guard is constexpr and the runtime
    // rank equality is re-checked below. A mixed typed/runtime pair is left to
    // the paths further down.
    if constexpr ((HasCompileTimeRank<AType> && HasCompileTimeRank<BType> &&
                   std::remove_cvref_t<AType>::Rank == std::remove_cvref_t<BType>::Rank) ||
                  (!HasCompileTimeRank<AType> && !HasCompileTimeRank<BType>)) {
        if (c_idx.empty() && a_idx == b_idx && links.size() == a_idx.size() && detail::tensor_rank(A) == detail::tensor_rank(B)) {
            T temp;
            if constexpr (IsComplexV<T>) {
                // true_dot(X, Y) is sum conj(X) * Y.
                if (conj_a && conj_b) {
                    temp = std::conj(linear_algebra::dot(A, B));
                } else if (conj_a) {
                    temp = linear_algebra::true_dot(A, B);
                } else if (conj_b) {
                    temp = linear_algebra::true_dot(B, A);
                } else {
                    temp = linear_algebra::dot(A, B);
                }
            } else {
                // Conjugation is the identity on a real type, so the flags
                // carry no meaning here and plain dot is already correct.
                temp = linear_algebra::dot(A, B);
            }

            bool const conjugating = IsComplexV<T> && (conj_a || conj_b);
            if constexpr (HasCompileTimeRank<AType>) {
                ProfileAnnotate("dispatch", conjugating ? "true_dot" : "dot");
                last_dispatch_route() = conjugating ? "true_dot" : "dot";
            } else {
                ProfileAnnotate("dispatch", conjugating ? "true_dot_runtime" : "dot_runtime");
                last_dispatch_route() = conjugating ? "true_dot_runtime" : "dot_runtime";
            }
            C->data()[0] = c_pf * C->data()[0] + ab_pf * temp;
            return;
        }
    }

    // ── Rank-1 special-case BLAS fast paths ─────────────────────────────────
    // These call helpers (string_gemv_mat_vec, linear_algebra::ger, etc.)
    // that are themselves rank-specific (MatrixConcept / VectorConcept), so
    // they only compile for typed Tensor<T, K> operands and stay gated
    // behind HasCompileTimeRank. PackedGemm (below) handles the rank-2+
    // GEMM-shaped cases and works uniformly for typed and runtime-rank
    // tensors via the runtime ContractionSpec entry point.
    // Conjugation skips the non-conj BLAS fast paths below: those helpers
    // (gemv, ger, string_gemm, direct_product) don't conjugate. Conjugated
    // contractions go to PackedGemm (native via spec.conj_a/conj_b) for
    // gemm-shaped cases, else the conj-aware generic loop.
    //
    // The two rank-2 GEMM routes are the exception: a matrix times a matrix is
    // the one shape here that a thread's node width could have spread, and a
    // vendor GEMM issued under such a width is clamped to one thread by the BLAS
    // wrappers' fence. When PackedGemm is the preferred route they stand aside
    // and it takes the shape, whose packed loops fork from the ICV the width
    // raised.
    //
    // Same answer the packed engine reaches, from the same call site's pin (@ref
    // packed_gemm::prefer_packed_route): a caller whose node has a pinned route
    // gets it here too, so the shape cannot take one route at this gate and the
    // other inside try_packed_gemm. A caller with no site - eager, or an
    // unplanned graph - reads the thread regime exactly as before.
    if (!conj_a && !conj_b) {
        [[maybe_unused]] bool const route_prefers_packed = packed_gemm::prefer_packed_route(pg_site);
        if constexpr (HasCompileTimeRank<AType> && HasCompileTimeRank<BType> && HasCompileTimeRank<CType>) {
            constexpr size_t a_rank = std::remove_cvref_t<AType>::Rank;
            constexpr size_t b_rank = std::remove_cvref_t<BType>::Rank;
            constexpr size_t c_rank = std::remove_cvref_t<CType>::Rank;

            // ── DOT product: scalar output, all indices contracted ──────────
            if constexpr (a_rank == 1 && b_rank == 1 && c_rank == 1) {
                if (c_idx.empty() || (links.size() == a_idx.size())) {
                    ProfileAnnotate("dispatch", "dot");
                    last_dispatch_route() = "dot";
                    T temp                = linear_algebra::dot(A, B);
                    C->data()[0]          = c_pf * C->data()[0] + ab_pf * temp;
                    return;
                }
            }

            // ── GEMV: matrix × vector → vector ──────────────────────────────
            if constexpr (a_rank == 2 && b_rank == 1 && c_rank == 1) {
                if (links.size() == 1) {
                    ProfileAnnotate("dispatch", "gemv_mat_vec");
                    last_dispatch_route() = "gemv_mat_vec";
                    string_gemv_mat_vec(parsed, c_pf, C, ab_pf, A, B, a_idx, links[0]);
                    return;
                }
            }

            // ── GEMV: vector × matrix → vector ──────────────────────────────
            if constexpr (a_rank == 1 && b_rank == 2 && c_rank == 1) {
                if (links.size() == 1) {
                    ProfileAnnotate("dispatch", "gemv_vec_mat");
                    last_dispatch_route() = "gemv_vec_mat";
                    // Reinterpret as B^T * A or B * A depending on where the link is
                    char trans = (b_idx[1] == links[0]) ? 'n' : 't';
                    linear_algebra::gemv(trans, ab_pf, B, A, c_pf, C);
                    return;
                }
            }

            // ── GER: vector × vector → matrix (outer product) ───────────────
            if constexpr (a_rank == 1 && b_rank == 1 && c_rank == 2) {
                if (links.empty()) {
                    ProfileAnnotate("dispatch", "ger");
                    last_dispatch_route() = "ger";
                    if (c_pf != T{1}) {
                        linear_algebra::scale(c_pf, C);
                    }
                    // ger(x, y, C) computes C[i,j] = x[i]*y[j], so the operand whose
                    // index labels C's first axis must be x. Swap for a transposed
                    // output (spec like "ji <- i ; j", where C's axes are ordered
                    // opposite to the A-then-B operand order).
                    if (c_idx[0] == a_idx[0]) {
                        linear_algebra::ger(ab_pf, A, B, C);
                    } else {
                        linear_algebra::ger(ab_pf, B, A, C);
                    }
                    return;
                }
            }

            // ── GEMM: matrix × matrix → matrix ──────────────────────────────
            if constexpr (a_rank == 2 && b_rank == 2 && c_rank == 2) {
                if (links.size() == 1 && !route_prefers_packed) {
                    ProfileAnnotate("dispatch", "gemm_direct");
                    last_dispatch_route() = "gemm_direct";
                    string_gemm(parsed, links[0], c_pf, C, ab_pf, A, B);
                    return;
                }
            }

            // ── Direct product: same indices on all three, no links ─────────
            // Elementwise at ANY rank, not just rank 2 - the gate used to sit
            // inside the rank-2 block above, so "i <- i ; i" and
            // "ijk <- ijk ; ijk" fell all the way to the serial generic loop
            // even though linear_algebra::direct_product takes any rank.
            if constexpr (a_rank == b_rank && b_rank == c_rank) {
                if (links.empty() && a_idx == b_idx && a_idx == c_idx) {
                    ProfileAnnotate("dispatch", "direct_product");
                    last_dispatch_route() = "direct_product";
                    linear_algebra::direct_product(ab_pf, A, B, c_pf, C);
                    return;
                }
            }
        }

        // ── Rank-erased BLAS fast paths ─────────────────────────────────────────
        // Mirror of the typed BLAS ladder above, reached whenever that one did
        // not apply. We build a zero-copy TensorView<T, K> over each operand's
        // data, whose impl carries the same dims and strides, then call the
        // same rank-specialized BLAS helpers. The upcast is just a pointer plus
        // a small metadata array, with no allocation and no copy, and it reads
        // only data()/dim()/stride() - which typed and runtime-rank tensors
        // both have, so it does not care which it is handed.
        //
        // That is what makes a MIXED triple work. This used to require all
        // three operands to be runtime-rank, so one typed and one runtime
        // operand satisfied neither ladder and fell through to PackedGemm -
        // which DEFERS a plain single-M/N/K GEMM back to direct BLAS, leaving
        // the serial generic loop to run it. A rank-2 matmul was hitting the
        // odometer loop purely because its operands were declared differently.
        //
        // Running it for an all-typed triple is harmless: every branch here
        // tests the same shape conditions the typed ladder already returned on,
        // so it only ever sees shapes that one declined.
        if constexpr (!(HasCompileTimeRank<AType> && HasCompileTimeRank<BType> && HasCompileTimeRank<CType>)) {
            std::size_t const a_rank = detail::tensor_rank(A);
            std::size_t const b_rank = detail::tensor_rank(B);
            std::size_t const c_rank = detail::tensor_rank(*C);

            auto upcast = [](auto const &t, auto rank_tag) {
                constexpr std::size_t K = decltype(rank_tag)::value;
                using ValueType         = typename std::remove_cvref_t<decltype(t)>::ValueType;
                std::array<size_t, K> dims;
                std::array<size_t, K> strides;
                for (std::size_t i = 0; i < K; ++i) {
                    dims[i]    = t.dim(i);
                    strides[i] = t.stride(i);
                }
                ::einsums::detail::TensorImpl<ValueType> impl(const_cast<ValueType *>(t.data()), dims, strides);
                return TensorView<ValueType, K>(impl);
            };

            // ── DOT product ──────────────────────────────────────────────
            if (a_rank == 1 && b_rank == 1 && c_rank <= 1) {
                if (c_idx.empty() || (links.size() == a_idx.size())) {
                    ProfileAnnotate("dispatch", "dot_runtime");
                    last_dispatch_route() = "dot_runtime";
                    auto av               = upcast(A, std::integral_constant<std::size_t, 1>{});
                    auto bv               = upcast(B, std::integral_constant<std::size_t, 1>{});
                    T    temp             = linear_algebra::dot(av, bv);
                    C->data()[0]          = c_pf * C->data()[0] + ab_pf * temp;
                    return;
                }
            }

            // ── GEMV: matrix × vector → vector ───────────────────────────
            if (a_rank == 2 && b_rank == 1 && c_rank == 1) {
                if (links.size() == 1) {
                    ProfileAnnotate("dispatch", "gemv_mat_vec_runtime");
                    last_dispatch_route() = "gemv_mat_vec_runtime";
                    auto av               = upcast(A, std::integral_constant<std::size_t, 2>{});
                    auto bv               = upcast(B, std::integral_constant<std::size_t, 1>{});
                    auto cv               = upcast(*C, std::integral_constant<std::size_t, 1>{});
                    string_gemv_mat_vec(parsed, c_pf, &cv, ab_pf, av, bv, a_idx, links[0]);
                    return;
                }
            }

            // ── GEMV: vector × matrix → vector ───────────────────────────
            if (a_rank == 1 && b_rank == 2 && c_rank == 1) {
                if (links.size() == 1) {
                    ProfileAnnotate("dispatch", "gemv_vec_mat_runtime");
                    last_dispatch_route() = "gemv_vec_mat_runtime";
                    auto av               = upcast(A, std::integral_constant<std::size_t, 1>{});
                    auto bv               = upcast(B, std::integral_constant<std::size_t, 2>{});
                    auto cv               = upcast(*C, std::integral_constant<std::size_t, 1>{});
                    char trans            = (b_idx[1] == links[0]) ? 'n' : 't';
                    linear_algebra::gemv(trans, ab_pf, bv, av, c_pf, &cv);
                    return;
                }
            }

            // ── GER: vector × vector → matrix ────────────────────────────
            if (a_rank == 1 && b_rank == 1 && c_rank == 2) {
                if (links.empty()) {
                    ProfileAnnotate("dispatch", "ger_runtime");
                    last_dispatch_route() = "ger_runtime";
                    auto av               = upcast(A, std::integral_constant<std::size_t, 1>{});
                    auto bv               = upcast(B, std::integral_constant<std::size_t, 1>{});
                    auto cv               = upcast(*C, std::integral_constant<std::size_t, 2>{});
                    if (c_pf != T{1}) {
                        linear_algebra::scale(c_pf, &cv);
                    }
                    // See the compile-time GER path: swap operands for a transposed
                    // output so the operand indexing C's first axis is x.
                    if (c_idx[0] == a_idx[0]) {
                        linear_algebra::ger(ab_pf, av, bv, &cv);
                    } else {
                        linear_algebra::ger(ab_pf, bv, av, &cv);
                    }
                    return;
                }
            }

            // ── GEMM: matrix × matrix → matrix ───────────────────────────
            if (a_rank == 2 && b_rank == 2 && c_rank == 2) {
                if (links.size() == 1 && !route_prefers_packed) {
                    ProfileAnnotate("dispatch", "gemm_direct_runtime");
                    last_dispatch_route() = "gemm_direct_runtime";
                    auto av               = upcast(A, std::integral_constant<std::size_t, 2>{});
                    auto bv               = upcast(B, std::integral_constant<std::size_t, 2>{});
                    auto cv               = upcast(*C, std::integral_constant<std::size_t, 2>{});
                    string_gemm(parsed, links[0], c_pf, &cv, ab_pf, av, bv);
                    return;
                }
            }

            // ── Direct product at ANY rank ───────────────────────────────
            // No upcast to a fixed K is needed: direct_product wants
            // SameRank<A, B, C>, which three runtime-rank operands satisfy
            // statically, so the runtime ranks are simply checked here. This
            // used to sit inside the rank-2 block above, matching the typed
            // ladder's old gate and sending every other rank to the generic
            // loop.
            if (a_rank == b_rank && b_rank == c_rank && links.empty() && a_idx == b_idx && a_idx == c_idx) {
                ProfileAnnotate("dispatch", "direct_product_runtime");
                last_dispatch_route() = "direct_product_runtime";
                linear_algebra::direct_product(ab_pf, A, B, c_pf, C);
                return;
            }
        }
    } // end of the !conj_a && !conj_b BLAS fast-path gate

    // ── PackedGemm path ─────────────────────────────────────────────────────
    // Handles arbitrary-rank GEMM-shaped contractions, including those with
    // batch (Hadamard) indices appearing in A, B, AND C. Works uniformly for
    // typed Tensor<T, K> and RuntimeTensor<T, Alloc> via the runtime
    // ContractionSpec entry point. Returns false (defers) for cases the
    // direct rank-1/rank-2 paths above already handle, or for shapes
    // PackedGemm can't form (no M-dims, no N-dims, no link indices).
    {
        // A caller that repeats this contraction (a graph node) hands in a
        // site holding the spec it built last time. The spec is a pure
        // function of the index lists and the conjugation flags, so when those
        // still agree it is lent back instead of assembling six vector<string>
        // - about fifteen allocations - on a path a tiled expansion drives
        // thousands of times per replay.
        bool const reuse_spec = pg_site != nullptr && pg_site->resolved &&
                                packed_gemm::spec_matches_indices(pg_site->key.spec, c_idx, a_idx, b_idx, links, conj_a, conj_b);

        packed_gemm::ContractionSpec built;
        if (!reuse_spec) {
            built.c_indices    = c_idx;
            built.a_indices    = a_idx;
            built.b_indices    = b_idx;
            built.link_indices = links;

            // Unique C indices in C's own order (PackedGemm's target space is
            // ordered, so a sorted set is the wrong shape). Index lists are
            // rank-bounded, so the linear scan beats a std::set - which is
            // what this used to build, on top of a sorted-unique vector that
            // was discarded unused a line later.
            built.target_indices.reserve(c_idx.size());
            for (auto const &t : c_idx) {
                if (std::find(built.target_indices.begin(), built.target_indices.end(), t) == built.target_indices.end()) {
                    built.target_indices.push_back(t);
                }
            }
            built.all_indices = built.target_indices;
            for (auto const &l : built.link_indices)
                built.all_indices.push_back(l);
            built.conj_a = conj_a; // PackedGemm conjugates during packing/transpose (native, no copy)
            built.conj_b = conj_b;
        }
        packed_gemm::ContractionSpec const &spec = reuse_spec ? pg_site->key.spec : built;

        if (packed_gemm::try_packed_gemm<AType, BType, CType>(spec, c_pf, C, ab_pf, A, B, /*allow_scatter=*/true, pg_site)) {
            ProfileAnnotate("dispatch", "packed_gemm");
            last_dispatch_route() = "packed_gemm";
            return;
        }
    }

    // ── Generic fallback: runtime nested-loop contraction ──────────
    // Reached when no fast path applies: pure outer products, mixed-dtype
    // edge cases, or contractions that even PackedGemm can't form into a
    // valid GEMM shape (no M-dims, no N-dims, no links).
    ProfileAnnotate("dispatch", "generic_loop");
    last_dispatch_route() = "generic_loop";
    generic_string_einsum(parsed, links, c_pf, C, ab_pf, A, B, conj_a, conj_b);
}

// ═════════════════════════════════════════════════════════════════════════════
// String-based permute dispatch
// ═════════════════════════════════════════════════════════════════════════════

/**
 * @brief Execute a tensor permutation described by a string specification.
 *
 * C[c_indices] = beta * C[c_indices] + alpha * A[a_indices]
 *
 * Builds the index permutation mapping at runtime and performs a stride-based copy.
 */
template <BasicTensorConcept AType, BasicTensorConcept CType>
    requires std::is_same_v<typename AType::ValueType, typename CType::ValueType>
void string_permute(ParsedPermuteSpec const &parsed, typename AType::ValueType beta, CType *C, typename AType::ValueType alpha,
                    AType const &A) {
    using T = typename AType::ValueType;

    auto const &c_idx = parsed.c_indices;
    auto const &a_idx = parsed.a_indices;

    size_t const rank = c_idx.size();

    // Build permutation: perm[i] = position of c_idx[i] in a_idx
    // i.e., C dimension i corresponds to A dimension perm[i]
    std::vector<size_t> perm(rank);
    for (size_t i = 0; i < rank; i++) {
        bool found = false;
        for (size_t j = 0; j < rank; j++) {
            if (c_idx[i] == a_idx[j]) {
                perm[i] = j;
                found   = true;
                break;
            }
        }
        if (!found) {
            EINSUMS_THROW_EXCEPTION(std::runtime_error, "String permute '{}': output index '{}' not found in input indices", parsed.raw,
                                    c_idx[i]);
        }
    }

    // Zero-extent: nothing to permute, but the beta prefactor still applies
    // to the (empty) C. Handled here so the HPTT path never sees a zero dim.
    size_t total = 1;
    for (size_t d = 0; d < rank; d++)
        total *= C->dim(d);
    if (total == 0) {
        return;
    }

    // Canonically dense tensors take HPTT through the shared plan cache; the
    // scalar loop below exists for strided views only. HPTT computes
    // C = beta*C + alpha*perm(A) natively, so beta/alpha need no pre-pass.
    // The string overload of compile_permute matches indices character by
    // character, so multi-character labels are re-encoded onto 'a'..'z'.
    // Contiguity alone is NOT enough: a permute_view spans the whole buffer
    // but presents reordered strides, and the stride-ratio outerSize
    // derivation assumes strides monotone in the layout flag's direction
    // (the same trap the gemm_hint's layout_matches_flag guards).
    auto const canonical_dense = [](auto const &impl) {
        if (!impl.is_contiguous()) {
            return false;
        }
        bool const   row_major = impl.is_row_major();
        size_t const irank     = impl.rank();
        size_t       prev      = 0;
        bool         first     = true;
        for (size_t n = 0; n < irank; ++n) {
            size_t const d = row_major ? irank - 1 - n : n;
            if (impl.dim(d) <= 1) {
                continue; // extent-1 axes are never traversed; ignore their strides
            }
            size_t const st = impl.stride(d);
            if (!first && st < prev) {
                return false;
            }
            prev  = st;
            first = false;
        }
        return true;
    };
    if (rank >= 2 && canonical_dense(C->impl()) && canonical_dense(A.impl())) {
        std::string a_chars(rank, ' ');
        std::string c_chars(rank, ' ');
        for (size_t j = 0; j < rank; j++) {
            a_chars[j] = static_cast<char>('a' + j);
        }
        for (size_t i = 0; i < rank; i++) {
            c_chars[i] = static_cast<char>('a' + perm[i]);
        }
        tensor_algebra::detail::permute<false, T>(beta, c_chars, &C->impl(), alpha, a_chars, A.impl());
        return;
    }

    // Scale C by beta
    if (beta == T{0}) {
        C->zero();
    } else if (beta != T{1}) {
        linear_algebra::scale(beta, C);
    }

    // If alpha is zero, nothing more to do
    if (alpha == T{0})
        return;

    std::vector<size_t> c_coords(rank, 0);

    for (size_t flat = 0; flat < total; flat++) {
        // Decode flat index into C coordinates
        if (flat > 0) {
            for (int d = static_cast<int>(rank) - 1; d >= 0; d--) {
                c_coords[d]++;
                if (c_coords[d] < C->dim(d))
                    break;
                c_coords[d] = 0;
            }
        }

        // Compute A offset using permuted coordinates
        size_t a_offset = 0;
        for (size_t d = 0; d < rank; d++) {
            a_offset += c_coords[d] * A.stride(perm[d]);
        }

        // Compute C offset
        size_t c_offset = 0;
        for (size_t d = 0; d < rank; d++) {
            c_offset += c_coords[d] * C->stride(d);
        }

        C->data()[c_offset] += alpha * A.data()[a_offset];
    }
}

EINSUMS_NAMESPACE_END(compute_graph::dispatch)
