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

#include <Einsums/Config.hpp>

#include <Einsums/BLAS/ThreadControl.hpp>
#include <Einsums/ComputeGraph/Detail/ErasedEinsum.hpp>
#include <Einsums/ComputeGraph/Detail/ImplLayout.hpp>
#include <Einsums/ComputeGraph/Detail/MixedPrecision.hpp>
#include <Einsums/ComputeGraph/EinsumSpec.hpp>
#include <Einsums/ComputeGraph/TensorRank.hpp>
#include <Einsums/Concepts/TensorConcepts.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/LinearAlgebra.hpp>
#include <Einsums/PackedGemm/PackedGemm.hpp>
#include <Einsums/Profile.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/TensorPermute/Permute.hpp>

#include <fmt/format.h>

#include <algorithm>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph::dispatch)

// ── Generic nested-loop contraction ─────────────────────────────────────

// last_dispatch_route() and the string_permute_impl declaration live in Detail/ErasedEinsum.hpp.

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
 * 51 kflop.
 *
 * The TARGET axes are then ordered by their step through C and merged where
 * they compose, so the fastest-moving loop walks C's narrowest stride instead
 * of whichever index the caller happened to spell last. The link axes keep the
 * caller's order: the sum accumulates exactly as it did, so the result is
 * bit-for-bit what it was even though the order C's elements are visited in is
 * now decided by the strides rather than by the spec.
 */
template <BasicTensorConcept AType, BasicTensorConcept BType, BasicTensorConcept CType>
    requires detail::EinsumElement<typename AType::ValueType> && detail::EinsumElement<typename BType::ValueType> &&
             detail::EinsumElement<typename CType::ValueType> &&
             detail::storable_v<detail::PromoteT<typename AType::ValueType, typename BType::ValueType>, typename CType::ValueType>
void generic_string_einsum(ParsedEinsumSpec const &parsed, std::vector<std::string> const &links, typename CType::ValueType c_pf, CType *C,
                           detail::PromoteT<typename AType::ValueType, typename BType::ValueType> ab_pf, AType const &A, BType const &B,
                           bool conj_a = false, bool conj_b = false) {
    // One loop for every combination of element types. The operands are read as the accumulator
    // type TR (promoted from TA and TB, detail/MixedPrecision.hpp) and the sum is stored into C as TC.
    // When all four are one type every conversion below is the identity, and the arithmetic, the
    // summation order and so the result are exactly those of the single-type loop.
    using TA = typename AType::ValueType;
    using TB = typename BType::ValueType;
    using TC = typename CType::ValueType;
    using TR = detail::PromoteT<TA, TB>;

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
    // anything, so an aliased operand would be read back as zeros (e.g.
    // "ab <- ab ; b" with C aliasing A).
    // Snapshot any operand whose storage overlaps C's and read the copy, which
    // leaves the loop below (and so the summation order, and so the exact
    // floating-point result) untouched.
    //
    // Guarded on a non-empty iteration space: the loop reads nothing when
    // either total is zero, and that is also the only way an operand can carry
    // a zero extent, which the span arithmetic below cannot represent.
    std::vector<TA> a_snapshot;
    std::vector<TB> b_snapshot;
    TA const       *a_data = A.data();
    TB const       *b_data = B.data();
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
        // Compared in bytes, since the operands' element types may differ.
        auto const bytes_of = [&](auto const &t) {
            auto const *lo = reinterpret_cast<unsigned char const *>(t.data());
            return std::pair{lo, lo + span_of(t) * sizeof(*t.data())};
        };
        auto const [c_lo, c_hi] = bytes_of(*C);
        auto const overlaps_c   = [&](auto const &t) {
            auto const [lo, hi] = bytes_of(t);
            return lo < c_hi && c_lo < hi;
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
    if (c_pf == TC{0}) {
        C->zero();
    } else if (c_pf != TC{1}) {
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
    // Order the TARGET axes for the layout, then merge the ones that compose.
    //
    // The odometer below runs its LAST axis fastest. In C's spelled order that
    // is the widest stride of a column-major tensor, a step the prefetcher
    // cannot follow. Sorting by C's step is layout agnostic; C decides because
    // a scattered store also pays read-for-ownership.
    //
    // Merging then folds any adjacent pair whose steps compose in A, B and C
    // alike. An elementwise contraction collapses to a single axis, which also
    // takes `advance` and its carry chain out of the per-element path.
    //
    // Only the TARGET axes move. The link axes keep the caller's order, so the
    // sum's accumulation order, and so the result bits, do not depend on it.
    auto order_for_layout = [](std::vector<Axis> axes) {
        std::stable_sort(axes.begin(), axes.end(), [](Axis const &l, Axis const &r) { return l.c_step > r.c_step; });

        while (axes.size() >= 2) {
            Axis const inner = axes.back();
            Axis const outer = axes[axes.size() - 2];
            if (outer.a_step != inner.a_step * inner.extent || outer.b_step != inner.b_step * inner.extent ||
                outer.c_step != inner.c_step * inner.extent) {
                break;
            }
            axes.pop_back();
            axes.back() =
                Axis{.extent = inner.extent * outer.extent, .a_step = inner.a_step, .b_step = inner.b_step, .c_step = inner.c_step};
        }
        return axes;
    };

    std::vector<Axis> const target_axes = order_for_layout(axes_of(target_infos));
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
        TR     sum    = TR{0};
        size_t a_off  = a_target;
        size_t b_off  = b_target;
        size_t unused = 0;
        std::ranges::fill(link_value, 0);

        for (size_t link_flat = 0; link_flat < link_total; link_flat++) {
            auto a_val = static_cast<TR>(a_data[a_off]);
            auto b_val = static_cast<TR>(b_data[b_off]);
            if constexpr (IsComplexV<TR>) {
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

        if constexpr (std::is_same_v<TC, TR>) {
            C->data()[c_offset] += ab_pf * sum;
        } else {
            // Add in a type that holds both C's existing value and the sum (a complex C with a real sum
            // keeps its imaginary part; a float C with a double sum adds in double), then round to C's
            // type once.
            using TS = detail::PromoteT<TR, TC>;
            TC &out  = C->data()[c_offset];
            out      = static_cast<TC>(static_cast<TS>(out) + static_cast<TS>(ab_pf * sum));
        }
        advance(target_axes, target_value, a_target, b_target, c_offset);
    }
}

/// Zero-extent operands: nothing to contract, but BLAS-style semantics still apply the output
/// prefactor. An empty C is a pure no-op; an empty input with a non-empty C (zero-extent link or
/// trace letter) means C = c_pf * C, with c_pf == 0 assigning zero rather than multiplying so stale
/// NaNs never survive. Handled once, before any kernel, so no fast path (BLAS wrappers, PackedGemm
/// tiling, generic loop) needs its own empty-tensor bookkeeping. Returns true when it handled the
/// call.
template <typename CType, typename AType, typename BType>
bool einsum_empty_operands(typename CType::ValueType c_pf, CType *C, AType const &A, BType const &B) {
    using TC              = typename CType::ValueType;
    auto const total_size = [](auto const &t) {
        size_t total = 1;
        for (size_t d = 0; d < detail::tensor_rank(t); d++) {
            total *= t.dim(d);
        }
        return total;
    };
    if (total_size(*C) == 0) {
        WAGGLE_ANNOTATE("dispatch", "empty_output_noop");
        last_dispatch_route() = "empty_output_noop";
        return true;
    }
    if (total_size(A) == 0 || total_size(B) == 0) {
        WAGGLE_ANNOTATE("dispatch", "empty_input_scale_only");
        last_dispatch_route() = "empty_input_scale_only";
        if (c_pf == TC{0}) {
            C->zero();
        } else if (c_pf != TC{1}) {
            linear_algebra::scale(c_pf, C);
        }
        return true;
    }
    return false;
}

/// Output aliasing an input is rejected: contractions read operands while writing C, so overlap
/// silently corrupts results. The one provably safe shape is carved out: when C's index list is IDENTICAL to the aliased operand's,
/// every element is read exactly once immediately before its own overwrite (pure elementwise update,
/// e.g. "ij <- ij ; ij" with C aliasing A). A and B sharing a buffer is always fine - inputs are
/// read-only. Must run after the zero-size check so the span arithmetic never sees a zero dimension.
///
/// Interval overlap alone is NOT proof of element overlap: disjoint column-major slices of one parent
/// interleave in memory. Only provable overlap rejects: both regions contiguous, or identical base
/// pointers (see the eager guard in TensorAlgebra's Backends/Dispatch.hpp for the full rationale).
/// Regions are compared in bytes, so operands of different element types are handled alike.
template <typename CType, typename AType, typename BType>
void reject_output_alias(ParsedEinsumSpec const &parsed, CType const &C, AType const &A, BType const &B) {
    struct Region {
        unsigned char const *lo;
        unsigned char const *hi;
        bool                 contiguous;
    };
    auto const region_of = [](auto const &t) -> Region {
        auto const *lo     = reinterpret_cast<unsigned char const *>(t.data());
        size_t      last   = 0;
        size_t      nelems = 1;
        for (size_t d = 0; d < detail::tensor_rank(t); d++) {
            last += (t.dim(d) - 1) * t.stride(d);
            nelems *= t.dim(d);
        }
        return {lo, lo + (last + 1) * sizeof(*t.data()), last + 1 == nelems};
    };
    auto const c_region   = region_of(C);
    auto const overlaps_c = [&](auto const &x) {
        auto const r = region_of(x);
        if (!(r.lo < c_region.hi && c_region.lo < r.hi)) {
            return false;
        }
        return (r.contiguous && c_region.contiguous) || r.lo == c_region.lo;
    };
    if ((overlaps_c(A) && parsed.a_indices != parsed.c_indices) || (overlaps_c(B) && parsed.b_indices != parsed.c_indices)) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                "einsum: output tensor overlaps an input operand. In-place einsum is only supported for pure "
                                "elementwise updates (the aliased operand's index list identical to the output's); for this "
                                "contraction, pass a separate output tensor or copy the input first.");
    }
}

/// @brief Fold an operand's repeated letters into one axis each, as a view of the same storage.
///
/// A letter that appears twice in one operand ('iik') walks every occurrence together, so it is a
/// single axis whose step is the sum of the occurrences' strides: A(i, i, k) is the rank-2 view with
/// strides (s0 + s1, s2). Folding costs no copy, and the folded operand has each letter once, which
/// is what every fast path in @ref string_einsum assumes. @p folded receives the letters in order of
/// first appearance, @p impl the view.
///
/// Returns false, touching nothing the caller uses, when two occurrences of a letter disagree on
/// the extent. The capture and eager paths reject such a spec before dispatch
/// (validate_einsum_dims), so this only keeps a caller that skipped them on the loop, which walks
/// the first occurrence's extent as it always has.
template <typename T, typename TensorType>
bool fold_repeated_letters(std::vector<std::string> const &idx, TensorType const &t, T *data, std::vector<std::string> &folded,
                           einsums::detail::TensorImpl<T> &impl) {
    std::vector<std::string> letters;
    std::vector<size_t>      dims;
    std::vector<size_t>      strides;
    for (size_t p = 0; p < idx.size(); p++) {
        auto const at = std::ranges::find(letters, idx[p]);
        if (at == letters.end()) {
            letters.push_back(idx[p]);
            dims.push_back(t.dim(p));
            strides.push_back(t.stride(p));
            continue;
        }
        auto const k = static_cast<size_t>(at - letters.begin());
        if (dims[k] != t.dim(p)) {
            return false;
        }
        strides[k] += t.stride(p);
    }
    folded = std::move(letters);
    impl   = einsums::detail::TensorImpl<T>(data, dims, strides);
    return true;
}

/// @brief Sum an operand over its lone letters, the ones in no other operand and not in C.
///
/// A lone letter is a reduction over that operand alone ("ij <- ikm ; j" sums A over k and m), and
/// summing it first turns a loop over every (target, lone) pair into one pass over the operand
/// followed by a smaller contraction the fast paths can take. @p kept receives the remaining
/// letters in their original order, @p scratch the summed values, column-major, and @p impl a view
/// of them.
///
/// Returns false when the operand has no lone letter, or when every letter is lone: that operand
/// would sum to a scalar, which the routes do not take as an operand, and the loop already reads
/// it once per target element, the same cost as reducing it.
template <typename T, typename TensorType>
bool reduce_lone_letters(std::vector<std::string> const &idx, TensorType const &t, std::vector<std::string> const &other,
                         std::vector<std::string> const &c_idx, std::vector<std::string> &kept, std::vector<T> &scratch,
                         einsums::detail::TensorImpl<T> &impl) {
    auto const contains = [](std::vector<std::string> const &v, std::string const &x) { return std::ranges::find(v, x) != v.end(); };

    size_t const        rank = idx.size();
    std::vector<size_t> out_step(rank, 0);
    std::vector<size_t> dims;
    size_t              total = 1;
    for (size_t p = 0; p < rank; p++) {
        if (contains(other, idx[p]) || contains(c_idx, idx[p])) {
            kept.push_back(idx[p]);
            out_step[p] = total;
            dims.push_back(t.dim(p));
            total *= t.dim(p);
        }
    }
    if (kept.size() == rank || kept.empty()) {
        kept.clear();
        return false;
    }

    scratch.assign(total, T{0});
    T const *data = t.data();

    // One pass over the operand: every element is added into the slot its kept letters name. Axis 0
    // is a tight inner loop, which vectorizes for the usual column-major operand whether that axis
    // is kept (a unit step into the result) or summed (a fixed slot); the outer axes step an
    // odometer once per run. The dispatcher has already dealt with a zero extent.
    size_t const        n0 = t.dim(0), in0 = t.stride(0), out0 = out_step[0];
    size_t              runs = 1;
    std::vector<size_t> value(rank, 0);
    for (size_t p = 1; p < rank; p++) {
        runs *= t.dim(p);
    }
    size_t in_off = 0, out_off = 0;
    for (size_t r = 0; r < runs; r++) {
        T const *in  = data + in_off;
        T       *out = scratch.data() + out_off;
        if (out0 == 0) {
            T sum = T{0};
            for (size_t i = 0; i < n0; i++) {
                sum += in[i * in0];
            }
            *out += sum;
        } else {
            for (size_t i = 0; i < n0; i++) {
                out[i * out0] += in[i * in0];
            }
        }
        for (size_t p = 1; p < rank; p++) {
            in_off += t.stride(p);
            out_off += out_step[p];
            if (++value[p] < t.dim(p)) {
                break;
            }
            in_off -= t.stride(p) * t.dim(p);
            out_off -= out_step[p] * t.dim(p);
            value[p] = 0;
        }
    }

    impl = einsums::detail::TensorImpl<T>(scratch.data(), dims, false);
    return true;
}

// ── Main dispatch function ──────────────────────────────────────────────────

/**
 * @brief Execute a contraction described by a parsed string spec, on runtime-rank operands.
 *
 * Classifies the contraction at run time and takes the first route that fits: a dot product, GEMV,
 * GER, GEMM or direct product on the operands' TensorImpls (an operand whose letters are C's in
 * another order is permuted into C's first), then PackedGemm, then the generic loop.
 * Repeated letters are folded into strided views first (@ref fold_repeated_letters) and the folded
 * contraction takes the same routes; an operand with lone summed letters is summed over them first
 * (@ref reduce_lone_letters) and the smaller contraction takes the same routes too.
 *
 * Every caller hands it runtime-rank views: the graph's replay executors, and eager ``cg::einsum``
 * through @ref erased_string_einsum, which is what code holding typed tensors calls.
 */
template <BasicTensorConcept AType, BasicTensorConcept BType, BasicTensorConcept CType>
    requires std::is_same_v<typename AType::ValueType, typename BType::ValueType> &&
             std::is_same_v<typename AType::ValueType, typename CType::ValueType> &&
             (!HasCompileTimeRank<AType> && !HasCompileTimeRank<BType> && !HasCompileTimeRank<CType>)
void string_einsum(ParsedEinsumSpec const &parsed, typename AType::ValueType c_pf, CType *C, typename AType::ValueType ab_pf,
                   AType const &A, BType const &B, bool conj_a = false, bool conj_b = false,
                   std::vector<std::string> const *precomputed_links = nullptr, packed_gemm::ContractionSite *pg_site = nullptr) {
    using T = typename AType::ValueType;

    // Conjugation is the identity on a real type. Dropping the flags here keeps
    // a real spec written with them on the BLAS routes below, which a set flag
    // would otherwise skip for PackedGemm or the generic loop.
    if constexpr (!IsComplexV<T>) {
        conj_a = false;
        conj_b = false;
    }

    // A permutation operator contracts ONCE and accumulates the result into C
    // several times, transposed and signed. The contraction cannot go straight
    // into C, because the permuted accumulations would then read what they are
    // writing, so it goes to a temporary shaped like C.
    //
    // This is the un-lowered path: the AntisymmetrizerExpansion pass turns a
    // captured node into an explicit contraction plus permutes, which is what
    // makes the temporary a graph-managed buffer rather than one allocated per
    // call. What runs here is the eager path and the replay of a graph nobody
    // optimized, and both have to be correct on their own.
    if (!parsed.operators.empty()) {
        ParsedEinsumSpec base = parsed;
        base.operators.clear();

        std::size_t total = 1;
        for (std::size_t d = 0; d < detail::tensor_rank(*C); d++) {
            total *= C->dim(d);
        }

        std::vector<T>           scratch(total);
        std::vector<std::size_t> dims(detail::tensor_rank(*C));
        for (std::size_t d = 0; d < dims.size(); d++) {
            dims[d] = C->dim(d);
        }
        einsums::detail::TensorImpl<T> temp_impl(scratch.data(), dims, C->impl().is_row_major());

        einsums::RuntimeTensorView<T> temp(temp_impl);

        // c_pf = 0: the base result is the whole content of the temporary, so
        // the accumulation below is the only place C is touched.
        string_einsum(base, T{0}, &temp, ab_pf, A, B, conj_a, conj_b, precomputed_links, pg_site);

        // Name the route as the operator wrapped around whatever kernel the base
        // contraction reached, so a test can assert BOTH that the operator fired
        // and that the contraction underneath it still took its fast path. A
        // flat "antisymmetrized" would hide a fast-path regression.
        static thread_local std::string route;
        route                  = std::string("antisymmetrized:") + last_dispatch_route();
        char const *base_route = route.c_str();

        auto const terms = expand_permutation_operators(parsed.c_indices, parsed.operators);

        ParsedPermuteSpec term_spec;
        term_spec.a_indices = parsed.c_indices; // the temporary carries C's index order
        term_spec.raw       = parsed.raw;
        // ab_pf was already applied by the base contraction, so each term carries
        // only its sign. Applying it here as well would square it.
        for (std::size_t t = 0; t < terms.size(); ++t) {
            term_spec.c_indices = terms[t].c_indices;
            string_permute_impl<T>(term_spec, t == 0 ? c_pf : T{1}, &C->impl(), static_cast<T>(terms[t].sign), temp_impl);
        }

        last_dispatch_route() = base_route;
        return;
    }

    WAGGLE_ZONE("cg::einsum: {} <- {} ; {}", fmt::join(parsed.c_indices, ","), fmt::join(parsed.a_indices, ","),
                fmt::join(parsed.b_indices, ","));
    // As integers, not std::to_string: the string form was BUILT before
    // annotate() could check whether anything was recording, so every
    // contraction paid three of them whether or not they went anywhere.
    WAGGLE_ANNOTATE("a_rank", static_cast<int64_t>(detail::tensor_rank(A)));
    WAGGLE_ANNOTATE("b_rank", static_cast<int64_t>(detail::tensor_rank(B)));
    WAGGLE_ANNOTATE("c_rank", static_cast<int64_t>(detail::tensor_rank(*C)));

    auto const &c_idx = parsed.c_indices;
    auto const &a_idx = parsed.a_indices;
    auto const &b_idx = parsed.b_indices;

    // Graph executors pass the links computed once at capture
    // (EinsumIndices::link_indices) so replays don't recompute them; the
    // eager path derives them here.
    std::vector<std::string> const  links_storage = precomputed_links == nullptr ? parsed.link_indices() : std::vector<std::string>{};
    std::vector<std::string> const &links         = precomputed_links != nullptr ? *precomputed_links : links_storage;

    if (einsum_empty_operands(c_pf, C, A, B)) {
        return;
    }

    reject_output_alias(parsed, *C, A, B);

    // Repeated letters within one operand ('ij <- iik ; kj') are diagonal
    // accesses. Every fast path below classifies indices assuming each
    // letter appears at most once per operand - the outer-product/GER
    // routes silently computed wrong values for these specs - so a repeated
    // letter is folded into one strided axis first (fold_repeated_letters)
    // and the folded contraction is dispatched like any other: a diagonal
    // feeding a GEMM runs as that GEMM, instead of the scalar loop that used
    // to take every such spec at 100-200x the GEMM's time.
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
        bool const                     c_repeats = has_repeated_letter(c_idx);
        ParsedEinsumSpec               folded    = parsed;
        einsums::detail::TensorImpl<T> a_view, b_view, c_view;

        // The inputs are only read; the views are non-const because TensorImpl is.
        bool fold = fold_repeated_letters(a_idx, A, const_cast<T *>(A.data()), folded.a_indices, a_view) &&
                    fold_repeated_letters(b_idx, B, const_cast<T *>(B.data()), folded.b_indices, b_view) &&
                    fold_repeated_letters(c_idx, *C, C->data(), folded.c_indices, c_view);

        // c_pf scales ALL of C, not only the diagonal a repeated output letter
        // writes, so that case scales C up front and folds with c_pf = 1. An
        // input sharing C's storage would then be read after the scaling; the
        // loop copies such an input first, so it keeps that case. The interval
        // test is conservative, which costs only the fold.
        if (fold && c_repeats) {
            auto const bytes_of = [](auto const &t) {
                size_t last = 0;
                for (size_t d = 0; d < detail::tensor_rank(t); d++) {
                    last += (t.dim(d) - 1) * t.stride(d);
                }
                auto const *lo = reinterpret_cast<unsigned char const *>(t.data());
                return std::pair{lo, lo + (last + 1) * sizeof(*t.data())};
            };
            auto const [c_lo, c_hi] = bytes_of(*C);
            auto const overlaps_c   = [&](auto const &t) {
                auto const [lo, hi] = bytes_of(t);
                return lo < c_hi && c_lo < hi;
            };
            fold = !overlaps_c(A) && !overlaps_c(B);
        }

        if (fold) {
            T folded_c_pf = c_pf;
            if (c_repeats) {
                if (c_pf == T{0}) {
                    C->zero();
                } else if (c_pf != T{1}) {
                    linear_algebra::scale(c_pf, C);
                }
                folded_c_pf = T{1};
            }
            einsums::RuntimeTensorView<T> a_folded(a_view), b_folded(b_view), c_folded(c_view);
            string_einsum(folded, folded_c_pf, &c_folded, ab_pf, a_folded, b_folded, conj_a, conj_b, nullptr, pg_site);

            // Name the fold AND the kernel the folded contraction reached, so a
            // test can assert both, as the antisymmetrized route does.
            static thread_local std::string route;
            route                 = std::string("diagonal:") + last_dispatch_route();
            last_dispatch_route() = route.c_str();
            return;
        }

        WAGGLE_ANNOTATE("dispatch", "generic_loop_repeated_indices");
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
    // declines it). The operand is summed over its lone letters first
    // (reduce_lone_letters) and the smaller contraction is dispatched like any
    // other (otherwise "ij <- ikm ; j" is n^4 work in the loop for an n^3
    // answer). This runs before every fast path so
    // both the empty-link case ("ij <- ijk ; ij") and the link+lone case land here.
    auto const has_lone_summed_index = [&] {
        for (auto const *idx : {&a_idx, &b_idx}) {
            for (auto const &s : *idx) {
                if (auto const role = index_role(s, c_idx, a_idx, b_idx); role == IndexRole::ALone || role == IndexRole::BLone) {
                    return true;
                }
            }
        }
        return false;
    };
    if (has_lone_summed_index()) {
        ParsedEinsumSpec               reduced = parsed;
        std::vector<T>                 a_scratch, b_scratch;
        einsums::detail::TensorImpl<T> a_impl, b_impl;
        std::vector<std::string>       a_kept, b_kept;

        bool const a_reduced = reduce_lone_letters(a_idx, A, b_idx, c_idx, a_kept, a_scratch, a_impl);
        bool const b_reduced = reduce_lone_letters(b_idx, B, a_idx, c_idx, b_kept, b_scratch, b_impl);
        // An operand whose letters are all lone reduces to a scalar and is not reduced (see
        // reduce_lone_letters), so a lone letter can survive; the loop takes those.
        bool const lone_left = [&] {
            for (auto const *idx : {a_reduced ? &a_kept : &a_idx, b_reduced ? &b_kept : &b_idx}) {
                for (auto const &s : *idx) {
                    auto const role = index_role(s, c_idx, a_reduced ? a_kept : a_idx, b_reduced ? b_kept : b_idx);
                    if (role == IndexRole::ALone || role == IndexRole::BLone) {
                        return true;
                    }
                }
            }
            return false;
        }();

        if ((a_reduced || b_reduced) && !lone_left) {
            if (a_reduced) {
                reduced.a_indices = a_kept;
            }
            if (b_reduced) {
                reduced.b_indices = b_kept;
            }
            // The reduced operands are fresh buffers, so they alias nothing; an unreduced operand
            // is passed as a view of itself so the call has one type combination.
            einsums::detail::TensorImpl<T> const a_view = a_reduced ? a_impl : A.impl();
            einsums::detail::TensorImpl<T> const b_view = b_reduced ? b_impl : B.impl();
            einsums::RuntimeTensorView<T>        a_arg(a_view), b_arg(b_view), c_arg(C->impl());
            string_einsum(reduced, c_pf, &c_arg, ab_pf, a_arg, b_arg, conj_a, conj_b, nullptr, pg_site);

            static thread_local std::string route;
            route                 = std::string("lone_reduced:") + last_dispatch_route();
            last_dispatch_route() = route.c_str();
            return;
        }

        WAGGLE_ANNOTATE("dispatch", "generic_loop_lone_summed");
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
            temp = linear_algebra::dot(A, B);
        }

        bool const conjugating = conj_a || conj_b;
        WAGGLE_ANNOTATE("dispatch", conjugating ? "true_dot_runtime" : "dot_runtime");
        last_dispatch_route() = conjugating ? "true_dot_runtime" : "dot_runtime";
        // A zero output prefactor assigns rather than multiplies, as on every other route: 0 * NaN is
        // NaN, so a dot into a never-written output would keep whatever it held.
        C->data()[0] = c_pf == T{0} ? ab_pf * temp : c_pf * C->data()[0] + ab_pf * temp;
        return;
    }

    // ── BLAS fast paths ─────────────────────────────────────────────────────
    // Each route passes the operands' TensorImpls straight to the rank-erased
    // linear_algebra kernels, which check the ranks at run time.
    //
    // Conjugation of a complex operand skips them: none of these kernels
    // conjugates. Conjugated contractions go to PackedGemm (native via spec.conj_a/conj_b) for
    // gemm-shaped cases, else the conj-aware generic loop.
    //
    // The rank-2 GEMM route is the exception to taking the first fit: a matrix
    // times a matrix is the one shape here that a thread's node width could have
    // spread, and a vendor GEMM issued under such a width is clamped to one thread
    // by the BLAS wrappers' fence. When PackedGemm is the preferred route it
    // stands aside and PackedGemm takes the shape, whose packed loops fork from
    // the ICV the width raised.
    //
    // Same answer the packed engine reaches, from the same call site's pin (@ref
    // packed_gemm::prefer_packed_route): a caller whose node has a pinned route
    // gets it here too, so the shape cannot take one route at this gate and the
    // other inside try_packed_gemm. A caller with no site - eager, or an
    // unplanned graph - reads the thread regime.
    if (!conj_a && !conj_b) {
        bool const        route_prefers_packed = packed_gemm::prefer_packed_route(pg_site);
        std::size_t const a_rank               = detail::tensor_rank(A);
        std::size_t const b_rank               = detail::tensor_rank(B);
        std::size_t const c_rank               = detail::tensor_rank(*C);

        // Every route below hands the operands' own TensorImpls to the rank-erased kernels.
        // Building TensorView<T, K> wrappers instead costs about 230 ns for three operands,
        // a third of a small contraction's eager call.
        namespace la = linear_algebra::detail;

        // The GEMV and GEMM routes hand their matrices to BLAS, which needs a unit stride along
        // one axis. Without one (a diagonal folded out of a rank-3 operand, a stepped slice) the
        // rank-erased kernels fall back to a hand-written loop, so such a contraction goes on to
        // PackedGemm instead, which packs from any strides and keeps it (packed_gemm::blas_addressable).
        auto const addressable = [](auto const &t) { return packed_gemm::blas_addressable(t); };

        // ── GEMV: matrix × vector → vector ───────────────────────────
        if (a_rank == 2 && b_rank == 1 && c_rank == 1) {
            if (links.size() == 1 && addressable(A)) {
                WAGGLE_ANNOTATE("dispatch", "gemv_mat_vec_runtime");
                last_dispatch_route() = "gemv_mat_vec_runtime";
                char const trans      = (a_idx[0] == links[0]) ? 't' : 'n';
                la::gemv(trans, ab_pf, A.impl(), B.impl(), c_pf, &C->impl());
                return;
            }
        }

        // ── GEMV: vector × matrix → vector ───────────────────────────
        if (a_rank == 1 && b_rank == 2 && c_rank == 1) {
            if (links.size() == 1 && addressable(B)) {
                WAGGLE_ANNOTATE("dispatch", "gemv_vec_mat_runtime");
                last_dispatch_route() = "gemv_vec_mat_runtime";
                char const trans      = (b_idx[1] == links[0]) ? 'n' : 't';
                la::gemv(trans, ab_pf, B.impl(), A.impl(), c_pf, &C->impl());
                return;
            }
        }

        // ── GER: vector × vector → matrix ────────────────────────────
        if (a_rank == 1 && b_rank == 1 && c_rank == 2) {
            if (links.empty()) {
                WAGGLE_ANNOTATE("dispatch", "ger_runtime");
                last_dispatch_route() = "ger_runtime";
                if (c_pf != T{1}) {
                    la::scale(c_pf, &C->impl());
                }
                // ger(x, y, C) computes C[i,j] = x[i]*y[j], so the operand whose
                // index labels C's first axis must be x. Swap for a transposed
                // output (spec like "ji <- i ; j", where C's axes are ordered
                // opposite to the A-then-B operand order).
                if (c_idx[0] == a_idx[0]) {
                    la::ger(ab_pf, A.impl(), B.impl(), &C->impl());
                } else {
                    la::ger(ab_pf, B.impl(), A.impl(), &C->impl());
                }
                return;
            }
        }

        // ── GEMM: matrix × matrix → matrix ───────────────────────────
        if (a_rank == 2 && b_rank == 2 && c_rank == 2) {
            if (links.size() == 1 && !route_prefers_packed && addressable(A) && addressable(B) && addressable(*C)) {
                WAGGLE_ANNOTATE("dispatch", "gemm_direct_runtime");
                last_dispatch_route() = "gemm_direct_runtime";
                // C = [freeA, freeB] is op(A) op(B); the transposed output C = [freeB, freeA]
                // is op(B) op(A). Honoring C's order is what keeps a transposed-output
                // contraction such as "ia <- ma ; mi" off gemm's dimension check.
                std::string const &k     = links[0];
                std::string const &freeA = (a_idx[0] == k) ? a_idx[1] : a_idx[0];
                if (c_idx[0] == freeA) {
                    char const ta = (a_idx[0] == k) ? 't' : 'n';
                    char const tb = (b_idx[1] == k) ? 't' : 'n';
                    la::gemm(ta, tb, ab_pf, A.impl(), B.impl(), c_pf, &C->impl());
                } else {
                    char const tb = (b_idx[0] == k) ? 't' : 'n';
                    char const ta = (a_idx[1] == k) ? 't' : 'n';
                    la::gemm(tb, ta, ab_pf, B.impl(), A.impl(), c_pf, &C->impl());
                }
                return;
            }
        }

        // ── Direct product at ANY rank ───────────────────────────────
        // The rank-erased kernel takes any rank, so the ranks are only
        // checked here.
        if (a_rank == b_rank && b_rank == c_rank && links.empty() && a_idx == b_idx && a_idx == c_idx) {
            WAGGLE_ANNOTATE("dispatch", "direct_product_runtime");
            last_dispatch_route() = "direct_product_runtime";
            la::direct_product(ab_pf, A.impl(), B.impl(), c_pf, &C->impl());
            return;
        }

        // ── Direct product with operands in another letter order ─────
        // "ij <- ij ; ji" is a direct product too, but no route above takes it:
        // the one above compares index lists, and PackedGemm finds no M, N or K
        // axis to pack. With no links and equal ranks, the repeated and lone
        // letters already handled above, each operand's letters are C's in
        // another order. Each such operand is permuted into a buffer laid out
        // like C, so the kernel runs its vectorized lock-step pass: a strided
        // view of the operand would send it to the per-element strided loop.
        if (a_rank == b_rank && b_rank == c_rank && links.empty()) {
            std::size_t         total = 1;
            std::vector<size_t> dims(c_rank);
            for (std::size_t d = 0; d < c_rank; d++) {
                dims[d] = C->dim(d);
                total *= dims[d];
            }
            bool const row_major = C->impl().is_row_major();

            std::vector<T>                 a_scratch, b_scratch;
            einsums::detail::TensorImpl<T> a_impl = A.impl(), b_impl = B.impl();
            auto const                     in_c_order = [&](std::vector<std::string> const &idx, einsums::detail::TensorImpl<T> const &src,
                                                            std::vector<T> &scratch, einsums::detail::TensorImpl<T> &out) {
                if (idx == c_idx) {
                    return;
                }
                scratch.resize(total);
                out = einsums::detail::TensorImpl<T>(scratch.data(), dims, row_major);
                ParsedPermuteSpec spec;
                spec.c_indices = c_idx;
                spec.a_indices = idx;
                spec.raw       = parsed.raw;
                string_permute_impl<T>(spec, T{0}, &out, T{1}, src);
            };
            in_c_order(a_idx, A.impl(), a_scratch, a_impl);
            in_c_order(b_idx, B.impl(), b_scratch, b_impl);

            WAGGLE_ANNOTATE("dispatch", "direct_product_permuted_runtime");
            last_dispatch_route() = "direct_product_permuted_runtime";
            la::direct_product(ab_pf, a_impl, b_impl, c_pf, &C->impl());
            return;
        }
    } // end of the !conj_a && !conj_b BLAS fast-path gate

    // ── PackedGemm path ─────────────────────────────────────────────────────
    // Handles arbitrary-rank GEMM-shaped contractions, including those with
    // batch (Hadamard) indices appearing in A, B, AND C, through the runtime
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
            // The target and all-index lists are left empty: try_packed_gemm derives both in C's
            // own order, which is the order its target space needs.
            built.conj_a = conj_a; // PackedGemm conjugates during packing/transpose (native, no copy)
            built.conj_b = conj_b;
        }
        packed_gemm::ContractionSpec const &spec = reuse_spec ? pg_site->key.spec : built;

        if (packed_gemm::try_packed_gemm<AType, BType, CType>(spec, c_pf, C, ab_pf, A, B, /*allow_scatter=*/true, pg_site)) {
            WAGGLE_ANNOTATE("dispatch", "packed_gemm");
            last_dispatch_route() = "packed_gemm";
            return;
        }
    }

    // ── Generic fallback: runtime nested-loop contraction ──────────
    // Reached when no fast path applies: pure outer products, mixed-dtype
    // edge cases, or contractions that even PackedGemm can't form into a
    // valid GEMM shape (no M-dims, no N-dims, no links).
    WAGGLE_ANNOTATE("dispatch", "generic_loop");
    last_dispatch_route() = "generic_loop";
    generic_string_einsum(parsed, links, c_pf, C, ab_pf, A, B, conj_a, conj_b);
}

/**
 * @brief An einsum whose operands do not all share one element type.
 *
 * The products and the sum are formed in the accumulator type promoted from A and B (complex if
 * either is, at the wider precision; see Detail/MixedPrecision.hpp), and the result is rounded to
 * C's type. There are no BLAS or PackedGemm routines over mixed types, so every such call runs the
 * generic loop; the zero-extent rule and the output-aliasing policy are the ones string_einsum
 * applies. Permutation operators are not supported here: their temporary goes through the
 * single-type permute.
 *
 * @throws std::invalid_argument for a spec with permutation operators, or for output aliasing.
 */
template <BasicTensorConcept AType, BasicTensorConcept BType, BasicTensorConcept CType>
    requires detail::EinsumElement<typename AType::ValueType> && detail::EinsumElement<typename BType::ValueType> &&
             detail::EinsumElement<typename CType::ValueType> &&
             (!(std::is_same_v<typename AType::ValueType, typename BType::ValueType> &&
                std::is_same_v<typename AType::ValueType, typename CType::ValueType>))
void mixed_string_einsum(ParsedEinsumSpec const &parsed, typename CType::ValueType c_pf, CType *C,
                         detail::PromoteT<typename AType::ValueType, typename BType::ValueType> ab_pf, AType const &A, BType const &B,
                         bool conj_a = false, bool conj_b = false, std::vector<std::string> const *precomputed_links = nullptr) {
    static_assert(detail::storable_v<detail::PromoteT<typename AType::ValueType, typename BType::ValueType>, typename CType::ValueType>,
                  "einsum: a complex operand makes the contraction complex, and a real output cannot hold it; make the output complex");
    if (!parsed.operators.empty()) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                "einsum '{}': permutation operators are not supported when the operands' element types differ; contract "
                                "into a tensor of one type first, or convert the operands",
                                parsed.raw);
    }
    WAGGLE_ZONE("cg::einsum (mixed precision): {} <- {} ; {}", fmt::join(parsed.c_indices, ","), fmt::join(parsed.a_indices, ","),
                fmt::join(parsed.b_indices, ","));

    std::vector<std::string> const  links_storage = precomputed_links == nullptr ? parsed.link_indices() : std::vector<std::string>{};
    std::vector<std::string> const &links         = precomputed_links != nullptr ? *precomputed_links : links_storage;

    if (einsum_empty_operands(c_pf, C, A, B)) {
        return;
    }
    reject_output_alias(parsed, *C, A, B);

    WAGGLE_ANNOTATE("dispatch", "generic_loop_mixed_precision");
    last_dispatch_route() = "generic_loop_mixed_precision";
    generic_string_einsum(parsed, links, c_pf, C, ab_pf, A, B, conj_a, conj_b);
}

// ═════════════════════════════════════════════════════════════════════════════
// String-based permute dispatch
// ═════════════════════════════════════════════════════════════════════════════

/**
 * @brief Execute a tensor permutation described by a string specification, over
 *        rank-erased impls.
 *
 * C[c_indices] = beta * C[c_indices] + alpha * A[a_indices]
 *
 * Builds the index permutation mapping at runtime and performs a stride-based copy.
 *
 * This is where the operation actually lives, and it is written against
 * ``TensorImpl`` rather than against tensor objects because an impl carries
 * data, dims and strides as runtime values: one definition then serves every
 * rank and every static tensor type. It is also what lets @ref build_executor
 * run a permute without first wrapping its operands in a tensor object, which
 * would copy two ``ShapeVector`` objects per operand on every replay.
 *
 * @tparam T Element type shared by both operands.
 * @param[in] parsed The index lists, and the raw spec for diagnostics.
 * @param[in] beta Prefactor on the destination; 0 overwrites it.
 * @param[in,out] C Destination geometry.
 * @param[in] alpha Prefactor on the source.
 * @param[in] A Source geometry.
 * @throws std::runtime_error When an output index does not appear in the input.
 */
template <typename T>
void string_permute_impl(ParsedPermuteSpec const &parsed, T beta, einsums::detail::TensorImpl<T> *C, T alpha,
                         einsums::detail::TensorImpl<T> const &A) {
    // A permutation operator expands into several signed accumulations of the
    // same source. No temporary is needed here, unlike the einsum case: every
    // term reads A, which this operation never writes, so the terms accumulate
    // into C directly. The destination prefactor belongs to the FIRST term only,
    // which is what keeps `C = beta*C + alpha*P[...](A)` from scaling C once per
    // term. The identity term is always first, so that term is also the one that
    // may legitimately overwrite.
    if (!parsed.operators.empty()) {
        auto const terms = expand_permutation_operators(parsed.c_indices, parsed.operators);

        ParsedPermuteSpec term_spec;
        term_spec.a_indices = parsed.a_indices;
        term_spec.raw       = parsed.raw;
        for (std::size_t t = 0; t < terms.size(); ++t) {
            term_spec.c_indices = terms[t].c_indices;
            string_permute_impl<T>(term_spec, t == 0 ? beta : T{1}, C, static_cast<T>(terms[t].sign) * alpha, A);
        }
        return;
    }

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
    // Contiguity alone is NOT enough: a permute_view spans the whole buffer
    // but presents reordered strides, and the stride-ratio outerSize
    // derivation assumes strides monotone in the layout flag's direction
    // (the same trap detail::strides_follow_layout guards).
    if (rank >= 2 && compute_graph::detail::canonical_dense(*C) && compute_graph::detail::canonical_dense(A)) {
        std::vector<int> const c_to_a(perm.begin(), perm.end());
        tensor_permute::detail::permute<false, T>(beta, std::span<int const>{c_to_a}, C, alpha, A);
        return;
    }

    // Scale C by beta. `copy_to` rather than a tensor's own `zero()`: it writes
    // exactly the elements this geometry describes, which is what a strided
    // destination needs, and it lands on the same all-zero bits an owning
    // tensor's memset would.
    if (beta == T{0}) {
        einsums::detail::copy_to(T{0}, *C);
    } else if (beta != T{1}) {
        linear_algebra::detail::scale(beta, C);
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
