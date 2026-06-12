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

#include <Einsums/ComputeGraph/EinsumSpec.hpp>
#include <Einsums/ComputeGraph/TensorRank.hpp>
#include <Einsums/Concepts/TensorConcepts.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/LinearAlgebra.hpp>
#include <Einsums/PackedGemm/EinsumPackedGemm.hpp>
#include <Einsums/Profile.hpp>

#include <fmt/format.h>

#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace einsums::compute_graph::dispatch {

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
void string_gemm(ParsedEinsumSpec const &parsed, T c_pf, CType *C, T ab_pf, AType const &A, BType const &B) {
    auto links = parsed.link_indices();
    if (links.size() != 1) {
        EINSUMS_THROW_EXCEPTION(std::runtime_error, "String einsum '{}': GEMM requires exactly 1 link index, got {}", parsed.raw,
                                links.size());
    }

    std::string const &k = links[0];

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

/**
 * @brief Generic runtime nested-loop contraction for arbitrary rank/pattern.
 *
 * Handles any contraction pattern by iterating over all target index
 * combinations (outer loops) and link index combinations (inner summation).
 * Uses stride-based offset computation for efficient element access.
 *
 * Performance note: This is O(product(target_dims) * product(link_dims)),
 * with no BLAS optimization. Use for patterns not covered by specialized
 * BLAS dispatch (rank-3+, multi-link, etc.).
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
        int         pos_in_a{-1}; // Position in A's index list, or -1
        int         pos_in_b{-1};
        int         pos_in_c{-1};
        bool        is_link{false}; // True if this is a contraction (link) index
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
        // Find position in each tensor's index list
        for (size_t p = 0; p < a_idx.size(); p++) {
            if (a_idx[p] == name) {
                info.pos_in_a = static_cast<int>(p);
                break;
            }
        }
        for (size_t p = 0; p < b_idx.size(); p++) {
            if (b_idx[p] == name) {
                info.pos_in_b = static_cast<int>(p);
                break;
            }
        }
        for (size_t p = 0; p < c_idx.size(); p++) {
            if (c_idx[p] == name) {
                info.pos_in_c = static_cast<int>(p);
                break;
            }
        }
        // Get dimension size from whichever tensor has this index
        if (info.pos_in_a >= 0)
            info.dim_size = A.dim(info.pos_in_a);
        else if (info.pos_in_b >= 0)
            info.dim_size = B.dim(info.pos_in_b);
        else if (info.pos_in_c >= 0)
            info.dim_size = C->dim(info.pos_in_c);
        all_indices.push_back(info);
    };

    for (auto const &t : targets)
        add_index(t, false);
    for (auto const &l : links)
        add_index(l, true);

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

    // Scale C by c_pf
    if (c_pf == T{0}) {
        C->zero();
    } else if (c_pf != T{1}) {
        linear_algebra::scale(c_pf, C);
    }

    // Iterate over all target index combinations
    std::vector<size_t> idx_values(all_indices.size(), 0);

    for (size_t target_flat = 0; target_flat < target_total; target_flat++) {
        // Decode target_flat into per-index values
        {
            size_t remaining = target_flat;
            for (int t = static_cast<int>(target_infos.size()) - 1; t >= 0; t--) {
                // NOLINTNEXTLINE(misc-redundant-expression)
                size_t pos = &all_indices[0] - &all_indices[0]; // find position
                for (size_t ai = 0; ai < all_indices.size(); ai++) {
                    if (&all_indices[ai] == target_infos[t]) {
                        pos = ai;
                        break;
                    }
                }
                idx_values[pos] = remaining % target_infos[t]->dim_size;
                remaining /= target_infos[t]->dim_size;
            }
        }

        // Compute C offset for this target combination
        size_t c_offset = 0;
        for (auto const &info : all_indices) {
            if (info.pos_in_c >= 0) {
                size_t idx_pos = &info - &all_indices[0];
                c_offset += idx_values[idx_pos] * C->stride(info.pos_in_c);
            }
        }

        // Sum over link indices
        T sum = T{0};
        for (size_t link_flat = 0; link_flat < link_total; link_flat++) {
            // Decode link_flat into per-index values
            {
                size_t remaining = link_flat;
                for (int l = static_cast<int>(link_infos.size()) - 1; l >= 0; l--) {
                    size_t pos = 0;
                    for (size_t ai = 0; ai < all_indices.size(); ai++) {
                        if (&all_indices[ai] == link_infos[l]) {
                            pos = ai;
                            break;
                        }
                    }
                    idx_values[pos] = remaining % link_infos[l]->dim_size;
                    remaining /= link_infos[l]->dim_size;
                }
            }

            // Compute A offset
            size_t a_offset = 0;
            for (auto const &info : all_indices) {
                if (info.pos_in_a >= 0) {
                    size_t idx_pos = &info - &all_indices[0];
                    a_offset += idx_values[idx_pos] * A.stride(info.pos_in_a);
                }
            }

            // Compute B offset
            size_t b_offset = 0;
            for (auto const &info : all_indices) {
                if (info.pos_in_b >= 0) {
                    size_t idx_pos = &info - &all_indices[0];
                    b_offset += idx_values[idx_pos] * B.stride(info.pos_in_b);
                }
            }

            T a_val = A.data()[a_offset];
            T b_val = B.data()[b_offset];
            if constexpr (IsComplexV<T>) {
                if (conj_a) {
                    a_val = std::conj(a_val);
                }
                if (conj_b) {
                    b_val = std::conj(b_val);
                }
            }
            sum += a_val * b_val;
        }

        C->data()[c_offset] += ab_pf * sum;
    }
}

// ── Main dispatch function ──────────────────────────────────────────────────

template <BasicTensorConcept AType, BasicTensorConcept BType, BasicTensorConcept CType>
    requires requires {
        requires std::is_same_v<typename AType::ValueType, typename BType::ValueType>;
        requires std::is_same_v<typename AType::ValueType, typename CType::ValueType>;
    }
void string_einsum(ParsedEinsumSpec const &parsed, typename AType::ValueType c_pf, CType *C, typename AType::ValueType ab_pf,
                   AType const &A, BType const &B, bool conj_a = false, bool conj_b = false) {
    using T = typename AType::ValueType;

    LabeledSection("cg::einsum: {} <- {} ; {}", fmt::join(parsed.c_indices, ","), fmt::join(parsed.a_indices, ","),
                   fmt::join(parsed.b_indices, ","));
    ProfileAnnotate("a_rank", std::to_string(detail::tensor_rank(A)));
    ProfileAnnotate("b_rank", std::to_string(detail::tensor_rank(B)));
    ProfileAnnotate("c_rank", std::to_string(detail::tensor_rank(*C)));

    auto const &c_idx = parsed.c_indices;
    auto const &a_idx = parsed.a_indices;
    auto const &b_idx = parsed.b_indices;

    std::set<std::string> const a_set(a_idx.begin(), a_idx.end());
    std::set<std::string> const b_set(b_idx.begin(), b_idx.end());
    std::set<std::string>       c_set(c_idx.begin(), c_idx.end());

    std::vector<std::string> links;
    for (auto const &idx : a_set) {
        if (b_set.count(idx) && !c_set.count(idx)) {
            links.push_back(idx);
        }
    }

    // ── Rank-1 special-case BLAS fast paths ─────────────────────────────────
    // These call helpers (string_gemv_mat_vec, linear_algebra::ger, etc.)
    // that are themselves rank-specific (MatrixConcept / VectorConcept), so
    // they only compile for typed Tensor<T, K> operands and stay gated
    // behind HasCompileTimeRank. PackedGemm (below) handles the rank-2+
    // GEMM-shaped cases and works uniformly for typed and runtime-rank
    // tensors via the runtime ContractionSpec entry point.
    // Conjugation skips the non-conj BLAS fast paths below: those helpers (dot,
    // gemv, ger, string_gemm, direct_product) don't conjugate. Conjugated
    // contractions go to PackedGemm (native via spec.conj_a/conj_b) for
    // gemm-shaped cases, else the conj-aware generic loop.
    if (!conj_a && !conj_b) {
        if constexpr (HasCompileTimeRank<AType> && HasCompileTimeRank<BType> && HasCompileTimeRank<CType>) {
            constexpr size_t a_rank = std::remove_cvref_t<AType>::Rank;
            constexpr size_t b_rank = std::remove_cvref_t<BType>::Rank;
            constexpr size_t c_rank = std::remove_cvref_t<CType>::Rank;

            // ── DOT product: scalar output, all indices contracted ──────────
            if constexpr (a_rank == 1 && b_rank == 1 && c_rank == 1) {
                if (c_idx.empty() || (links.size() == a_idx.size())) {
                    ProfileAnnotate("dispatch", "dot");
                    T temp       = linear_algebra::dot(A, B);
                    C->data()[0] = c_pf * C->data()[0] + ab_pf * temp;
                    return;
                }
            }

            // ── GEMV: matrix × vector → vector ──────────────────────────────
            if constexpr (a_rank == 2 && b_rank == 1 && c_rank == 1) {
                if (links.size() == 1) {
                    ProfileAnnotate("dispatch", "gemv_mat_vec");
                    string_gemv_mat_vec(parsed, c_pf, C, ab_pf, A, B, a_idx, links[0]);
                    return;
                }
            }

            // ── GEMV: vector × matrix → vector ──────────────────────────────
            if constexpr (a_rank == 1 && b_rank == 2 && c_rank == 1) {
                if (links.size() == 1) {
                    ProfileAnnotate("dispatch", "gemv_vec_mat");
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
                if (links.size() == 1) {
                    ProfileAnnotate("dispatch", "gemm_direct");
                    string_gemm(parsed, c_pf, C, ab_pf, A, B);
                    return;
                }
                // Direct product: same indices on all tensors, no links
                if (links.empty() && a_set == b_set && a_set == c_set && a_idx == b_idx && a_idx == c_idx) {
                    ProfileAnnotate("dispatch", "direct_product");
                    linear_algebra::direct_product(ab_pf, A, B, c_pf, C);
                    return;
                }
            }
        }

        // ── Runtime-rank BLAS fast paths ────────────────────────────────────────
        // Mirror of the typed BLAS ladder above for runtime-rank operands. We
        // build a zero-copy TensorView<T, K> over the RuntimeTensor's data, whose
        // impl carries the same dims and strides, then call the same rank-
        // specialized BLAS helpers. The upcast is just a pointer plus a small
        // metadata array, with no allocation and no copy. It only fires when all
        // three operands are runtime-rank; mixed typed/runtime calls fall through
        // to PackedGemm or the generic loop.
        if constexpr (!HasCompileTimeRank<AType> && !HasCompileTimeRank<BType> && !HasCompileTimeRank<CType>) {
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
                    auto av      = upcast(A, std::integral_constant<std::size_t, 1>{});
                    auto bv      = upcast(B, std::integral_constant<std::size_t, 1>{});
                    T    temp    = linear_algebra::dot(av, bv);
                    C->data()[0] = c_pf * C->data()[0] + ab_pf * temp;
                    return;
                }
            }

            // ── GEMV: matrix × vector → vector ───────────────────────────
            if (a_rank == 2 && b_rank == 1 && c_rank == 1) {
                if (links.size() == 1) {
                    ProfileAnnotate("dispatch", "gemv_mat_vec_runtime");
                    auto av = upcast(A, std::integral_constant<std::size_t, 2>{});
                    auto bv = upcast(B, std::integral_constant<std::size_t, 1>{});
                    auto cv = upcast(*C, std::integral_constant<std::size_t, 1>{});
                    string_gemv_mat_vec(parsed, c_pf, &cv, ab_pf, av, bv, a_idx, links[0]);
                    return;
                }
            }

            // ── GEMV: vector × matrix → vector ───────────────────────────
            if (a_rank == 1 && b_rank == 2 && c_rank == 1) {
                if (links.size() == 1) {
                    ProfileAnnotate("dispatch", "gemv_vec_mat_runtime");
                    auto av    = upcast(A, std::integral_constant<std::size_t, 1>{});
                    auto bv    = upcast(B, std::integral_constant<std::size_t, 2>{});
                    auto cv    = upcast(*C, std::integral_constant<std::size_t, 1>{});
                    char trans = (b_idx[1] == links[0]) ? 'n' : 't';
                    linear_algebra::gemv(trans, ab_pf, bv, av, c_pf, &cv);
                    return;
                }
            }

            // ── GER: vector × vector → matrix ────────────────────────────
            if (a_rank == 1 && b_rank == 1 && c_rank == 2) {
                if (links.empty()) {
                    ProfileAnnotate("dispatch", "ger_runtime");
                    auto av = upcast(A, std::integral_constant<std::size_t, 1>{});
                    auto bv = upcast(B, std::integral_constant<std::size_t, 1>{});
                    auto cv = upcast(*C, std::integral_constant<std::size_t, 2>{});
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
                if (links.size() == 1) {
                    ProfileAnnotate("dispatch", "gemm_direct_runtime");
                    auto av = upcast(A, std::integral_constant<std::size_t, 2>{});
                    auto bv = upcast(B, std::integral_constant<std::size_t, 2>{});
                    auto cv = upcast(*C, std::integral_constant<std::size_t, 2>{});
                    string_gemm(parsed, c_pf, &cv, ab_pf, av, bv);
                    return;
                }
                if (links.empty() && a_set == b_set && a_set == c_set && a_idx == b_idx && a_idx == c_idx) {
                    ProfileAnnotate("dispatch", "direct_product_runtime");
                    auto av = upcast(A, std::integral_constant<std::size_t, 2>{});
                    auto bv = upcast(B, std::integral_constant<std::size_t, 2>{});
                    auto cv = upcast(*C, std::integral_constant<std::size_t, 2>{});
                    linear_algebra::direct_product(ab_pf, av, bv, c_pf, &cv);
                    return;
                }
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
        packed_gemm::ContractionSpec spec;
        spec.c_indices      = c_idx;
        spec.a_indices      = a_idx;
        spec.b_indices      = b_idx;
        spec.link_indices   = links;
        spec.target_indices = std::vector<std::string>(c_set.begin(), c_set.end());
        // Preserve target order from c_idx (sets aren't ordered).
        spec.target_indices.clear();
        std::set<std::string> seen_target;
        for (auto const &t : c_idx) {
            if (seen_target.insert(t).second)
                spec.target_indices.push_back(t);
        }
        spec.all_indices = spec.target_indices;
        for (auto const &l : spec.link_indices)
            spec.all_indices.push_back(l);
        spec.conj_a = conj_a; // PackedGemm conjugates during packing/transpose (native, no copy)
        spec.conj_b = conj_b;

        if (packed_gemm::try_packed_gemm<AType, BType, CType>(spec, c_pf, C, ab_pf, A, B)) {
            ProfileAnnotate("dispatch", "packed_gemm");
            return;
        }
    }

    // ── Generic fallback: runtime nested-loop contraction ──────────
    // Reached when no fast path applies: pure outer products, mixed-dtype
    // edge cases, or contractions that even PackedGemm can't form into a
    // valid GEMM shape (no M-dims, no N-dims, no links).
    ProfileAnnotate("dispatch", "generic_loop");
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

    // Scale C by beta
    if (beta == T{0}) {
        C->zero();
    } else if (beta != T{1}) {
        linear_algebra::scale(beta, C);
    }

    // If alpha is zero, nothing more to do
    if (alpha == T{0})
        return;

    // Iterate over all elements via C's index space
    size_t total = 1;
    for (size_t d = 0; d < rank; d++)
        total *= C->dim(d);

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

} // namespace einsums::compute_graph::dispatch
