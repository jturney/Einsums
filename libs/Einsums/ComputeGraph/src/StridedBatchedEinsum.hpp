//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/BLAS.hpp>
#include <Einsums/ComputeGraph/Detail/ImplLayout.hpp>
#include <Einsums/ComputeGraph/EinsumSpec.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/TensorImpl/TensorImpl.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <ranges>
#include <string>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(compute_graph::detail)

/**
 * @brief How an einsum that is a batch of equal matrix products maps onto one ``gemm_batch`` call.
 *
 * A contraction is a strided batch when one or more indices appear in A, B and C (the batch), the
 * other indices form a plain matrix product (an A target, one link, a B target), and the batch axes
 * are the outermost ones in memory. Both storage conventions qualify:
 *   - row-major with the batch axes FIRST, the ML/CUDA convention ("bij;bjk->bik", shape (B, M, K));
 *   - column-major with the batch axes LAST, Einsums' default layout ("ijb;jkb->ikb", shape (M, K, B)).
 * Several batch indices flatten into one batch with a uniform stride, as long as they sit at the same
 * positions in all three operands. The 2D slice at each flat batch index is then a contiguous block,
 * which is exactly what ``cublasDgemmStridedBatched``-style calls take.
 *
 * The plan is a function of the index lists and the operands' geometry and nothing else, so it is
 * valid for as long as both are what they were when it was made.
 */
struct StridedBatchPlan {
    char         trans_a{'N'};
    char         trans_b{'N'};
    int          m{0};
    int          n{0};
    int          k{0};
    int          lda{0};
    int          ldb{0};
    int          ldc{0};
    int          batch_count{0};
    std::int64_t stride_a{0};
    std::int64_t stride_b{0};
    std::int64_t stride_c{0};
    bool         swap_ab{false}; ///< Row-major: BLAS computes the transposed product, so A and B trade places.
};

/// The geometry a plan was made against: an operand's extents, strides and storage order.
struct OperandGeometry {
    std::vector<std::size_t> dims;
    std::vector<std::size_t> strides;
    bool                     row_major{false};

    template <typename T>
    static OperandGeometry of(::einsums::detail::TensorImpl<T> const &impl) {
        OperandGeometry g{.dims = {}, .strides = {}, .row_major = impl.is_row_major()};
        g.dims.reserve(impl.rank());
        g.strides.reserve(impl.rank());
        for (std::size_t d = 0; d < impl.rank(); ++d) {
            g.dims.push_back(impl.dim(d));
            g.strides.push_back(impl.stride(d));
        }
        return g;
    }

    template <typename T>
    [[nodiscard]] bool matches(::einsums::detail::TensorImpl<T> const &impl) const {
        if (impl.rank() != dims.size() || impl.is_row_major() != row_major) {
            return false;
        }
        for (std::size_t d = 0; d < dims.size(); ++d) {
            if (impl.dim(d) != dims[d] || impl.stride(d) != strides[d]) {
                return false;
            }
        }
        return true;
    }
};

/**
 * @brief The strided-batch plan for ``C <- A ; B`` over these operands, when there is one.
 *
 * The caller rules out what the index lists do not show: permutation operators and conjugation,
 * neither of which a ``gemm_batch`` call can express.
 * @param[in] c_indices,a_indices,b_indices The contraction's index lists.
 * @param[in] links The contracted indices.
 * @param[in] a,b,c The operands' current geometry.
 * @return The plan, or nothing when the contraction is not a strided batch of matrix products.
 */
template <typename T>
std::optional<StridedBatchPlan> plan_strided_batch(std::vector<std::string> const &c_indices, std::vector<std::string> const &a_indices,
                                                   std::vector<std::string> const &b_indices, std::vector<std::string> const &links,
                                                   ::einsums::detail::TensorImpl<T> const &a, ::einsums::detail::TensorImpl<T> const &b,
                                                   ::einsums::detail::TensorImpl<T> const &c) {
    std::size_t const rank = a.rank();
    if (rank < 3 || b.rank() != rank || c.rank() != rank || a_indices.size() != rank || b_indices.size() != rank ||
        c_indices.size() != rank || links.size() != 1) {
        return std::nullopt;
    }
    std::string const &link = links[0];

    // The batch indices, in A's order, so "abij;abjk->abik" gives [a, b]. Each operand holds the
    // batch plus two more (A and B: a target and the link; C: both targets), so there are rank - 2.
    std::vector<std::string> batch;
    for (auto const &idx : a_indices) {
        if (index_role(idx, c_indices, a_indices, b_indices) == IndexRole::Batch) {
            batch.push_back(idx);
        }
    }
    if (batch.size() != rank - 2) {
        return std::nullopt;
    }

    // The batch sits at the same positions in all three operands, or flattening it gives no single stride.
    auto const position = [](std::vector<std::string> const &list, std::string const &idx) {
        return static_cast<std::size_t>(std::ranges::find(list, idx) - list.begin());
    };
    std::vector<std::size_t> positions;
    for (auto const &name : batch) {
        std::size_t const pa = position(a_indices, name);
        if (pa != position(b_indices, name) || pa != position(c_indices, name)) {
            return std::nullopt;
        }
        positions.push_back(pa);
    }

    if (!canonical_dense(a) || !canonical_dense(b) || !canonical_dense(c)) {
        return std::nullopt;
    }
    std::size_t const nbatch = batch.size();
    bool const        row_mode =
        a.is_row_major() && b.is_row_major() && c.is_row_major() && std::ranges::equal(positions, std::views::iota(std::size_t{0}, nbatch));
    bool const col_mode = a.is_column_major() && b.is_column_major() && c.is_column_major() &&
                          std::ranges::equal(positions, std::views::iota(rank - nbatch, rank));
    if (!row_mode && !col_mode) {
        return std::nullopt;
    }

    // The two slice axes: the last two in row-major mode, the first two in column-major mode.
    std::size_t const first  = row_mode ? rank - 2 : 0;
    auto const        slice  = [first](std::vector<std::string> const &list) { return std::vector{list[first], list[first + 1]}; };
    auto const        a_rest = slice(a_indices);
    auto const        b_rest = slice(b_indices);
    auto const        c_rest = slice(c_indices);

    // C's slice has to be in canonical (M, N) order: M shared with A first, N shared with B second.
    // A transposed output (e.g. "kji <- jli ; lki", slice (N, M)) would mis-map m and n against the
    // operands, so it goes to the general path.
    std::string const &m_index = a_rest[0] == link ? a_rest[1] : a_rest[0];
    std::string const &n_index = b_rest[0] == link ? b_rest[1] : b_rest[0];
    if (c_rest[0] != m_index || c_rest[1] != n_index) {
        return std::nullopt;
    }

    auto const dim = [first](::einsums::detail::TensorImpl<T> const &impl, std::size_t local) {
        return static_cast<int>(impl.dim(first + local));
    };
    char const natural_a = a_rest[0] == link ? 'T' : 'N';
    char const natural_b = b_rest[1] == link ? 'T' : 'N';

    StridedBatchPlan p;
    if (col_mode) {
        p.trans_a = natural_a;
        p.trans_b = natural_b;
        p.m       = dim(c, 0);
        p.n       = dim(c, 1);
        p.lda     = dim(a, 0);
        p.ldb     = dim(b, 0);
        p.ldc     = dim(c, 0);
    } else {
        p.trans_a = natural_b;
        p.trans_b = natural_a;
        p.m       = dim(c, 1);
        p.n       = dim(c, 0);
        p.lda     = dim(b, 1);
        p.ldb     = dim(a, 1);
        p.ldc     = dim(c, 1);
    }
    p.k = natural_a == 'N' ? dim(a, 1) : dim(a, 0);

    std::int64_t batch_count = 1;
    for (std::size_t const pos : positions) {
        batch_count *= static_cast<std::int64_t>(a.dim(pos));
    }
    p.batch_count = static_cast<int>(batch_count);
    p.stride_a    = static_cast<std::int64_t>(dim(a, 0)) * dim(a, 1);
    p.stride_b    = static_cast<std::int64_t>(dim(b, 0)) * dim(b, 1);
    p.stride_c    = static_cast<std::int64_t>(dim(c, 0)) * dim(c, 1);
    p.swap_ab     = row_mode;
    return p;
}

/**
 * @brief The per-slice pointer tables a strided batch hands ``gemm_batch``.
 *
 * Pure functions of the three base pointers (base + i*stride), so they are rebuilt only when a
 * rebind moves a base, not on every replay. One per node; a node never runs concurrently with
 * itself, so no synchronization is needed.
 */
template <typename T>
struct StridedBatchTables {
    T const               *base_a{nullptr};
    T const               *base_b{nullptr};
    T                     *base_c{nullptr};
    std::vector<T const *> a;
    std::vector<T const *> b;
    std::vector<T *>       c;
};

/// Run @p plan as ``C = beta*C + alpha*A*B`` per slice.
template <typename T>
void run_strided_batch(StridedBatchPlan const &plan, T alpha, T beta, T const *base_a, T const *base_b, T *base_c,
                       StridedBatchTables<T> &tables) {
    if (base_a != tables.base_a || base_b != tables.base_b || base_c != tables.base_c ||
        tables.a.size() != static_cast<std::size_t>(plan.batch_count)) {
        tables.a.resize(plan.batch_count);
        tables.b.resize(plan.batch_count);
        tables.c.resize(plan.batch_count);
        for (int i = 0; i < plan.batch_count; ++i) {
            tables.a[i] = base_a + i * plan.stride_a;
            tables.b[i] = base_b + i * plan.stride_b;
            tables.c[i] = base_c + i * plan.stride_c;
        }
        tables.base_a = base_a;
        tables.base_b = base_b;
        tables.base_c = base_c;
    }
    T const **blas_a = plan.swap_ab ? tables.b.data() : tables.a.data();
    T const **blas_b = plan.swap_ab ? tables.a.data() : tables.b.data();
    blas::gemm_batch<T>(plan.trans_a, plan.trans_b, plan.m, plan.n, plan.k, alpha, blas_a, plan.lda, blas_b, plan.ldb, beta,
                        tables.c.data(), plan.ldc, plan.batch_count);
}

EINSUMS_NAMESPACE_END(compute_graph::detail)
