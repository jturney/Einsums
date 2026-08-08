//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

/// @file
/// The C++ backend of the `compute_pno_overlaps` stage.
///
/// This is the other half of `dlpno/pno_overlaps.py`. Both implement the same
/// contract, both are selectable at runtime, and they are compared against each
/// other rather than one being retired.
///
/// The aggregates below mirror `dlpno/contracts.py` field for field. That
/// correspondence is the contract, it is checked by hand today, and it is what
/// `python -m einsums.stages check` will verify once it exists.

#pragma once

#include <Einsums/Tensor/RuntimeTensor.hpp>

#include <cstdint>
#include <utility>
#include <vector>

namespace dlpno {

/// Which pairs couple to which partners, and where every block goes.
///
/// Structure of arrays: the per-coupling vectors are all the same length and
/// are indexed by coupling ordinal. The Python side chose that shape for this
/// side's benefit - a dict of lists of tuples would arrive here as a pointer
/// chase per coupling, and the whole reason this stage is in C++ is to walk
/// 32,948 couplings without one.
struct CouplingPlan {
    /// `(bucket of the pair, bucket of the partner)` per class, ascending. The
    /// class ORDINAL is the identity everywhere else; this only recovers the
    /// two block dimensions.
    std::vector<std::pair<std::int64_t, std::int64_t>> classes;
    /// Columns in the pair-major layout, excluding the reserved zero slot.
    std::vector<std::int64_t> slots_pair;
    /// Columns in the partner-major layout, including the reserved zero slot.
    std::vector<std::int64_t> slots_partner;
    /// Partner-major slot -> pair-major slot. Padding points past the end of
    /// the pair-major side, at the reserved zero, so a gather writes its whole
    /// destination and the padded columns are zero rather than stale.
    std::vector<std::vector<std::int64_t>> inv_perm;

    // Per coupling, all the same length.
    std::vector<std::int64_t> pair;
    std::vector<std::int64_t> partner;
    std::vector<std::int64_t> cls;
    std::vector<std::int64_t> dest_slot;
    std::vector<std::int64_t> src_slot;
    /// Sign of the Fock element. Rides on the transposed copy only: S appears
    /// on both sides of the congruence, so a sign in both would square away.
    std::vector<double> sign;
    /// The Fock element itself; its square root scales the pair-major copy.
    std::vector<double> factor;

    /// Pairs with at least one coupling, ascending.
    std::vector<std::int64_t> coupled_pairs;
};

/// The overlap tensors, one per shape class, in both layouts.
struct PnoOverlaps {
    /// Pair-major, `(M_b, M_p * (slots_pair + 1))`, scaled by `sqrt|factor|`.
    std::vector<einsums::RuntimeTensor<double>> S_cls;
    /// Partner-major, `(M_p, M_b, slots_partner)`, carrying `sqrt|factor|` from
    /// the copy it was lifted from and the sign of its own coupling.
    std::vector<einsums::RuntimeTensor<double>> S_T;
};

/// Overlaps between every coupled pair of PNO bases, scaled by the Fock element.
///
/// Runs to completion and returns values; it does NOT emit into an ambient
/// capture. That matches the Python backend, which is an `eager` stage for a
/// structural reason rather than an unfinished one: the phase runs two graphs
/// with a host-side scaling between them, and the second graph's inputs are the
/// first graph's outputs after that scaling. There is no single graph to emit
/// into until the scaling itself becomes a captured op.
///
/// So this stage is not what proves the shared-graph contract. That is a
/// separate, deliberately minimal stage; see the shared-capture test.
///
/// @param X_pno           Per pair, the PNO transform on its own domain.
/// @param S_pao           PAO overlap matrix, `(npao, npao)`.
/// @param lmopair_to_paos Per pair, the PAO indices its domain covers.
/// @param n_pno           Per pair, its PNO count; zero means the pair is dead.
/// @param bucket_of       Per pair, which padding bucket it is stored at.
/// @param bucket_dims     Per bucket, the padded block dimension.
/// @param plan            The layout, from `plan_pno_couplings` on the Python side.
[[nodiscard]] PnoOverlaps compute_pno_overlaps(std::vector<einsums::RuntimeTensor<double>> const &X_pno,
                                               einsums::RuntimeTensor<double> const              &S_pao,
                                               std::vector<std::vector<std::int64_t>> const      &lmopair_to_paos,
                                               std::vector<std::int64_t> const &n_pno, std::vector<std::int64_t> const &bucket_of,
                                               std::vector<std::int64_t> const &bucket_dims, CouplingPlan const &plan);

} // namespace dlpno
