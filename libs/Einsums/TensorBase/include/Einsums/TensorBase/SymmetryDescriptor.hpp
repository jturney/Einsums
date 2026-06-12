//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <array>
#include <cstdint>
#include <initializer_list>
#include <vector>

namespace einsums {

/// Maximum tensor rank that a single symmetry operation can describe. Tensor
/// rank doesn't need to match this value, since permutations are stored padded.
/// The value mirrors the practical upper bound used elsewhere in the library,
/// such as 4-index integrals, rank-5 tests, and HPTT limits.
inline constexpr int kMaxSymmetryRank = 8;

/// One generator of a tensor's symmetry group.
///
/// ``permutation[i]`` is the new position of the index that currently sits at
/// position ``i``. For a tensor ``T(i0, i1, ...)`` the operation maps
/// ``T(i0, i1, ...) -> sign * T(i_{perm[0]}, i_{perm[1]}, ...)``. For the
/// Hermitian case the right-hand side is additionally complex-conjugated.
///
/// Padding: for tensors of rank < ``kMaxSymmetryRank`` the unused trailing
/// positions must be filled with their natural index (``permutation[i] = i``).
struct EINSUMS_EXPORT SymmetryOp {
    std::array<int8_t, kMaxSymmetryRank> permutation{};
    int8_t                               sign{+1};         ///< +1 symmetric, -1 antisymmetric
    bool                                 conjugate{false}; ///< true for Hermitian / anti-Hermitian

    /// Identity operation: all positions map to themselves.
    [[nodiscard]] static SymmetryOp identity() {
        SymmetryOp op;
        for (int i = 0; i < kMaxSymmetryRank; ++i)
            op.permutation[i] = static_cast<int8_t>(i);
        return op;
    }

    /// Swap two index positions. All other positions are identity.
    [[nodiscard]] static SymmetryOp swap(int a, int b, int8_t sign = +1, bool conjugate = false) {
        auto op           = identity();
        op.permutation[a] = static_cast<int8_t>(b);
        op.permutation[b] = static_cast<int8_t>(a);
        op.sign           = sign;
        op.conjugate      = conjugate;
        return op;
    }

    /// Swap two index groups (useful for the bra/ket swap in chemists'-notation
    /// ERIs: ``(ij|kl) = (kl|ij)``). ``g1`` and ``g2`` must be the same length;
    /// positions not listed in either group are identity.
    [[nodiscard]] static SymmetryOp group_swap(std::initializer_list<int> g1, std::initializer_list<int> g2, int8_t sign = +1,
                                               bool conjugate = false);

    [[nodiscard]] bool operator==(SymmetryOp const &other) const noexcept {
        return permutation == other.permutation && sign == other.sign && conjugate == other.conjugate;
    }
};

/// Describes the symmetry of a tensor as a set of generators.
///
/// The full symmetry group is the closure of the generators under composition.
/// We store only the generators both because it keeps the descriptor small
/// (ERIs' 8-fold symmetry fits in three generators) and because callers
/// usually want to reason about which invariants hold, not enumerate 48
/// group elements.
///
/// @par Common patterns
/// - ``SymmetryDescriptor::symmetric_pair(i, j)``: ``T(..,i,..,j,..) = T(..,j,..,i,..)``.
/// - ``SymmetryDescriptor::antisymmetric_pair(i, j)``: negated on swap.
/// - ``SymmetryDescriptor::hermitian_pair(i, j)``: symmetric plus complex conjugate.
/// - ``SymmetryDescriptor::eri_8fold()``: 8-fold ERI symmetry in chemists' notation.
/// - ``SymmetryDescriptor::ccsd_t2()``: antisym in (0,1) and antisym in (2,3).
///
/// @par Tolerance
/// Floating-point arithmetic drifts away from exact symmetry. ``tolerance``
/// is the maximum absolute element-wise deviation that ``check_symmetry()``
/// accepts as "still symmetric."
struct EINSUMS_EXPORT SymmetryDescriptor {
    std::vector<SymmetryOp> ops;
    double                  tolerance{1e-12};

    SymmetryDescriptor() = default;
    explicit SymmetryDescriptor(std::vector<SymmetryOp> gens, double tol = 1e-12) : ops(std::move(gens)), tolerance(tol) {}

    [[nodiscard]] bool        empty() const noexcept { return ops.empty(); }
    [[nodiscard]] std::size_t size() const noexcept { return ops.size(); }

    /// Append another generator to this descriptor (mutates in place).
    SymmetryDescriptor &add(SymmetryOp op) {
        ops.push_back(op);
        return *this;
    }

    [[nodiscard]] bool operator==(SymmetryDescriptor const &other) const noexcept {
        // Exact generator-list equality (including ordering). Canonicalization
        // for semantic equality across different generator sets is intentionally
        // deferred; it's only needed once CSE starts merging symmetric nodes
        // and we can design that comparator alongside.
        return ops == other.ops && tolerance == other.tolerance;
    }

    // ── Named factories ─────────────────────────────────────────────────────

    /// Symmetric in a single index pair: ``T(..,a,..,b,..) = T(..,b,..,a,..)``.
    [[nodiscard]] static SymmetryDescriptor symmetric_pair(int a, int b);

    /// Antisymmetric in a single index pair: ``T(..,a,..,b,..) = -T(..,b,..,a,..)``.
    [[nodiscard]] static SymmetryDescriptor antisymmetric_pair(int a, int b);

    /// Hermitian in a single index pair (complex tensors):
    /// ``T(..,a,..,b,..) = conj(T(..,b,..,a,..))``.
    [[nodiscard]] static SymmetryDescriptor hermitian_pair(int a, int b);

    /// Anti-Hermitian / skew-Hermitian: ``T(..,a,..,b,..) = -conj(T(..,b,..,a,..))``.
    [[nodiscard]] static SymmetryDescriptor anti_hermitian_pair(int a, int b);

    /// 8-fold symmetry of real two-electron integrals in chemists' notation
    /// ``(μν|λσ) = (νμ|λσ) = (μν|σλ) = (νμ|σλ) = (λσ|μν) = (σλ|μν) = (λσ|νμ) = (σλ|νμ)``.
    /// The three generators are: swap(0,1), swap(2,3), group_swap({0,1},{2,3}).
    [[nodiscard]] static SymmetryDescriptor eri_8fold();

    /// 4-fold symmetry of complex two-electron integrals: two real-pair swaps,
    /// no bra/ket swap (which would also complex-conjugate).
    [[nodiscard]] static SymmetryDescriptor eri_4fold();

    /// CCSD T2 amplitudes ``T[a,b,i,j]``: antisym in virtual pair (0,1) and
    /// antisym in occupied pair (2,3), so ``T[a,b,i,j] = -T[b,a,i,j] =
    /// -T[a,b,j,i] = T[b,a,j,i]``.
    [[nodiscard]] static SymmetryDescriptor ccsd_t2();
};

} // namespace einsums
