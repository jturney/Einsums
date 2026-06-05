//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/// @file SymmetryOps.hpp
/// @brief Free functions that enforce / verify a tensor's declared symmetry.
///
/// ``Tensor::set_symmetry`` only attaches metadata — it does not touch the
/// data. ``symmetrize()`` walks the data and mutates it to satisfy the
/// descriptor in place; ``check_symmetry()`` walks the data and reports
/// whether the descriptor holds to within a tolerance. Both are rank-N
/// generic, composed from the descriptor's generators.

#include <Einsums/Concepts/Complex.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorBase/SymmetryDescriptor.hpp>

#include <array>
#include <cmath>
#include <complex>
#include <cstddef>

namespace einsums {

namespace detail {

/// Apply a SymmetryOp's permutation to a multi-index.
template <size_t Rank>
inline std::array<size_t, Rank> permute_index(std::array<size_t, Rank> const &idx, SymmetryOp const &op) {
    std::array<size_t, Rank> out{};
    for (size_t i = 0; i < Rank; ++i)
        out[op.permutation[i]] = idx[i];
    return out;
}

/// Conditionally conjugate a value. No-op for non-complex types.
template <typename T>
inline T maybe_conjugate(T v, bool doit) {
    if constexpr (einsums::IsComplexV<T>) {
        return doit ? std::conj(v) : v;
    } else {
        (void)doit;
        return v;
    }
}

/// Visit every multi-index of a rank-``Rank`` tensor with dimensions
/// ``dims``. Calls ``fn(idx)`` once per element in natural order.
template <size_t Rank, typename F>
void for_each_index(std::array<size_t, Rank> const &dims, F &&fn) {
    std::array<size_t, Rank> idx{};
    while (true) {
        fn(idx);
        // Increment like a multi-digit odometer from position Rank-1.
        size_t k = Rank;
        while (k > 0) {
            --k;
            if (++idx[k] < dims[k])
                break;
            idx[k] = 0;
            if (k == 0)
                return;
        }
    }
}

} // namespace detail

/// Enforce the declared symmetry on ``T`` in place by averaging all elements
/// related by each generator. The result satisfies ``check_symmetry()`` to
/// within the descriptor's tolerance (up to round-off).
///
/// For a single symmetric generator ``T(i,j) = T(j,i)``: replaces
/// ``(T(i,j), T(j,i))`` with ``(T(i,j) + T(j,i))/2``.
///
/// For antisymmetric: subtracts and halves. For Hermitian: averages with
/// the conjugate of the partner. Generators are applied sequentially; the
/// final tensor satisfies each individually.
template <typename T, size_t Rank, typename Alloc>
void symmetrize(GeneralTensor<T, Rank, Alloc> &tensor) {
    auto const *desc = tensor.symmetry();
    if (!desc || desc->empty())
        return;

    std::array<size_t, Rank> dims{};
    for (size_t i = 0; i < Rank; ++i)
        dims[i] = static_cast<size_t>(tensor.dim(i));

    auto at = [&](std::array<size_t, Rank> const &idx) -> T & {
        return std::apply([&](auto... i) -> T & { return tensor(static_cast<int>(i)...); }, idx);
    };

    for (auto const &op : desc->ops) {
        detail::for_each_index<Rank>(dims, [&](std::array<size_t, Rank> const &idx) {
            auto partner = detail::permute_index<Rank>(idx, op);
            if (partner == idx) {
                // Fixed point under the permutation. The only values
                // consistent with the declared symmetry are:
                //  - antisymmetric (sign=-1): must be zero
                //  - Hermitian (sign=+1, conj): real part only (imag→0)
                //  - anti-Hermitian (sign=-1, conj): imag part only (real→0)
                //  - symmetric (sign=+1, no conj): no constraint
                T &a = at(idx);
                if (op.sign < 0 && !op.conjugate) {
                    a = T{};
                } else if constexpr (einsums::IsComplexV<T>) {
                    if (op.conjugate && op.sign > 0)
                        a = T{a.real(), typename T::value_type{0}};
                    else if (op.conjugate && op.sign < 0)
                        a = T{typename T::value_type{0}, a.imag()};
                }
                return;
            }
            // Visit each unordered pair once.
            if (idx > partner)
                return;
            T &a  = at(idx);
            T &b  = at(partner);
            T  bc = detail::maybe_conjugate(b, op.conjugate);
            T  avg;
            if (op.sign > 0) {
                avg = (a + bc) / static_cast<T>(2);
            } else {
                avg = (a - bc) / static_cast<T>(2);
            }
            a = avg;
            b = detail::maybe_conjugate(static_cast<T>(static_cast<T>(op.sign) * avg), op.conjugate);
        });
    }
}

/// Verify that ``tensor`` satisfies its declared symmetry to within
/// ``tolerance`` (or the descriptor's own tolerance if the argument is
/// negative). Returns true when every pair agrees; false on first
/// violation.
template <typename T, size_t Rank, typename Alloc>
[[nodiscard]] bool check_symmetry(GeneralTensor<T, Rank, Alloc> const &tensor, double tolerance = -1.0) {
    auto const *desc = tensor.symmetry();
    if (!desc || desc->empty())
        return true;

    double tol = tolerance >= 0.0 ? tolerance : desc->tolerance;

    std::array<size_t, Rank> dims{};
    for (size_t i = 0; i < Rank; ++i)
        dims[i] = static_cast<size_t>(tensor.dim(i));

    auto at = [&](std::array<size_t, Rank> const &idx) -> T const & {
        return std::apply([&](auto... i) -> T const & { return tensor(static_cast<int>(i)...); }, idx);
    };

    bool ok = true;
    for (auto const &op : desc->ops) {
        detail::for_each_index<Rank>(dims, [&](std::array<size_t, Rank> const &idx) {
            if (!ok)
                return;
            auto partner = detail::permute_index<Rank>(idx, op);
            if (partner == idx || idx > partner)
                return;
            T const &a       = at(idx);
            T const &b       = at(partner);
            T        bc      = detail::maybe_conjugate(b, op.conjugate);
            T        expect  = static_cast<T>(op.sign) * bc;
            auto     diff    = a - expect;
            double   diffmag = static_cast<double>(std::abs(diff));
            if (diffmag > tol)
                ok = false;
        });
        if (!ok)
            return false;
    }
    return ok;
}

} // namespace einsums
