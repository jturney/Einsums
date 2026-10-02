//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/// @file SymmetryOps.hpp
/// @brief Free functions that enforce / verify a tensor's declared symmetry.
///
/// ``set_symmetry`` only attaches metadata. ``symmetrize()`` enforces it in place and
/// ``check_symmetry()`` verifies it to a tolerance, for static- and runtime-rank tensors.

#include <Einsums/Concepts/Complex.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/Tensor/RuntimeTensor.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorBase/SymmetryDescriptor.hpp>

#include <array>
#include <cmath>
#include <complex>
#include <concepts>
#include <cstddef>
#include <iterator>
#include <optional>
#include <utility>
#include <vector>

EINSUMS_NAMESPACE_BEGIN()

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

// ── Runtime-rank forms ──────────────────────────────────────────────────────

/// A runtime-rank tensor the walks below can traverse: extents, strides and a
/// base pointer, which is all an offset odometer needs.
///
/// No descriptor required, as @ref GeneralRuntimeTensorView has none.
template <typename TensorType>
concept RuntimeRankWalkable = requires(TensorType const &t) {
    typename std::remove_cvref_t<TensorType>::ValueType;
    { t.rank() } -> std::convertible_to<size_t>;
    { t.dim(0) } -> std::convertible_to<size_t>;
    { t.stride(0) } -> std::convertible_to<size_t>;
    { t.data() };
};

/// A walkable runtime-rank tensor that also CARRIES a descriptor, which the
/// forms reading a tensor's own declared symmetry need.
template <typename TensorType>
concept RuntimeRankSymmetryTensor = RuntimeRankWalkable<TensorType> && requires(TensorType const &t) {
    { t.symmetry() } -> std::convertible_to<SymmetryDescriptor const *>;
};

namespace detail {

/// The two axes a generator swaps, when it swaps exactly two and fixes the rest.
///
/// Returned in increasing order, which is what lets the caller decide pair
/// membership from the two axes' index values alone.
inline std::optional<std::pair<size_t, size_t>> transposed_axes(SymmetryOp const &op, size_t rank) {
    long long first  = -1;
    long long second = -1;
    for (size_t i = 0; i < rank; ++i) {
        auto const to = static_cast<size_t>(op.permutation[i]);
        if (to == i) {
            continue;
        }
        if (first < 0) {
            first = static_cast<long long>(i);
        } else if (second < 0) {
            second = static_cast<long long>(i);
        } else {
            return std::nullopt; // three or more axes move
        }
    }
    if (first < 0 || second < 0) {
        return std::nullopt;
    }
    auto const p = static_cast<size_t>(first);
    auto const q = static_cast<size_t>(second);
    if (static_cast<size_t>(op.permutation[p]) != q || static_cast<size_t>(op.permutation[q]) != p) {
        return std::nullopt; // the two moved axes form a cycle, not a swap
    }
    return std::pair{p, q};
}

/// Visit each unordered pair of elements that @p op relates, by OFFSET.
///
/// Two offsets stepped by an odometer: @c off and its partner @c poff (axis @c k moves them by
/// @c stride[k] and @c stride[op.permutation[k]]). @p visit receives ``(off, poff, fixed)`` and
/// returns false to stop, so a generator that does not hold costs only the distance to its first
/// violation.
/// @return false when @p visit stopped the walk, true when it ran to completion.
template <typename F>
bool for_each_symmetry_pair(std::vector<size_t> const &dims, std::vector<size_t> const &strides, SymmetryOp const &op, F &&visit) {
    size_t const rank = dims.size();
    for (auto const extent : dims) {
        if (extent == 0) {
            return true; // an empty tensor has no elements, and no symmetry to violate
        }
    }

    // Fast path for a two-axis swap (most generators): membership is `a < b` and the partner is a
    // constant distance away. 1.1 against 4.9 ns/element at 64 MB on rank 6, the gap growing with size.
    if (auto const axes = transposed_axes(op, rank); axes.has_value()) {
        auto const [p, q] = *axes;

        // Sweep in memory order (every axis sorted by stride), computing the partner:
        //     poff = off + (idx[q] - idx[p]) * (stride[p] - stride[q])
        // with membership idx[p] < idx[q].
        std::vector<size_t> order(rank);
        for (size_t i = 0; i < rank; ++i) {
            order[i] = i;
        }
        std::ranges::sort(order, [&](size_t l, size_t r) { return strides[l] > strides[r]; });

        std::ptrdiff_t const step = static_cast<std::ptrdiff_t>(strides[p]) - static_cast<std::ptrdiff_t>(strides[q]);

        std::vector<size_t> idx(rank, 0);
        size_t              off = 0;
        while (true) {
            if (idx[p] <= idx[q]) {
                bool const           fixed = idx[p] == idx[q];
                std::ptrdiff_t const shift = static_cast<std::ptrdiff_t>(idx[q]) - static_cast<std::ptrdiff_t>(idx[p]);
                if (!visit(off, static_cast<size_t>(static_cast<std::ptrdiff_t>(off) + shift * step), fixed)) {
                    return false;
                }
            }

            size_t k        = rank;
            bool   finished = true;
            while (k > 0) {
                --k;
                size_t const axis = order[k];
                if (++idx[axis] < dims[axis]) {
                    off += strides[axis];
                    finished = false;
                    break;
                }
                idx[axis] = 0;
                off -= (dims[axis] - 1) * strides[axis];
            }
            if (finished) {
                return true;
            }
        }
    }

    // Partner strides, and the inverse permutation the pair-ordering test needs:
    // partner[j] == idx[inverse[j]].
    std::vector<size_t> partner_stride(rank);
    std::vector<size_t> inverse(rank);
    for (size_t k = 0; k < rank; ++k) {
        partner_stride[k]                               = strides[static_cast<size_t>(op.permutation[k])];
        inverse[static_cast<size_t>(op.permutation[k])] = k;
    }

    std::vector<size_t> idx(rank, 0);
    size_t              off  = 0;
    size_t              poff = 0;

    while (true) {
        // Which member of the pair is this? Below its partner means visit it,
        // equal means a fixed point, above means the partner already handled it.
        int cmp = 0;
        for (size_t j = 0; j < rank && cmp == 0; ++j) {
            size_t const partner_j = idx[inverse[j]];
            if (idx[j] < partner_j) {
                cmp = -1;
            } else if (idx[j] > partner_j) {
                cmp = 1;
            }
        }
        if (cmp <= 0 && !visit(off, poff, cmp == 0)) {
            return false;
        }

        size_t k        = rank;
        bool   finished = true;
        while (k > 0) {
            --k;
            if (++idx[k] < dims[k]) {
                off += strides[k];
                poff += partner_stride[k];
                finished = false;
                break;
            }
            idx[k] = 0;
            off -= (dims[k] - 1) * strides[k];
            poff -= (dims[k] - 1) * partner_stride[k];
        }
        if (finished) {
            return true;
        }
    }
}

/// Whether @p op maps every axis onto one of the same length.
///
/// A runtime-rank tensor can have any shape, and a permutation across unequal axes would index out
/// of range.
inline bool symmetry_op_axes_conform(std::vector<size_t> const &dims, SymmetryOp const &op) {
    for (size_t i = 0; i < dims.size(); ++i) {
        if (dims[i] != dims[static_cast<size_t>(op.permutation[i])]) {
            return false;
        }
    }
    return true;
}

/// The extents of a runtime-rank tensor, as the walks want them.
/// The extents of a runtime-rank tensor, as the walks want them.
template <RuntimeRankWalkable TensorType>
std::vector<size_t> symmetry_dims(TensorType const &tensor) {
    std::vector<size_t> dims(tensor.rank());
    for (size_t i = 0; i < dims.size(); ++i) {
        dims[i] = static_cast<size_t>(tensor.dim(static_cast<int>(i)));
    }
    return dims;
}

/// The strides of a runtime-rank tensor, in elements.
template <RuntimeRankWalkable TensorType>
std::vector<size_t> symmetry_strides(TensorType const &tensor) {
    std::vector<size_t> strides(tensor.rank());
    for (size_t i = 0; i < strides.size(); ++i) {
        strides[i] = static_cast<size_t>(tensor.stride(static_cast<int>(i)));
    }
    return strides;
}

} // namespace detail

/// Verify that @p tensor satisfies @p desc to within @p tolerance (or the
/// descriptor's own tolerance when the argument is negative).
///
/// For testing a candidate symmetry, as an optimizer does. False, not a throw, when the descriptor
/// cannot apply (rank past @ref kMaxSymmetryRank, or unequal permuted axes).
template <RuntimeRankWalkable TensorType>
[[nodiscard]] bool check_symmetry(TensorType const &tensor, SymmetryDescriptor const &desc, double tolerance = -1.0) {
    using T = typename std::remove_cvref_t<TensorType>::ValueType;

    if (desc.empty()) {
        return true;
    }
    if (tensor.rank() > static_cast<size_t>(kMaxSymmetryRank)) {
        return false;
    }

    double const              tol     = tolerance >= 0.0 ? tolerance : desc.tolerance;
    std::vector<size_t> const dims    = detail::symmetry_dims(tensor);
    std::vector<size_t> const strides = detail::symmetry_strides(tensor);
    T const                  *data    = tensor.data();

    for (auto const &op : desc.ops) {
        if (!detail::symmetry_op_axes_conform(dims, op)) {
            return false;
        }
        bool const complete = detail::for_each_symmetry_pair(dims, strides, op, [&](size_t off, size_t poff, bool fixed) {
            if (fixed) {
                // A fixed point: antisymmetric forces zero, (anti-)Hermitian forces real (imaginary).
                T const value = data[off];
                if (op.sign < 0 && !op.conjugate) {
                    return static_cast<double>(std::abs(value)) <= tol;
                }
                if constexpr (einsums::IsComplexV<T>) {
                    if (op.conjugate && op.sign > 0) {
                        return static_cast<double>(std::abs(value.imag())) <= tol;
                    }
                    if (op.conjugate && op.sign < 0) {
                        return static_cast<double>(std::abs(value.real())) <= tol;
                    }
                }
                return true;
            }
            T const expect = static_cast<T>(op.sign) * detail::maybe_conjugate(data[poff], op.conjugate);
            return static_cast<double>(std::abs(data[off] - expect)) <= tol;
        });
        if (!complete) {
            return false;
        }
    }
    return true;
}

/// Verify that @p tensor satisfies its OWN declared symmetry.
/// @see check_symmetry(TensorType const &, SymmetryDescriptor const &, double)
template <RuntimeRankSymmetryTensor TensorType>
[[nodiscard]] bool check_symmetry(TensorType const &tensor, double tolerance = -1.0) {
    auto const *desc = tensor.symmetry();
    if (desc == nullptr) {
        return true;
    }
    return check_symmetry(tensor, *desc, tolerance);
}

/// Enforce a runtime-rank tensor's declared symmetry in place, by averaging the
/// elements each generator relates. The runtime-rank twin of the
/// @ref symmetrize overload taking a compile-time-rank tensor.
///
/// Throws when the descriptor cannot apply, rather than leaving the invariant unenforced.
template <RuntimeRankSymmetryTensor TensorType>
void symmetrize(TensorType &tensor) {
    using T = typename std::remove_cvref_t<TensorType>::ValueType;

    auto const *desc = tensor.symmetry();
    if (desc == nullptr || desc->empty()) {
        return;
    }
    if (tensor.rank() > static_cast<size_t>(kMaxSymmetryRank)) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                "symmetrize: rank {} exceeds the {} a SymmetryOp can describe, so the declared symmetry cannot be enforced",
                                tensor.rank(), kMaxSymmetryRank);
    }

    std::vector<size_t> const dims    = detail::symmetry_dims(tensor);
    std::vector<size_t> const strides = detail::symmetry_strides(tensor);
    T                        *data    = tensor.data();

    for (auto const &op : desc->ops) {
        if (!detail::symmetry_op_axes_conform(dims, op)) {
            EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                    "symmetrize: a generator permutes axes of unequal extent, so the declared symmetry cannot hold");
        }
        detail::for_each_symmetry_pair(dims, strides, op, [&](size_t off, size_t poff, bool fixed) {
            if (fixed) {
                // The case analysis the statically ranked form documents.
                T const value = data[off];
                if (op.sign < 0 && !op.conjugate) {
                    data[off] = T{};
                } else if constexpr (einsums::IsComplexV<T>) {
                    if (op.conjugate && op.sign > 0) {
                        data[off] = T{value.real(), typename T::value_type{0}};
                    } else if (op.conjugate && op.sign < 0) {
                        data[off] = T{typename T::value_type{0}, value.imag()};
                    }
                }
                return true;
            }
            T const a   = data[off];
            T const bc  = detail::maybe_conjugate(data[poff], op.conjugate);
            T const avg = op.sign > 0 ? static_cast<T>((a + bc) / static_cast<T>(2)) : static_cast<T>((a - bc) / static_cast<T>(2));

            data[off]  = avg;
            data[poff] = detail::maybe_conjugate(static_cast<T>(static_cast<T>(op.sign) * avg), op.conjugate);
            return true;
        });
    }
}

EINSUMS_NAMESPACE_END()
