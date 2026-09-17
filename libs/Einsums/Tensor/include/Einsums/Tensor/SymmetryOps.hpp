//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/// @file SymmetryOps.hpp
/// @brief Free functions that enforce / verify a tensor's declared symmetry.
///
/// ``Tensor::set_symmetry`` only attaches metadata; it does not touch the
/// data. ``symmetrize()`` walks the data and mutates it to satisfy the
/// descriptor in place; ``check_symmetry()`` walks the data and reports
/// whether the descriptor holds to within a tolerance. Both are rank-N
/// generic, composed from the descriptor's generators.
///
/// Each comes in two forms. The statically ranked one walks a
/// @ref GeneralTensor with the rank in the type; the runtime-rank one walks the
/// @ref GeneralRuntimeTensor family, which is what the Python-facing path and
/// the ComputeGraph hold, and which could already CARRY a descriptor through
/// @c set_symmetry with no way to enforce or verify it.

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
/// Stated here as a requires-expression rather than borrowed from Concepts,
/// which has no runtime-rank concept, and deliberately narrow: it names exactly
/// what the traversal uses and nothing else. In particular it does NOT require a
/// declared descriptor, because @ref GeneralRuntimeTensorView has none and a view
/// over an impl is how a ComputeGraph pass reaches a bound tensor's data. The
/// caller-supplied-descriptor overload has to accept one.
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

/// Visit each unordered pair of elements that @p op relates, by OFFSET.
///
/// Two running offsets stepped by an odometer, not an index vector rebuilt per
/// element: @c off is the element's own, @c poff its partner's, and incrementing
/// axis @c k moves them by @c stride[k] and @c stride[op.permutation[k]]
/// respectively. The index vector survives only to decide which member of a pair
/// to visit, and is never allocated inside the loop.
///
/// Measured on a rank-6 tensor of a million elements: 20.2 ns an element before,
/// 1.47 ns after, the same cause and the same order of improvement
/// @ref compute_graph::dispatch::generic_string_einsum records at "roughly 25 ns
/// an element". A symmetry walk validates operands as large as the ones the
/// arithmetic touches, so it has to cost what a pass over them costs.
///
/// The early stop is the larger of the two wins and was a defect rather than an
/// optimization: the verdict used to be carried in a captured flag with nothing
/// to end the iteration, so a generator that did NOT hold still swept the whole
/// tensor, 2.30 ms where it is now 0.4 microseconds. Probing candidate generators
/// against tensors that mostly do not carry them is the use, so being cheap when
/// the answer is no is the property that matters most.
///
/// @p visit receives ``(off, poff, fixed)`` and returns false to STOP, which is
/// what makes a generator that does not hold cost the distance to its first
/// violation instead of a full sweep.
/// @return false when @p visit stopped the walk, true when it ran to completion.
template <typename F>
bool for_each_symmetry_pair(std::vector<size_t> const &dims, std::vector<size_t> const &strides, SymmetryOp const &op, F &&visit) {
    size_t const rank = dims.size();
    for (auto const extent : dims) {
        if (extent == 0) {
            return true; // an empty tensor has no elements, and no symmetry to violate
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
/// The statically ranked walks never ask, because their callers pass square
/// tensors. A runtime-rank tensor reaching here can have any shape, and a
/// permutation across axes of different extents does not merely fail to hold:
/// it indexes out of range. Checked once per generator rather than per element.
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
/// The descriptor is passed IN rather than read off the tensor, which is what
/// lets a caller ask whether a symmetry it is considering actually holds. That
/// is the question an optimizer asks before rewriting arithmetic on the strength
/// of one, and it is a different question from "does this tensor's own declared
/// symmetry hold", which the one-argument overload below asks.
///
/// Returns false, rather than throwing, when the descriptor cannot apply to this
/// tensor at all: a rank past @ref kMaxSymmetryRank, which a @ref SymmetryOp
/// cannot describe, or a generator permuting axes of unequal extent. Both mean
/// the symmetry does not hold here, which is what the caller asked.
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
                // A fixed point constrains the element against ITSELF: an
                // antisymmetric generator forces zero, an (anti-)Hermitian one
                // forces the value real or imaginary. A symmetric generator says
                // nothing, which is why the statically ranked check skips these
                // outright and this one cannot.
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
/// elements each generator relates. The runtime-rank twin of
/// @ref symmetrize(GeneralTensor<T, Rank, Alloc> &).
///
/// Throws rather than silently declining when the descriptor cannot apply: an
/// enforcement that quietly did nothing would leave the caller believing an
/// invariant it does not have, which is the failure this function exists to
/// prevent.
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
