//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/// @file Permute.hpp
/// @brief Tensor permutation with the axes named by characters, on rank-erased tensors.
///
/// The axes of each operand are spelled as a string, one character per axis: ``"ji"`` for the
/// output and ``"ij"`` for the input is a matrix transpose. Nothing here is templated on the
/// index letters, so a permutation can be built at run time and rewritten as data, which is what
/// the ComputeGraph string specs need. The kernel is HPTT, through a per-thread plan cache.

#include <Einsums/Config.hpp>

#include <Einsums/BufferAllocator/BufferAllocator.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/Errors/Error.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/HPTT/HPTT.hpp>
#include <Einsums/HPTT/HPTTTypes.hpp>
#include <Einsums/HPTT/Transpose.hpp>
#include <Einsums/Profile.hpp>
#include <Einsums/StringUtil/StringOps.hpp>
#include <Einsums/TensorImpl/TensorImpl.hpp>
#include <Einsums/TensorPermute/Detail/HpttPlanCache.hpp>

#include <algorithm>
#include <cctype>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

EINSUMS_NAMESPACE_BEGIN(tensor_permute)

namespace detail {

/// The plan for C = beta * C + alpha * permute(A), where C's axis @c i is A's axis
/// ``c_to_a[i]``. Every public form ends here once it knows that permutation; the caller has
/// already checked that it is one.
template <bool ConjA, typename T>
std::shared_ptr<hptt::Transpose<T>> build_permute_plan(T beta, std::span<int const> c_to_a, einsums::detail::TensorImpl<T> *C, T alpha,
                                                       einsums::detail::TensorImpl<T> const &A, hptt::SelectionMethod method) {
    // C has A's extents, axis for axis. HPTT takes its sizes from A alone and C's only from its
    // strides, so a C of any other shape is written wrong or out of bounds rather than refused.
    // Checked before the empty return, so an empty operand against a non-empty one still throws.
    for (std::size_t i = 0; i < c_to_a.size(); ++i) {
        if (C->dim(static_cast<int>(i)) != A.dim(c_to_a[i])) {
            EINSUMS_THROW_EXCEPTION(DimensionError, "permute: output axis {} has extent {}, but the input axis it takes has extent {}", i,
                                    C->dim(static_cast<int>(i)), A.dim(c_to_a[i]));
        }
    }

    // An empty operand has nothing to permute, and HPTT rejects a zero extent. There is no plan.
    if (A.size() == 0 || C->size() == 0) {
        return nullptr;
    }

    // HPTT describes both operands in A's layout. A C in the other layout is read as its transpose,
    // axes reversed. A row-major A with a C of neither layout is described column-major.
    bool const                                    row_major = A.is_row_major() && (C->is_row_major() || C->is_column_major());
    bool const                                    flip = row_major ? !C->is_row_major() : !(A.is_column_major() && C->is_column_major());
    std::optional<einsums::detail::TensorImpl<T>> swapped;
    if (flip) {
        swapped.emplace(row_major ? C->to_row_major() : C->to_column_major());
    }
    einsums::detail::TensorImpl<T> const &C_view = flip ? *swapped : *C;

    std::size_t const   rank = A.rank();
    BufferVector<int>   perms(rank);
    std::vector<size_t> size(rank);
    std::vector<size_t> outerSizeA(rank);
    std::vector<size_t> offsetA(rank, 0);
    std::vector<size_t> outerSizeC(rank);
    std::vector<size_t> offsetC(rank, 0);
    for (std::size_t i = 0; i < rank; ++i) {
        perms[i] = c_to_a[flip ? rank - 1 - i : i];
    }

    // An extent-1 axis's stride is arbitrary (a view keeps its parent's) and can make HPTT reject the
    // plan, so it is replaced by one consistent with its neighbours: 1 on the fastest axis; else the
    // stride of the nearest slower axis with extent > 1; else the packed stride past the faster one.
    auto const effective_strides = [&](einsums::detail::TensorImpl<T> const &t) {
        auto const          axis = [&](std::size_t p) { return static_cast<int>(row_major ? rank - 1 - p : p); };
        std::vector<size_t> s(rank, 0);
        std::size_t         slower = 0; // stride of the nearest slower stepped axis, 0 for none
        for (std::size_t p = rank; p-- > 0;) {
            int const d = axis(p);
            if (t.dim(d) > 1) {
                s[d]   = t.stride(d);
                slower = s[d];
            } else if (p == 0) {
                s[d] = 1;
            } else {
                s[d] = slower;
            }
        }
        for (std::size_t p = 1; p < rank; ++p) {
            if (s[axis(p)] == 0) {
                s[axis(p)] = s[axis(p - 1)] * t.dim(axis(p - 1));
            }
        }
        return s;
    };
    std::vector<size_t> const strideA = effective_strides(A);
    std::vector<size_t> const strideC = effective_strides(C_view);

    // The outer sizes are what each axis's stride says the allocation spans, measured from the
    // fastest axis: the last in row-major, the first in column-major. HPTT places axis i at the
    // inner stride times the outer sizes of the axes faster than it, so an axis's outer size is
    // the ratio of the next slower axis's stride to its own; the inner stride is already in both.
    size_t const innerStrideA = row_major ? strideA[rank - 1] : strideA[0];
    size_t const innerStrideC = row_major ? strideC[rank - 1] : strideC[0];
    for (std::size_t i = 0; i < rank; ++i) {
        size[i] = A.dim(static_cast<int>(i));
    }
    if (row_major) {
        outerSizeA[0] = A.dim(0);
        outerSizeC[0] = C_view.dim(0);
        for (std::size_t i0 = 1; i0 < rank; i0++) {
            outerSizeA[i0] = strideA[i0 - 1] / strideA[i0];
            outerSizeC[i0] = strideC[i0 - 1] / strideC[i0];
        }
    } else {
        outerSizeA[rank - 1] = A.dim(-1);
        outerSizeC[rank - 1] = C_view.dim(-1);
        for (std::size_t i0 = 0; i0 + 1 < rank; i0++) {
            outerSizeA[i0] = strideA[i0 + 1] / strideA[i0];
            outerSizeC[i0] = strideC[i0 + 1] / strideC[i0];
        }
    }

    auto plan = detail::get_or_create_hptt_plan<T>(perms.data(), static_cast<int>(rank), alpha, A.data(), size.data(), outerSizeA.data(),
                                                   offsetA.data(), innerStrideA, beta, C->data(), outerSizeC.data(), offsetC.data(),
                                                   innerStrideC, row_major, method);
    plan->set_conj_a(ConjA);
    return plan;
}

/// The kernel behind compile_permute: each operand's axes named by one character per axis. The
/// public forms parse a spec into these.
template <bool ConjA = false, typename T>
std::shared_ptr<hptt::Transpose<T>> compile_permute(T beta, std::string const &C_indices, einsums::detail::TensorImpl<T> *C, T alpha,
                                                    std::string const &A_indices, einsums::detail::TensorImpl<T> const &A,
                                                    hptt::SelectionMethod method = hptt::ESTIMATE) {
    WAGGLE_ZONE("permute: {} <- {}", C_indices, A_indices);

    // One letter per axis, and the same letters on both sides. Checked in both directions: a letter
    // on either side alone would reach HPTT as an invalid permutation.
    if (C_indices.size() != C->rank() || A_indices.size() != A.rank()) {
        EINSUMS_THROW_EXCEPTION(RankError, "permute: '{}' names {} axes of a rank-{} output and '{}' names {} axes of a rank-{} input",
                                C_indices, C_indices.size(), C->rank(), A_indices, A_indices.size(), A.rank());
    }
    if (!difference(A_indices, C_indices).empty() || !difference(C_indices, A_indices).empty()) {
        EINSUMS_THROW_EXCEPTION(RankError, "permute: '{}' and '{}' do not name the same axes", C_indices, A_indices);
    }

    std::vector<int> c_to_a(C_indices.size());
    for (std::size_t i = 0; i < C_indices.size(); ++i) {
        c_to_a[i] = static_cast<int>(A_indices.find(C_indices[i]));
    }
    return build_permute_plan<ConjA, T>(beta, c_to_a, C, alpha, A, method);
}

/// As the character form, with the permutation given directly: C's axis @c i is A's axis
/// ``c_to_a[i]``. For a caller that already holds the permutation, which the character form would
/// only encode for HPTT to decode again.
template <bool ConjA = false, typename T>
std::shared_ptr<hptt::Transpose<T>> compile_permute(T beta, std::span<int const> c_to_a, einsums::detail::TensorImpl<T> *C, T alpha,
                                                    einsums::detail::TensorImpl<T> const &A,
                                                    hptt::SelectionMethod                 method = hptt::ESTIMATE) {
    WAGGLE_ZONE("permute");
    std::size_t const rank = A.rank();
    if (c_to_a.size() != rank || C->rank() != rank) {
        EINSUMS_THROW_EXCEPTION(RankError, "permute: a permutation of {} axes between a rank-{} output and a rank-{} input", c_to_a.size(),
                                C->rank(), rank);
    }
    std::vector<bool> used(rank, false);
    for (int const axis : c_to_a) {
        if (axis < 0 || static_cast<std::size_t>(axis) >= rank || used[static_cast<std::size_t>(axis)]) {
            EINSUMS_THROW_EXCEPTION(RankError, "permute: the axis map is not a permutation of {} axes", rank);
        }
        used[static_cast<std::size_t>(axis)] = true;
    }
    return build_permute_plan<ConjA, T>(beta, c_to_a, C, alpha, A, method);
}

/// Build and run the plan for C = beta * C + alpha * permute(A), axes named by characters.
template <bool ConjA = false, typename T>
void permute(T beta, std::string const &C_indices, einsums::detail::TensorImpl<T> *C, T alpha, std::string const &A_indices,
             einsums::detail::TensorImpl<T> const &A) {
    auto plan = compile_permute<ConjA>(beta, C_indices, C, alpha, A_indices, A);
    if (plan != nullptr) {
        plan->execute();
    }
}

/// Build and run the plan for C = beta * C + alpha * permute(A), with C's axis @c i being A's
/// axis ``c_to_a[i]``.
template <bool ConjA = false, typename T>
void permute(T beta, std::span<int const> c_to_a, einsums::detail::TensorImpl<T> *C, T alpha, einsums::detail::TensorImpl<T> const &A) {
    auto plan = compile_permute<ConjA>(beta, c_to_a, C, alpha, A);
    if (plan != nullptr) {
        plan->execute();
    }
}

/// Parse a permute spec, ``"ji <- ij"`` or ``"ij -> ji"``, into one character per axis for each side.
/// Names are single characters, or comma-separated for multi-character names (``"nu,mu <- mu,nu"``),
/// which are re-encoded onto characters. The kernel checks that the two sides name the same axes.
inline std::pair<std::string, std::string> parse_permute_spec(std::string_view spec) {
    std::string_view c_text, a_text;
    if (auto const at = spec.find("<-"); at != std::string_view::npos) {
        c_text = spec.substr(0, at);
        a_text = spec.substr(at + 2);
    } else if (auto const at2 = spec.find("->"); at2 != std::string_view::npos) {
        a_text = spec.substr(0, at2);
        c_text = spec.substr(at2 + 2);
    } else {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "permute: '{}' has no '<-' or '->'", spec);
    }
    auto names_of = [&](std::string_view text) {
        std::vector<std::string> names;
        bool const               commas = text.find(',') != std::string_view::npos;
        std::string              current;
        auto                     flush = [&] {
            if (!current.empty()) {
                names.push_back(current);
                current.clear();
            } else if (commas) {
                EINSUMS_THROW_EXCEPTION(std::invalid_argument, "permute: empty index name in '{}'", spec);
            }
        };
        for (char const ch : text) {
            if (ch == ' ' || ch == '\t') {
                continue;
            }
            if (ch == ',') {
                flush();
            } else if (std::isalnum(static_cast<unsigned char>(ch)) || ch == '_') {
                current.push_back(ch);
                if (!commas) {
                    flush();
                }
            } else {
                EINSUMS_THROW_EXCEPTION(std::invalid_argument, "permute: '{}' in '{}' is not an index name", ch, spec);
            }
        }
        if (commas) {
            flush();
        }
        return names;
    };
    auto const c_names = names_of(c_text);
    auto const a_names = names_of(a_text);

    std::vector<std::string> seen;
    auto                     encode = [&](std::vector<std::string> const &names) {
        std::string chars;
        for (auto const &name : names) {
            auto it = std::find(seen.begin(), seen.end(), name);
            if (it == seen.end()) {
                seen.push_back(name);
                it = seen.end() - 1;
            }
            chars.push_back(static_cast<char>('a' + (it - seen.begin())));
        }
        return chars;
    };
    std::string a_chars = encode(a_names);
    std::string c_chars = encode(c_names);
    if (seen.size() > 26) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "permute: '{}' names more than 26 axes", spec);
    }
    return {std::move(c_chars), std::move(a_chars)};
}

} // namespace detail

/**
 * @brief Build the HPTT plan for @f$ C = \beta C + \alpha\,\mathrm{permute}(A) @f$ without running it.
 *
 * The spec names each operand's axes, output first: ``"ji <- ij"`` (or ``"ij -> ji"``) transposes a
 * matrix. The same letters must appear on both sides. The plan comes from a per-thread cache keyed
 * on the shapes, so building the same permutation again is cheap. Run it with the plan's own
 * ``execute()``, or rebind it to new data of the same shapes with the three-argument ``permute``.
 *
 * @tparam ConjA If true, conjugate the elements of @p A as they are permuted.
 * @tparam T The element type of both tensors.
 * @param spec The axes of @p C and of @p A.
 * @param beta The scale applied to the existing contents of @p C.
 * @param C The output tensor.
 * @param alpha The scale applied to the permuted @p A.
 * @param A The input tensor. It must not share storage with @p C.
 * @param method How hard HPTT searches for a fast plan.
 * @return The plan, or null when either operand is empty: there is nothing to permute, and running
 *         the null plan through the three-argument ``permute`` does nothing.
 * @throws RankError when a side does not have one letter per axis of its operand, or the two sides
 *         do not name the same axes.
 * @throws DimensionError when an axis of @p C does not have the extent of the axis of @p A it takes.
 *
 * @versionadded{2.0.0}
 */
template <bool ConjA = false, typename T>
std::shared_ptr<hptt::Transpose<T>> compile_permute(std::string_view spec, T beta, einsums::detail::TensorImpl<T> *C, T alpha,
                                                    einsums::detail::TensorImpl<T> const &A,
                                                    hptt::SelectionMethod                 method = hptt::ESTIMATE) {
    auto const [c_chars, a_chars] = detail::parse_permute_spec(spec);
    return detail::compile_permute<ConjA>(beta, c_chars, C, alpha, a_chars, A, method);
}

/**
 * @brief Run a plan from @ref compile_permute on new data of the same shapes. A null plan, which
 * compile_permute returns for empty operands, does nothing.
 *
 * @versionadded{2.0.0}
 */
template <typename T>
void permute(einsums::detail::TensorImpl<T> *C, einsums::detail::TensorImpl<T> const &A, std::shared_ptr<hptt::Transpose<T>> plan) {
    if (plan == nullptr) {
        return; // compile_permute's answer for empty operands
    }
    plan->set_input_ptr(A.data());
    plan->set_output_ptr(C->data());

    plan->execute();
}

/**
 * @brief Compute @f$ C = \beta C + \alpha\,\mathrm{permute}(A) @f$, the axes named by a spec.
 *
 * @code
 * tensor_permute::permute("kij <- ijk", 0.0, &C, 1.0, A);  // C(k, i, j) = A(i, j, k)
 * @endcode
 *
 * Empty operands are valid and leave @p C as it is.
 *
 * @tparam ConjA If true, conjugate the elements of @p A as they are permuted.
 * @tparam T The element type of both tensors.
 * @throws RankError when a side does not have one letter per axis of its operand, or the two sides
 *         do not name the same axes.
 * @throws DimensionError when an axis of @p C does not have the extent of the axis of @p A it takes.
 *
 * @versionadded{2.0.0}
 */
template <bool ConjA = false, typename T>
void permute(std::string_view spec, T beta, einsums::detail::TensorImpl<T> *C, T alpha, einsums::detail::TensorImpl<T> const &A) {
    auto const [c_chars, a_chars] = detail::parse_permute_spec(spec);
    detail::permute<ConjA>(beta, c_chars, C, alpha, a_chars, A);
}

/**
 * @brief Compute @f$ C = A^T @f$ for rank-2 tensors.
 *
 * @throws RankError when either operand is not rank 2.
 * @throws DimensionError when @p C does not have the extents of the transposed @p A.
 *
 * @versionadded{2.0.0}
 */
template <bool ConjA = false, typename T>
void transpose(einsums::detail::TensorImpl<T> *C, einsums::detail::TensorImpl<T> const &A) {
    if (C->rank() != 2 || A.rank() != 2) {
        EINSUMS_THROW_EXCEPTION(RankError, "transpose needs rank-2 tensors, got output rank {} and input rank {}", C->rank(), A.rank());
    }
    if (C->dim(0) != A.dim(1) || C->dim(1) != A.dim(0)) {
        EINSUMS_THROW_EXCEPTION(DimensionError, "transpose: the output tensor does not have the extents of the transposed input");
    }
    detail::permute<ConjA>(T{0}, "ij", C, T{1}, "ji", A);
}

/**
 * @brief A tensor type that exposes the TensorImpl it wraps, as every dense tensor, view and
 *        RuntimeTensor does.
 *
 * The overloads below take such tensors directly, so callers never reach for ``impl()``, while
 * this module still depends on nothing above TensorImpl.
 *
 * @versionadded{2.0.0}
 */
template <typename T>
concept HasTensorImpl = requires(T &t, T const &ct) {
    typename T::ValueType;
    t.impl();
    ct.impl();
};

/**
 * @brief Compute @f$ C = \beta C + \alpha\,\mathrm{permute}(A) @f$ on tensors, the axes named by a
 *        spec. The same signature as ``compute_graph::permute``.
 *
 * @code
 * tensor_permute::permute("kij <- ijk", 0.0, &C, 1.0, A);  // C(k, i, j) = A(i, j, k)
 * @endcode
 *
 * @throws RankError when a side does not have one letter per axis of its operand, or the two sides
 *         do not name the same axes.
 * @throws DimensionError when an axis of @p C does not have the extent of the axis of @p A it takes.
 *
 * @versionadded{2.0.0}
 */
template <bool ConjA = false, HasTensorImpl CType, HasTensorImpl AType>
    requires std::is_same_v<typename CType::ValueType, typename AType::ValueType>
void permute(std::string_view spec, typename AType::ValueType beta, CType *C, typename AType::ValueType alpha, AType const &A) {
    permute<ConjA>(spec, beta, &C->impl(), alpha, A.impl());
}

/**
 * @brief Compute @f$ C = \mathrm{permute}(A) @f$ on tensors, overwriting @p C.
 *
 * @versionadded{2.0.0}
 */
template <bool ConjA = false, HasTensorImpl CType, HasTensorImpl AType>
    requires std::is_same_v<typename CType::ValueType, typename AType::ValueType>
void permute(std::string_view spec, CType *C, AType const &A) {
    using T = typename AType::ValueType;
    permute<ConjA>(spec, T{0}, &C->impl(), T{1}, A.impl());
}

/**
 * @brief Compute @f$ C = A^T @f$ for rank-2 tensors.
 *
 * @throws RankError when either operand is not rank 2.
 * @throws DimensionError when @p C does not have the extents of the transposed @p A.
 *
 * @versionadded{2.0.0}
 */
template <bool ConjA = false, HasTensorImpl CType, HasTensorImpl AType>
    requires std::is_same_v<typename CType::ValueType, typename AType::ValueType>
void transpose(CType *C, AType const &A) {
    transpose<ConjA>(&C->impl(), A.impl());
}

EINSUMS_NAMESPACE_END(tensor_permute)
