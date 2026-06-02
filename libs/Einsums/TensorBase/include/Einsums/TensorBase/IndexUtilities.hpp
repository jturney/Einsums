//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config/CompilerSpecific.hpp>
#include <Einsums/Errors/Error.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/Iterator/Zip.hpp>

#include <cstdarg>
#include <cstddef>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <vector>

namespace einsums {

template <typename UniqueIndex, int BDim, typename BType>
inline size_t get_dim_ranges_for_many_b(BType const & /*B*/, std::tuple<> const & /*B_indices*/) {
    return 1;
}

template <typename UniqueIndex, int BDim, typename BType, typename BHead>
inline auto get_dim_ranges_for_many_b(BType const &B, std::tuple<BHead> const & /*B_indices*/) -> size_t
    requires std::is_same_v<BHead, UniqueIndex>
{
    return B.dim(BDim);
}

template <typename UniqueIndex, int BDim, typename BType, typename BHead, typename... BIndices>
inline size_t get_dim_ranges_for_many_b(BType const &B, std::tuple<BHead, BIndices...> const & /*B_indices*/) {
    if constexpr (std::is_same_v<BHead, UniqueIndex>) {
        return B.dim(BDim);
    } else {
        return get_dim_ranges_for_many_b<UniqueIndex, BDim + 1>(B, std::tuple<BIndices...>());
    }
}

template <typename UniqueIndex, int ADim, typename AType, typename BType, typename... BIndices>
inline size_t get_dim_ranges_for_many_a(AType const & /*A*/, std::tuple<> const & /*A_indices*/, BType const &B,
                                        std::tuple<BIndices...> const &B_indices) {
    return get_dim_ranges_for_many_b<UniqueIndex, 0>(B, B_indices);
}

template <typename UniqueIndex, int ADim, typename AType, typename BType, typename AHead, typename... BIndices>
inline size_t get_dim_ranges_for_many_a(AType const &A, std::tuple<AHead> const & /*A_indices*/, BType const &B,
                                        std::tuple<BIndices...> const &B_indices) {
    if constexpr (std::is_same_v<AHead, UniqueIndex>) {
        return A.dim(ADim);
    } else {
        return get_dim_ranges_for_many_b<UniqueIndex, 0>(B, B_indices);
    }
}

template <typename UniqueIndex, int ADim, typename AType, typename BType, typename AHead, typename... AIndices, typename... BIndices>
inline auto get_dim_ranges_for_many_a(AType const &A, std::tuple<AHead, AIndices...> const & /*A_indices*/, BType const &B,
                                      std::tuple<BIndices...> const &B_indices) -> size_t
    requires(sizeof...(AIndices) != 0)
{
    if constexpr (std::is_same_v<AHead, UniqueIndex>) {
        return A.dim(ADim);
    } else {
        return get_dim_ranges_for_many_a<UniqueIndex, ADim + 1>(A, std::tuple<AIndices...>(), B, B_indices);
    }
}

template <typename UniqueIndex, int CDim, typename CType, typename AType, typename BType, typename... AIndices, typename... BIndices>
inline size_t get_dim_ranges_for_many_c(CType const & /*C*/, std::tuple<> const & /*C_indices*/, AType const &A,
                                        std::tuple<AIndices...> const &A_indices, BType const &B,
                                        std::tuple<BIndices...> const &B_indices) {
    return get_dim_ranges_for_many_a<UniqueIndex, 0>(A, A_indices, B, B_indices);
}

template <typename UniqueIndex, int CDim, typename CType, typename AType, typename BType, typename CHead, typename... AIndices,
          typename... BIndices>
inline size_t get_dim_ranges_for_many_c(CType const &C, std::tuple<CHead> const & /*C_indices*/, AType const &A,
                                        std::tuple<AIndices...> const &A_indices, BType const &B,
                                        std::tuple<BIndices...> const &B_indices) {
    if constexpr (std::is_same_v<CHead, UniqueIndex>) {
        return C.dim(CDim);
    } else {
        return get_dim_ranges_for_many_a<UniqueIndex, 0>(A, A_indices, B, B_indices);
    }
}

template <typename UniqueIndex, int CDim, typename CType, typename AType, typename BType, typename CHead, typename... CIndices,
          typename... AIndices, typename... BIndices>
inline auto get_dim_ranges_for_many_c(CType const &C, std::tuple<CHead, CIndices...> const & /*C_indices*/, AType const &A,
                                      std::tuple<AIndices...> const &A_indices, BType const &B, std::tuple<BIndices...> const &B_indices)
    -> size_t
    requires(sizeof...(CIndices) != 0)
{
    if constexpr (std::is_same_v<CHead, UniqueIndex>) {
        return C.dim(CDim);
    } else {
        return get_dim_ranges_for_many_c<UniqueIndex, CDim + 1>(C, std::tuple<CIndices...>(), A, A_indices, B, B_indices);
    }
}

/**
 * @brief Finds the dimensions for the requested indices.
 *
 * @param C The C tensor.
 * @param C_indices The indices for the C tensor.
 * @param A The A tensor.
 * @param A_indices The indices for the A tensor.
 * @param B The B tensor.
 * @param B_indices The indices for the B tensor.
 * @param All_unique_indices The list of all indices with duplicates removed.
 */
template <typename CType, typename AType, typename BType, typename... CIndices, typename... AIndices, typename... BIndices,
          typename... AllUniqueIndices>
inline auto get_dim_ranges_for_many(CType const &C, std::tuple<CIndices...> const &C_indices, AType const &A,
                                    std::tuple<AIndices...> const &A_indices, BType const &B, std::tuple<BIndices...> const &B_indices,
                                    std::tuple<AllUniqueIndices...> const & /*All_unique_indices*/) {
    return std::array{get_dim_ranges_for_many_c<AllUniqueIndices, 0>(C, C_indices, A, A_indices, B, B_indices)...};
}

namespace detail {

/**
 * @brief Adds elements from two sources into the target.
 *
 * Useful in adding offsets to a set of indices.
 *
 * @tparam N The rank of the data.
 * @tparam Target
 * @tparam Source1
 * @tparam Source2
 * @param target The output
 * @param source1 The first source
 * @param source2 The second source
 */
template <size_t N, typename Target, typename Source1, typename Source2>
void add_elements(Target &target, Source1 const &source1, Source2 const &source2) {
    if constexpr (N > 1) {
        add_elements<N - 1>(target, source1, source2);
    }
    target[N - 1] = source1[N - 1] + std::get<N - 1>(source2);
}

} // namespace detail

/**
 * @brief Converts a single sentinel value into a list of indices.
 *
 * @param sentinel The sentinel to convert.
 * @param unique_strides The strides of the unique indices that the sentinel is iterating over.
 * @param out_inds The converted indices.
 */
template <size_t num_unique_inds>
EINSUMS_HOSTDEV inline void sentinel_to_indices(size_t sentinel, size_t const *unique_strides, size_t *out_inds) {
    size_t hold = sentinel;

    if (unique_strides[0] < unique_strides[num_unique_inds - 1]) {
#pragma unroll
        for (ptrdiff_t i = num_unique_inds - 1; i >= 0; i--) {
            if (unique_strides[i] != 0) {
                out_inds[i] = hold / unique_strides[i];
                hold %= unique_strides[i];
            } else {
                [[unlikely]] out_inds[i] = 0;
            }
        }
    } else {
#pragma unroll
        for (ptrdiff_t i = 0; i < num_unique_inds; i++) {
            if (unique_strides[i] != 0) {
                out_inds[i] = hold / unique_strides[i];
                hold %= unique_strides[i];
            } else {
                [[unlikely]] out_inds[i] = 0;
            }
        }
    }
}

template <size_t num_unique_inds, typename stride_int1, typename stride_int2>
void sentinel_to_indices(size_t sentinel, std::array<stride_int1, num_unique_inds> const &unique_strides,
                         std::array<stride_int2, num_unique_inds> &out_inds) {
    size_t hold = sentinel;

    if (unique_strides[0] < unique_strides[num_unique_inds - 1]) {
#pragma unroll
        for (ptrdiff_t i = num_unique_inds - 1; i >= 0; i--) {
            if (unique_strides[i] != 0) {
                out_inds[i] = hold / unique_strides[i];
                hold %= unique_strides[i];
            } else {
                [[unlikely]] out_inds[i] = 0;
            }
        }
    } else {
#pragma unroll
        for (ptrdiff_t i = 0; i < num_unique_inds; i++) {
            if (unique_strides[i] != 0) {
                out_inds[i] = hold / unique_strides[i];
                hold %= unique_strides[i];
            } else {
                [[unlikely]] out_inds[i] = 0;
            }
        }
    }
}

template <typename StorageType1, typename StorageType2>
void sentinel_to_indices(size_t sentinel, StorageType1 const &unique_strides, StorageType2 &out_inds) {
    size_t hold = sentinel;

    // if (out_inds.size() != unique_strides.size()) {
    //     out_inds.resize(unique_strides.size());
    // }

    if (unique_strides[0] < unique_strides[unique_strides.size() - 1]) {
        for (ptrdiff_t i = unique_strides.size() - 1; i >= 0; i--) {
            if (unique_strides[i] != 0) {
                out_inds[i] = hold / unique_strides[i];
                hold %= unique_strides[i];
            } else {
                [[unlikely]] out_inds[i] = 0;
            }
        }
    } else {
        for (ptrdiff_t i = 0; i < unique_strides.size(); i++) {
            if (unique_strides[i] != 0) {
                out_inds[i] = hold / unique_strides[i];
                hold %= unique_strides[i];
            } else {
                [[unlikely]] out_inds[i] = 0;
            }
        }
    }
}

EINSUMS_HOSTDEV inline void sentinel_to_indices_mult_imp(size_t ordinal, size_t index) {
}

template <typename Extra>
EINSUMS_HOSTDEV void sentinel_to_indices_mult_imp(size_t ordinal, size_t index, Extra &&extra) = delete;

template <typename... Rest>
EINSUMS_HOSTDEV void sentinel_to_indices_mult_imp(size_t ordinal, size_t index, size_t const *strides, size_t *indices, Rest &&...rest) {
    indices[index] = strides[index] * ordinal;

    sentinel_to_indices_mult_imp(ordinal, index, std::forward<Rest>(rest)...);
}

/**
 * @brief Implementation details for the sentinel_to_indices function with multiple output lists.
 */
template <typename Stride, typename Indices, typename... Rest>
void sentinel_to_indices_mult_imp(size_t ordinal, size_t index, Stride const &strides, Indices &indices, Rest &&...rest) {
    indices[index] = strides[index] * ordinal;

    sentinel_to_indices_mult_imp(ordinal, index, std::forward<Rest>(rest)...);
}

/**
 * @brief Converts a single sentinel value into lists of indices.
 *
 * @param sentinel The sentinel to convert.
 * @param index_strides The strides of the unique indices that the sentinel is iterating over.
 * @param strides_inds The converted indices.
 */
template <size_t num_indices, typename StorageType, typename... StridesInds>
    requires(sizeof...(StridesInds) % 2 == 0)
EINSUMS_HOSTDEV inline void sentinel_to_indices(size_t sentinel, size_t const *index_strides, StridesInds &&...strides_inds) {
    size_t hold = sentinel;

    if (index_strides[0] < index_strides[num_indices - 1]) {
#pragma unroll
        for (ptrdiff_t i = num_indices - 1; i >= 0; i--) {
            size_t ordinal;
            if (index_strides[i] != 0) {
                ordinal = hold / index_strides[i];
                hold %= index_strides[i];
            } else {
                [[unlikely]] ordinal = 0;
            }

            sentinel_to_indices_mult_imp(ordinal, i, std::forward<StridesInds>(strides_inds)...);
        }
    } else {
#pragma unroll
        for (ptrdiff_t i = 0; i < num_indices; i++) {
            size_t ordinal;
            if (index_strides[i] != 0) {
                ordinal = hold / index_strides[i];
                hold %= index_strides[i];
            } else {
                [[unlikely]] ordinal = 0;
            }

            sentinel_to_indices_mult_imp(ordinal, i, std::forward<StridesInds>(strides_inds)...);
        }
    }
}

template <size_t num_indices, typename StorageType, typename... StridesInds>
    requires(sizeof...(StridesInds) % 2 == 0)
void sentinel_to_indices(size_t sentinel, std::array<size_t, num_indices> const &index_strides, StridesInds &&...strides_inds) {
    size_t hold = sentinel;

    if (index_strides[0] < index_strides[num_indices - 1]) {
#pragma unroll
        for (ptrdiff_t i = num_indices - 1; i >= 0; i--) {
            size_t ordinal;
            if (index_strides[i] != 0) {
                ordinal = hold / index_strides[i];
                hold %= index_strides[i];
            } else {
                [[unlikely]] ordinal = 0;
            }

            sentinel_to_indices_mult_imp(ordinal, i, std::forward<StridesInds>(strides_inds)...);
        }
    } else {
#pragma unroll
        for (ptrdiff_t i = 0; i < num_indices; i++) {
            size_t ordinal;
            if (index_strides[i] != 0) {
                ordinal = hold / index_strides[i];
                hold %= index_strides[i];
            } else {
                [[unlikely]] ordinal = 0;
            }

            sentinel_to_indices_mult_imp(ordinal, i, std::forward<StridesInds>(strides_inds)...);
        }
    }
}

template <typename StorageType, typename... StridesInds>
    requires(sizeof...(StridesInds) % 2 == 0)
void sentinel_to_indices(size_t sentinel, StorageType const &index_strides, StridesInds &&...strides_inds) {
    size_t hold = sentinel;

    if (index_strides[0] < index_strides[index_strides.size() - 1]) {
        for (ptrdiff_t i = index_strides.size() - 1; i >= 0; i--) {
            size_t ordinal;
            if (index_strides[i] != 0) {
                ordinal = hold / index_strides[i];
                hold %= index_strides[i];
            } else {
                [[unlikely]] ordinal = 0;
            }

            sentinel_to_indices_mult_imp(ordinal, i, std::forward<StridesInds>(strides_inds)...);
        }
    } else {
        for (ptrdiff_t i = 0; i < index_strides.size(); i++) {
            size_t ordinal;
            if (index_strides[i] != 0) {
                ordinal = hold / index_strides[i];
                hold %= index_strides[i];
            } else {
                [[unlikely]] ordinal = 0;
            }

            sentinel_to_indices_mult_imp(ordinal, i, std::forward<StridesInds>(strides_inds)...);
        }
    }
}

EINSUMS_HOSTDEV inline void sentinel_to_sentinels_mult_imp(size_t, size_t index) {
}

template <typename Extra>
EINSUMS_HOSTDEV void sentinel_to_sentinels_mult_imp(size_t ordinal, size_t index, Extra extra) = delete;

template <typename Sentinel, typename... Rest>
EINSUMS_HOSTDEV void sentinel_to_sentinels_mult_imp(size_t ordinal, size_t index, size_t const *strides, Sentinel &sentinel,
                                                    Rest &&...rest) {
    sentinel += strides[index] * ordinal;

    sentinel_to_sentinels_mult_imp(ordinal, index, std::forward<Rest>(rest)...);
}

/**
 * @brief Implementation details for the sentinel_to_sentinels function.
 */
template <typename Stride, typename Sentinel, typename... Rest>
void sentinel_to_sentinels_mult_imp(size_t ordinal, size_t index, Stride const &strides, Sentinel &sentinel, Rest &&...rest) {
    sentinel += strides[index] * ordinal;

    sentinel_to_sentinels_mult_imp(ordinal, index, std::forward<Rest>(rest)...);
}

EINSUMS_HOSTDEV inline void sentinel_to_sentinels_zero_imp() {
}

template <typename Extra>
EINSUMS_HOSTDEV void sentinel_to_sentinels_zero_imp(Extra &&extra) = delete;

/**
 * @brief Implementation details for the sentinel_to_sentinels function. This clears the outputs.
 */
template <typename Stride, typename Sentinel, typename... Rest>
EINSUMS_HOSTDEV void sentinel_to_sentinels_zero_imp(Stride && /*strides*/, Sentinel &sentinel, Rest &&...rest) {
    sentinel = 0;

    sentinel_to_sentinels_zero_imp(std::forward<Rest>(rest)...);
}

/**
 * @brief Converts a single sentinel value into several sentinels for tensors with different strides.
 *
 * @param sentinel The sentinel to convert.
 * @param index_strides The strides of the unique indices that the sentinel is iterating over.
 * @param strides_inds The converted sentinels.
 */
template <size_t num_indices, typename StorageType, typename... StridesInds>
    requires(sizeof...(StridesInds) % 2 == 0)
EINSUMS_HOSTDEV inline void sentinel_to_sentinels(size_t sentinel, size_t const *index_strides, StridesInds &&...strides_inds) {
    size_t hold = sentinel;

    sentinel_to_sentinels_zero_imp(std::forward<StridesInds>(strides_inds)...);

    if (index_strides[0] < index_strides[num_indices - 1]) {
#pragma unroll
        for (ptrdiff_t i = num_indices - 1; i >= 0; i--) {
            size_t ordinal = hold / index_strides[i];
            hold %= index_strides[i];

            sentinel_to_sentinels_mult_imp(ordinal, i, std::forward<StridesInds>(strides_inds)...);
        }
    } else {
#pragma unroll
        for (ptrdiff_t i = 0; i < num_indices; i++) {
            size_t ordinal = hold / index_strides[i];
            hold %= index_strides[i];

            sentinel_to_sentinels_mult_imp(ordinal, i, std::forward<StridesInds>(strides_inds)...);
        }
    }
}

template <size_t num_indices, typename StorageType, typename... StridesInds>
    requires(sizeof...(StridesInds) % 2 == 0)
void sentinel_to_sentinels(size_t sentinel, std::array<int64_t, num_indices> const &index_strides, StridesInds &&...strides_inds) {
    size_t hold = sentinel;

    sentinel_to_sentinels_zero_imp(std::forward<StridesInds>(strides_inds)...);

    if (index_strides[0] < index_strides[num_indices - 1]) {
#pragma unroll
        for (ptrdiff_t i = num_indices - 1; i >= 0; i--) {
            size_t ordinal = hold / index_strides[i];
            hold %= index_strides[i];

            sentinel_to_sentinels_mult_imp(ordinal, i, std::forward<StridesInds>(strides_inds)...);
        }
    } else {
#pragma unroll
        for (ptrdiff_t i = 0; i < num_indices; i++) {
            size_t ordinal = hold / index_strides[i];
            hold %= index_strides[i];

            sentinel_to_sentinels_mult_imp(ordinal, i, std::forward<StridesInds>(strides_inds)...);
        }
    }
}

template <typename StorageType, typename... StridesInds>
    requires(sizeof...(StridesInds) % 2 == 0)
void sentinel_to_sentinels(size_t sentinel, StorageType const &index_strides, StridesInds &&...strides_inds) {
    size_t hold = sentinel;

    sentinel_to_sentinels_zero_imp(std::forward<StridesInds>(strides_inds)...);

    if (index_strides[0] < index_strides[index_strides.size() - 1]) {
        for (ptrdiff_t i = index_strides.size() - 1; i >= 0; i--) {
            size_t ordinal = hold / index_strides[i];
            hold %= index_strides[i];

            sentinel_to_sentinels_mult_imp(ordinal, i, std::forward<StridesInds>(strides_inds)...);
        }
    } else {
        for (ptrdiff_t i = 0; i < index_strides.size(); i++) {
            size_t ordinal = hold / index_strides[i];
            hold %= index_strides[i];

            sentinel_to_sentinels_mult_imp(ordinal, i, std::forward<StridesInds>(strides_inds)...);
        }
    }
}

/**
 * @brief The opposite of sentinel_to_indices. Calculates a sentinel given indices and strides.
 */
template <size_t num_unique_inds>
EINSUMS_HOSTDEV inline size_t indices_to_sentinel(size_t const *unique_strides, size_t const *inds) noexcept {
    size_t out = 0;

#pragma unroll
    for (size_t i = 0; i < num_unique_inds; i++) {
        out += inds[i] * unique_strides[i];
    }

    return out;
}

/**
 * @brief The opposite of sentinel_to_indices. Calculates a sentinel given indices and strides.
 */
template <size_t num_unique_inds>
inline size_t indices_to_sentinel(std::array<std::int64_t, num_unique_inds> const &unique_strides,
                                  std::array<size_t, num_unique_inds> const       &inds) noexcept {
    size_t out = 0;

#pragma unroll
    for (size_t i = 0; i < num_unique_inds; i++) {
        out += inds[i] * unique_strides[i];
    }

    return out;
}

/**
 * @brief The opposite of sentinel_to_indices. Calculates a sentinel given indices and strides.
 */
template <size_t num_unique_inds>
inline size_t indices_to_sentinel(std::array<size_t, num_unique_inds> const &unique_strides,
                                  std::array<size_t, num_unique_inds> const &inds) noexcept {
    size_t out = 0;

#pragma unroll
    for (size_t i = 0; i < num_unique_inds; i++) {
        out += inds[i] * unique_strides[i];
    }

    return out;
}

namespace detail {

/**
 * @brief Implementation detail for indices_to_sentinel.
 */
template <size_t index, size_t num_unique_inds, typename FirstIndex, typename... MultiIndex>
    requires(std::is_integral_v<std::decay_t<MultiIndex>> && ... && std::is_integral_v<std::decay_t<FirstIndex>>)
constexpr inline size_t indices_to_sentinel(std::array<std::int64_t, num_unique_inds> const &unique_strides, FirstIndex &&first_index,
                                            MultiIndex &&...indices) noexcept {
    if constexpr (index >= num_unique_inds) {
        return 0;
    } else if constexpr (sizeof...(MultiIndex) != 0) {
        return std::get<index>(unique_strides) * first_index +
               indices_to_sentinel<index + 1>(unique_strides, std::forward<MultiIndex>(indices)...);
    } else {
        return std::get<index>(unique_strides) * first_index;
    }
}
} // namespace detail

/**
 * @brief The opposite of sentinel_to_indices. Calculates a sentinel given indices and strides.
 */
template <size_t num_unique_inds, typename... MultiIndex>
    requires(std::is_integral_v<std::decay_t<MultiIndex>> && ...)
constexpr inline size_t indices_to_sentinel(std::array<std::int64_t, num_unique_inds> const &unique_strides,
                                            MultiIndex &&...indices) noexcept {
    static_assert(sizeof...(MultiIndex) == num_unique_inds);
    return detail::indices_to_sentinel<0>(unique_strides, std::forward<MultiIndex>(indices)...);
}

/**
 * @brief The opposite of sentinel_to_indices. Calculates a sentinel given indices and strides.
 */
template <typename StorageType1, typename StorageType2>
    requires(!std::is_arithmetic_v<StorageType1> && !std::is_arithmetic_v<StorageType2>)
inline size_t indices_to_sentinel(StorageType1 const &unique_strides, StorageType2 const &inds) {
    size_t out = 0;

    for (auto [ind, stride] : Zip(inds, unique_strides)) {
        out += ind * stride;
    }

    return out;
}

namespace detail {

/**
 * @brief Implementation details for indices_to_sentinel_negative_check.
 */
template <size_t index, size_t num_unique_inds, typename FirstIndex, typename... MultiIndex>
    requires requires { requires(std::is_integral_v<std::decay_t<MultiIndex>> && ... && std::is_integral_v<std::decay_t<FirstIndex>>); }
inline size_t indices_to_sentinel_negative_check(std::array<std::int64_t, num_unique_inds> const &unique_strides,
                                                 std::array<std::int64_t, num_unique_inds> const &dims, FirstIndex &&first_index,
                                                 MultiIndex &&...indices) {
    if constexpr (index >= num_unique_inds) {
        return 0;
    } else {
        std::int64_t dim        = std::get<index>(dims);
        auto         temp_index = first_index;

        if constexpr (std::is_signed_v<FirstIndex>) {
            if (temp_index < 0) {
                temp_index += dim;
            }
        }

        if (temp_index < 0 || temp_index >= dim) {
            EINSUMS_THROW_EXCEPTION(std::out_of_range,
                                    "Index out of range! Index {} in rank {} was either greater than or equal to {} or less than {}",
                                    first_index, index, dim, -(ptrdiff_t)dim);
        }
        if constexpr (sizeof...(MultiIndex) != 0) {
            return std::get<index>(unique_strides) * temp_index +
                   indices_to_sentinel_negative_check<index + 1>(unique_strides, dims, std::forward<MultiIndex>(indices)...);
        } else {
            return std::get<index>(unique_strides) * temp_index;
        }
    }
}
} // namespace detail

/**
 * @brief Converts indices to a sentinel. Checks for negative numbers and converts them.
 *
 * When checking for negative numbers, it will add the appropriate dimension to bring the index into
 * the range for that dimension. Can not be used on GPU since it throws errors. Also, running all those if-statements
 * would be very slow.
 */
template <size_t num_unique_inds, typename... MultiIndex>
    requires(std::is_integral_v<std::decay_t<MultiIndex>> && ...)
inline size_t indices_to_sentinel_negative_check(std::array<std::int64_t, num_unique_inds> const &unique_strides,
                                                 std::array<std::int64_t, num_unique_inds> const &dims, MultiIndex &&...indices) {
    static_assert(sizeof...(MultiIndex) == num_unique_inds);
    auto offset = detail::indices_to_sentinel_negative_check<0>(unique_strides, dims, std::forward<MultiIndex>(indices)...);

    return offset;
}

/**
 * @brief Converts indices to a sentinel. Checks for negative numbers and converts them.
 *
 * When checking for negative numbers, it will add the appropriate dimension to bring the index into
 * the range for that dimension. Can not be used on GPU since it throws errors. Also, running all those if-statements
 * would be very slow.
 */
template <typename StorageType1, typename StorageType2, typename StorageType3>
    requires requires {
        requires !std::is_arithmetic_v<StorageType1>;
        requires !std::is_arithmetic_v<StorageType2>;
        requires !std::is_arithmetic_v<StorageType3>;
    }
inline size_t indices_to_sentinel_negative_check(StorageType1 const &unique_strides, StorageType2 const &unique_dims,
                                                 StorageType3 const &inds) {
    size_t out = 0;

    if (inds.size() > unique_strides.size() || inds.size() > unique_dims.size()) {
        EINSUMS_THROW_EXCEPTION(too_many_args, "Too many indices supplied!");
    }

    auto ind_iter    = inds.begin();
    auto dim_iter    = unique_dims.begin();
    auto stride_iter = unique_strides.begin();

    for (size_t i = 0; i < inds.size(); i++, ind_iter++, dim_iter++, stride_iter++) {
        ptrdiff_t ind = *ind_iter;

        if (ind < 0) [[unlikely]] {
            ind += *dim_iter;
        }

        if (ind < 0 || ind >= *dim_iter) [[unlikely]] {
            EINSUMS_THROW_EXCEPTION(std::out_of_range,
                                    "Index out of range! Index {} in rank {} was either greater than or equal to {} or less than {}",
                                    *ind_iter, i, *dim_iter, -(ptrdiff_t)*dim_iter);
        }

        out += ind * *stride_iter;
    }

    return out;
}

/**
 * @brief Compute the strides for turning a sentinel into a list of indices.
 *
 * @param dims The list of dimensions.
 * @param out The calculated strides.
 * @param row_major If true, then the first stride will be the largest. If false, then the first stride will be the smallest.
 * @return The size calculated from the dimensions. Can be safely ignored.
 */
template <typename Alloc1, typename Alloc2>
size_t dims_to_strides(std::vector<size_t, Alloc1> const &dims, std::vector<size_t, Alloc2> &out, bool row_major) {
    size_t stride = 1;

    out.resize(dims.size());

    if (row_major) {
        for (int i = dims.size() - 1; i >= 0; i--) {
            out[i] = stride;
            stride *= dims[i];
        }
    } else {
        for (int i = 0; i < dims.size(); i++) {
            out[i] = stride;
            stride *= dims[i];
        }
    }

    return stride;
}

/**
 * @brief Compute the strides for turning a sentinel into a list of indices.
 *
 * @param dims The list of dimensions.
 * @param out The calculated strides.
 * @param row_major If true, then the first stride will be the largest. If false, then the first stride will be the smallest.
 * @return The size calculated from the dimensions. Can be safely ignored.
 */
template <typename arr_type1, typename arr_type2, size_t Dims>
    requires(std::is_integral_v<arr_type1> && std::is_integral_v<arr_type2>)
constexpr size_t dims_to_strides(std::array<arr_type1, Dims> const &dims, std::array<arr_type2, Dims> &out, bool row_major) {
    size_t stride = 1;

    if (row_major) {
        for (int i = Dims - 1; i >= 0; i--) {
            out[i] = stride;
            stride *= dims[i];
        }
    } else {
        for (int i = 0; i < Dims; i++) {
            out[i] = stride;
            stride *= dims[i];
        }
    }

    return stride;
}

/**
 * @brief Compute the strides for turning a sentinel into a list of indices.
 *
 * @param dims The list of dimensions.
 * @param out The calculated strides.
 * @param row_major If true, then the first stride will be the largest. If false, then the first stride will be the smallest.
 * @return The size calculated from the dimensions. Can be safely ignored.
 */
template <typename Alloc1, typename Alloc2>
size_t dims_to_strides(std::vector<size_t, Alloc1> const &dims, std::vector<size_t, Alloc2> &out) {
    size_t stride = 1;

    out.resize(dims.size());

    if (GlobalConfigMap::get_singleton().get_bool("row-major")) {
        for (int i = dims.size() - 1; i >= 0; i--) {
            out[i] = stride;
            stride *= dims[i];
        }
    } else {
        for (int i = 0; i < dims.size(); i++) {
            out[i] = stride;
            stride *= dims[i];
        }
    }

    return stride;
}

/**
 * @brief Compute the strides for turning a sentinel into a list of indices.
 *
 * @param dims The list of dimensions.
 * @param out The calculated strides.
 * @param row_major If true, then the first stride will be the largest. If false, then the first stride will be the smallest.
 * @return The size calculated from the dimensions. Can be safely ignored.
 */
template <typename arr_type1, typename arr_type2, size_t Dims>
    requires(std::is_integral_v<arr_type1> && std::is_integral_v<arr_type2>)
constexpr size_t dims_to_strides(std::array<arr_type1, Dims> const &dims, std::array<arr_type2, Dims> &out) {
    size_t stride = 1;

    if (GlobalConfigMap::get_singleton().get_bool("row-major")) {
        for (int i = Dims - 1; i >= 0; i--) {
            out[i] = stride;
            stride *= dims[i];
        }
    } else {
        for (int i = 0; i < Dims; i++) {
            out[i] = stride;
            stride *= dims[i];
        }
    }

    return stride;
}

namespace detail {

/**
 * @brief Implementation details for dims_to_strides.
 */
template <ptrdiff_t index, bool RowMajor, size_t Dims, typename arr_type2, typename... TupleDims>
    requires requires {
        requires sizeof...(TupleDims) == Dims;
        requires std::is_integral_v<arr_type2>;
    }
constexpr size_t dims_to_strides(std::tuple<TupleDims...> const &dims, std::array<arr_type2, Dims> &out) {
    if constexpr (index < 0 || index >= sizeof...(TupleDims)) {
        return 1;
    } else if constexpr (RowMajor) {
        size_t stride = dims_to_strides<index + 1, RowMajor>(dims, out);

        out[index] = stride;

        return stride * std::get<index>(dims);
    } else {
        size_t stride = dims_to_strides<index - 1, RowMajor>(dims, out);

        out[index] = stride;

        return stride * std::get<index>(dims);
    }
}

} // namespace detail

/**
 * @brief Compute the strides for turning a sentinel into a list of indices.
 *
 * @param dims The list of dimensions.
 * @param out The calculated strides.
 * @param row_major If true, then the first stride will be the largest. If false, then the first stride will be the smallest.
 * @return The size calculated from the dimensions. Can be safely ignored.
 */
template <typename arr_type2, size_t Dims, typename... TupleDims>
    requires requires {
        requires sizeof...(TupleDims) == Dims;
        requires std::is_integral_v<arr_type2>;
    }
constexpr size_t dims_to_strides(std::tuple<TupleDims...> const &dims, std::array<arr_type2, Dims> &out, bool row_major) {

    if (row_major) {
        return detail::dims_to_strides<0, true>(dims, out);
    } else {
        return detail::dims_to_strides<Dims - 1, false>(dims, out);
    }
}

/**
 * @brief Compute the strides for turning a sentinel into a list of indices.
 *
 * @param dims The list of dimensions.
 * @param out The calculated strides.
 * @param row_major If true, then the first stride will be the largest. If false, then the first stride will be the smallest.
 * @return The size calculated from the dimensions. Can be safely ignored.
 */
template <typename arr_type2, size_t Dims, typename... TupleDims>
    requires requires {
        requires sizeof...(TupleDims) == Dims;
        requires std::is_integral_v<arr_type2>;
    }
constexpr size_t dims_to_strides(std::tuple<TupleDims...> const &dims, std::array<arr_type2, Dims> &out) {

    if (GlobalConfigMap::get_singleton().get_bool("row-major")) {
        return detail::dims_to_strides<0, true>(dims, out);
    } else {
        return detail::dims_to_strides<Dims - 1, false>(dims, out);
    }
}

template <int __I, typename Head, typename Index>
int compile_index_table(std::tuple<Head> const &, Index const &, int &out) {
    if constexpr (std::is_same_v<Head, Index>) {
        out = __I;
    } else {
        out = -1;
    }
    return 0;
}

template <int __I, typename Head, typename... UniqueIndices, typename Index>
auto compile_index_table(std::tuple<Head, UniqueIndices...> const &, Index const &index, int &out) -> int
    requires(sizeof...(UniqueIndices) != 0)
{
    if constexpr (std::is_same_v<Head, Index>) {
        out = __I;
    } else {
        compile_index_table<__I + 1>(std::tuple<UniqueIndices...>(), index, out);
    }
    return 0;
}

template <typename... UniqueIndices, typename... Indices, size_t... __I>
void compile_index_table(std::tuple<UniqueIndices...> const &from_inds, std::tuple<Indices...> const &to_inds, int *out,
                         std::index_sequence<__I...>) {
    std::array<int, sizeof...(Indices)> arr{compile_index_table<0>(from_inds, std::get<__I>(to_inds), out[__I])...};
}

/**
 * @brief Turn a list of indices into a link table.
 *
 * Takes a list of indices and creates a mapping so that an index list for a tensor can reference the unique index list.
 */
template <typename... UniqueIndices, typename... Indices>
void compile_index_table(std::tuple<UniqueIndices...> const &from_inds, std::tuple<Indices...> const &to_inds, int *out) {
    compile_index_table(from_inds, to_inds, out, std::make_index_sequence<sizeof...(Indices)>());
}

template <typename... UniqueIndices, typename... Indices, size_t... __I>
void compile_index_table(std::tuple<UniqueIndices...> const &from_inds, std::tuple<Indices...> const &to_inds,
                         std::array<int, sizeof...(Indices)> &out, std::index_sequence<__I...>) {
    std::array<int, sizeof...(Indices)> arr{compile_index_table<0>(from_inds, std::get<__I>(to_inds), out[__I])...};
}

template <typename... UniqueIndices, typename... Indices>
void compile_index_table(std::tuple<UniqueIndices...> const &from_inds, std::tuple<Indices...> const &to_inds,
                         std::array<int, sizeof...(Indices)> &out) {
    compile_index_table(from_inds, to_inds, out, std::make_index_sequence<sizeof...(Indices)>());
}

} // namespace einsums