//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <einsums/config.hpp>

#include <einsums/tensor/tensor.hpp>

#include <array>
#include <cstdint>
#include <tuple>

namespace einsums::linear_algebra::detail {

template <typename UniqueIndex, int BDim, typename BType>
inline size_t get_dim_ranges_for_many_b(const BType &B, const std::tuple<> &B_indices) {
    return 1;
}

template <typename UniqueIndex, int BDim, typename BType, typename BHead>
inline auto get_dim_ranges_for_many_b(const BType             &B,
                                      const std::tuple<BHead> &B_indices) -> std::enable_if<std::is_same_v<BHead, UniqueIndex>, size_t> {
    return B.dim(BDim);
}

template <typename UniqueIndex, int BDim, typename BType, typename BHead, typename... BIndices>
inline size_t get_dim_ranges_for_many_b(const BType &B, const std::tuple<BHead, BIndices...> &B_indices) {
    if constexpr (std::is_same_v<BHead, UniqueIndex>) {
        return B.dim(BDim);
    } else {
        return get_dim_ranges_for_many_b<UniqueIndex, BDim + 1>(B, std::tuple<BIndices...>());
    }
}

template <typename UniqueIndex, int ADim, typename AType, typename BType, typename... BIndices>
inline size_t get_dim_ranges_for_many_a(const AType &A, const std::tuple<> &A_indices, const BType &B,
                                        const std::tuple<BIndices...> &B_indices) {
    return get_dim_ranges_for_many_b<UniqueIndex, 0>(B, B_indices);
}

template <typename UniqueIndex, int ADim, typename AType, typename BType, typename AHead, typename... BIndices>
inline size_t get_dim_ranges_for_many_a(const AType &A, const std::tuple<AHead> &A_indices, const BType &B,
                                        const std::tuple<BIndices...> &B_indices) {
    if constexpr (std::is_same_v<AHead, UniqueIndex>) {
        return A.dim(ADim);
    } else {
        return get_dim_ranges_for_many_b<UniqueIndex, 0>(B, B_indices);
    }
}

template <typename UniqueIndex, int ADim, typename AType, typename BType, typename AHead, typename... AIndices, typename... BIndices>
inline auto get_dim_ranges_for_many_a(const AType &A, const std::tuple<AHead, AIndices...> &A_indices, const BType &B,
                                      const std::tuple<BIndices...> &B_indices) -> std::enable_if_t<sizeof...(AIndices) != 0, size_t> {
    if constexpr (std::is_same_v<AHead, UniqueIndex>) {
        return A.dim(ADim);
    } else {
        return get_dim_ranges_for_many_a<UniqueIndex, ADim + 1>(A, std::tuple<AIndices...>(), B, B_indices);
    }
}

template <typename UniqueIndex, int CDim, typename CType, typename AType, typename BType, typename... AIndices, typename... BIndices>
inline size_t get_dim_ranges_for_many_c(const CType &C, const std::tuple<> &C_indices, const AType &A,
                                        const std::tuple<AIndices...> &A_indices, const BType &B,
                                        const std::tuple<BIndices...> &B_indices) {
    return get_dim_ranges_for_many_a<UniqueIndex, 0>(A, A_indices, B, B_indices);
}

template <typename UniqueIndex, int CDim, typename CType, typename AType, typename BType, typename CHead, typename... AIndices,
          typename... BIndices>
inline size_t get_dim_ranges_for_many_c(const CType &C, const std::tuple<CHead> &C_indices, const AType &A,
                                        const std::tuple<AIndices...> &A_indices, const BType &B,
                                        const std::tuple<BIndices...> &B_indices) {
    if constexpr (std::is_same_v<CHead, UniqueIndex>) {
        return C.dim(CDim);
    } else {
        return get_dim_ranges_for_many_a<UniqueIndex, 0>(A, A_indices, B, B_indices);
    }
}

template <typename UniqueIndex, int CDim, typename CType, typename AType, typename BType, typename CHead, typename... CIndices,
          typename... AIndices, typename... BIndices>
inline auto get_dim_ranges_for_many_c(const CType &C, const std::tuple<CHead, CIndices...> &C_indices, const AType &A,
                                      const std::tuple<AIndices...> &A_indices, const BType &B,
                                      const std::tuple<BIndices...> &B_indices) -> std::enable_if_t<sizeof...(CIndices) != 0, size_t> {
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
inline auto get_dim_ranges_for_many(const CType &C, const std::tuple<CIndices...> &C_indices, const AType &A,
                                    const std::tuple<AIndices...> &A_indices, const BType &B, const std::tuple<BIndices...> &B_indices,
                                    const std::tuple<AllUniqueIndices...> &All_unique_indices) {
    return std::array{get_dim_ranges_for_many_c<AllUniqueIndices, 0>(C, C_indices, A, A_indices, B, B_indices)...};
}

/**
 * @brief Converts a single sentinel value into a list of indices.
 *
 * @param sentinel The sentinel to convert.
 * @param unique_strides The strides of the unique indices that the sentinel is iterating over.
 * @param out_inds The converted indices.
 */
template <size_t num_unique_inds>
EINSUMS_HOST_DEVICE inline void sentinel_to_indices(size_t sentinel, const size_t *unique_strides, size_t *out_inds) {
    size_t hold = sentinel;

#pragma unroll
    for (ssize_t i = 0; i < num_unique_inds; i++) {
        if (unique_strides[i] != 0) {
            out_inds[i] = hold / unique_strides[i];
            hold %= unique_strides[i];
        } else {
            out_inds[i] = 0;
        }
    }
}

template <size_t num_unique_inds>
inline void sentinel_to_indices(size_t sentinel, const std::array<size_t, num_unique_inds> &unique_strides,
                                std::array<size_t, num_unique_inds> &out_inds) {
    size_t hold = sentinel;

#pragma unroll
    for (ssize_t i = 0; i < num_unique_inds; i++) {
        if (unique_strides[i] != 0) {
            out_inds[i] = hold / unique_strides[i];
            hold %= unique_strides[i];
        } else {
            out_inds[i] = 0;
        }
    }
}

/**
 * @brief Compute the strides for turning a sentinel into a list of indices.
 *
 * @param dims The list of dimensions.
 * @param out The calculated strides.
 */
template <size_t Dims>
void dims_to_strides(const std::array<size_t, Dims> &dims, std::array<size_t, Dims> &out) {
    size_t stride = 1;

    for (int i = Dims - 1; i >= 0; i--) {
        out[i] = stride;
        stride *= dims[i];
    }
}

template <size_t Dims>
void dims_to_strides(const einsums::dim<Dims> &dims, std::array<size_t, Dims> &out) {
    size_t stride = 1;

    for (int i = Dims - 1; i >= 0; i--) {
        out[i] = stride;
        stride *= dims[i];
    }
}

template <int I, typename Head, typename Index>
int compile_index_table(const std::tuple<Head> &, const Index &, int &out) {
    if constexpr (std::is_same_v<Head, Index>) {
        out = I;
    } else {
        out = -1;
    }
    return 0;
}

template <int I, typename Head, typename... UniqueIndices, typename Index>
auto compile_index_table(const std::tuple<Head, UniqueIndices...> &, const Index &index,
                         int &out) -> std::enable_if_t<sizeof...(UniqueIndices) != 0, int> {
    if constexpr (std::is_same_v<Head, Index>) {
        out = I;
    } else {
        compile_index_table<I + 1>(std::tuple<UniqueIndices...>(), index, out);
    }
    return 0;
}

template <typename... UniqueIndices, typename... Indices, size_t... I>
void compile_index_table(const std::tuple<UniqueIndices...> &from_inds, const std::tuple<Indices...> &to_inds, int *out,
                         std::index_sequence<I...>) {
    std::array<int, sizeof...(Indices)> arr{compile_index_table<0>(from_inds, std::get<I>(to_inds), out[I])...};
}

/**
 * @brief Turn a list of indices into a link table.
 *
 * Takes a list of indices and creates a mapping so that an index list for a tensor can reference the unique index list.
 */
template <typename... UniqueIndices, typename... Indices>
void compile_index_table(const std::tuple<UniqueIndices...> &from_inds, const std::tuple<Indices...> &to_inds, int *out) {
    compile_index_table(from_inds, to_inds, out, std::make_index_sequence<sizeof...(Indices)>());
}

template <typename... UniqueIndices, typename... Indices, size_t... I>
void compile_index_table(const std::tuple<UniqueIndices...> &from_inds, const std::tuple<Indices...> &to_inds,
                         std::array<int, sizeof...(Indices)> &out, std::index_sequence<I...>) {
    std::array<int, sizeof...(Indices)> arr{compile_index_table<0>(from_inds, std::get<I>(to_inds), out[I])...};
}

template <typename... UniqueIndices, typename... Indices>
void compile_index_table(const std::tuple<UniqueIndices...> &from_inds, const std::tuple<Indices...> &to_inds,
                         std::array<int, sizeof...(Indices)> &out) {
    compile_index_table(from_inds, to_inds, out, std::make_index_sequence<sizeof...(Indices)>());
}
} // namespace einsums::linear_algebra::detail