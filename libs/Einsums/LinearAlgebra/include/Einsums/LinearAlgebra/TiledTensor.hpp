//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Config.hpp>

#include <Einsums/BLAS.hpp>
#include <Einsums/Concepts/TensorConcepts.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/LinearAlgebra/Base.hpp>
#include <Einsums/Print.hpp>

#include <cstring>

namespace einsums::linear_algebra::detail {

template <TiledTensorConcept AType, typename T = RemoveComplexT<typename AType::ValueType>>
void sum_square(AType const &A, T *scale, T *sumsq) {

    for (auto const &[key, tile] : A.tiles()) {
        sum_square(tile, scale, sumsq);
    }
}

template <TiledTensorConcept AType, TiledTensorConcept BType>
    requires(SameRank<AType, BType>)
auto dot(AType const &A, BType const &B) -> BiggestTypeT<typename AType::ValueType, typename BType::ValueType> {
    constexpr size_t Rank = AType::Rank;
    using T               = BiggestTypeT<typename AType::ValueType, typename BType::ValueType>;
    std::array<size_t, Rank> strides;

    size_t prod = 1;

    for (int i = 0; i < Rank; i++) {
        if (A.grid_size(i) != B.grid_size(i)) {
            EINSUMS_THROW_EXCEPTION(tensor_compat_error, "Tiled tensors have incompatible tiles.");
        }

        strides[Rank - 1 - i] = prod;
        prod *= A.grid_size(i);
    }
    T out = 0;

#pragma omp parallel for reduction(+ : out)
    for (size_t index = 0; index < A.grid_size(); index++) {
        std::array<size_t, Rank> index_arr;
        size_t                   temp_index = index;

        for (int i = 0; i < Rank; i++) {
            index_arr[i] = temp_index / strides[i];
            temp_index %= strides[i];
        }

        if (!A.has_tile(index_arr) || !B.has_tile(index_arr) || A.has_zero_size(index_arr) || B.has_zero_size(index_arr)) {
            continue;
        }
        out += dot(A.tile(index_arr), B.tile(index_arr));
    }

    return out;
}

template <TiledTensorConcept AType, TiledTensorConcept BType>
    requires(SameRank<AType, BType>)
auto true_dot(AType const &A, BType const &B) -> BiggestTypeT<typename AType::ValueType, typename BType::ValueType> {
    constexpr size_t Rank = AType::Rank;
    using T               = BiggestTypeT<typename AType::ValueType, typename BType::ValueType>;
    std::array<size_t, Rank> strides;

    size_t prod = 1;

    for (int i = 0; i < Rank; i++) {
        if (A.grid_size(i) != B.grid_size(i)) {
            EINSUMS_THROW_EXCEPTION(tensor_compat_error, "Tiled tensors have incompatible tiles.");
        }

        strides[Rank - 1 - i] = prod;
        prod *= A.grid_size(i);
    }
    T out = 0;

#pragma omp parallel for reduction(+ : out)
    for (size_t index = 0; index < A.grid_size(); index++) {
        std::array<size_t, Rank> index_arr;
        size_t                   temp_index = index;

        for (int i = 0; i < Rank; i++) {
            index_arr[i] = temp_index / strides[i];
            temp_index %= A.grid_size(i);
        }

        if (!A.has_tile(index_arr) || !B.has_tile(index_arr) || A.has_zero_size(index_arr) || B.has_zero_size(index_arr)) {
            continue;
        }
        out += true_dot(A.tile(index_arr), B.tile(index_arr));
    }

    return out;
}

template <TiledTensorConcept AType, TiledTensorConcept BType, TiledTensorConcept CType, typename U>
    requires requires {
        requires MatrixConcept<AType>;
        requires SameUnderlyingAndRank<AType, BType, CType>;
        requires std::convertible_to<U, typename AType::ValueType>;
    }
void gemm(char transA, char transB, U const alpha, AType const &A, BType const &B, U const beta, CType *C) {
    bool tA = std::tolower(transA) != 'n', tB = std::tolower(transB) != 'n';

    if (!strchr("CNTcnt", transA) || !strchr("CNTcnt", transB)) {
        EINSUMS_THROW_EXCEPTION(
            std::invalid_argument,
            "The transpose arguments were invalid! Expected c, n, or t, case insensitive. Got transA: {} and transB: {}.", transA, transB);
    }
    // Check for compatibility.
    if (C->grid_size(0) != A.grid_size(tA ? 1 : 0) || C->grid_size(1) != B.grid_size(tB ? 0 : 1)) {
        EINSUMS_THROW_CODED_EXCEPTION(tensor_compat_error, 0, "Output tensor needs to have a compatible tile grid with the inputs.");
    }
    if (A.grid_size(tA ? 0 : 1) != B.grid_size(tB ? 1 : 0)) {
        EINSUMS_THROW_CODED_EXCEPTION(tensor_compat_error, 1, "Input tensors need to have compatible tile grids.");
    }
    for (int i = 0; i < C->grid_size(0); i++) {
        if (C->tile_size(0)[i] != A.tile_size(tA ? 1 : 0)[i]) {
            EINSUMS_THROW_EXCEPTION(dimension_error, "Tile sizes need to match between all three tensors.");
        }
    }
    for (int i = 0; i < C->grid_size(1); i++) {
        if (C->tile_size(1)[i] != B.tile_size(tB ? 0 : 1)[i]) {
            EINSUMS_THROW_EXCEPTION(dimension_error, "Tile sizes need to match between all three tensors.");
        }
    }
    for (int i = 0; i < A.grid_size(tA ? 0 : 1); i++) {
        if (A.tile_size(tA ? 0 : 1)[i] != B.tile_size(tB ? 1 : 0)[i]) {
            EINSUMS_THROW_EXCEPTION(dimension_error, "Tile sizes need to match between all three tensors.");
        }
    }
    int x_size = C->grid_size(0), y_size = C->grid_size(1), z_size = A.grid_size(tA ? 0 : 1);

    // For every block in C, do matrix multiplication.
    // #pragma omp parallel for collapse(2)
    for (int i = 0; i < x_size; i++) {
        for (int j = 0; j < y_size; j++) {
            if (C->has_zero_size(i, j)) {
                continue;
            }
            // Check to see if C will be modified.
            bool modified = false;
            for (int k = 0; k < z_size; k++) {
                if (tA && tB) {
                    modified |= A.has_tile(k, i) && B.has_tile(j, k) && !A.has_zero_size(k, i) && !B.has_zero_size(j, k);
                } else if (!tA && tB) {
                    modified |= A.has_tile(i, k) && B.has_tile(j, k) && !A.has_zero_size(i, k) && !B.has_zero_size(j, k);
                } else if (tA && !tB) {
                    modified |= A.has_tile(k, i) && B.has_tile(k, j) && !A.has_zero_size(k, i) && !B.has_zero_size(k, j);
                } else {
                    modified |= A.has_tile(i, k) && B.has_tile(k, j) && !A.has_zero_size(i, k) && !B.has_zero_size(k, j);
                }
            }

            // If C is modified, then loop through and matrix multiply. Otherwise, scale or delete depending on beta.
            if (modified) {
                C->lock();
                bool  created = !C->has_tile(i, j);
                auto &C_tile  = C->tile(i, j);
                C->unlock();
                if (beta == U{0.0} || created) {
                    C_tile.zero();
                } else {
                    C_tile *= beta;
                }

                for (int k = 0; k < z_size; k++) {
                    if (tA && tB) {
                        if (A.has_tile(k, i) && B.has_tile(j, k) && !A.has_zero_size(k, i) && !B.has_zero_size(j, k)) {
                            gemm(transA, transB, alpha, A.tile(k, i), B.tile(j, k), U{1.0}, &C_tile);
                        }
                    } else if (tA && !tB) {
                        if (A.has_tile(k, i) && B.has_tile(k, j) && !A.has_zero_size(k, i) && !B.has_zero_size(k, j)) {
                            gemm(transA, transB, alpha, A.tile(k, i), B.tile(k, j), U{1.0}, &C_tile);
                        }
                    } else if (!tA && tB) {
                        if (A.has_tile(i, k) && B.has_tile(j, k) && !A.has_zero_size(i, k) && !B.has_zero_size(j, k)) {
                            gemm(transA, transB, alpha, A.tile(i, k), B.tile(j, k), U{1.0}, &C_tile);
                        }
                    } else {
                        if (A.has_tile(i, k) && B.has_tile(k, j) && !A.has_zero_size(i, k) && !B.has_zero_size(k, j)) {
                            gemm(transA, transB, alpha, A.tile(i, k), B.tile(k, j), U{1.0}, &C_tile);
                        }
                    }
                }
            } else {
                C->lock();
                if (C->has_tile(i, j) && !C->has_zero_size(i, j)) {
                    if (beta == U{0.0}) {
                        // C->tiles().erase(std::array<int, 2>{i, j});
                        C->tile(i, j).zero();
                    } else {
                        C->tile(i, j) *= beta;
                    }
                }
                C->unlock();
            }
        }
    }
}

template <bool TransA, bool TransB, TiledTensorConcept AType, TiledTensorConcept BType, TiledTensorConcept CType, typename U>
    requires requires {
        requires MatrixConcept<AType>;
        requires SameUnderlyingAndRank<AType, BType, CType>;
        requires std::convertible_to<U, typename AType::ValueType>;
    }
void gemm(U const alpha, AType const &A, BType const &B, U const beta, CType *C) {
    gemm((TransA) ? 't' : 'n', (TransB) ? 't' : 'n', alpha, A, B, beta, C);
}

template <TiledTensorConcept AType, VectorConcept XType, VectorConcept YType, typename U>
    requires requires {
        requires MatrixConcept<AType>;
        requires SameUnderlying<AType, XType, YType>;
        requires std::convertible_to<U, typename AType::ValueType>;
    }
void gemv(char transA, U const alpha, AType const &A, XType const &z, U const beta, YType *y) {
    char tA    = std::tolower(transA);
    bool trans = tA != 'n';

    if (!strchr("ntc", tA)) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "The transA parameter was invalid! Expected n, t, or c, case insensitive, got {}.",
                                transA);
    }

    if constexpr (IsTiledTensorV<XType>) {
        if (transA) {
            if (A.tile_sizes(0) != z.tile_sizes(0)) {
                EINSUMS_THROW_CODED_EXCEPTION(tensor_compat_error, 0, "Tiled tensors need to have compatible tile sizes.");
            }
        } else {
            if (A.tile_sizes(1) != z.tile_sizes(0)) {
                EINSUMS_THROW_CODED_EXCEPTION(tensor_compat_error, 0, "Tiled tensors need to have compatible tile sizes.");
            }
        }
    }
    if constexpr (IsTiledTensorV<YType>) {
        if (trans) {
            if (A.tile_sizes(1) != y->tile_sizes(0)) {
                EINSUMS_THROW_CODED_EXCEPTION(tensor_compat_error, 1, "Tiled tensors need to have compatible tile sizes.");
            }
        } else {
            if (A.tile_sizes(0) != y->tile_sizes(0)) {
                EINSUMS_THROW_CODED_EXCEPTION(tensor_compat_error, 1, "Tiled tensors need to have compatible tile sizes.");
            }
        }
    }

    if (beta == U(0.0)) {
        y->zero();
    } else {
        *y *= beta;
    }

    int loop_bound = (trans) ? A.grid_size(1) : A.grid_size(0);
    EINSUMS_OMP_PARALLEL_FOR
    for (int i = 0; i < loop_bound; i++) {
        if (A.tile_size((trans) ? 1 : 0)[i] == 0) {
            continue;
        }
        if constexpr (IsTiledTensorV<YType>) {
            y->lock();
            auto &y_tile = y->tile(i);
            y->unlock();
            for (int j = 0; j < A.grid_size((trans) ? 0 : 1); j++) {
                if (trans) {
                    if (!A.has_tile(j, i) || A.has_zero_size(j, i)) {
                        continue;
                    }
                } else {
                    if (!A.has_tile(i, j) || A.has_zero_size(i, j)) {
                        continue;
                    }
                }
                if constexpr (IsTiledTensorV<XType>) {
                    if (trans) {
                        gemv(transA, alpha, A.tile(j, i), z.tile(j), U(1.0), &y_tile);
                    } else {
                        gemv(transA, alpha, A.tile(i, j), z.tile(j), U(1.0), &y_tile);
                    }
                } else {
                    if (trans) {
                        gemv(transA, alpha, A.tile(j, i),
                             z(Range{A.tile_offset((trans) ? 0 : 1)[j],
                                     A.tile_offset((trans) ? 0 : 1)[j] + A.tile_size((trans) ? 0 : 1)[j]}),
                             U(1.0), &y_tile);
                    } else {
                        gemv(transA, alpha, A.tile(i, j),
                             z(Range{A.tile_offset((trans) ? 0 : 1)[j],
                                     A.tile_offset((trans) ? 0 : 1)[j] + A.tile_size((trans) ? 0 : 1)[j]}),
                             U(1.0), &y_tile);
                    }
                }
            }
        } else {
            auto y_tile =
                (*y)(Range{A.tile_offset((trans) ? 1 : 0)[i], A.tile_offset((trans) ? 1 : 0)[i] + A.tile_size((trans) ? 1 : 0)[i]});
            for (int j = 0; j < A.grid_size((trans) ? 0 : 1); j++) {
                if (trans) {
                    if (!A.has_tile(j, i) || A.has_zero_size(j, i)) {
                        continue;
                    }
                } else {
                    if (!A.has_tile(i, j) || A.has_zero_size(i, j)) {
                        continue;
                    }
                }
                if constexpr (IsTiledTensorV<XType>) {
                    if (trans) {
                        gemv(transA, alpha, A.tile(j, i), z.tile(j), U(1.0), &y_tile);
                    } else {
                        gemv(transA, alpha, A.tile(i, j), z.tile(j), U(1.0), &y_tile);
                    }
                } else {
                    if (trans) {
                        gemv(transA, alpha, A.tile(j, i),
                             z(Range{A.tile_offset((trans) ? 0 : 1)[j],
                                     A.tile_offset((trans) ? 0 : 1)[j] + A.tile_size((trans) ? 0 : 1)[j]}),
                             U(1.0), &y_tile);
                    } else {
                        gemv(transA, alpha, A.tile(i, j),
                             z(Range{A.tile_offset((trans) ? 0 : 1)[j],
                                     A.tile_offset((trans) ? 0 : 1)[j] + A.tile_size((trans) ? 0 : 1)[j]}),
                             U(1.0), &y_tile);
                    }
                }
            }
        }
    }
}

template <bool TransA, TiledTensorConcept AType, VectorConcept XType, VectorConcept YType, typename U>
    requires requires {
        requires MatrixConcept<AType>;
        requires SameUnderlying<AType, XType, YType>;
        requires std::convertible_to<U, typename AType::ValueType>;
    }
void gemv(U const alpha, AType const &A, XType const &z, U const beta, YType *y) {
    gemv((TransA) ? 't' : 'n', alpha, A, z, beta, y);
}

template <TiledTensorConcept AType, TiledTensorConcept BType, TiledTensorConcept CType>
    requires SameUnderlyingAndRank<AType, BType, CType>
void direct_product(typename AType::ValueType alpha, AType const &A, BType const &B, typename AType::ValueType beta, CType *C) {
    using T               = typename AType::ValueType;
    constexpr size_t Rank = AType::Rank;
    EINSUMS_OMP_PARALLEL_FOR
    for (size_t sentinel = 0; sentinel < A.grid_size(); sentinel++) {
        std::array<int, Rank> index = std::array<int, Rank>{};
        size_t                hold  = sentinel;

        // Calculate the index.
        for (int i = 0; i < Rank; i++) {
            index[i] = hold % A.grid_size(i);
            hold /= A.grid_size(i);
        }
        if (A.has_tile(index) && B.has_tile(index) && !A.has_zero_size(index) && !B.has_zero_size(index)) {
            C->lock();
            auto &C_tile = C->tile(index);
            C->unlock();
            direct_product(alpha, A.tile(index), B.tile(index), beta, &C_tile);
        }
    }
}

template <TiledTensorConcept AType, TiledTensorConcept XYType>
    requires requires {
        requires MatrixConcept<AType>;
        requires VectorConcept<XYType>;
        requires SameUnderlying<AType, XYType>;
    }
void ger(typename AType::ValueType alpha, XYType const &X, XYType const &Y, AType *A) {
    if (A->grid_size(0) != X.grid_size(0) && A->grid_size(1) != Y.grid_size(0)) {
        EINSUMS_THROW_EXCEPTION(tensor_compat_error, "Tiled tensors have incompatible grids!");
    }

#pragma omp parallel for collapse(2)
    for (int i = 0; i < X.grid_size(); i++) {
        for (int j = 0; j < Y.grid_size(); j++) {
            if (X.has_zero_size(i) || Y.has_zero_size(j) || !X.has_tile(i) || !Y.has_tile(j)) {
                continue;
            }
            A->lock();
            auto &a_tile = A->tile(i, j);
            A->unlock();

            ger(alpha, X.tile[i], Y.tile[j], &a_tile);
        }
    }
}

template <TiledTensorConcept AType, TiledTensorConcept XYType>
    requires requires {
        requires MatrixConcept<AType>;
        requires VectorConcept<XYType>;
        requires SameUnderlying<AType, XYType>;
    }
void gerc(typename AType::ValueType alpha, XYType const &X, XYType const &Y, AType *A) {
    if (A->grid_size(0) != X.grid_size(0) && A->grid_size(1) != Y.grid_size(0)) {
        EINSUMS_THROW_EXCEPTION(tensor_compat_error, "Tiled tensors have incompatible grids!");
    }

#pragma omp parallel for collapse(2)
    for (int i = 0; i < X.grid_size(); i++) {
        for (int j = 0; j < Y.grid_size(); j++) {
            if (X.has_zero_size(i) || Y.has_zero_size(j) || !X.has_tile(i) || !Y.has_tile(j)) {
                continue;
            }
            A->lock();
            auto &a_tile = A->tile(i, j);
            A->unlock();

            gerc(alpha, X.tile[i], Y.tile[j], &a_tile);
        }
    }
}

template <TiledTensorConcept AType>
void scale(typename AType::ValueType alpha, AType *A) {
    EINSUMS_OMP_PARALLEL_FOR
    for (auto it = A->tiles().begin(); it != A->tiles().end(); it++) {
        scale(alpha, &(it->second));
    }
}

} // namespace einsums::linear_algebra::detail