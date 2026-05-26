//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Einsums/Assert.hpp>
#include <Einsums/BLAS.hpp>
#include <Einsums/Concepts/Complex.hpp>
#include <Einsums/Concepts/SubscriptChooser.hpp>
#include <Einsums/Concepts/TensorConcepts.hpp>
#include <Einsums/Config/CompilerSpecific.hpp>
#include <Einsums/Errors/Error.hpp>
#include <Einsums/LinearAlgebra/Bases/direct_division.hpp>
#include <Einsums/LinearAlgebra/Bases/direct_product.hpp>
#include <Einsums/LinearAlgebra/Bases/dot.hpp>
#include <Einsums/LinearAlgebra/Bases/gemm.hpp>
#include <Einsums/LinearAlgebra/Bases/gemv.hpp>
#include <Einsums/LinearAlgebra/Bases/ger.hpp>
#include <Einsums/LinearAlgebra/Bases/norm.hpp>
#include <Einsums/LinearAlgebra/Bases/sum_square.hpp>
#include <Einsums/LinearAlgebra/Bases/syev.hpp>
#include <Einsums/LinearAlgebra/Bases/triangular.hpp>
#include <Einsums/LinearAlgebra/SymmetryDispatch.hpp>
#include <Einsums/Profile.hpp>
#include <Einsums/Tensor/Tensor.hpp>
#include <Einsums/TensorBase/IndexUtilities.hpp>
#include <Einsums/TensorImpl/TensorImpl.hpp>
#include <Einsums/TensorImpl/TensorImplOperations.hpp>
#include <Einsums/TensorUtilities/CreateTensorLike.hpp>

#include <optional>
#include <stdexcept>

namespace einsums::linear_algebra::detail {

template <typename T>
void sum_square(einsums::detail::TensorImpl<T> const &a, RemoveComplexT<T> *scale, RemoveComplexT<T> *sumsq) noexcept {
    impl_sum_square(a, scale, sumsq);
}

template <CoreBasicTensorConcept AType>
void sum_square(AType const &a, RemoveComplexT<typename AType::ValueType> *scale, RemoveComplexT<typename AType::ValueType> *sumsq) {
    sum_square(a.impl(), scale, sumsq);
}

template <typename AType, typename BType, typename CType, typename AlphaType, typename BetaType>
void gemm(char transA, char transB, AlphaType const alpha, einsums::detail::TensorImpl<AType> const &A,
          einsums::detail::TensorImpl<BType> const &B, BetaType const beta, einsums::detail::TensorImpl<CType> *C) {
    char const tA = std::tolower(transA), tB = std::tolower(transB);
    // Check for gemmability.
    if (A.rank() != 2 || B.rank() != 2 || C->rank() != 2) {
        EINSUMS_THROW_EXCEPTION(rank_error, "The inputs to gemm need to be matrices! Got ranks {}, {}, and {}.", A.rank(), B.rank(),
                                C->rank());
    }

    if (!std::strchr("cnt", tA) || !std::strchr("cnt", tB)) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                "One of the transpose inputs was invalid. Expected 'c', 'n', or 't', case insensitive. Got {:?} and {:?}.",
                                transA, transB);
    }
    if (tA == 'n' && tB == 'n') {
        if (A.dim(1) != B.dim(0)) {
            EINSUMS_THROW_EXCEPTION(tensor_compat_error, "The link dimensions do not match! Got {} and {}.", A.dim(1), B.dim(0));
        }

        if (A.dim(0) != C->dim(0)) {
            EINSUMS_THROW_EXCEPTION(tensor_compat_error, "The first target dimensions do not match! Got {} and {}.", A.dim(0), C->dim(0));
        }

        if (B.dim(1) != C->dim(1)) {
            EINSUMS_THROW_EXCEPTION(tensor_compat_error, "The second target dimensions do not match! Got {} and {}.", B.dim(1), C->dim(1));
        }
    } else if (tA != 'n' && tB == 'n') {
        if (A.dim(0) != B.dim(0)) {
            EINSUMS_THROW_EXCEPTION(tensor_compat_error, "The link dimensions do not match! Got {} and {}.", A.dim(0), B.dim(0));
        }

        if (A.dim(1) != C->dim(0)) {
            EINSUMS_THROW_EXCEPTION(tensor_compat_error, "The first target dimensions do not match! Got {} and {}.", A.dim(1), C->dim(0));
        }

        if (B.dim(1) != C->dim(1)) {
            EINSUMS_THROW_EXCEPTION(tensor_compat_error, "The second target dimensions do not match! Got {} and {}.", B.dim(1), C->dim(1));
        }
    } else if (tA == 'n' && tB != 'n') {
        if (A.dim(1) != B.dim(1)) {
            EINSUMS_THROW_EXCEPTION(tensor_compat_error, "The link dimensions do not match! Got {} and {}.", A.dim(1), B.dim(1));
        }

        if (A.dim(0) != C->dim(0)) {
            EINSUMS_THROW_EXCEPTION(tensor_compat_error, "The first target dimensions do not match! Got {} and {}.", A.dim(0), C->dim(0));
        }

        if (B.dim(0) != C->dim(1)) {
            EINSUMS_THROW_EXCEPTION(tensor_compat_error, "The second target dimensions do not match! Got {} and {}.", B.dim(0), C->dim(1));
        }
    } else if (tA != 'n' && tB != 'n') {
        if (A.dim(0) != B.dim(1)) {
            EINSUMS_THROW_EXCEPTION(tensor_compat_error, "The link dimensions do not match! Got {} and {}.", A.dim(0), B.dim(1));
        }

        if (A.dim(1) != C->dim(0)) {
            EINSUMS_THROW_EXCEPTION(tensor_compat_error, "The first target dimensions do not match! Got {} and {}.", A.dim(1), C->dim(0));
        }

        if (B.dim(0) != C->dim(1)) {
            EINSUMS_THROW_EXCEPTION(tensor_compat_error, "The second target dimensions do not match! Got {} and {}.", B.dim(0), C->dim(1));
        }
    }

    if (A.dim(0) == 0 || A.dim(1) == 0 || B.dim(0) == 0 || B.dim(1) == 0 || C->dim(0) == 0 || C->dim(1) == 0) {
        return;
    }

    impl_gemm(transA, transB, alpha, A, B, beta, C);
}

template <bool TransA, bool TransB, typename AType, typename BType, typename CType, typename AlphaType, typename BetaType>
void gemm(AlphaType const alpha, einsums::detail::TensorImpl<AType> const &A, einsums::detail::TensorImpl<BType> const &B,
          BetaType const beta, einsums::detail::TensorImpl<CType> *C) {
    gemm(TransA ? 't' : 'n', TransB ? 't' : 'n', alpha, A, B, beta, C);
}

template <bool TransA, bool TransB, typename U, CoreBasicTensorConcept AType, CoreBasicTensorConcept BType, CoreBasicTensorConcept CType>
    requires requires {
        requires CoreBasicTensorConcept<AType>;
        requires SameUnderlying<AType, BType, CType>;
        requires(std::convertible_to<U, typename AType::ValueType>);
    }
void gemm(U const alpha, AType const &A, BType const &B, U const beta, CType *C) {
    gemm<TransA, TransB>(alpha, A.impl(), B.impl(), beta, &C->impl());
}

template <typename U, CoreBasicTensorConcept AType, CoreBasicTensorConcept BType, CoreBasicTensorConcept CType>
    requires requires {
        requires CoreBasicTensorConcept<AType>;
        requires SameUnderlying<AType, BType, CType>;
        requires(std::convertible_to<U, typename AType::ValueType>);
    }
void gemm(char transA, char transB, U const alpha, AType const &A, BType const &B, U const beta, CType *C) {
    using T = typename AType::ValueType;
    // Symmetry-aware fast path: if either operand declared a rank-2
    // symmetric or Hermitian descriptor, dispatch to ``symm`` / ``hemm``
    // instead of the general gemm. Falls through when no match.
    auto const *desc_a = A.symmetry();
    auto const *desc_b = B.symmetry();
    if (desc_a || desc_b) {
        if (try_symmetric_gemm<T>(transA, transB, static_cast<T>(alpha), A.impl(), desc_a, B.impl(), desc_b, static_cast<T>(beta),
                                  &C->impl())) {
            return;
        }
    }
    gemm(transA, transB, alpha, A.impl(), B.impl(), beta, &C->impl());
}

template <typename AType, typename XType, typename YType, typename AlphaType, typename BetaType>
void gemv(char transA, AlphaType alpha, einsums::detail::TensorImpl<AType> const &A, einsums::detail::TensorImpl<XType> const &X,
          BetaType beta, einsums::detail::TensorImpl<YType> *Y) {
    if (A.rank() != 2 || X.rank() != 1 || Y->rank() != 1) {
        EINSUMS_THROW_EXCEPTION(
            rank_error, "The ranks of the tensors passed to gemv are incompatible! Requires a matrix and to vectors. Got {}, {}, and {}.",
            A.rank(), X.rank(), Y->rank());
    }
    if (!std::strchr("cntCNT", transA)) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "The transpose character is invalid! Expected either 'c', 'n', or 't'. Got {:?}.",
                                transA);
    }
    if (std::tolower(transA) == 'n') {
        if (A.dim(0) != Y->dim(0)) {
            EINSUMS_THROW_EXCEPTION(tensor_compat_error,
                                    "The dimensions of the input matrix and output tensor do not match! Got {} and {}.", A.dim(1),
                                    Y->dim(0));
        }
        if (A.dim(1) != X.dim(0)) {
            EINSUMS_THROW_EXCEPTION(tensor_compat_error, "The dimensions of the input matrix and input tensor do not match! Got {} and {}.",
                                    A.dim(0), X.dim(0));
        }
    } else {
        if (A.dim(1) != Y->dim(0)) {
            EINSUMS_THROW_EXCEPTION(tensor_compat_error,
                                    "The dimensions of the input matrix and output tensor do not match! Got {} and {}.", A.dim(0),
                                    Y->dim(0));
        }
        if (A.dim(0) != X.dim(0)) {
            EINSUMS_THROW_EXCEPTION(tensor_compat_error, "The dimensions of the input matrix and input tensor do not match! Got {} and {}.",
                                    A.dim(1), X.dim(0));
        }
    }

    if (A.dim(0) == 0 || A.dim(1) == 0 || X.dim(0) == 0 || Y->dim(0) == 0) {
        return;
    }
    impl_gemv(transA, alpha, A, X, beta, Y);
}

template <bool TransA, typename AType, typename XType, typename YType, typename AlphaType, typename BetaType>
void gemv(AlphaType alpha, einsums::detail::TensorImpl<AType> const &A, einsums::detail::TensorImpl<XType> const &X, BetaType beta,
          einsums::detail::TensorImpl<YType> *Y) {

    gemv((TransA) ? 't' : 'n', alpha, A, X, beta, Y);
}

template <bool TransA, typename U, CoreBasicTensorConcept AType, CoreBasicTensorConcept XType, CoreBasicTensorConcept YType>
    requires requires {
        requires SameUnderlying<AType, XType, YType>;
        requires CoreBasicTensorConcept<AType>;
        requires CoreBasicTensorConcept<XType>;
        requires CoreBasicTensorConcept<YType>;
        requires std::convertible_to<U, typename AType::ValueType>;
    }
void gemv(U const alpha, AType const &A, XType const &z, U const beta, YType *y) {
    gemv<TransA>(alpha, A.impl(), z.impl(), beta, &y->impl());
}

template <typename U, CoreBasicTensorConcept AType, CoreBasicTensorConcept XType, CoreBasicTensorConcept YType>
    requires requires {
        requires SameUnderlying<AType, XType, YType>;
        requires CoreBasicTensorConcept<AType>;
        requires CoreBasicTensorConcept<XType>;
        requires CoreBasicTensorConcept<YType>;
        requires std::convertible_to<U, typename AType::ValueType>;
    }
void gemv(char transA, U const alpha, AType const &A, XType const &z, U const beta, YType *y) {
    gemv(transA, alpha, A.impl(), z.impl(), beta, &y->impl());
}

template <bool ComputeEigenvectors = true, typename AType>
void syev(einsums::detail::TensorImpl<AType> *A, einsums::detail::TensorImpl<RemoveComplexT<AType>> *W) {
    if (A->rank() != 2 || W->rank() != 1) {
        EINSUMS_THROW_EXCEPTION(rank_error,
                                "The inputs to syev/heev need to be a pointer to a matrix and a pointer to a vector. Got ranks {} and {}.",
                                A->rank(), W->rank());
    }

    if (A->dim(0) != A->dim(1)) {
        EINSUMS_THROW_EXCEPTION(dimension_error, "The input matrix to syev/heev needs to be square and symmetric.");
    }

    if (A->dim(0) != W->dim(0)) {
        EINSUMS_THROW_EXCEPTION(tensor_compat_error, "The input and output to syev/heev have incompatible dimensions.");
    }

    if (A->dim(0) == 0) {
        return;
    }

    size_t const n   = A->dim(0);
    size_t       lda = 0;

    blas::int_t lwork;

    BufferVector<AType> work;

    constexpr char jobz = (ComputeEigenvectors) ? 'v' : 'n';

    if constexpr (IsComplexV<AType>) {
        // Check if we can use LAPACK.
        if (A->is_gemmable(&lda)) {
            // Transpose A if necessary.
            if constexpr (ComputeEigenvectors) {
                if (A->is_row_major()) {
                    einsums::detail::impl_conj(*A);
                }
            }

            // Query buffer params.
            AType lwork_complex = AType{(RemoveComplexT<AType>)(2 * n - 1)};

            blas::int_t info;

            info = blas::heev(jobz, 'u', n, A->data(), lda, (RemoveComplexT<AType> *)nullptr, &lwork_complex, -1,
                              (RemoveComplexT<AType> *)nullptr);

            if (info < 0) {
                EINSUMS_THROW_EXCEPTION(
                    std::invalid_argument,
                    "One of the arguments passed to syev/heev was invalid! 1 (jobz): {}, 2 (uplo): 'u', 3 (n): {}, 5 (lda): {}, 8 (lwork): "
                    "-1. This is  the query call, so lwork should be -1.",
                    jobz, n, lda);
            } else if (info > 0) {
                EINSUMS_THROW_EXCEPTION(std::runtime_error, "The query call to syev/heev returned an unknown error!");
            }

            lwork = (blas::int_t)std::real(lwork_complex);

            work.resize(lwork);

            // Set up real work buffer.
            BufferVector<RemoveComplexT<AType>> rwork(std::max((ptrdiff_t)1, 3 * (ptrdiff_t)n - 2));

            // Check if we need to make a temporary buffer for the eigenvalues.
            if (W->get_incx() != 1) {

                // Make a temporary buffer for the eigenvalues, then copy after.
                BufferVector<RemoveComplexT<AType>> temp_vals(A->dim(0));
                info = blas::heev(jobz, 'u', n, A->data(), lda, temp_vals.data(), work.data(), lwork, rwork.data());

                if (info < 0) {
                    EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                            "One of the arguments passed to syev/heev was invalid! 1 (jobz): {}, 2 (uplo): 'u', 3 (n): {}, "
                                            "5 (lda): {}, 8 (lwork): "
                                            "{}.",
                                            jobz, n, lda, lwork);
                } else if (info > 0) {
                    EINSUMS_THROW_EXCEPTION(std::runtime_error, "The eigenvalue algorithm did not converge!");
                }

                for (int i = 0; i < W->dim(0); i++) {
                    W->subscript(i) = temp_vals[i];
                }
            } else {
                // No temporary buffer needed.
                info = blas::heev(jobz, 'u', n, A->data(), lda, W->data(), work.data(), lwork, rwork.data());

                if (info < 0) {
                    EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                            "One of the arguments passed to syev/heev was invalid! 1 (jobz): {}, 2 (uplo): 'u', 3 (n): {}, "
                                            "5 (lda): {}, 8 (lwork): "
                                            "{}.",
                                            jobz, n, lda, lwork);
                } else if (info > 0) {
                    EINSUMS_THROW_EXCEPTION(std::runtime_error, "The eigenvalue algorithm did not converge!");
                }
            }

            // We might need to transpose the eigenvectors.
            if (A->is_row_major() && ComputeEigenvectors) {
                for (size_t i = 0; i < A->dim(0); i++) {
                    for (size_t j = i + 1; j < A->dim(1); j++) {
                        AType temp                  = A->subscript_no_check(i, j);
                        A->subscript_no_check(i, j) = A->subscript_no_check(j, i);
                        A->subscript_no_check(j, i) = temp;
                    }
                }
            }
        } else {
            // We can't use LAPACK, so use our own version. Note that the eigenvectors may be off by some complex phase factor.
            // They are still eigenvectors, they just don't match when directly compared with the results of LAPACK.
            // Also, the sizes of the buffers are different.

            if (jobz == 'v' && std::is_same_v<std::complex<float>, AType>) {
                EINSUMS_LOG_WARN("Computing eigenvectors of single-precision matrices with non-unit smallest stride is not fully stable. "
                                 "It works, but there may be some deviation from LAPACK. The stability is likely to improve in the future. "
                                 "Consider copying the data into a freshly constructed Tensor<std::complex<float>,2> instead.");
            }
            lwork = impl_heev_get_work_length(jobz, A, W);

            work.resize(lwork);
            BufferVector<RemoveComplexT<AType>> rwork(std::max((ptrdiff_t)1, 2 * (ptrdiff_t)n - 1));

            impl_strided_heev(jobz, A, W, work.data(), rwork.data());
        }
    } else {
        // Check if we can use LAPACK.
        if (A->is_gemmable(&lda)) {
            // Query buffer params.
            AType       lwork_real = (AType)(3 * n - 2);
            blas::int_t info       = 0;

            info = blas::syev(jobz, 'u', n, A->data(), lda, (AType *)nullptr, &lwork_real, -1);

            if (info < 0) {
                EINSUMS_THROW_EXCEPTION(
                    std::invalid_argument,
                    "One of the arguments passed to syev/heev was invalid! 1 (jobz): {}, 2 (uplo): 'u', 3 (n): {}, 5 (lda): {}, 8 (lwork): "
                    "-1. This is  the query call, so lwork should be -1.",
                    jobz, n, lda);
            } else if (info > 0) {
                EINSUMS_THROW_EXCEPTION(std::runtime_error, "The query call to syev/heev returned an unknown error!");
            }

            lwork = (blas::int_t)lwork_real;

            work.resize(lwork);

            // Check
            if (W->get_incx() != 1) {
                BufferVector<RemoveComplexT<AType>> temp(A->dim(0));
                info = blas::syev(jobz, 'u', n, A->data(), lda, temp.data(), work.data(), lwork);

                if (info < 0) {
                    EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                            "One of the arguments passed to syev/heev was invalid! 1 (jobz): {}, 2 (uplo): 'u', 3 (n): {}, "
                                            "5 (lda): {}, 8 (lwork): "
                                            "{}.",
                                            jobz, n, lda, lwork);
                } else if (info > 0) {
                    EINSUMS_THROW_EXCEPTION(std::runtime_error, "The eigenvalue algorithm did not converge!");
                }

                for (int i = 0; i < W->dim(0); i++) {
                    W->subscript(i) = temp[i];
                }
            } else {
                info = blas::syev(jobz, 'u', n, A->data(), lda, W->data(), work.data(), lwork);

                if (info < 0) {
                    EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                            "One of the arguments passed to syev/heev was invalid! 1 (jobz): {}, 2 (uplo): 'u', 3 (n): {}, "
                                            "5 (lda): {}, 8 (lwork): "
                                            "{}.",
                                            jobz, n, lda, lwork);
                } else if (info > 0) {
                    EINSUMS_THROW_EXCEPTION(std::runtime_error, "The eigenvalue algorithm did not converge!");
                }
            }

            // We might need to transpose the eigenvectors.
            if (A->is_row_major()) {
                for (size_t i = 0; i < A->dim(0); i++) {
                    for (size_t j = i + 1; j < A->dim(1); j++) {
                        std::swap(A->subscript_no_check(i, j), A->subscript_no_check(j, i));
                    }
                }
            }
        } else {
            // We can't use LAPACK, so use our own version. Note that the eigenvectors may be off by some complex phase factor.
            // They are still eigenvectors, they just don't match when directly compared with the results of LAPACK.
            // Also, the sizes of the buffers are different.
            if (jobz == 'v' && std::is_same_v<float, AType>) {
                EINSUMS_LOG_WARN("Computing eigenvectors of single-precision matrices with non-unit smallest stride is not fully stable. "
                                 "It works, but there may be some deviation from LAPACK. The stability is likely to improve in the future. "
                                 "Consider copying the data into a freshly constructed Tensor<float,2> instead.");
            }
            lwork = impl_syev_get_work_length(jobz, A, W);

            work.resize(lwork);
            impl_strided_syev(jobz, A, W, work.data());
        }
    }
}

template <bool ComputeEigenvectors = true, typename AType>
void heev(einsums::detail::TensorImpl<AType> *A, einsums::detail::TensorImpl<AType> *W) {
    syev(A, W);
}

template <bool ComputeEigenvectors = true, CoreBasicTensorConcept AType, CoreBasicTensorConcept WType>
    requires requires {
        requires SameUnderlying<AType, WType>;
        requires RankTensorConcept<AType, 2>;
        requires RankTensorConcept<WType, 1>;
        requires NotComplex<AType>;
    }
void syev(AType *A, WType *W) {
    syev<ComputeEigenvectors>(&A->impl(), &W->impl());
}

template <typename T>
void geev(einsums::detail::TensorImpl<T> *A, einsums::detail::TensorImpl<AddComplexT<T>> *W,
          einsums::detail::TensorImpl<AddComplexT<T>> *lvecs, einsums::detail::TensorImpl<AddComplexT<T>> *rvecs) {

    char jobvl = 'n', jobvr = 'n';

#ifdef EINSUMS_DEBUG
    jobvl = (lvecs == nullptr) ? 'n' : 'v';
    jobvr = (rvecs == nullptr) ? 'n' : 'v';

    EINSUMS_LOG_TRACE("Performing geev with jobvl = {} and jobvr = {}.", jobvl, jobvr);

    jobvl = 'n';
    jobvr = 'n';
#endif

    if (A->rank() != 2) {
        EINSUMS_THROW_EXCEPTION(rank_error, "The input tensor needs to be a matrix!");
    }

    if (W->rank() != 1) {
        EINSUMS_THROW_EXCEPTION(rank_error, "The eigenvalue output needs to be a vector!");
    }

    if (A->dim(0) != A->dim(1)) {
        EINSUMS_THROW_EXCEPTION(dimension_error, "The input matrix needs to be square!");
    }

    if (A->dim(0) > W->dim(0)) {
        EINSUMS_THROW_EXCEPTION(tensor_compat_error, "The eigenvalue does not have enough space to hold the eigenvalues!");
    }

    if (lvecs != nullptr) {
        jobvl = 'v';

        if (lvecs->rank() != 2) {
            EINSUMS_THROW_EXCEPTION(rank_error, "The left eigenvector output needs to be a matrix or nullpointer!");
        }

        if (lvecs->dim(0) != lvecs->dim(1)) {
            EINSUMS_THROW_EXCEPTION(dimension_error, "The left eigenvector output needs to be a square matrix!");
        }

        if (lvecs->dim(0) != A->dim(0)) {
            EINSUMS_THROW_EXCEPTION(dimension_error, "The left eigenvectors needs to have the same shape as the input matrix.");
        }
    }

    if (rvecs != nullptr) {
        jobvr = 'v';

        if (rvecs->rank() != 2) {
            EINSUMS_THROW_EXCEPTION(rank_error, "The right eigenvector output needs to be a matrix or nullpointer!");
        }

        if (rvecs->dim(0) != rvecs->dim(1)) {
            EINSUMS_THROW_EXCEPTION(dimension_error, "The right eigenvector output needs to be a square matrix!");
        }

        if (rvecs->dim(0) != A->dim(0)) {
            EINSUMS_THROW_EXCEPTION(dimension_error, "The right eigenvectors needs to have the same shape as the input matrix.");
        }
    }

    Tensor<T, 2> A_temp{false, "Temporary", A->dim(0), A->dim(1)};
    einsums::detail::copy_to(*A, A_temp.impl());

    if constexpr (!IsComplexV<T>) {
        Tensor<T, 2> lvecs_temp, rvecs_temp;
        blas::int_t  ldvl = 1, ldvr = 1;

        if (jobvl == 'v') {
            lvecs_temp = Tensor<T, 2>{false, "Temporary lvecs", A->dim(0), A->dim(1)};
            ldvl       = lvecs_temp.impl().get_lda();
        }

        if (jobvr == 'v') {
            rvecs_temp = Tensor<T, 2>{false, "Temporary rvecs", A->dim(0), A->dim(1)};
            ldvr       = rvecs_temp.impl().get_lda();
        }

        auto status = blas::geev(jobvl, jobvr, A_temp.dim(0), A_temp.data(), A_temp.impl().get_lda(), W->data(), lvecs_temp.data(), ldvl,
                                 rvecs_temp.data(), ldvr);

        if (status < 0) {
            EINSUMS_THROW_EXCEPTION(
                std::invalid_argument,
                "The {} argument to geev was invalid! 1 (jobvl): {}, 2 (jobvr): {}, 3 (n): {}, 5 (lda): {}, 8 (ldvl): {}, 10 (ldvr): {}.",
                print::ordinal(-status), jobvl, jobvr, A_temp.dim(0), A_temp.impl().get_lda(), ldvl, ldvr);
        } else if (status > 0) {
            EINSUMS_THROW_EXCEPTION(std::runtime_error, "The eigenvalue algorithm did not converge!");
        }

        // Process the eigenvectors.
        if (jobvl == 'v') {
            for (int i = 0; i < A->dim(0); i++) {
                if (std::imag(W->subscript_no_check(i)) != T{0.0}) {
                    for (int j = 0; j < A->dim(0); j++) {
                        lvecs->subscript_no_check(j, i)     = AddComplexT<T>(lvecs_temp(j, i), lvecs_temp(j, i + 1));
                        lvecs->subscript_no_check(j, i + 1) = AddComplexT<T>(lvecs_temp(j, i), -lvecs_temp(j, i + 1));
                    }
                    i++;
                } else {
                    for (int j = 0; j < A->dim(0); j++) {
                        lvecs->subscript_no_check(j, i) = AddComplexT<T>(lvecs_temp(j, i));
                    }
                }
            }
        }

        if (jobvr == 'v') {
            for (int i = 0; i < A->dim(0); i++) {
                if (std::imag(W->subscript_no_check(i)) != T{0.0}) {
                    for (int j = 0; j < A->dim(0); j++) {
                        rvecs->subscript(j, i)     = AddComplexT<T>(rvecs_temp(j, i), rvecs_temp(j, i + 1));
                        rvecs->subscript(j, i + 1) = AddComplexT<T>(rvecs_temp(j, i), -rvecs_temp(j, i + 1));
                    }
                    i++;
                } else {
                    for (int j = 0; j < A->dim(0); j++) {
                        rvecs->subscript(j, i) = AddComplexT<T>(rvecs_temp(j, i));
                    }
                }
            }
        }
    } else {
        Tensor<T, 2> *lvecs_temp = nullptr, *rvecs_temp = nullptr;
        blas::int_t   ldvl = 1, ldvr = 1;

        T *lvecs_data = nullptr, *rvecs_data = nullptr;

        if (jobvl == 'v') {
            if (lvecs->is_gemmable()) {
                lvecs_data = lvecs->data();
                ldvl       = lvecs->get_lda();
            } else {
                lvecs_temp = new Tensor<T, 2>{false, "Temporary lvecs", A->dim(0), A->dim(1)};
                ldvl       = lvecs_temp->impl().get_lda();
                lvecs_data = lvecs_temp->data();
            }
        }

        if (jobvr == 'v') {
            if (rvecs->is_gemmable()) {
                rvecs_data = rvecs->data();
                ldvr       = rvecs->get_lda();
            } else {
                rvecs_temp = new Tensor<T, 2>{false, "Temporary rvecs", A->dim(0), A->dim(1)};
                ldvr       = rvecs_temp->impl().get_lda();
                rvecs_data = rvecs_temp->data();
            }
        }

        auto status =
            blas::geev(jobvl, jobvr, A->dim(0), A_temp.data(), A_temp.impl().get_lda(), W->data(), lvecs_data, ldvl, rvecs_data, ldvr);

        if (status < 0) {
            EINSUMS_THROW_EXCEPTION(
                std::invalid_argument,
                "The {} argument to geev was invalid! 1 (jobvl): {}, 2 (jobvr): {}, 3 (n): {}, 5 (lda): {}, 8 (ldvl): {}, 10 (ldvr): {}.",
                print::ordinal(-status), jobvl, jobvr, A_temp.dim(0), A_temp.impl().get_lda(), ldvl, ldvr);
        } else if (status > 0) {
            EINSUMS_THROW_EXCEPTION(std::runtime_error, "The eigenvalue algorithm did not converge!");
        }

        if (lvecs_temp != nullptr) {
            einsums::detail::copy_to(lvecs_temp->impl(), *lvecs);
            delete lvecs_temp;
        } else if (jobvl == 'v' && lvecs->is_row_major()) {
            for (size_t i = 0; i < A->dim(0); i++) {
                for (size_t j = 0; j < i; j++) {
                    std::swap(lvecs->subscript_no_check(i, j), lvecs->subscript_no_check(j, i));
                }
            }
        }

        if (rvecs_temp != nullptr) {
            einsums::detail::copy_to(rvecs_temp->impl(), *rvecs);
            delete rvecs_temp;
        } else if (jobvr == 'v' && rvecs->is_row_major()) {
            for (size_t i = 0; i < A->dim(0); i++) {
                for (size_t j = 0; j < i; j++) {
                    std::swap(rvecs->subscript_no_check(i, j), rvecs->subscript_no_check(j, i));
                }
            }
        }
    }
}

template <CoreBasicTensorConcept AType, VectorConcept WType, typename LVecPtr, typename RVecPtr>
    requires requires {
        requires InSamePlace<AType, WType>;
        requires RankTensorConcept<AType, 2>;
        requires RankTensorConcept<WType, 1>;
        requires std::is_same_v<typename WType::ValueType, AddComplexT<typename AType::ValueType>>;
        requires std::is_null_pointer_v<LVecPtr> ||
                     (MatrixConcept<std::remove_pointer_t<LVecPtr>> && CoreBasicTensorConcept<std::remove_pointer_t<LVecPtr>> &&
                      std::is_same_v<typename std::remove_pointer_t<LVecPtr>::ValueType, AddComplexT<typename AType::ValueType>>);
        requires std::is_null_pointer_v<RVecPtr> ||
                     (MatrixConcept<std::remove_pointer_t<RVecPtr>> && CoreBasicTensorConcept<std::remove_pointer_t<RVecPtr>> &&
                      std::is_same_v<typename std::remove_pointer_t<RVecPtr>::ValueType, AddComplexT<typename AType::ValueType>>);
    }
void geev(AType *A, WType *W, LVecPtr lvecs, RVecPtr rvecs) {
    einsums::detail::TensorImpl<AddComplexT<typename AType::ValueType>> *lvec_ptr = nullptr, *rvec_ptr = nullptr;

    if constexpr (!std::is_null_pointer_v<LVecPtr>) {
        if (lvecs != nullptr) {
            lvec_ptr = &lvecs->impl();
        }
    }

    if constexpr (!std::is_null_pointer_v<RVecPtr>) {
        if (rvecs != nullptr) {
            rvec_ptr = &rvecs->impl();
        }
    }

    geev(&A->impl(), &W->impl(), lvec_ptr, rvec_ptr);
}

template <bool ComputeEigenvectors = true, CoreBasicTensorConcept AType, CoreBasicTensorConcept WType>
    requires requires {
        requires NotComplex<WType>;
        requires std::is_same_v<typename AType::ValueType, AddComplexT<typename WType::ValueType>>;
        requires MatrixConcept<AType>;
        requires VectorConcept<WType>;
    }
void heev(AType *A, WType *W) {
    syev<ComputeEigenvectors>(&A->impl(), &W->impl());
}

template <typename T>
[[nodiscard]] auto gesv(einsums::detail::TensorImpl<T> *A, einsums::detail::TensorImpl<T> *B) -> int {
    if (A->rank() != 2) {
        EINSUMS_THROW_EXCEPTION(rank_error, "The coefficient matrix needs to be rank-2!");
    }
    if (B->rank() > 2) {
        EINSUMS_THROW_EXCEPTION(rank_error, "The output matrix needs to be rank 1 or 2!");
    }

    if (A->dim(0) != A->dim(1) || A->dim(0) != B->dim(0)) {
        EINSUMS_THROW_EXCEPTION(tensor_compat_error,
                                "The coefficient matrix needs to be square and the number of rows of the result matrix needs to match! A "
                                "dims: ({}, {}), B dim: {}.",
                                A->dim(0), A->dim(1), B->dim(0));
    }

    if (A->dim(0) == 0) {
        return 0;
    }

    if (A->dim(0) == 1) {
        if (B->rank() == 1) {
            B->subscript(0) /= A->subscript(0, 0);
        } else {
            for (int i = 0; i < B->dim(1); i++) {
                B->subscript(0, i) /= A->subscript(0, 0);
            }
        }
        return 0;
    }

    if (A->is_column_major() && B->is_column_major() && A->is_gemmable() && (B->rank() == 1 || B->is_gemmable())) {
        auto n   = A->dim(0);
        auto lda = A->get_lda();

        size_t nrhs, ldb;

        if (B->rank() == 1) {
            nrhs = 1;
            ldb  = n;
        } else {
            nrhs = B->dim(1);
            ldb  = B->get_ldb();
        }

        int                       lwork = n;
        BufferVector<blas::int_t> ipiv(lwork);

        int info = blas::gesv(n, nrhs, A->data(), lda, ipiv.data(), B->data(), ldb);

        if (info < 0) {
            EINSUMS_THROW_EXCEPTION(std::invalid_argument, "gesv: argument {} has an invalid value! n: {}, nrhs: {}, lda: {}, ldb: {}.",
                                    print::ordinal(-info), n, nrhs, lda, ldb);
        } else if (info > 0) {
            EINSUMS_THROW_EXCEPTION(std::runtime_error, "gesv: factor U({0},{0}) is exactly zero. The solution could not be computed.",
                                    info - 1);
        }

        return info;
    } else {
        BufferVector<blas::int_t> ipiv(A->dim(0));
        int                       info;
        if (B->rank() == 2 && B->dim(1) == 0) {
            info = impl_lu_decomp(*A, ipiv);
        } else {
            info = impl_solve(*A, *B, ipiv);
        }

        if (info < 0) {
            EINSUMS_THROW_EXCEPTION(std::invalid_argument, "gesv: argument {} has an invalid value!", print::ordinal(-info));
        } else if (info > 0) {
            EINSUMS_THROW_EXCEPTION(std::runtime_error, "gesv: factor U({0},{0}) is exactly zero. The solution could not be computed.",
                                    info - 1);
        }

        return info;
    }
}

template <CoreBasicTensorConcept AType, CoreBasicTensorConcept BType>
    requires requires {
        requires SameUnderlying<AType, BType>;
        requires MatrixConcept<AType>;
        requires MatrixConcept<BType> || VectorConcept<BType>;
    }
[[nodiscard]] auto gesv(AType *A, BType *B) -> int {
    return gesv(&A->impl(), &B->impl());
}

template <typename T>
void scale(T scale, einsums::detail::TensorImpl<T> *A) {
    einsums::detail::impl_scal(scale, *A);
}

template <CoreBasicTensorConcept AType>
void scale(typename AType::ValueType scale, AType *A) {
    detail::scale(scale, &A->impl());
}

template <typename T>
void scale_row(ptrdiff_t row, T scale, einsums::detail::TensorImpl<T> *A) {
    if (A->rank() != 2) {
        EINSUMS_THROW_EXCEPTION(rank_error, "The input to scale_row needs to be a rank-2 tensor!");
    }
    blas::scal(A->dim(1), scale, A->data(row, 0), A->stride(1));
}

template <CoreBasicTensorConcept AType>
    requires(MatrixConcept<AType>)
void scale_row(ptrdiff_t row, typename AType::ValueType scale, AType *A) {
    blas::scal(A->dim(1), scale, A->data(row, 0ul), A->stride(1));
}

template <typename T>
void scale_column(ptrdiff_t col, T scale, einsums::detail::TensorImpl<T> *A) {
    if (A->rank() != 2) {
        EINSUMS_THROW_EXCEPTION(rank_error, "The input to scale_column needs to be a rank-2 tensor!");
    }
    blas::scal(A->dim(0), scale, A->data(0, col), A->stride(0));
}

template <CoreBasicTensorConcept AType>
void scale_column(ptrdiff_t col, typename AType::ValueType scale, AType *A) {
    blas::scal(A->dim(0), scale, A->data(0, col), A->stride(0));
}

template <typename T, typename TOther>
BiggestTypeT<T, TOther> dot(einsums::detail::TensorImpl<T> const &A, einsums::detail::TensorImpl<TOther> const &B) {
    if (A.rank() != B.rank()) {
        EINSUMS_THROW_EXCEPTION(rank_error, "The ranks of the tensors passed to dot must be the same!");
    }

    if (A.dims() != B.dims()) {
        EINSUMS_THROW_EXCEPTION(dimension_error, "The dimensions of the tensors passed to dot must be the same!");
    }

    return impl_dot(A, B);
}

template <CoreBasicTensorConcept AType, CoreBasicTensorConcept BType>
    requires requires { requires SameRank<AType, BType>; }
[[nodiscard]] auto dot(AType const &A, BType const &B) -> BiggestTypeT<typename AType::ValueType, typename BType::ValueType> {
    return dot(A.impl(), B.impl());
}

template <typename T>
T true_dot(einsums::detail::TensorImpl<T> const &A, einsums::detail::TensorImpl<T> const &B) {
    if (A.rank() != B.rank()) {
        EINSUMS_THROW_EXCEPTION(rank_error, "The ranks of the tensors passed to true_dot must be the same!");
    }

    if (A.dims() != B.dims()) {
        EINSUMS_THROW_EXCEPTION(dimension_error, "The dimensions of the tensors passed to true_dot must be the same!");
    }

    return impl_true_dot(A, B);
}

template <CoreBasicTensorConcept AType, CoreBasicTensorConcept BType>
    requires requires { requires SameRank<AType, BType>; }
[[nodiscard]] auto true_dot(AType const &A, BType const &B) -> BiggestTypeT<typename AType::ValueType, typename BType::ValueType> {
    return true_dot(A.impl(), B.impl());
}

template <typename AType, typename BType, typename CType>
[[nodiscard]] auto dot(einsums::detail::TensorImpl<AType> const &A, einsums::detail::TensorImpl<BType> const &B,
                       einsums::detail::TensorImpl<CType> const &C) -> BiggestTypeT<AType, BType, CType> {
    return impl_dot(A, B, C);
}

template <CoreBasicTensorConcept AType, CoreBasicTensorConcept BType, CoreBasicTensorConcept CType>
    requires SameRank<AType, BType, CType>
[[nodiscard]] auto dot(AType const &A, BType const &B, CType const &C)
    -> BiggestTypeT<typename AType::ValueType, typename BType::ValueType, typename CType::ValueType> {

    return dot(A.impl(), B.impl(), C.impl());
}

template <typename T>
void axpy(T alpha, einsums::detail::TensorImpl<T> const &X, einsums::detail::TensorImpl<T> *Y) {
    einsums::detail::impl_axpy(alpha, X, *Y);
}

template <CoreBasicTensorConcept XType, CoreBasicTensorConcept YType>
    requires SameUnderlyingAndRank<XType, YType>
void axpy(typename XType::ValueType alpha, XType const &X, YType *Y) {
    axpy(alpha, X.impl(), &Y->impl());
}

template <typename T>
void axpby(T alpha, einsums::detail::TensorImpl<T> const &X, T beta, einsums::detail::TensorImpl<T> *Y) {
    einsums::detail::impl_scal(beta, *Y);
    einsums::detail::impl_axpy(alpha, X, *Y);
}

template <CoreBasicTensorConcept XType, CoreBasicTensorConcept YType>
    requires SameUnderlyingAndRank<XType, YType>
void axpby(typename XType::ValueType alpha, XType const &X, typename YType::ValueType beta, YType *Y) {
    axpby(alpha, X.impl(), beta, &Y->impl());
}

template <typename T>
void ger(T alpha, einsums::detail::TensorImpl<T> const &X, einsums::detail::TensorImpl<T> const &Y, einsums::detail::TensorImpl<T> *A) {
    impl_ger(alpha, X, Y, *A);
}

template <CoreBasicTensorConcept AType, CoreBasicTensorConcept XYType>
    requires requires {
        requires MatrixConcept<AType>;
        requires VectorConcept<XYType>;
        requires SameUnderlying<AType, XYType>;
    }
void ger(typename XYType::ValueType alpha, XYType const &X, XYType const &Y, AType *A) {
    ger(alpha, X.impl(), Y.impl(), &A->impl());
}

template <typename T>
void gerc(T alpha, einsums::detail::TensorImpl<T> const &X, einsums::detail::TensorImpl<T> const &Y, einsums::detail::TensorImpl<T> *A) {
    impl_gerc(alpha, X, Y, *A);
}

template <CoreBasicTensorConcept AType, CoreBasicTensorConcept XYType>
    requires requires {
        requires MatrixConcept<AType>;
        requires VectorConcept<XYType>;
        requires SameUnderlying<AType, XYType>;
    }
void gerc(typename XYType::ValueType alpha, XYType const &X, XYType const &Y, AType *A) {
    gerc(alpha, X.impl(), Y.impl(), &A->impl());
}

template <bool TransA, bool TransB, CoreBasicTensorConcept AType, CoreBasicTensorConcept BType, CoreBasicTensorConcept CType>
    requires requires {
        requires SameUnderlyingAndRank<AType, BType, CType>;
        requires MatrixConcept<AType>;
    }
void symm_gemm(AType const &A, BType const &B, CType *C) {
    int  temp_rows, temp_cols;
    bool shape_test = true;
    if constexpr (TransA && TransB) {
        shape_test = B.dim(0) == A.dim(0) && A.dim(1) == B.dim(0) && C->dim(0) == B.dim(1) && C->dim(1) == B.dim(1);
    } else if constexpr (TransA && !TransB) {
        shape_test = B.dim(1) == A.dim(0) && A.dim(1) == B.dim(1) && C->dim(0) == B.dim(0) && C->dim(1) == B.dim(0);
    } else if constexpr (!TransA && TransB) {
        shape_test = B.dim(0) == A.dim(1) && A.dim(0) == B.dim(0) && C->dim(0) == B.dim(1) && C->dim(1) == B.dim(1);
    } else {
        shape_test = B.dim(1) == A.dim(1) && A.dim(0) == B.dim(1) && C->dim(0) == B.dim(0) && C->dim(1) == B.dim(0);
    }

    if (!shape_test) {
        EINSUMS_THROW_EXCEPTION(tensor_compat_error, "The shapes of the input and output tensors are incompatible!");
    }

    if constexpr (TransA) {
        temp_rows = A.dim(1);
    } else {
        temp_rows = A.dim(0);
    }

    if constexpr (TransB) {
        temp_cols = B.dim(0);
    } else {
        temp_cols = B.dim(1);
    }

    *C = typename CType::ValueType(0.0);

    Tensor<typename AType::ValueType, 2> temp{"temp", temp_rows, temp_cols};

    gemm<TransA, TransB>(typename AType::ValueType{1.0}, A, B, typename CType::ValueType{0.0}, &temp);
    gemm<!TransB, false>(typename AType::ValueType{1.0}, B, temp, typename CType::ValueType{0.0}, C);
}

template <typename AType, typename BType, typename CType>
void direct_product(CType alpha, einsums::detail::TensorImpl<AType> const &A, einsums::detail::TensorImpl<BType> const &B, CType beta,
                    einsums::detail::TensorImpl<CType> *C) {
    impl_direct_product(alpha, A, B, beta, C);
}

template <CoreBasicTensorConcept AType, CoreBasicTensorConcept BType, CoreBasicTensorConcept CType>
    requires SameUnderlyingAndRank<AType, BType, CType>
void direct_product(typename AType::ValueType alpha, AType const &A, BType const &B, typename CType::ValueType beta, CType *C) {
    direct_product(alpha, A.impl(), B.impl(), beta, &C->impl());
}

template <typename AType, typename BType, typename CType>
void direct_division(CType alpha, einsums::detail::TensorImpl<AType> const &A, einsums::detail::TensorImpl<BType> const &B, CType beta,
                     einsums::detail::TensorImpl<CType> *C) {
    impl_direct_division(alpha, A, B, beta, C);
}

template <CoreBasicTensorConcept AType, CoreBasicTensorConcept BType, CoreBasicTensorConcept CType>
    requires SameUnderlyingAndRank<AType, BType, CType>
void direct_division(typename AType::ValueType alpha, AType const &A, BType const &B, typename CType::ValueType beta, CType *C) {
    direct_division(alpha, A.impl(), B.impl(), beta, &C->impl());
}

template <CoreBasicTensorConcept AType>
    requires MatrixConcept<AType>
[[nodiscard]] auto pow(AType const &a, typename AType::ValueType alpha,
                       typename AType::ValueType cutoff = std::numeric_limits<typename AType::ValueType>::epsilon())
    -> Tensor<typename AType::ValueType, 2> {
    EINSUMS_ASSERT(a.dim(0) == a.dim(1));

    if (a.dim(0) == 0) {
        return Tensor<typename AType::ValueType, 2>{"pow result", 0, 0};
    }

    if (a.dim(0) == 1) {
        Tensor<typename AType::ValueType, 2> out{"pow result", 1, 1};

        out(0, 0) = std::pow(a(0, 0), alpha);
        return out;
    }

    using T = typename AType::ValueType;

    size_t                               n      = a.dim(0);
    Tensor<typename AType::ValueType, 2> a1     = a;
    Tensor<typename AType::ValueType, 2> result = create_tensor_like(a);
    result.set_name("pow result");
    Tensor<RemoveComplexT<T>, 1> e{"e", n};
    result.zero();

    // Diagonalize
    if constexpr (IsComplexTensor<AType>) {
        heev<true>(&a1, &e);
    } else {
        syev<true>(&a1, &e);
    }

    Tensor<typename AType::ValueType, 2> a2(a1);

    // Determine the largest magnitude of the eigenvalues to use as a scaling factor for the cutoff.

    T max_e{0.0};
    // Block tensors don't have sorted eigenvalues, so we can't make assumptions about ordering.
    for (int i = 0; i < n; i++) {
        if (std::fabs(e(i)) > max_e) {
            max_e = std::fabs(e(i));
        }
    }

    for (size_t i = 0; i < n; i++) {
        if (alpha < 0.0 && std::fabs(e(i)) < cutoff * max_e) {
            e(i) = 0.0;
        } else {
            e(i) = std::pow(e(i), alpha);
            if (!std::isfinite(e(i))) {
                e(i) = 0.0;
            }
        }

        scale_column(i, e(i), &a2);
    }

    gemm('n', 'c', 1.0, a2, a1, 0.0, &result);

    return result;
}

template <typename T, typename Pivots, bool is_resizable = requires(Pivots c, typename Pivots::size_type size) { c.resize(size); }>
    requires requires(Pivots a, size_t ind) {
        typename Pivots::value_type;
        typename Pivots::size_type;

        { a.size() } -> std::same_as<typename Pivots::size_type>;
        { a.data() } -> std::same_as<typename Pivots::value_type *>;
        a[ind];
        requires std::same_as<blas::int_t, typename Pivots::value_type>;
    }
[[nodiscard]] auto getrf(einsums::detail::TensorImpl<T> *A, Pivots *pivot) -> int {
    LabeledSection0();

    if (A->rank() != 2) {
        EINSUMS_THROW_EXCEPTION(rank_error, "Can only decompose rank-2 tensors!");
    }

    if (A->dim(0) == 0 || A->dim(1) == 0) {
        return 0;
    }

    if constexpr (is_resizable) {
        if (pivot->size() < std::min(A->dim(0), A->dim(1))) {
            // println("getrf: resizing pivot vector from {} to {}", pivot->size(), std::min(A->dim(0), A->dim(1)));
            pivot->resize(std::min(A->dim(0), A->dim(1)));
        }
    } else {
        if (pivot->size() < std::min(A->dim(0), A->dim(1))) {
            EINSUMS_THROW_EXCEPTION(std::length_error, "The length of the pivot array is too small and can not be resized!");
        }
    }

    int result;

    if (A->is_gemmable() && A->is_column_major()) {
        result = blas::getrf(A->dim(0), A->dim(1), A->data(), A->get_lda(), pivot->data());
    } else {
        result = impl_lu_decomp(*A, *pivot);
    }

    if (result < 0) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "getrf: {} argument has an invalid value!, m: {}, n: {}, lda: {}.",
                                print::ordinal(-result), A->dim(0), A->dim(1), A->get_lda());
    } else if (result > 0) {
        EINSUMS_LOG_INFO("The matrix passed into the LU decomposition routine was singular. The decomposition was completed, but the "
                         "result should not be used to solve equations or to find the inverse of the matrix.");
    }

    return result;
}

/**
 * @brief Computes the LU factorization of a general m-by-n matrix.
 *
 * The routine computes the LU factorization of a general m-by-n matrix A as
 * \f[
 * A = P*L*U
 * \f]
 * where P is a permutation matrix, L is lower triangular with unit diagonal elements and U is upper triangular. The routine uses
 * partial pivoting, with row interchanges.
 *
 * @tparam TensorType
 * @param A
 * @param pivot
 * @return
 */
template <MatrixConcept TensorType, typename Pivots>
    requires requires(Pivots a, size_t ind) {
        typename Pivots::value_type;
        typename Pivots::size_type;

        { a.size() } -> std::same_as<typename Pivots::size_type>;
        { a.data() } -> std::same_as<typename Pivots::value_type *>;
        a[ind];
        requires std::same_as<blas::int_t, typename Pivots::value_type>;
        requires(CoreTensorConcept<TensorType>);
    }
[[nodiscard]] auto getrf(TensorType *A, Pivots *pivot) -> int {
    return getrf(&A->impl(), pivot);
}

/**
 * @brief Computes the inverse of a matrix using the LU factorization computed by getrf.
 *
 * The routine computes the inverse \f$inv(A)\f$ of a general matrix \f$A\f$. Before calling this routine, call getrf to factorize
 * \f$A\f$.
 *
 * @tparam TensorType The type of the tensor
 * @param A The matrix to invert
 * @param pivot The pivot vector from getrf
 * @return int If 0, the execution is successful.
 */
template <typename T, typename Pivots>
    requires requires(Pivots a, size_t ind) {
        typename Pivots::value_type;
        typename Pivots::size_type;

        { a.size() } -> std::same_as<typename Pivots::size_type>;
        { a.data() } -> std::same_as<typename Pivots::value_type *>;
        a[ind];
        requires std::same_as<blas::int_t, typename Pivots::value_type>;
    }
void getri(einsums::detail::TensorImpl<T> *A, Pivots const &pivot) {
    LabeledSection0();

    if (A->rank() != 2) {
        EINSUMS_THROW_EXCEPTION(rank_error, "Can only compute the inverses of matrices.");
    }

    if (A->dim(0) != A->dim(1)) {
        EINSUMS_THROW_EXCEPTION(dimension_error, "Can only compute the inverses of square matrices.");
    }

    int result;

    if (A->is_gemmable() && A->is_column_major()) {
        result = blas::getri(A->dim(0), A->data(), A->get_lda(), pivot.data());
    } else {
        BufferVector<T> work(A->dim(0));
        result = impl_invert_lu(*A, pivot, work.data());
    }

    if (result < 0) {
        EINSUMS_THROW_EXCEPTION(std::invalid_argument, "getri: {} argument has an invalid value! n: {}, lda: {}.", print::ordinal(-result),
                                A->dim(0), A->get_lda());
    } else if (result > 0) {
        EINSUMS_THROW_EXCEPTION(std::runtime_error, "The matrix passed into the inversion function is singular! An inverse could not be "
                                                    "computed. This means that the return value from getrf was ignored.");
    }
}

/**
 * @brief Computes the inverse of a matrix using the LU factorization computed by getrf.
 *
 * The routine computes the inverse \f$inv(A)\f$ of a general matrix \f$A\f$. Before calling this routine, call getrf to factorize
 * \f$A\f$.
 *
 * @tparam TensorType The type of the tensor
 * @param A The matrix to invert
 * @param pivot The pivot vector from getrf
 * @return int If 0, the execution is successful.
 */
template <MatrixConcept TensorType, typename Pivots>
    requires requires(Pivots a, size_t ind) {
        typename Pivots::value_type;
        typename Pivots::size_type;

        { a.size() } -> std::same_as<typename Pivots::size_type>;
        { a.data() } -> std::same_as<typename Pivots::value_type *>;
        a[ind];
        requires std::same_as<blas::int_t, typename Pivots::value_type>;
        requires(CoreTensorConcept<TensorType>);
    }
void getri(TensorType *A, Pivots const &pivot) {
    getri(&A->impl(), pivot);
}

template <typename T>
void invert(einsums::detail::TensorImpl<T> *A) {
    LabeledSection0();

    if (A->rank() != 2) {
        EINSUMS_THROW_EXCEPTION(rank_error, "Can only compute the inverses of matrices.");
    }

    if (A->dim(0) != A->dim(1)) {
        EINSUMS_THROW_EXCEPTION(dimension_error, "Can only compute the inverses of square matrices.");
    }

    BufferVector<blas::int_t> pivot(A->dim(0));
    int                       result;

    if (A->is_gemmable()) {
        result = blas::getrf(A->dim(0), A->dim(1), A->data(), A->get_lda(), pivot.data());
    } else {
        result = impl_lu_decomp(*A, pivot);
    }

    if (result < 0) {
        EINSUMS_LOG_WARN("getrf: argument {} has an invalid value", -result);
    }

    if (result > 0) {
        EINSUMS_THROW_EXCEPTION(std::runtime_error,
                                "invert: getrf: the ({}, {}) element of the factor U or L is zero, and the inverse could not be computed",
                                result - 1, result - 1);
    }

    if (A->is_gemmable()) {
        result = blas::getri(A->dim(0), A->data(), A->get_lda(), pivot.data());
    } else {
        BufferVector<T> work(A->dim(0));
        result = impl_invert_lu(*A, pivot, work.data());
    }

    if (result < 0) {
        EINSUMS_LOG_WARN("getri: argument {} has an invalid value", -result);
    }

    if (result > 0) {
        EINSUMS_THROW_EXCEPTION(std::runtime_error,
                                "invert: getri: the ({}, {}) element of the factor U or L i zero, and the inverse could not be computed",
                                result - 1, result - 1);
    }
}

/**
 * @brief Inverts a matrix.
 *
 * Utilizes the LAPACK routines getrf and getri to invert a matrix.
 *
 * @tparam TensorType The type of the tensor
 * @param A Matrix to invert. On exit, the inverse of A.
 */
template <MatrixConcept TensorType>
    requires(CoreTensorConcept<TensorType>)
void invert(TensorType *A) {
    invert(&A->impl());
}

template <typename T>
[[nodiscard]] auto svd(einsums::detail::TensorImpl<T> const &_A, char jobu, char jobvt)
    -> std::tuple<std::optional<Tensor<T, 2>>, Tensor<RemoveComplexT<T>, 1>, std::optional<Tensor<T, 2>>> {
    LabeledSection0();

    using option = std::optional<Tensor<T, 2>>;

    if (_A.rank() != 2) {
        EINSUMS_THROW_EXCEPTION(rank_error, "Can only decompose matrices!");
    }

    // Calling svd will destroy the original data. Make a copy of it.
    Tensor<T, 2> A{false, "A temp", _A.dim(0), _A.dim(1)};
    einsums::detail::copy_to(_A, A.impl());

    size_t m   = A.dim(0);
    size_t n   = A.dim(1);
    size_t lda = A.impl().get_lda();

    Tensor<T, 2> U, Vt;
    T           *U_data = nullptr, *Vt_data = nullptr;
    size_t       ldu = 1, ldvt = 1;

    // Test if it is absolutely necessary to zero out these tensors first.
    if (std::tolower(jobu) != 'n') {
        U = create_tensor<T>(false, "U (stored columnwise)", m, m);
        U.zero();
        U_data = U.data();
        ldu    = U.impl().get_lda();
    }
    auto S = create_tensor<RemoveComplexT<T>>("S", std::min(m, n));
    S.zero();
    if (std::tolower(jobvt) != 'n') {
        Vt = create_tensor<T>(false, "Vt (stored rowwise)", n, n);
        Vt.zero();
        Vt_data = Vt.data();
        ldvt    = Vt.impl().get_lda();
    }
    auto superb = create_tensor<T>(false, "superb", std::min(m, n));
    superb.zero();

    //    int info{0};
    int info = blas::gesvd(jobu, jobvt, m, n, A.data(), lda, S.data(), U_data, ldu, Vt_data, ldvt, superb.data());

    if (info < 0) {
        EINSUMS_THROW_EXCEPTION(
            std::invalid_argument,
            "svd: Argument {} has an invalid value.\n#3 (m) = {}, #4 (n) = {}, #6 (lda) = {}, #9 (ldu) = {}, #11 (ldvt) = {}", -info, m, n,
            lda, ldu, ldvt);
    } else if (info > 0) {
        EINSUMS_THROW_EXCEPTION(std::runtime_error, "svd: Algorithm did not converge!", info);
    }

    if (std::tolower(jobu) != 'n' && U.is_row_major()) {
        U.impl() = U.impl().transpose_view();
        for (size_t i = 0; i < m; i++) {
            for (size_t j = 0; j < i; j++) {
                std::swap(U(i, j), U(j, i));
            }
        }
    }

    if (std::tolower(jobvt) != 'n' && Vt.is_row_major()) {
        Vt.impl() = Vt.impl().transpose_view();
        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < i; j++) {
                std::swap(Vt(i, j), Vt(j, i));
            }
        }
    }

    return std::make_tuple((std::tolower(jobu) != 'n') ? option(std::move(U)) : option(), S,
                           (std::tolower(jobvt) != 'n') ? option(std::move(Vt)) : option());
}

template <MatrixConcept AType>
    requires(CoreTensorConcept<AType>)
[[nodiscard]] auto svd(AType const &A, char jobu, char jobvt)
    -> std::tuple<std::optional<Tensor<typename AType::ValueType, 2>>, Tensor<RemoveComplexT<typename AType::ValueType>, 1>,
                  std::optional<Tensor<typename AType::ValueType, 2>>> {
    return svd(A.impl(), jobu, jobvt);
}

template <typename T>
[[nodiscard]] auto norm(char norm_type, einsums::detail::TensorImpl<T> const &a) -> RemoveComplexT<T> {
    LabeledSection0();

    if (a.rank() > 2) {
        EINSUMS_THROW_EXCEPTION(rank_error, "A norm can only be taken on a matrix or vector!");
    }

    if constexpr (blas::IsBlasableV<T>) {
        switch (norm_type) {
        case 'm':
        case 'M':
            if (a.rank() == 1) {
                return impl_max_abs_norm(a);
            } else if (a.is_gemmable()) {
                return impl_max_abs_norm_gemmable(a);
            } else {
                return impl_max_abs_norm(a);
            }
        case '1':
        case 'o':
        case 'O':
            if (a.rank() == 1) {
                return blas::sum1(a.dim(0), a.data(), a.get_incx());
            } else if (a.is_gemmable()) {
                return impl_one_norm_gemmable(a);
            } else {
                return impl_one_norm(a);
            }
        case 'i':
        case 'I':
            if (a.rank() == 1) {
                return impl_max_abs_norm(a);
            } else if (a.is_gemmable()) {
                return impl_infinity_norm_gemmable(a);
            } else {
                return impl_infinity_norm(a);
            }
        case 'e':
        case 'f':
        case 'E':
        case 'F':
            if (a.rank() == 1) {
                return blas::nrm2(a.dim(0), a.data(), a.get_incx());
            } else if (a.is_gemmable()) {
                return impl_frobenius_norm_gemmable(a);
            } else {
                return impl_frobenius_norm(a);
            }
        case '2':
            if (a.rank() == 1) {
                return blas::nrm2(a.dim(0), a.data(), a.get_incx());
            } else {
                auto result = svd(a, 'n', 'n');
                return std::get<1>(result)(0);
            }
        default:
            EINSUMS_THROW_EXCEPTION(enum_error, "The norm type passed to norm is not valid!");
        }
    } else {
        switch (norm_type) {
        case 'm':
        case 'M':
            return impl_max_abs_norm(a);
        case '1':
        case 'o':
        case 'O':
            return impl_one_norm(a);
        case 'i':
        case 'I':
            return impl_infinity_norm(a);
        case 'e':
        case 'f':
        case 'E':
        case 'F':
            return impl_frobenius_norm(a);
        case '2':
            if (a.rank() == 1) {
                return impl_frobenius_norm(a);
            } else {
                EINSUMS_THROW_EXCEPTION(not_implemented, "We haven't implemented the spectral norm for matrices that don't have a LAPACK-"
                                                         "compatible type. If you need this, reach out to us so we can implement it.");
            }
        default:
            EINSUMS_THROW_EXCEPTION(enum_error, "The norm type passed to norm is not valid!");
        }
    }
}

template <MatrixConcept AType>
    requires(CoreTensorConcept<AType>)
[[nodiscard]] auto norm(char norm_type, AType const &a) -> RemoveComplexT<typename AType::ValueType> {
    return norm(norm_type, a.impl());
}

template <typename T>
[[nodiscard]] auto vec_norm(einsums::detail::TensorImpl<T> const &a) -> RemoveComplexT<T> {
    RemoveComplexT<T> norm = 0.0, scale = 1.0;

    sum_square(a, &scale, &norm);

    return std::sqrt(norm) * scale;
}

template <TensorConcept AType>
[[nodiscard]] auto vec_norm(AType const &a) -> RemoveComplexT<typename AType::ValueType> {
    return vec_norm(a.impl());
}

template <typename T>
[[nodiscard]] auto svd_dd(einsums::detail::TensorImpl<T> const &_A, char job)
    -> std::tuple<std::optional<Tensor<T, 2>>, Tensor<RemoveComplexT<T>, 1>, std::optional<Tensor<T, 2>>> {
    LabeledSection0();

    using option = std::optional<Tensor<T, 2>>;

    //    DisableOMPThreads const nothreads;

    if (_A.rank() != 2) {
        EINSUMS_THROW_EXCEPTION(rank_error, "The input tensor to svd_dd needs to be rank-2!");
    }

    // Calling svd will destroy the original data. Make a copy of it.
    Tensor<T, 2> A{false, "A temp", _A.dim(0), _A.dim(1)};
    einsums::detail::copy_to(_A, A.impl());

    size_t m = A.dim(0);
    size_t n = A.dim(1);

    Tensor<T, 2> U, Vt;
    T           *U_data = nullptr, *Vt_data = nullptr;
    size_t       ldu = 1, ldvt = 1;

    // Test if it absolutely necessary to zero out these tensors first.
    if (std::tolower(job) != 'n') {
        U = create_tensor<T>(false, "U (stored columnwise)", m, m);
        zero(U);
        Vt = create_tensor<T>(false, "Vt (stored rowwise)", n, n);
        zero(Vt);
        U_data  = U.data();
        Vt_data = Vt.data();
        ldu     = U.impl().get_lda();
        ldvt    = Vt.impl().get_lda();
    }

    auto S = create_tensor<RemoveComplexT<T>>("S", std::min(m, n));
    zero(S);

    int info = blas::gesdd(static_cast<char>(job), static_cast<int>(m), static_cast<int>(n), A.data(), A.impl().get_lda(), S.data(), U_data,
                           ldu, Vt_data, ldvt);

    if (info != 0) {
        if (info < 0) {
            EINSUMS_THROW_EXCEPTION(
                std::invalid_argument,
                "svd_dd: Argument {} has an invalid value.\n#2 (m) = {}, #3 (n) = {}, #5 (lda) = {}, #8 (ldu) = {}, #10 (ldvt) = {}", -info,
                m, n, A.impl().get_lda(), ldu, ldvt);
        } else {
            EINSUMS_THROW_EXCEPTION(std::runtime_error, "svd_dd: error value {}", info);
        }
    }

    if (std::tolower(job) != 'n' && Vt.is_row_major()) {
        U.impl() = U.impl().transpose_view();
        for (size_t i = 0; i < m; i++) {
            for (size_t j = 0; j < i; j++) {
                std::swap(U(i, j), U(j, i));
            }
        }
    }
    if (std::tolower(job) != 'n' && Vt.is_row_major()) {
        Vt.impl() = Vt.impl().transpose_view();
        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < i; j++) {
                std::swap(Vt(i, j), Vt(j, i));
            }
        }
    }

    return std::make_tuple((std::tolower(job) != 'n') ? option(std::move(U)) : option(), S,
                           (std::tolower(job) != 'n') ? option(std::move(Vt)) : option());
}

template <MatrixConcept AType>
    requires(CoreTensorConcept<AType>)
[[nodiscard]] auto svd_dd(AType const &A, char job)
    -> std::tuple<std::optional<Tensor<typename AType::ValueType, 2>>, Tensor<RemoveComplexT<typename AType::ValueType>, 1>,
                  std::optional<Tensor<typename AType::ValueType, 2>>> {
    return svd_dd(A.impl(), job);
}

template <typename T>
auto qr(einsums::detail::TensorImpl<T> const &_A) -> std::tuple<Tensor<T, 2>, Tensor<T, 2>> {
    LabeledSection0();

    // Copy A because it will be overwritten by the QR call.
    Tensor<T, 2>      A = _A;
    blas::int_t const m = A.dim(0);
    blas::int_t const n = A.dim(1);

    blas::int_t info;

    Tensor<T, 1> tau("tau", std::min(m, n));
    // Compute QR factorization of Y
    if (A.is_row_major()) {
        info = blas::gelqf(n, m, A.data(), A.impl().get_lda(), tau.data());
        if (info != 0) {
            EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                    "{} parameter to gelqf has an illegal value. #1: (m) {}, #2: (n) {}, #4: (lda) {}.",
                                    print::ordinal(-info), n, m, A.impl().get_lda());
        }
    } else {
        info = blas::geqrf(m, n, A.data(), A.impl().get_lda(), tau.data());
        if (info != 0) {
            EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                    "{} parameter to geqrf has an illegal value. #1: (m) {}, #2: (n) {}, #4: (lda) {}.",
                                    print::ordinal(-info), m, n, A.impl().get_lda());
        }
    }

    Tensor<T, 2> Q{"Q", m, m};
    Q.zero();

    // Extract the elementary reflectors from A.
    for (size_t i = 0; i < tau.dim(0); i++) {
        Q(i, i) = T{1.0};
        for (size_t j = i + 1; j < m; j++) {
            Q(j, i) = A(j, i);
            A(j, i) = T{0.0};
        }
    }

    // Extract Matrix Q out of QR factorization
    if (Q.is_row_major()) {
        if constexpr (IsComplexV<T>) {
            info = blas::unglq(m, m, tau.dim(0), Q.data(), Q.impl().get_lda(), tau.data());
        } else {
            info = blas::orglq(m, m, tau.dim(0), Q.data(), Q.impl().get_lda(), tau.data());
        }

        if (info != 0) {
            EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                    "{} parameter to {{or,un}}gqr was invalid! #1: (m) {}, #2: (n) {}, #3: (k) {}, #5: (lda) {}.",
                                    print::ordinal(-info), m, m, tau.dim(0), Q.impl().get_lda());
        }
    } else {
        if constexpr (IsComplexV<T>) {
            info = blas::ungqr(m, m, tau.dim(0), Q.data(), Q.impl().get_lda(), tau.data());
        } else {
            info = blas::orgqr(m, m, tau.dim(0), Q.data(), Q.impl().get_lda(), tau.data());
        }

        if (info != 0) {
            EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                    "{} parameter to {{or,un}}gqr was invalid! #1: (m) {}, #2: (n) {}, #3: (k) {}, #5: (lda) {}.",
                                    print::ordinal(-info), m, m, tau.dim(0), Q.impl().get_lda());
        }
    }

    return {Q, A};
}

template <MatrixConcept AType>
    requires(CoreTensorConcept<AType>)
auto qr(AType const &A) -> std::tuple<Tensor<typename AType::ValueType, 2>, Tensor<typename AType::ValueType, 2>> {
    return qr(A.impl());
}

} // namespace einsums::linear_algebra::detail
