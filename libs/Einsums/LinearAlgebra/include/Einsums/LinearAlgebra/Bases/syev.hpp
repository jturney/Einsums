//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once
#include <Einsums/BLAS.hpp>
#include <Einsums/Concepts/Complex.hpp>
#include <Einsums/Errors/ThrowException.hpp>
#include <Einsums/Print.hpp>
#include <Einsums/TensorImpl/TensorImpl.hpp>

#include <fmt/ranges.h>

#include <stdexcept>

namespace einsums {
namespace linear_algebra {
namespace detail {

template <NotComplex T>
void impl_hessenberg_reduce(einsums::detail::TensorImpl<T> *A, T *vec1, T *vec2, T *tau, bool keep_tau) {
    size_t const dim = A->dim(0), row_stride = A->stride(0), col_stride = A->stride(1);

    T *A_data = A->data();

    for (size_t k = 0; k < dim - 2; k++) {
        T alpha_sq = 0, scale = 0, alpha = 0;

        T key = A->subscript(k + 1, k);

        // Calculate alpha squared and the sign of alpha. The sign is -1 for non-negative values (and zero) and 1 for negative values of the
        // key.
        if (key < 0) {
            alpha = 1;
        } else {
            alpha = -1;
        }
        blas::lassq(dim - k - 1, A->data(k + 1, k), row_stride, &scale, &alpha_sq);

        alpha *= std::sqrt(alpha_sq) * scale;

        // Calculate r.
        T r = std::sqrt((alpha_sq * scale * scale - key * alpha) / 2);

        // Now, we need to calculate the values of the vectors.

        for (size_t i = 0; i < k; i++) {
            vec1[i] = T{0.0};
        }

        vec1[k + 1] = (A_data[(k + 1) * row_stride + k * col_stride] - alpha) / (2 * r);

        for (size_t i = k + 2; i < dim; i++) {
            vec1[i] = A_data[i * row_stride + k * col_stride] / (2 * r);
        }

        // Now, we need to calculate Av.
        EINSUMS_OMP_PARALLEL_FOR_SIMD
        for (size_t i = 0; i < dim; i++) {
            vec2[i] = T{0.0};
        }

        // The first k + 1 elements of vec1 are all zero. This also means we can ignore the first k rows of A.
        EINSUMS_OMP_PRAGMA(parallel for collapse(2))
        for (size_t i = 0; i < dim; i++) {
            for (size_t j = k + 1; j < dim; j++) {
                vec2[i] = std::fma(A_data[i * row_stride + j * col_stride], vec1[j], vec2[i]);
            }
        }

        // Scale by 2.
        blas::scal(dim, T{2.0}, vec2, 1);

        // Next, update A to be A - 2Avv^T.
        EINSUMS_OMP_PRAGMA(parallel for collapse(2))
        for (size_t i = 0; i < dim; i++) {
            for (size_t j = k + 1; j < dim; j++) {
                A_data[i * row_stride + j * col_stride] = std::fma(-vec2[i], vec1[j], A_data[i * row_stride + j * col_stride]);
            }
        }

        // Now, do it in reverse. Calculate v^T(A - Avv^T)
        EINSUMS_OMP_PARALLEL_FOR_SIMD
        for (size_t i = 0; i < dim; i++) {
            vec2[i] = T{0.0};
        }

        // In this case, the first k columns of A - Avv^T should be zero when vec1 is non-zero.
        // This means that the first k entries of vec2 will be zero.
        EINSUMS_OMP_PRAGMA(parallel for collapse(2))
        for (size_t j = k + 1; j < dim; j++) {
            for (size_t i = k; i < dim; i++) {
                vec2[i] = std::fma(A_data[j * row_stride + i * col_stride], vec1[j], vec2[i]);
            }
        }

        // Scale by 2.
        blas::scal(dim, T{2.0}, vec2, 1);

        // Finally, update A to be (A - 2Avv^T) - 2vv^T(A - 2Avv^T), which is (I - 2vv^T)A(I - 2vv^T).
        EINSUMS_OMP_PRAGMA(parallel for collapse(2))
        for (size_t i = k + 1; i < dim; i++) {
            for (size_t j = k; j < dim; j++) {
                A_data[i * row_stride + j * col_stride] = std::fma(-vec2[j], vec1[i], A_data[i * row_stride + j * col_stride]);
            }
        }

        // LAPACK: On exit, the diagonal and first super diagonal are overwritten by the corresponding elements.
        // This is by construction, so nothing to do.

        // LAPACK: the elements above the first super diagonal are overwritten by the elements of the elementary reflectors.
        // The way it defines it is that the first non-zero element is 1, so we need to rescale the reflectors.

        EINSUMS_OMP_PARALLEL_FOR
        for (size_t i = k + 2; i < dim; i++) {
            A_data[i * row_stride + k * col_stride] = vec1[i] / vec1[k + 1];
        }

        if (keep_tau) {
            // Set tau.
            tau[k] = 2 * vec1[k + 1] * vec1[k + 1];
        }
    }

    if (keep_tau) {
        tau[dim - 2] = T{0.0};
    }
}

template <Complex T>
void impl_hessenberg_reduce(einsums::detail::TensorImpl<T> *A, T *vec1, T *vec2, T *tau, bool keep_tau) {
    using Real = RemoveComplexT<T>;

    size_t const dim = A->dim(0), row_stride = A->stride(0), col_stride = A->stride(1);

    T *A_data = A->data();

    constexpr Real epsilon = std::numeric_limits<Real>::epsilon();
    constexpr Real small   = Real{1.0} / std::numeric_limits<Real>::max();
    constexpr Real safe_min =
        (small >= std::numeric_limits<Real>::min()) ? small * (1.0 + epsilon) / epsilon : std::numeric_limits<Real>::min() / epsilon;

    for (size_t k = 0; k < dim - 1; k++) {
        RemoveComplexT<T> alpha_sq = 0.0, scale = 0.0, alpha = 0.0;
        T                 tau_val;

        T key = A->subscript(k + 1, k);

        // Calculate alpha squared and the sign of alpha. The sign is -1 for non-negative values (and zero) and 1 for negative values of
        // the key.
        blas::lassq(dim - k - 1, A->data(k + 1, k), row_stride, &scale, &alpha_sq);

        alpha = std::sqrt(alpha_sq) * scale;

        // Check to see if the current column is already solved.
        if (alpha == Real{0.0}) {
            if (keep_tau) {
                tau[k] = T{0.0};
            }
            continue;
        }

        int tries = 0;

        // If the alpha value is too small, we need to rescale.
        for (tries = 0; tries < 10 && alpha < safe_min; tries++) {
            blas::scal(dim - k - 1, Real{1.0} / safe_min, A->data(k + 1, k), row_stride);
            alpha /= safe_min;
        }

        if (tries != 0) {
            blas::lassq(dim - k - 1, A->data(k + 1, k), row_stride, &scale, &alpha_sq);

            alpha = std::sqrt(alpha_sq) * scale;
        }

        if (std::real(key) < 0) {
            alpha = -alpha;
        }

        // Calculate tau.
        tau_val = (key + alpha) / alpha;

        for (int undo = 0; undo < tries; undo++) {
            tau_val *= safe_min;
        }

        // Now, we need to calculate the values of the vectors.
        for (size_t i = 0; i < k; i++) {
            vec1[i] = RemoveComplexT<T>{0.0};
        }

        vec1[k + 1] = RemoveComplexT<T>{1.0};

        for (size_t i = k + 2; i < dim; i++) {
            vec1[i] = A_data[i * row_stride + k * col_stride] / (key + alpha);
        }

        // Now, we need to calculate Av.
        EINSUMS_OMP_PARALLEL_FOR_SIMD
        for (size_t i = 0; i < dim; i++) {
            vec2[i] = T{0.0};
        }

        // The first k + 1 elements of vec1 are all zero. This also means we can ignore the first k rows of A.
        EINSUMS_OMP_PRAGMA(parallel for collapse(2))
        for (size_t i = 0; i < dim; i++) {
            for (size_t j = k + 1; j < dim; j++) {
                vec2[i] += A_data[i * row_stride + j * col_stride] * vec1[j];
            }
        }

        // Scale by the scale factor.
        blas::scal(dim, tau_val, vec2, 1);

        // Next, update A to be A - tau Avv^H.
        EINSUMS_OMP_PRAGMA(parallel for collapse(2))
        for (size_t i = 0; i < dim; i++) {
            for (size_t j = k + 1; j < dim; j++) {
                A_data[i * row_stride + j * col_stride] -= vec2[i] * std::conj(vec1[j]);
            }
        }

        // Now, do it in reverse. Calculate v^H(A - Avv^H)
        EINSUMS_OMP_PARALLEL_FOR_SIMD
        for (size_t i = 0; i < dim; i++) {
            vec2[i] = T{0.0};
        }

        // In this case, the first k columns of A - tau^* Avv^H should be zero when vec1 is non-zero.
        // This means that the first k entries of vec2 will be zero.
        EINSUMS_OMP_PRAGMA(parallel for collapse(2))
        for (size_t i = k; i < dim; i++) {
            for (size_t j = k + 1; j < dim; j++) {
                vec2[i] += A_data[j * row_stride + i * col_stride] * std::conj(vec1[j]);
            }
        }

        // Scale by the scale factor.
        blas::scal(dim, std::conj(tau_val), vec2, 1);

        // Finally, update A to be (A - 2Avv^T) - 2vv^T(A - 2Avv^T), which is (I - 2vv^T)A(I - 2vv^T).
        EINSUMS_OMP_PRAGMA(parallel for collapse(2))
        for (size_t i = k + 1; i < dim; i++) {
            for (size_t j = k; j < dim; j++) {
                A_data[i * row_stride + j * col_stride] -= vec2[j] * vec1[i];
            }
        }

        // LAPACK: On exit, the diagonal and first super diagonal are overwritten by the corresponding elements.
        // This is by construction, so nothing to do.

        // LAPACK: the elements above the first super diagonal are overwritten by the elements of the elementary reflectors.
        // The way it defines it is that the first non-zero element is 1, so we need to rescale the reflectors.

        for (int i = 0; i < tries; i++) {
            alpha *= safe_min;
        }

        EINSUMS_OMP_PARALLEL_FOR
        for (size_t i = k + 2; i < dim; i++) {
            A_data[i * row_stride + k * col_stride] = vec1[i];
        }

        if (keep_tau) {
            // Set tau.
            tau[k] = tau_val;
        }
    }
}

template <typename T>
void impl_compute_q(einsums::detail::TensorImpl<T> *Q, T *vec1, T *vec2, T *tau) {
    T           *Q_data = Q->data();
    size_t const dim = Q->dim(1), row_stride = Q->stride(0), col_stride = Q->stride(1);

    // Set up the first level.
    Q->subscript(-1, -1) = T{1.0};

    // Loop.
    for (int n = dim - 2; n >= 0; n--) {
        // Set up the vector.
        EINSUMS_OMP_PARALLEL_FOR_SIMD
        for (int k = 0; k < dim; k++) {
            vec1[k] = T{0.0};
        }

        // Put a 1 in the next element.
        vec1[n + 1] = T{1.0};

        // Fill the rest of the elements.
        EINSUMS_OMP_PARALLEL_FOR
        for (int k = n + 2; k < dim; k++) {
            vec1[k] = Q_data[k * row_stride + n * col_stride];
        }

        // Now, set up the next level of the matrix.

        EINSUMS_OMP_PARALLEL_FOR
        for (int k = n + 1; k < dim; k++) {
            Q_data[k * row_stride + n * col_stride] = T{0.0};
        }

        EINSUMS_OMP_PARALLEL_FOR
        for (int k = n + 1; k < dim; k++) {
            Q_data[n * row_stride + k * col_stride] = T{0.0};
        }

        Q_data[n * (row_stride + col_stride)] = T{1.0};

        // Next, find Qv.
        EINSUMS_OMP_PARALLEL_FOR_SIMD
        for (int i = 0; i < dim; i++) {
            vec2[i] = T{0.0};
        }

        // Next, find Q - Qvv^H.
        if constexpr (IsComplexV<T>) {

            EINSUMS_OMP_PRAGMA(parallel for collapse(2))
            for (int i = n; i < dim; i++) {
                for (int j = n; j < dim; j++) {
                    vec2[i] += Q_data[i * row_stride + j * col_stride] * std::conj(vec1[j]);
                }
            }
            EINSUMS_OMP_PRAGMA(parallel for collapse(2))
            for (int i = n; i < dim; i++) {
                for (int j = n; j < dim; j++) {
                    // Might need conj(tau).
                    Q_data[i * row_stride + j * col_stride] -= tau[n] * vec2[j] * vec1[i];
                }
            }
        } else {
            EINSUMS_OMP_PRAGMA(parallel for collapse(2))
            for (int i = n; i < dim; i++) {
                for (int j = n; j < dim; j++) {
                    vec2[i] += Q_data[i * row_stride + j * col_stride] * vec1[j];
                }
            }
            EINSUMS_OMP_PRAGMA(parallel for collapse(2))
            for (int i = n; i < dim; i++) {
                for (int j = n; j < dim; j++) {
                    Q_data[i * row_stride + j * col_stride] -= tau[n] * vec2[i] * vec1[j];
                }
            }
        }
    }
}

template <typename T>
void givens_params(T x, T y, T *s, T *c, T *r) {
    if (y == T{0.0}) {
        if (x == T{0.0}) {
            *c = T{1.0};
        } else {
            *c = std::copysign(T{1.0}, x);
        }
        *s = T{0.0};
        *r = std::abs(x);
    } else if (x == T{0.0}) {
        *c = 0;
        *s = std::copysign(T{1.0}, y);
        *r = std::abs(y);
    } else if (std::abs(x) > std::abs(y)) {
        T t = y / x;
        T u = std::copysign(std::sqrt(1 + t * t), x);
        *c  = T{1.0} / u;
        *s  = -*c * t;
        *r  = y * u;
    } else {
        T t = x / y;
        T u = std::copysign(std::sqrt(1 + t * t), y);
        *s  = T{-1.0} / u;
        *c  = t / u;
        *r  = y * u;
    }
}

template <typename T>
void impl_compute_2by2_eigen(T *Q, size_t Q_rows, size_t row_stride, size_t col_stride, RemoveComplexT<T> *diag,
                             RemoveComplexT<T> *subdiag) {
    // Truly diagonal matrix.
    if (*subdiag == RemoveComplexT<T>{0.0}) {
        // Check if we need to swap the eigenvalues.
        if (diag[0] > diag[1]) {
            std::swap(diag[0], diag[1]);
            for (int i = 0; i < Q_rows; i++) {
                std::swap(Q[i * row_stride], Q[i * row_stride + col_stride]);
            }
        }
        // Otherwise, we don't have to do anything. We can just return here.
        return;
    }

    // Set up as a polynomial.
    RemoveComplexT<T> const b = diag[0] + diag[1], c = diag[0] * diag[1] - subdiag[0] * subdiag[0];

    // Find the eigenvalues using the quadratic formula.
    RemoveComplexT<T> const l1 = (b - std::sqrt(b * b - 4 * c)) / 2;
    RemoveComplexT<T> const l2 = (b + std::sqrt(b * b - 4 * c)) / 2;

    // Now, find the eigenvectors.
    RemoveComplexT<T> sine, cosine, norm;

    // Avoid roundoff errors. If we get too close to a diagonal matrix, there can be some issues.
    if (std::abs(l1 - diag[0]) < std::numeric_limits<RemoveComplexT<T>>::epsilon() * std::abs(l1)) {
        if (std::abs(l1 - diag[1]) < std::numeric_limits<RemoveComplexT<T>>::epsilon() * std::abs(l1)) {
            if (std::abs(l2 - diag[0]) < std::numeric_limits<RemoveComplexT<T>>::epsilon() * std::abs(l2)) {
                norm   = std::sqrt((diag[0] - diag[1]) * (l2 - diag[1]) + 2 * subdiag[0] * subdiag[0]);
                cosine = subdiag[0] / norm;
                sine   = -(l2 - diag[1]) / norm;
            } else {
                norm   = std::sqrt((diag[1] - diag[0]) * (l2 - diag[0]) + 2 * subdiag[0] * subdiag[0]);
                sine   = -subdiag[0] / norm;
                cosine = (l2 - diag[0]) / norm;
            }
        } else {
            norm   = std::sqrt((diag[0] - diag[1]) * (l1 - diag[1]) + 2 * subdiag[0] * subdiag[0]);
            cosine = (l1 - diag[1]) / norm;
            sine   = subdiag[0] / norm;
        }
    } else {
        norm   = std::sqrt((diag[1] - diag[0]) * (l1 - diag[0]) + 2 * subdiag[0] * subdiag[0]);
        sine   = (l1 - diag[0]) / norm;
        cosine = subdiag[0] / norm;
    }

    // Now, we can multiply Q on the left by this.
    for (size_t i = 0; i < Q_rows; i++) {
        T A = Q[i * row_stride], B = Q[i * row_stride + col_stride];
        Q[i * row_stride]              = A * cosine + B * sine;
        Q[i * row_stride + col_stride] = -A * sine + B * cosine;
    }

    diag[0]    = l1;
    diag[1]    = l2;
    subdiag[0] = RemoveComplexT<T>{0.0};

    return;
}

template <typename T>
void impl_qr_tridiag_eigen_step(size_t dim, T *Q, size_t Q_rows, size_t row_stride, size_t col_stride, RemoveComplexT<T> *diag,
                                RemoveComplexT<T> *subdiag) {

    // If the matrix is not 2x2, then do the shifted qr algorithm.
    // Start by computing the shift.
    RemoveComplexT<T> shift;

    auto temp1 = (diag[dim - 2] - diag[dim - 1]) / (2.0 * subdiag[dim - 2]);
    auto temp2 = std::hypot(RemoveComplexT<T>{1.0}, temp1);

    if (std::abs(temp1) < std::numeric_limits<RemoveComplexT<T>>::epsilon()) {
        shift = diag[dim - 1] - std::abs(subdiag[dim - 2]);
    } else {
        shift = diag[dim - 1] - subdiag[dim - 2] / (temp1 + std::copysign(temp2, temp1));
    }

    // Shift the diagonal elements.

    for (size_t i = 0; i < dim; i++) {
        diag[i] -= shift;
    }

    RemoveComplexT<T> x = diag[0], y = subdiag[0];

    // Use Givens rotations to compute the next step.
    for (size_t i = 0; i < dim - 1; i++) {
        RemoveComplexT<T> r, s, c;
        givens_params(x, y, &s, &c, &r);

        RemoveComplexT<T> w = c * x - s * y;
        RemoveComplexT<T> d = diag[i] - diag[i + 1];
        RemoveComplexT<T> z = (2 * c * subdiag[i] + d * s) * s;
        diag[i] -= z;
        diag[i + 1] += z;
        subdiag[i] = d * c * s + (c * c - s * s) * subdiag[i];

        x = subdiag[i];

        if (i > 0) {
            subdiag[i - 1] = w;
        }

        if (i < dim - 2) {
            y = -s * subdiag[i + 1];
            subdiag[i + 1] *= c;
        }

        // Operate on the eigenvectors.
        for (size_t j = 0; j < Q_rows; j++) {
            T A = Q[j * row_stride + i * col_stride], B = Q[j * row_stride + (i + 1) * col_stride];
            Q[j * row_stride + i * col_stride]       = A * c - B * s;
            Q[j * row_stride + (i + 1) * col_stride] = A * s + B * c;
        }
    }

    for (size_t i = 0; i < dim; i++) {
        diag[i] += shift;
    }
}

template <typename T>
void impl_qr_tridiag_iterate(size_t dim, T *Q, size_t Q_rows, size_t row_stride, size_t col_stride, RemoveComplexT<T> *diag,
                             RemoveComplexT<T> *subdiag) {

    using Real = RemoveComplexT<T>;

    constexpr Real epsilon  = std::numeric_limits<Real>::epsilon();
    constexpr Real small    = Real{1.0} / std::numeric_limits<Real>::max();
    constexpr Real safe_min = (small >= std::numeric_limits<Real>::min()) ? small * (1.0 + epsilon) : std::numeric_limits<Real>::min();
    Real           safe_scale_min = std::sqrt(safe_min) / (epsilon * epsilon);
    Real           safe_scale_max = std::sqrt(Real{1.0} / safe_min) / Real{3.0}; // This is what LAPACK uses. Seems good enough.

    ptrdiff_t lower_bound = 0;

    for (size_t iters = 1; iters <= dim * 30; iters++) {
        // We have solved the matrix, so exit.
        if (lower_bound >= dim - 1) {
            break;
        }

        // Split the matrix.
        // If we already solved the block above, clear this element.
        if (lower_bound > 0) {
            subdiag[lower_bound] = Real{0.0};
        }

        // Test for small elements and set the split.
        ptrdiff_t split = (ptrdiff_t)dim - 1;
        for (ptrdiff_t m = lower_bound; m < dim - 1; m++) {
            Real test = std::abs(subdiag[lower_bound]);
            if (test == Real{0.0}) {
                split = m;
                break;
            }
            if (test <= std::sqrt(std::abs(diag[m]) * std::abs(diag[m + 1])) * epsilon) {
                subdiag[lower_bound] = Real{0.0};
                split                = m;
                break;
            }
        }

        ptrdiff_t l = split, end = lower_bound;
        lower_bound = split + 1;

        if (end == 1) {
            continue;
        }

        // Scale here.

        Real norm = diag[split - 1];
        for (size_t i = end; i < split; i++) {
            if (diag[i] + subdiag[i] > norm) {
                norm = diag[i] + subdiag[i];
            }
        }

        if (norm == Real{0.0}) {
            continue;
        }

        enum { NO_SCALE, SCALED_UP, SCALED_DOWN } did_scale = NO_SCALE;

        if (norm > safe_scale_max) {
            did_scale = SCALED_DOWN;

            blas::int_t info = blas::lascl('G', 0, 0, norm, safe_scale_max, split - end + 1, 1, diag + end, dim);
            if (info < 0) {
                EINSUMS_THROW_EXCEPTION(std::invalid_argument, "The {} argument to lascl had an invalid value!", print::ordinal(-info));
            }

            info = blas::lascl('G', 0, 0, norm, safe_scale_max, split - end, 1, subdiag + end, dim);
            if (info < 0) {
                EINSUMS_THROW_EXCEPTION(std::invalid_argument, "The {} argument to lascl had an invalid value!", print::ordinal(-info));
            }
        } else if (norm < safe_scale_min) {
            did_scale = SCALED_UP;

            blas::int_t info = blas::lascl('G', 0, 0, norm, safe_scale_min, split - end + 1, 1, diag + end, dim);
            if (info < 0) {
                EINSUMS_THROW_EXCEPTION(std::invalid_argument, "The {} argument to lascl had an invalid value!", print::ordinal(-info));
            }

            info = blas::lascl('G', 0, 0, norm, safe_scale_min, split - end, 1, subdiag + end, dim);
            if (info < 0) {
                EINSUMS_THROW_EXCEPTION(std::invalid_argument, "The {} argument to lascl had an invalid value!", print::ordinal(-info));
            }
        }

        // Perform QR iteration.
        while (l >= end) {
            // Look for a small subdiagonal element.
            ptrdiff_t small_elem = end;
            if (l != end) {
                for (ptrdiff_t m = l; m > end; m--) {
                    Real test = std::abs(subdiag[m - 1]);
                    if (test < std::sqrt(std::abs(diag[m]) * std::abs(diag[m - 1])) * epsilon) {
                        small_elem = m;
                        break;
                    }
                }
            }

            // Clear the small element.
            if (small_elem > end) {
                subdiag[small_elem - 1] = Real{0.0};
            }

            if (small_elem == l) {
                l--;
                continue;
            }

            // Check to see if the remaining matrix is 2x2.
            if (small_elem == l - 1) {
                impl_compute_2by2_eigen(Q + small_elem * col_stride, Q_rows, row_stride, col_stride, diag + small_elem,
                                        subdiag + small_elem);
                l -= 2;
                if (l >= end) {
                    continue;
                }
                // Continue on with the next block.
                break;
            } else {
                impl_qr_tridiag_eigen_step(l + 1 - small_elem, Q + small_elem * col_stride, Q_rows, row_stride, col_stride,
                                           diag + small_elem, subdiag + small_elem);
            }
        }

        if (did_scale == SCALED_UP) {
            blas::int_t info = blas::lascl('G', 0, 0, safe_scale_min, norm, split - end + 1, 1, diag + end, dim);
            if (info < 0) {
                EINSUMS_THROW_EXCEPTION(std::invalid_argument, "The {} argument to lascl had an invalid value!", print::ordinal(-info));
            }

            info = blas::lascl('G', 0, 0, safe_scale_min, norm, split - end, 1, subdiag + end, dim);
            if (info < 0) {
                EINSUMS_THROW_EXCEPTION(std::invalid_argument, "The {} argument to lascl had an invalid value!", print::ordinal(-info));
            }
        } else if (did_scale == SCALED_DOWN) {
            blas::int_t info = blas::lascl('G', 0, 0, safe_scale_max, norm, split - end + 1, 1, diag + end, dim);
            if (info < 0) {
                EINSUMS_THROW_EXCEPTION(std::invalid_argument, "The {} argument to lascl had an invalid value!", print::ordinal(-info));
            }

            info = blas::lascl('G', 0, 0, safe_scale_max, norm, split - end, 1, subdiag + end, dim);
            if (info < 0) {
                EINSUMS_THROW_EXCEPTION(std::invalid_argument, "The {} argument to lascl had an invalid value!", print::ordinal(-info));
            }
        }
    }

    for (int i = 0; i < dim - 1; i++) {
        if (subdiag[i] != Real{0.0}) {
            EINSUMS_THROW_EXCEPTION(std::runtime_error, "The {} eigenvalue did not converge in {} iterations!", print::ordinal(i + 1),
                                    dim * 30);
        }
    }
}

template <typename T>
void impl_eig_sort(einsums::detail::TensorImpl<T> *Q, RemoveComplexT<T> *work) {
    size_t const dim = Q->dim(0), row_stride = Q->stride(0), col_stride = Q->stride(1);
    T           *Q_data = Q->data();

    for (size_t head = 0; head < dim; head++) {
        size_t            min_pos = head;
        RemoveComplexT<T> min_val = work[head];
        for (size_t curr_pos = head; curr_pos < dim; curr_pos++) {
            if (work[curr_pos] < min_val) {
                min_val = work[curr_pos];
                min_pos = curr_pos;
            }
        }

        if (min_pos != head) {
            EINSUMS_OMP_PARALLEL_FOR
            for (size_t i = 0; i < dim; i++) {
                std::swap(Q_data[i * row_stride + head * col_stride], Q_data[i * row_stride + min_pos * col_stride]);
            }
            std::swap(work[head], work[min_pos]);
        }
    }
}

template <NotComplex T>
size_t impl_syev_get_work_length(char jobz, einsums::detail::TensorImpl<T> const *A, einsums::detail::TensorImpl<T> const *W) {
    size_t const dim = A->dim(0);

    if (dim <= 1) {
        return 0;
    }
    if (dim == 2) {
        return 3;
    }

    if (std::tolower(jobz) == 'n') {
        return 2 * dim;
    } else {
        return 5 * dim - 2;
    }
}

template <Complex T>
size_t impl_heev_get_work_length(char jobz, einsums::detail::TensorImpl<T> const *A,
                                 einsums::detail::TensorImpl<RemoveComplexT<T>> const *W) {
    size_t const dim = A->dim(0);

    if (dim <= 1) {
        return 0;
    }
    if (dim == 2) {
        return 3;
    }

    if (std::tolower(jobz) == 'n') {
        return 2 * dim;
    } else {
        return 3 * dim - 1;
    }
}

template <NotComplex T>
void impl_strided_syev(char jobz, einsums::detail::TensorImpl<T> *A, einsums::detail::TensorImpl<T> *W, T *work) {
    size_t const row_stride = A->stride(0), col_stride = A->stride(1), dim = A->dim(0);
    T           *A_data = A->data();

    // Tau is only referenced when jobz != 'n', so it may point to past the end of the work array.
    T *vec1 = work, *vec2 = work + dim, *tau = work + 2 * dim;
    T *diag = work, *subdiag = work + dim;

    if (dim == 1) {
        W->subscript(0)    = A->subscript(0, 0);
        A->subscript(0, 0) = T{1.0};
    } else if (dim == 2) {
        work[0]            = A->subscript(0, 0);
        work[1]            = A->subscript(1, 1);
        work[2]            = A->subscript(1, 0);
        A->subscript(0, 0) = T{1.0};
        A->subscript(0, 1) = T{0.0};
        A->subscript(1, 0) = T{0.0};
        A->subscript(1, 1) = T{1.0};
        impl_compute_2by2_eigen(A_data, 2, row_stride, col_stride, work, work + 2);
        W->subscript(0) = work[0];
        W->subscript(1) = work[1];
    } else {

        // Reduce to tridiagonal form.
        impl_hessenberg_reduce(A, vec1, vec2, tau, std::tolower(jobz) != 'n');

        // Compute the eigenvalues and eigenvectors of the tridiagonal form.
        if (std::tolower(jobz) == 'n') {
            EINSUMS_OMP_PARALLEL_FOR
            for (size_t i = 0; i < dim; i++) {
                diag[i] = A_data[i * (row_stride + col_stride)];
            }

            EINSUMS_OMP_PARALLEL_FOR
            for (size_t i = 0; i < dim - 1; i++) {
                subdiag[i] = A_data[i * (row_stride + col_stride) + row_stride];
            }

            // Just the eigenvalues. We can use LAPACK's sterf to compute the eigenvalues of a tridiagonal matrix.
            auto info = blas::sterf(A->dim(0), diag, subdiag);

            if (info == -1) {
                EINSUMS_THROW_EXCEPTION(std::invalid_argument, "The length argument to sterf was invalid! It must be non-negative, got {}.",
                                        A->dim(0));
            } else if (info < 0) {
                EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                        "The {} argument to sterf was invalid! This is likely due to being passed a null pointer.",
                                        print::ordinal(-info));
            } else if (info > 0) {
                EINSUMS_THROW_EXCEPTION(
                    std::runtime_error,
                    "The algorithm failed to find all of the eigenvalues within {} iterations (30 times the dimension). A "
                    "total of {} elements have not converged to zero.",
                    30 * A->dim(0), info);
            }

            // Copy the eigenvalues.
            EINSUMS_OMP_PARALLEL_FOR
            for (size_t i = 0; i < A->dim(0); i++) {
                W->subscript(i) = diag[i];
            }
        } else {
            // Change where diag and subdiag point.
            diag    = work + 3 * dim - 1;
            subdiag = work + 4 * dim - 1;

            EINSUMS_OMP_PARALLEL_FOR
            for (size_t i = 0; i < dim; i++) {
                diag[i] = A_data[i * (row_stride + col_stride)];
            }

            EINSUMS_OMP_PARALLEL_FOR
            for (size_t i = 0; i < dim - 1; i++) {
                subdiag[i] = A_data[i * (row_stride + col_stride) + row_stride];
            }

            // Compute the Q matrix.
            impl_compute_q(A, vec1, vec2, tau);

            impl_qr_tridiag_iterate(A->dim(0), A->data(), A->dim(0), A->stride(0), A->stride(1), diag, subdiag);

            impl_eig_sort(A, diag);

            for (size_t i = 0; i < dim; i++) {
                W->subscript(i) = diag[i];
            }
        }
    }
}

template <Complex T>
void impl_strided_heev(char jobz, einsums::detail::TensorImpl<T> *A, einsums::detail::TensorImpl<RemoveComplexT<T>> *W, T *work,
                       RemoveComplexT<T> *rwork) {
    size_t const row_stride = A->stride(0), col_stride = A->stride(1), dim = A->dim(0);
    T           *A_data = A->data();

    T                 *vec1 = work, *vec2 = work + dim, *tau = work + (2 * dim);
    RemoveComplexT<T> *diag = rwork, *subdiag = rwork + dim;

    if (dim == 1) {
        W->subscript(0)    = std::real(A->subscript(0, 0));
        A->subscript(0, 0) = T{1.0};
    } else if (dim == 2) {
        RemoveComplexT<T> a = std::real(A->subscript(0, 0)), c = std::real(A->subscript(1, 1));
        T                 b = A->subscript(0, 1);

        RemoveComplexT<T> b_mag2 = std::real(b) * std::real(b) + std::imag(b) * std::imag(b);
        RemoveComplexT<T> B      = -a - c;
        RemoveComplexT<T> C      = a * c - b_mag2;
        RemoveComplexT<T> l1 = (-B - std::sqrt(B * B - 4 * C)) / 2, l2 = (-B + std::sqrt(B * B - 4 * C));

        // Now, find the eigenvectors.
        RemoveComplexT<T> norm;
        T                 sine, cosine;

        // Avoid roundoff errors. If we get too close to a diagonal matrix, there can be some issues.
        if (std::abs(l1 - a) < std::numeric_limits<RemoveComplexT<T>>::epsilon() * std::abs(l1)) {
            if (std::abs(l1 - c) < std::numeric_limits<RemoveComplexT<T>>::epsilon() * std::abs(l1)) {
                if (std::abs(l2 - a) < std::numeric_limits<RemoveComplexT<T>>::epsilon() * std::abs(l2)) {
                    norm   = std::sqrt((a - c) * (l2 - c) + 2 * b_mag2);
                    cosine = std::conj(b) / norm;
                    sine   = -(l2 - c) / norm;
                } else {
                    norm   = std::sqrt((c - a) * (l2 - a) + 2 * b_mag2);
                    sine   = -b / norm;
                    cosine = (l2 - a) / norm;
                }
            } else {
                norm   = std::sqrt((a - c) * (l1 - c) + 2 * b_mag2);
                cosine = (l1 - c) / norm;
                sine   = std::conj(b) / norm;
            }
        } else {
            norm   = std::sqrt((c - a) * (l1 - a) + 2 * b_mag2);
            sine   = (l1 - a) / norm;
            cosine = b / norm;
        }

        A->subscript(0, 0) = cosine;
        A->subscript(0, 1) = -sine;
        A->subscript(1, 0) = sine;
        A->subscript(1, 1) = cosine;
        W->subscript(0)    = l1;
        W->subscript(1)    = l2;
    } else {
        // Reduce to tridiagonal form.
        impl_hessenberg_reduce(A, vec1, vec2, tau, std::tolower(jobz) != 'n');

        EINSUMS_OMP_PARALLEL_FOR
        for (size_t i = 0; i < dim; i++) {
            diag[i] = std::real(A_data[i * (row_stride + col_stride)]);
        }

        EINSUMS_OMP_PARALLEL_FOR
        for (size_t i = 0; i < dim - 1; i++) {
            subdiag[i] = std::real(A_data[i * (row_stride + col_stride) + row_stride]);
        }

        // Compute the eigenvalues and eigenvectors of the tridiagonal form.
        if (std::tolower(jobz) == 'n') {

            // Just the eigenvalues. We can use LAPACK's sterf to compute the eigenvalues of a tridiagonal matrix.
            auto info = blas::sterf(A->dim(0), diag, subdiag);

            if (info == -1) {
                EINSUMS_THROW_EXCEPTION(std::invalid_argument, "The length argument to sterf was invalid! It must be non-negative, got {}.",
                                        A->dim(0));
            } else if (info < 0) {
                EINSUMS_THROW_EXCEPTION(std::invalid_argument,
                                        "The {} argument to sterf was invalid! This is likely due to being passed a null pointer.",
                                        print::ordinal(-info));
            } else if (info > 0) {
                EINSUMS_THROW_EXCEPTION(
                    std::runtime_error,
                    "The algorithm failed to find all of the eigenvalues within {} iterations (30 times the dimension). A "
                    "total of {} elements have not converged to zero.",
                    30 * A->dim(0), info);
            }

            // Copy the eigenvalues.
            EINSUMS_OMP_PARALLEL_FOR
            for (size_t i = 0; i < A->dim(0); i++) {
                W->subscript(i) = rwork[i];
            }
        } else {
            // Compute the Q matrix.

            impl_compute_q(A, vec1, vec2, tau);

            impl_qr_tridiag_iterate(A->dim(0), A->data(), A->dim(0), A->stride(0), A->stride(1), diag, subdiag);

            impl_eig_sort(A, diag);

            for (size_t i = 0; i < dim; i++) {
                W->subscript(i) = diag[i];
            }
        }
    }
}

} // namespace detail
} // namespace linear_algebra
} // namespace einsums