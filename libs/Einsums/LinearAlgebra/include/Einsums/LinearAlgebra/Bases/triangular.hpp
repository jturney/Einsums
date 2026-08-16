//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once
#include <Einsums/BLAS.hpp>
#include <Einsums/Config/Namespace.hpp>
#include <Einsums/TensorImpl/TensorImpl.hpp>

#include <cmath>

EINSUMS_NAMESPACE_BEGIN(linear_algebra::detail)

template <typename T, typename Pivots>
    requires requires(Pivots a, size_t ind) {
        typename Pivots::value_type;
        typename Pivots::size_type;

        { a.size() } -> std::same_as<typename Pivots::size_type>;
        { a.data() } -> std::same_as<typename Pivots::value_type *>;
        a[ind];
    }
int impl_lu_decomp(einsums::detail::TensorImpl<T> &A, Pivots &pivot) {
    size_t const m = A.dim(0), n = A.dim(1), min_dim = std::min(m, n);
    int          ret = 0;

    // Gaussian elimination.
    for (size_t k = 0; k < min_dim - 1; k++) {
        // Find the largest element not yet processed in this column.
        size_t max_row  = k;
        T      max_elem = A.subscript_no_check(k, k);

        for (size_t i = k; i < m; i++) {
            if (std::abs(A.subscript(i, k)) > std::abs(max_elem)) {
                max_row  = i;
                max_elem = A.subscript(i, k);
            }
        }

        // If the current column only has zeros, then skip this iteration.
        if (max_elem == T{0.0}) {
            if (ret == 0) {
                ret = (int)k + 1;
            }
            continue;
        }

        // Swap the current row and the biggest row.
        pivot[k] = max_row + 1; // Plus 1 to keep it compatible with LAPACK.

        // Checks to see if we actually need to swap.
        if (max_row != k) {
            for (size_t j = 0; j < n; j++) {
                std::swap(A.subscript_no_check(k, j), A.subscript_no_check(max_row, j));
            }
        }

        // Eliminate the rows.
        T row_scale = A.subscript_no_check(k, k);
        for (size_t i = k + 1; i < m; i++) {
            T curr_scale = A.subscript_no_check(i, k);
            for (size_t j = k; j < n; j++) {
                // Do it like this to hopefully avoid over/underflow.
                A.subscript_no_check(i, j) = A.subscript_no_check(i, j) - curr_scale * A.subscript_no_check(k, j) / row_scale;
            }

            // Set the value for the L matrix. The diagonal should be unit.
            A.subscript(i, k) = curr_scale / row_scale;
        }
    }

    pivot[min_dim - 1] = min_dim;

    return ret;
}

template <typename T, typename Pivots>
    requires requires(Pivots a, size_t ind) {
        typename Pivots::value_type;
        typename Pivots::size_type;

        { a.size() } -> std::same_as<typename Pivots::size_type>;
        { a.data() } -> std::same_as<typename Pivots::value_type *>;
        a[ind];
    }
int impl_solve(einsums::detail::TensorImpl<T> &A, einsums::detail::TensorImpl<T> &X, Pivots &pivot) {
    size_t const m = A.dim(0), n = A.dim(1), min_dim = std::min(m, n), nrhs = (X.rank() == 1) ? 1 : X.dim(1);

    // LU decomposition.
    int info = impl_lu_decomp(A, pivot);

    if (info != 0) {
        return info;
    }

    // Solve the system.

    if (X.rank() == 1) {
        // First, apply P^-1 and L^-1.
        for (size_t k = 0; k < min_dim - 1; k++) {
            // We know where the max row was. Swap.
            if (pivot[k] != k + 1) {
                std::swap(X.subscript_no_check(k), X.subscript_no_check(pivot[k] - 1));
            }

            // We know the scale factors for the rows. Apply them
            for (size_t i = k + 1; i < m; i++) {
                T scale = A.subscript_no_check(i, k);
                X.subscript_no_check(i) -= X.subscript(k) * scale;
            }
        }

        // Finally, apply U^-1.
        for (size_t k = n - 1; k > 0; k--) {
            T row_scale = A.subscript_no_check(k, k);
            for (ptrdiff_t i = (ptrdiff_t)k - 1; i >= 0; i--) {
                T scale = A.subscript_no_check(i, k);

                X.subscript_no_check(i) = (row_scale * X.subscript_no_check(i) - scale * X.subscript_no_check(k)) / row_scale;
            }

            X.subscript_no_check(k) /= row_scale;
        }
    } else {

        // First, apply P^-1 and L^-1.
        for (size_t k = 0; k < min_dim - 1; k++) {
            // We know where the max row was. Swap.
            if (pivot[k] != k + 1) {
                for (size_t j = 0; j < nrhs; j++) {
                    std::swap(X.subscript_no_check(k, j), X.subscript_no_check(pivot[k] - 1, j));
                }
            }
        }
        for (size_t k = 0; k < min_dim - 1; k++) {
            // We know the scale factors for the rows. Apply them
            for (size_t i = k + 1; i < m; i++) {
                T scale = A.subscript_no_check(i, k);
                for (size_t j = 0; j < nrhs; j++) {
                    X.subscript_no_check(i, j) -= X.subscript(k, j) * scale;
                }
            }
        }

        // Finally, apply U^-1.
        for (ptrdiff_t k = (ptrdiff_t)n - 1; k >= 0; k--) {
            T row_scale = A.subscript_no_check(k, k);
            for (ptrdiff_t i = (ptrdiff_t)k - 1; i >= 0; i--) {
                T scale = A.subscript_no_check(i, k);

                for (size_t j = 0; j < nrhs; j++) {
                    X.subscript_no_check(i, j) = (row_scale * X.subscript_no_check(i, j) - scale * X.subscript_no_check(k, j)) / row_scale;
                }
            }

            for (size_t j = 0; j < nrhs; j++) {
                X.subscript_no_check(k, j) /= row_scale;
            }
        }
    }

    return info;
}

template <typename T, typename Pivots>
    requires requires(Pivots a, size_t ind) {
        typename Pivots::value_type;
        typename Pivots::size_type;

        { a.size() } -> std::same_as<typename Pivots::size_type>;
        { a.data() } -> std::same_as<typename Pivots::value_type *>;
        a[ind];
    }
int impl_invert_lu(einsums::detail::TensorImpl<T> &A_lu, Pivots const &pivot, T *work) {
    // Assume A_lu has been already put into impl_lu_decomp, and pivot is the result.
    size_t const m = A_lu.dim(0), n = A_lu.dim(1), min_dim = std::min(m, n);

    // Check for singular values.
    for (size_t i = 0; i < n; i++) {
        if (A_lu.subscript_no_check(i, i) == T{0.0}) {
            return (int)i + 1;
        }
    }

    // First, compute the inverse of U.
    /*
     * The idea:
     * Pretend that we have the identity matrix in A.
     * Do Gaussian elimination starting from the lower right corner.
     * If we do it like this, we can ignore the elements to the right since in the full
     * calculation, these would be zeroed in U. We can then replace these with the calculated
     * elements of the inverse matrix.
     */
    for (ptrdiff_t k = (ptrdiff_t)n - 1; k >= 0; k--) {
        // Get the row scale.
        T row_scale                   = A_lu.subscript_no_check(k, k);
        A_lu.subscript_no_check(k, k) = T{1.0};

        // Eliminate the rows above.
        for (size_t i = 0; i < k; i++) {
            T scale                       = A_lu.subscript_no_check(i, k);
            A_lu.subscript_no_check(i, k) = -scale / row_scale;
            for (size_t j = k + 1; j < n; j++) {
                A_lu.subscript_no_check(i, j) =
                    (row_scale * A_lu.subscript_no_check(i, j) - scale * A_lu.subscript_no_check(k, j)) / row_scale;
            }
        }

        // Scale the current row.
        for (size_t j = k; j < n; j++) {
            A_lu.subscript_no_check(k, j) /= row_scale;
        }
    }

    // Next, solve for A^-1 column by column.
    for (ptrdiff_t k = n - 2; k >= 0; k--) {
        // Copy a column of L into the work array.
        // We don't need the 1 entry, since when you work it out, this will be the part
        // we are solving for.
        for (size_t i = k + 1; i < n; i++) {
            work[i]                       = A_lu.subscript_no_check(i, k);
            A_lu.subscript_no_check(i, k) = T{0.0};
        }

        // Now, do a matrix-vector multiply. Don't clear the column first, we need the values already there.
        for (size_t i = 0; i < n; i++) {
            for (size_t j = k + 1; j < n; j++) {
                A_lu.subscript_no_check(i, k) -= A_lu.subscript_no_check(i, j) * work[j];
            }
        }
    }

    // Undo the permutes.
    for (ptrdiff_t i = n - 1; i >= 0; i--) {
        if (pivot[i] != i + 1) {
            for (size_t j = 0; j < m; j++) {
                std::swap(A_lu.subscript_no_check(j, pivot[i] - 1), A_lu.subscript_no_check(j, i));
            }
        }
    }

    return 0;
}
EINSUMS_NAMESPACE_END(linear_algebra::detail)