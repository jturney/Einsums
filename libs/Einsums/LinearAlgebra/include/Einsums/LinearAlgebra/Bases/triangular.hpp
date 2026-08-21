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

/**
 * @brief Solve @f$AX = B@f$ from an LU factorization something else already computed.
 *
 * @p A_lu is READ, never written, so one factorization serves any number of right-hand sides. @p pivot follows the LAPACK
 * convention - one-based, @p pivot[k] naming the row that was swapped with row @p k - which is what both ::impl_lu_decomp and
 * LAPACK's @c getrf produce, and the multipliers below it are read in that permuted order. That is why every interchange is
 * applied to @p X before any elimination is: the factorization swapped whole rows, so the stored @c L is already permuted.
 *
 * @p X is a rank-1 tensor for a single right-hand side or a rank-2 tensor whose columns are the right-hand sides.
 */
template <typename T, typename Pivots>
    requires requires(Pivots a, size_t ind) {
        typename Pivots::value_type;
        typename Pivots::size_type;

        { a.size() } -> std::same_as<typename Pivots::size_type>;
        a[ind];
    }
void impl_lu_solve(einsums::detail::TensorImpl<T> const &A_lu, einsums::detail::TensorImpl<T> &X, Pivots const &pivot) {
    size_t const m = A_lu.dim(0), n = A_lu.dim(1), min_dim = std::min(m, n), nrhs = (X.rank() == 1) ? 1 : X.dim(1);
    bool const   vector = X.rank() == 1;

    auto x = [&X, vector](size_t i, size_t j) -> T & {
        if (vector) {
            return X.subscript_no_check(i);
        }
        return X.subscript_no_check(i, j);
    };

    // Every interchange first, then the unit-diagonal forward substitution.
    for (size_t k = 0; k + 1 < min_dim; k++) {
        if (static_cast<size_t>(pivot[k]) != k + 1) {
            for (size_t j = 0; j < nrhs; j++) {
                std::swap(x(k, j), x(static_cast<size_t>(pivot[k]) - 1, j));
            }
        }
    }

    for (size_t k = 0; k + 1 < min_dim; k++) {
        for (size_t i = k + 1; i < m; i++) {
            T const scale = A_lu.subscript_no_check(i, k);
            for (size_t j = 0; j < nrhs; j++) {
                x(i, j) -= x(k, j) * scale;
            }
        }
    }

    // Back substitution against U.
    for (ptrdiff_t k = (ptrdiff_t)n - 1; k >= 0; k--) {
        T const row_scale = A_lu.subscript_no_check((size_t)k, (size_t)k);

        for (ptrdiff_t i = k - 1; i >= 0; i--) {
            T const scale = A_lu.subscript_no_check((size_t)i, (size_t)k);

            for (size_t j = 0; j < nrhs; j++) {
                // Written this way rather than as a division of the subtrahend to keep an over- or underflow out of the intermediate.
                x((size_t)i, j) = (row_scale * x((size_t)i, j) - scale * x((size_t)k, j)) / row_scale;
            }
        }

        for (size_t j = 0; j < nrhs; j++) {
            x((size_t)k, j) /= row_scale;
        }
    }
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
    // LU decomposition.
    int info = impl_lu_decomp(A, pivot);

    if (info != 0) {
        return info;
    }

    impl_lu_solve(A, X, pivot);

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