# `det` sign bug for random matrices with odd-parity LU pivots

## Symptom

`einsums.linalg.det(A)` returns the determinant with the wrong sign for some
random matrices. Same magnitude as `numpy.linalg.det`, opposite sign.

First surfaced in CI on the `Linux / default • mkl • RelWithDebInfo` leg:

```
test_det_eager_matches_numpy:
  ACTUAL:   0.53243
  DESIRED: -0.53243
  Max absolute difference: 1.06486
  Max relative difference: 2.0
```

The test was patched to compare `np.abs()` on both sides as a CI unblock.
The underlying sign bug is still live and needs a real fix.

## Code under investigation

`libs/Einsums/LinearAlgebra/include/Einsums/LinearAlgebra.hpp`, lines ~2118-2155:

```cpp
template <MatrixConcept AType>
typename AType::ValueType det(AType const &A) {
    // ...
    RemoveViewT<AType> temp = A;
    BufferVector<blas::int_t> pivots;
    int singular = getrf(&temp, &pivots);
    if (singular > 0) return T{0.0};

    T ret{1.0};
    int parity = 0;

    // Count permutation parity from LAPACK ipiv (1-based)
    for (int i = 0; i < A.dim(0); i++) {
        if (pivots[i] != i + 1) {
            parity++;
        }
    }

    // Diagonal product
    #pragma omp parallel for simd reduction(* : ret)
    for (int i = 0; i < A.dim(0); i++) {
        ret *= temp(i, i);
    }

    if (parity % 2 == 1) {
        ret *= T{-1.0};
    }
    return ret;
}
```

The parity counter *looks* correct: LAPACK's `getrf` returns `ipiv[i] = j+1`
where `j` is the row swapped into position `i` at step `i`. `ipiv[i] == i+1`
means no swap. Counting the number of swaps gives the permutation parity, and
`(-1)^parity` is the sign correction.

So either:

1. The counter has a subtle off-by-one I'm not seeing.
2. The OpenMP reduction is somehow flipping a sign (unlikely — `*` reduction
   on `double` doesn't flip signs).
3. The `temp(i, i)` access on certain layouts reads the wrong cell.
4. MKL on Linux returns `ipiv` in a different convention than OpenBLAS /
   Accelerate (different base, different meaning).
5. There's an interaction with `MKL_ILP64` (64-bit `int_t`) and the way
   `pivots[i] != i + 1` is evaluated.

## Investigation steps

1. Reproduce locally with a known-bad seed. The CI seed isn't easy to extract
   from the pytest output, but `np.random.seed(0)` + `create_random_tensor`
   may reproduce.
2. Print `pivots` for the failing input and check the convention against
   LAPACK's `dgetrf` docs.
3. Compare `parity` against numpy's `np.linalg.slogdet(A)` which returns
   `(sign, logabsdet)` explicitly.
4. Check the MKL ILP64 build's `blas::int_t` size and verify
   `pivots[i] != i + 1` does the right integer promotion.
5. Confirm `LinearAlgebra/include/Einsums/LinearAlgebra/Base.hpp::getrf`
   doesn't transpose `ipiv` or reverse it.

## Related

- The `test_invert_eager` random-matrix flake fixed in `307b0a46` had a
  similar root cause flavor (random data tripping a code path that should
  have been robust).
- `BLAS::int_t` typedef and the MKL ILP64 path are documented in
  `libs/Einsums/BLAS/include/Einsums/BLAS/Types.hpp`.

## Status

- **Patched**: `test_det_eager_matches_numpy` compares `np.abs` on both
  sides (commit `<pending>`).
- **Sign computation**: not fixed. This bug is still in
  `einsums.linalg.det` for any caller relying on the sign.
- **Priority**: medium. Most chemistry workloads use `det` for sanity
  checks rather than sign-sensitive logic, but the bug needs a real fix.
