# Einsums in C++

|   |   |
|---|---|
| **Status** | [![codecov](https://codecov.io/github/Einsums/Einsums/graph/badge.svg?token=Z8WA6CEGQA)](https://codecov.io/github/Einsums/Einsums) ![GitHub branch check runs](https://img.shields.io/github/check-runs/Einsums/Einsums/main) |
| **Release** | ![GitHub Release](https://img.shields.io/github/v/release/Einsums/Einsums) ![GitHub commits since latest release](https://img.shields.io/github/commits-since/Einsums/Einsums/latest) |
| **Documentation** | [![Documentation](https://img.shields.io/badge/docs-latest-green?style=flat)](https://einsums.github.io/Einsums/) |
| **Connect With Us** | [![Discord](https://img.shields.io/discord/1357368862512906360?logo=discord&label=Discord)](https://discord.gg/8GvtkyWZUv) |

Provides compile-time contraction pattern analysis to determine optimal operation to perform.

## Requirements
A C++ compiler with C++20 support.

The following libraries are required to build Einsums:

* BLAS and LAPACK
* HDF5

The following libraries are also required, but will be fetched if they can not be found.

* fmtlib >= 11
* Catch2 >= 3
* gabime/spdlog >= 1

On my personal development machine, I use MKL for the above requirements. On GitHub Actions, stock BLAS, LAPACK, and FFTW3 are used.

Optional requirements:

* A Fast Fourier Transform library, either FFTW3 or DFT from MKL.
* HIP for graphics card support. Uses hipBlas, hipSolver, and the HIP language. Does not yet support hipFFT.
* cpptrace for backtraces.
* LibreTT for GPU transposes.
* pybind11 for the Python extension module.

## Examples
This will optimize at compile-time to a BLAS dgemm call.
```C++
#include "Einsums/TensorAlgebra.hpp"

using einsums;  // Provides Tensor and create_random_tensor
using einsums::tensor_algebra;  // Provides einsum and Indices
using einsums::index;  // Provides i, j, k

Tensor<2> A = create_random_tensor("A", 7, 7);
Tensor<2> B = create_random_tensor("B", 7, 7);
Tensor<2> C{"C", 7, 7};

einsum(Indices{i, j}, &C, Indices{i, k}, A, Indices{k, j}, B);
```

Two-Electron Contribution to the Fock Matrix
```C++
#include "Einsums/TensorAlgebra.hpp"

using namespace einsums;

void build_Fock_2e_einsum(Tensor<2> *F,
                          const Tensor<4> &g,
                          const Tensor<2> &D) {
    using namespace einsums::tensor_algebra;
    using namespace einsums::index;

    // Will compile-time optimize to BLAS gemv
    einsum(1.0, Indices{p, q}, F,
           2.0, Indices{p, q, r, s}, g, Indices{r, s}, D);

    // As written cannot be optimized.
    // A generic arbitrary contraction function will be used.
    einsum(1.0, Indices{p, q}, F,
          -1.0, Indices{p, r, q, s}, g, Indices{r, s}, D);
}
```

Here are some comparisons between different methods of building the Hartree-Fock G matrix out of the two-electron integrals and the density matrix.
The code for this is similar to the sample above.
The first plot uses timings for 100 ortbitals using several methods: C for loops with compiler loop vectorization; C for loops with
OpenMP loop vectorization and parallelization; Fortran do-concurrent loops; BLAS with a for loop for calculating the K matrix, gemv for the
J matrix, and axpy for the G matrix; BLAS with a for loop to permute the two-electron integrals, then gemv for the J and K matrices
and axpy for the G matrix; Einsums without permuting the two-electron integrals, using the generic algorithm for the K matrix; and
Einsums with a permutation of the two-electron integrals, using a selected algorithm for the K matrix.

![einsum Performance](/docs/sphinx/_static/index-images/Performance.png)

The following shows the difference in overall performance as the number of orbitals increases.

![einsums Growth](/docs/sphinx/_static/index-images/Performance_comp.png)

These timings were computed on a system with  an Intel Core i7-13700K with 32 GB of DDR5 RAM and an
AMD Radeon 7900X graphics card running Debian 12, kernel version 6.1.

W Intermediates in CCD
```C++
Wmnij = g_oooo;
// Compile-time optimizes to gemm
einsum(1.0,  Indices{m, n, i, j}, &Wmnij,
       0.25, Indices{i, j, e, f}, t_oovv,
             Indices{m, n, e, f}, g_oovv);

Wabef = g_vvvv;
// Compile-time optimizes to gemm
einsum(1.0,  Indices{a, b, e, f}, &Wabef,
       0.25, Indices{m, n, e, f}, g_oovv,
             Indices{m, n, a, b}, t_oovv);

Wmbej = g_ovvo;
// As written uses generic arbitrary contraction function
einsum(1.0, Indices{m, b, e, j}, &Wmbej,
      -0.5, Indices{j, n, f, b}, t_oovv,
            Indices{m, n, e, f}, g_oovv);
```

CCD Energy
```C++
/// Compile-time optimizes to a dot product
einsum(0.0,  Indices{}, &e_ccd,
       0.25, Indices{i, j, a, b}, new_t_oovv,
             Indices{i, j, a, b}, g_oovv);
```

## Python: a NumPy-style API

The `einsums` Python package (built with `EINSUMS_BUILD_PYTHON=ON`) wraps the
same engine with a NumPy-shaped surface, so tensor code reads like NumPy while
every operation dispatches to Einsums' own kernels (BLAS, etc.) — *not* to
NumPy. Tensors implement the buffer protocol, so `np.asarray(t)` is a zero-copy
view; that is the explicit escape hatch when you actually want NumPy.

```python
import numpy as np
import einsums

# Construction (NumPy-style; default dtype float64)
A = einsums.zeros((3, 4))
B = einsums.ones((4, 2))
M = einsums.asarray(np.arange(12.0).reshape(3, 4))   # ingest a NumPy array
Z = einsums.full((2, 2), 1 + 2j, dtype="complex128")

# Attributes
A.shape          # (3, 4)
A.ndim, len(A)   # 2, 3
A.dtype          # dtype('float64')

# Operators dispatch to Einsums (gemm / gemv / axpy / scale / ...), never NumPy
C = A @ B            # matrix-matrix (gemm) -> (3, 2)
y = A @ B[:, 0]      # matrix-vector (gemv) -> (3,)
S = einsums.ones((3, 3))
K = 2.0 * S - S.T    # scalar mul, subtraction, transpose view
M += 1.0             # in-place scalar shift
H = M * M            # element-wise (Hadamard)

# Indexing: reads return zero-copy views, assignment writes through
row   = M[1]                      # rank-reducing -> (4,)
block = M[1:3, :]                 # slice view -> (2, 4)
M[0, :] = einsums.full((4,), 5.0) # sub-view assignment
M[2:, :] = 0.0                    # scalar fill

# Transpose views, a dense copy, and reductions
At = M.transpose(1, 0)            # also M.T / M.swapaxes(0, 1)
Mc = M.copy()
total, mean, largest = M.sum(), M.mean(), M.max()
```

The same code composes inside a **ComputeGraph** capture, where each operation
is *recorded* into a graph that is then optimized and executed. Operators,
`.T`, indexing, and reductions are all capture-aware:

```python
import einsums.graph as cg

g = cg.Graph("workflow")
with cg.capture(g):
    G = A.T @ A          # gemm on a graph-registered transpose view -> (4, 4)
    e = (A * A).sum()    # reduction -> a 1-element graph tensor
g.execute()              # run the optimized graph
```

See [`examples/psi4-bridge/df_mp2_numpy_style.py`](examples/psi4-bridge/df_mp2_numpy_style.py)
(eager) and
[`df_mp2_graph_numpy_style.py`](examples/psi4-bridge/df_mp2_graph_numpy_style.py)
(captured) for a full density-fitted MP2 written this way, checked against psi4.
