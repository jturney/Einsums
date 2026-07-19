# Einsums in C++

|   |   |
|---|---|
| **Status** | [![codecov](https://codecov.io/github/Einsums/Einsums/graph/badge.svg?token=Z8WA6CEGQA)](https://codecov.io/github/Einsums/Einsums) ![GitHub branch check runs](https://img.shields.io/github/check-runs/Einsums/Einsums/main) |
| **Release** | ![GitHub Release](https://img.shields.io/github/v/release/Einsums/Einsums) ![GitHub commits since latest release](https://img.shields.io/github/commits-since/Einsums/Einsums/latest) |
| **Documentation** | [![Documentation](https://img.shields.io/badge/docs-latest-green?style=flat)](https://einsums.github.io/Einsums/) |
| **Connect With Us** | [![Discord](https://img.shields.io/discord/1357368862512906360?logo=discord&label=Discord)](https://discord.gg/8GvtkyWZUv) |

Einsums is a C++20 tensor algebra library.
It analyzes contraction patterns at compile time to dispatch each `einsum` call to the best available implementation: a direct BLAS call, a packed-GEMM backend, or a generic contraction algorithm.
On top of the eager API, a deferred-execution ComputeGraph records whole workflows, optimizes them with a pass pipeline, and replays them; this is the preferred mechanism for performance-critical code.
The eager API targets the host CPU; GPU offload and distributed execution are delivered through graph passes, so they apply only to captured workflows.
SIMD kernels are compiled for several x86-64 instruction-set levels and selected at runtime, so one binary runs everywhere and still uses the fastest kernels the CPU supports.
A NumPy-style Python package wraps the same engine.
CUDA and HIP GPU backends and MPI distributed execution are works in progress.

## Requirements

A C++ compiler with C++20 support.

The following libraries are required to build Einsums:

* BLAS and LAPACK
* HDF5
* OpenMP

The following libraries are also required, but will be fetched if they can not be found:

* fmtlib >= 12
* Catch2 >= 3
* gabime/spdlog >= 1

Optional requirements:

* A Fast Fourier Transform library, either FFTW3 or DFT from MKL.
* Python, for the `einsums` Python package (`-DEINSUMS_BUILD_PYTHON=ON`; the binding generator is vendored, and pybind11 is fetched as needed).
* CUDA or HIP for GPU support (work in progress).
* MPI (Open MPI or MPICH) for distributed execution (work in progress).
* cpptrace for backtraces.
* LibreTT for GPU transposes.

## Examples

This will optimize at compile time to a BLAS dgemm call.
```C++
#include <Einsums/TensorAlgebra.hpp>

using namespace einsums;                 // Tensor, create_random_tensor
using namespace einsums::tensor_algebra; // einsum, Indices
using namespace einsums::index;          // i, j, k

Tensor<double, 2> A = create_random_tensor("A", 7, 7);
Tensor<double, 2> B = create_random_tensor("B", 7, 7);
Tensor<double, 2> C{"C", 7, 7};

einsum(Indices{i, j}, &C, Indices{i, k}, A, Indices{k, j}, B);
```

Two-electron contribution to the Fock matrix:
```C++
#include <Einsums/TensorAlgebra.hpp>

using namespace einsums;

void build_Fock_2e_einsum(Tensor<double, 2> *F,
                          Tensor<double, 4> const &g,
                          Tensor<double, 2> const &D) {
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

W intermediates in CCD:
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

CCD energy:
```C++
// Compile-time optimizes to a dot product
einsum(0.0,  Indices{}, &e_ccd,
       0.25, Indices{i, j, a, b}, new_t_oovv,
             Indices{i, j, a, b}, g_oovv);
```

## ComputeGraph: capture, optimize, replay

The eager calls above run immediately.
ComputeGraph is the preferred way to write C++ workflows: the same operations are recorded into a graph, optimized once by a pass pipeline, and executed as many times as needed.

Capture and optimize an AO-to-MO integral transformation, a chain of four contractions whose intermediates the graph owns:

```C++
#include <Einsums/ComputeGraph.hpp>

namespace cg = einsums::compute_graph;

cg::Graph graph("ao_to_mo");

// Deferred intermediates: the pipeline sizes, allocates, and frees them.
auto &t1 = graph.declare_tensor<double, 4>("t1", nmo, nao, nao, nao);
auto &t2 = graph.declare_tensor<double, 4>("t2", nmo, nmo, nao, nao);
auto &t3 = graph.declare_tensor<double, 4>("t3", nmo, nmo, nmo, nao);
{
    cg::CaptureGuard const guard(graph);
    cg::einsum("iqrs <- pqrs ; pi", &t1, g, C);   // recorded, not executed
    cg::einsum("ijrs <- iqrs ; qj", &t2, t1, C);
    cg::einsum("ijks <- ijrs ; rk", &t3, t2, C);
    cg::einsum("ijkl <- ijks ; sl", &mo, t3, C);
}
auto pm = cg::PassManager::create_default();
graph.apply(pm);      // plan the chain, allocate intermediates, schedule frees
graph.execute();      // run the optimized graph
```

The optimizer materializes each intermediate at its planned size and frees it after its last consumer, so peak memory tracks one live intermediate rather than all three.

Inside a capture, `cg::` mirrors the eager linear-algebra API, so BLAS/LAPACK calls, elementwise updates, and einsums all record into one graph:

```C++
cg::Graph graph("mixed");
{
    cg::CaptureGuard const guard(graph);
    cg::gemm<false, false>(1.0, A, B, 0.0, &C);  // C = A B
    cg::scale(0.5, &C);                          // C *= 0.5
    cg::axpy(1.0, C, &D);                        // D += C
    cg::syev(&D, &evals);                        // eigenvectors overwrite D, eigenvalues fill evals
}
graph.execute();
```

Captured tensors are bound by reference, so replaying after updating the inputs recomputes with the new data.
That makes iterative methods natural: capture the loop body once, then execute per cycle.

```C++
cg::Graph scf_iter("scf_iteration");
{
    cg::CaptureGuard const guard(scf_iter);
    cg::einsum("pq <- pqrs ; rs", 0.0, &G, 2.0, g, D);   // G  = 2 J
    cg::einsum("pq <- prqs ; rs", 1.0, &G, -1.0, g, D);  // G -= K
}
auto pm = cg::PassManager::create_default();
scf_iter.apply(pm);   // StreamContractionFusion builds G in one pass over g

while (!converged) {
    update_density(&D);      // write new values into the captured tensor
    scf_iter.execute();      // replay the optimized graph on the new D
    // ... build F from H + G, diagonalize, check convergence ...
}
```

The pass pipeline includes common-subexpression elimination, dead-node elimination, GEMM batching, permute fusion, scale absorption, memory planning, and more.
Automatic GPU placement and distributed execution (both works in progress) are also graph passes: eager calls run on the host CPU, and capturing a workflow is what makes it eligible for GPU or multi-rank execution as those backends mature.
Operations whose returning forms would need a value immediately (`dot(A, B)`, `syev_eig`, `svd`) throw during capture; their pointer-writer forms record instead, keeping every result inside the graph.

## Performance

The first figure answers "why Einsums?": one Fock build (G = 2J − K from the two-electron integrals and the density matrix) through five execution strategies.
The gray family is hand code: straightforward unfused serial loops, the same loops fused into a single cache-ordered nest, and that fused nest parallelized with OpenMP.
The serial pair shows that fusion alone buys almost nothing on one core (a single core cannot saturate memory bandwidth, so halving the traffic does not help); the fused OpenMP nest - the way a careful C programmer would write it - is the strong baseline.
Writing the same math as two `einsum` calls trades that hand fusion for notation: each contraction runs on the measured-best engine, but the integrals are streamed twice, so eager einsum lands near serial hand code on this bandwidth-bound workload, a few-fold above the fused OpenMP nest.
Capturing the two calls as a ComputeGraph closes the gap and then some: the StreamContractionFusion pass sees that both contractions read the same tensor and fuses them into one storage-order pass feeding both accumulators - matching the hand-fused loops at small sizes and beating them at large ones, with no fusion written by the programmer.

The einsum line in the figure is exactly this code:

```cpp
using namespace einsums::tensor_algebra;
using namespace einsums::index;

// G = 2 J - K, written as the two contractions it is.
einsum(0.0, Indices{mu, nu}, &G, 2.0, Indices{mu, nu, lambda, sigma}, TEI, Indices{lambda, sigma}, D);
einsum(1.0, Indices{mu, nu}, &G, -1.0, Indices{mu, lambda, nu, sigma}, TEI, Indices{lambda, sigma}, D);
```

The stream-fused line is the same two contractions captured into a graph; the default optimization pipeline does the fusion.

```cpp
namespace cg = einsums::compute_graph;

cg::Graph graph("fock");
{
    cg::CaptureGuard const capture(graph);
    cg::einsum("i,j <- i,j,k,l ; k,l", 0.0, &G, 2.0, TEI, D);
    cg::einsum("i,j <- i,k,j,l ; k,l", 1.0, &G, -1.0, TEI, D);
}
graph.optimize(); // StreamContractionFusion merges both into one pass over TEI
graph.execute();
```

![why Einsums](/docs/sphinx/_static/index-images/why_einsums.png)

This figure is regenerated by `devtools/profiling/plot_why_einsums.py` (driver: `profile_strategies`).

The second figure measures the capture-once/replay-many pattern on a workload of 200 independent small matrix multiplications, the shape that dominates block-sparse and tiled tensor codes.
The eager loop pays per-call dispatch for every multiplication.
In the captured graph, ComputeGraph's GEMMBatching pass fuses all 200 contractions into a single `gemm_batch` node, so one replayed call amortizes the dispatch and runs the batch members in parallel.
Today `gemm_batch` is an OpenMP-parallel loop over the per-matrix GEMMs; wiring in a vendor batched kernel (such as MKL's `cblas_dgemm_batch`) would see further gains with no pass changes.

![eager vs ComputeGraph](/docs/sphinx/_static/index-images/eager_vs_graph.png)

For large, kernel-dominated operations the two paths converge - graph capture adds no measurable overhead when the passes find no structure to exploit - so capturing is never a penalty.
Both figures are regenerated by `devtools/profiling/profile_compare.py` and `devtools/profiling/profile_graph.py`, which print the measured timings as they draw; the plots above come from an Apple-silicon Mac (macOS, arm64).

The third figure shows the PackedGemm dispatch engine on scrambled contractions, index patterns that do not map directly onto a BLAS call, such as the CCSD-like ring term in the title, across three engine generations.
The generic nested loops handle any pattern but leave one to two orders of magnitude on the table.
Sort+GEMM recovers most of it: permute both operands into canonical order, run one vendor GEMM, permute the result back - at the price of full permuted copies of the operands on every call.
The PackedGemm engines instead pack cache-sized blocks and contract them on the best kernel the machine offers, on Apple M4, SME FMOPA outer-product tiles for real types and the same real tiles via the 1m method for complex types, selected at runtime through the SIMD dispatch ladder.
The result is faster for every supported datatype and allocates no operand-sized temporaries at all; shapes where Sort+GEMM or the generic loops remain faster (batched layouts, small outer products, matrix-vector forms) still dispatch to those engines, each preference backed by measurement.

![PackedGemm dispatch](/docs/sphinx/_static/index-images/packed_gemm_dispatch.png)

This figure is regenerated by `devtools/profiling/plot_packed_dispatch.py`.

## NumPy-style API

The `einsums` Python package, built with `EINSUMS_BUILD_PYTHON=ON`, wraps the same engine with a NumPy-shaped surface, so tensor code reads like NumPy while dispatching to Einsums' own kernels.
Tensors implement the buffer protocol, so `np.asarray(t)` is a zero-copy view.

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

The same code composes inside a ComputeGraph capture, where each operation is recorded into a graph that is then optimized and executed.
Operators, `.T`, indexing, and reductions are all capture-aware:

```python
import einsums.graph as cg

g = cg.Graph("workflow")
with cg.capture(g):
    G = A.T @ A          # gemm on a graph-registered transpose view -> (4, 4)
    e = (A * A).sum()    # reduction -> a 1-element graph tensor
g.execute()              # run the optimized graph
```

For a full density-fitted MP2 written this way and checked against Psi4, see the eager version in
[`examples/psi4-bridge/df_mp2_numpy_style.py`](examples/psi4-bridge/df_mp2_numpy_style.py)
and the captured version in
[`df_mp2_graph_numpy_style.py`](examples/psi4-bridge/df_mp2_graph_numpy_style.py).
