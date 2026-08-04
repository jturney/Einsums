# DLPNO on the einsums ComputeGraph

A port of psi4's DLPNO module (`psi4/src/psi4/dlpno`) with the tensor algebra expressed in einsums and the iterative solvers captured as ComputeGraphs.
DLPNO is a good fit for deferred execution: it is thousands of small dense operations whose shapes and dependency pattern are fixed for a whole calculation and change only in their values from iteration to iteration, which is exactly the capture-once, replay-many shape.

The module split follows psi4's, so the two can be read side by side.

| this package | psi4 | status |
| --- | --- | --- |
| `dlpno/sparse.py` | `dlpno/sparse.cc` | ported |
| `dlpno/base.py` | `dlpno/dlpno.cc` (class `DLPNO`) | orbitals, PAOs, DOI and dipole screening, domains, DF integrals, PNO transform |
| `dlpno/mp2.py` | `dlpno/mp2.cc` (class `DLPNOMP2`) | ported, iterations captured as graphs |
| - | `dlpno/ccsd.cc` (class `DLPNOCCSD`) | not started |
| - | `dlpno/triples.cc` (class `DLPNOCCSD_T`) | not started |

`dlpno/reference.py` and `dlpno/psi4_source.py` have no psi4 counterpart.
They define the psi4-free data contract the port starts from and the single adapter that fills it in, so nothing else in the package knows psi4 exists and neither library is built against the other.
This mirrors the buffer-level bridge in `einsums.interop.psi4`.

## Running

Needs `einsums` and `psi4` importable.
Point `PYTHONPATH` at an in-tree Einsums build and any psi4 install, and use the conda env's Python:

```bash
PYTHONPATH=/path/to/Einsums/build/lib:/path/to/psi4/cmake-build-debug/stage/lib \
    python examples/dlpno/run_pno_mp2.py --molecule water-dimer
```

Useful flags: `--basis`, `--localization {BOYS,PIPEK_MEZEY}`, `--t-cut-pno`, `--buckets`, `--threads`, `--no-optimize` (skip the graph passes), `--no-diis`.

Note `--threads`: importing psi4 clamps the process-wide OpenMP thread count to 1, so einsums runs serial unless it is set, and `OMP_NUM_THREADS` alone will not do it.
The scripts are not wired into CTest or pytest, since they need a psi4 install.

## What is validated

`run_pno_mp2.py` checks the port against two psi4 references at once.

**Untruncated, against canonical DF-MP2.**
With PNO truncation off and full domains, local MP2 in the PAO space *is* canonical MP2, just written in a non-canonical basis, so the correlation energy has to match psi4's DF-MP2 to machine precision.
It does, to 1e-13 or better on every molecule in the driver.
This is the check that pins the port down: the PAO construction, the local density fit, the PNO machinery, the residual, and the solver all have to be right simultaneously for it to pass.

**Truncated, against psi4's own DLPNO-MP2.**
Both sides apply the same three truncations, so this is an exact comparison rather than an ordering check.
At `T_CUT_PNO = 1e-8`, cc-pVDZ, no frozen core:

| molecule | pairs kept | vs psi4 DLPNO-MP2 |
| --- | --- | --- |
| water | 25 / 25 | 5.4e-13 |
| water-dimer | 100 / 100 | 5.3e-13 |
| water-dimer-far | **50 / 100** | 1.2e-12 |
| methanol | 81 / 81 | 2.5e-13 |
| ethanol | 169 / 169 | 1.7e-13 |

Only `water-dimer-far` exercises the pair prescreening: the compact geometries are small enough that psi4 keeps every pair too, so the two monomers are pushed 12 A apart to force the issue.
There the port drops exactly the 50 pairs psi4 drops, and its dipole estimate of what was discarded (-0.0000002432 Eh) matches psi4's `Screened LMO pair energy` (-0.000000243227) to the printed digits.

## Design notes

**The setup phases are captured too, but shaped differently.**
`compute_pno_overlaps` builds 4056 pair-basis overlaps from 338 captured nodes rather than 8112, by scattering each pair's PNO transform onto the full PAO axis (which makes the domain restriction implicit and deletes a per-coupling `S_pao` gather that was 42% of the phase) and concatenating a pair's partners so all its overlaps are one GEMM.
Node count is the constraint here, not flops: capture costs ~38 us a node and this phase runs once, so capturing the per-coupling form would have spent more building the graph than the eager version spent computing. 0.233 s -> 0.090 s.

**Five graphs per iteration.**
`mp2.py` captures the residual prologue, the Fock coupling, the iteration energy, the Jacobi amplitude step, and the antisymmetrized amplitudes as separate graphs, then replays them in that order each iteration.
The split is not cosmetic: see "Phases are separate graphs" below.

DIIS sits between the step and the antisymmetrization, because psi4 extrapolates the amplitudes before rebuilding `Tt`.
It runs on the host: it is a solver detail over flattened buffers rather than tensor algebra, and with every pair in one contiguous store it needs no gathering at all, just a `ravel` view of `T_all`.
It writes back into the same tensors the graphs already reference, so the replay picks the new values up.
Folding the whole loop into one graph with a loop node is the obvious next step.

**Everything per-pair lives in one contiguous store, padded to a uniform shape.**
This is the layout decision the whole calculation turns on, and it is the one psi4 makes differently.
psi4 keeps each pair's block in its own `SharedMatrix` sized to that pair's PNO count, which leaves the residual as a loop of individually dispatched small GEMMs over operands that share neither shape nor allocation.
Here each per-pair quantity is one `(npno_max, npno_max, n_pairs)` tensor.

Two things follow.
Everything elementwise over pairs (the amplitude update, the antisymmetrization, the energy) becomes a single operation on the whole store rather than one per pair: the Jacobi step is 1 node and the antisymmetrization is 2, whatever the molecule.
And because every coupling GEMM now has the same shape, `GEMMBatching` can collapse all `n_pairs * naocc` of them into a handful of `blas::gemm_batch` calls.

Padding is inert: integrals, amplitudes and overlaps are zero outside each pair's logical block and the energy denominators are one, so padded components stay zero for the life of the calculation.

**Emit the batch directly with `cg.batched_gemm`.**
The residual's Fock coupling groups its `(pair, k)` contractions by shape, sign and dependency level and emits one `cg.batched_gemm` per group.

It used to emit one einsum per coupling and let `GEMMBatching` fuse them, which reaches the same node but only after building the graph one node at a time: 8112 nodes for a graph that ends up ~30 wide, and capture is not free.
Emitting the fused form directly took the ethanol capture from 8215 nodes to 335.
The blocks must still be padded, per above, since one `gemm_batch` call takes a single m/n/k and a single leading dimension for the whole batch.

If you do write the per-contraction form, use `einsum` and not `linalg.gemm`: a 2D x 2D -> 2D einsum with one link index carries the `gemm_hint` that `GEMMBatching` groups on, while `linalg.gemm` captures as `OpKind::Gemm` and the pass skips it outright, silently costing the whole optimization.

Hand-stacking k into a trailing batch axis and calling one rank-3 batched einsum per pair was tried first and is *worse*: 2.1x against 4.2x on the water dimer.
It fixes the batch at one pair's worth of k, where the pass batches across every pair and every k at once.
The general lesson is to make the data uniform and leave the batching to the optimizer, not to pre-commit to a batch shape.

`bench_batching.py` measures this over identical data, on the coupling term alone:

| configuration | nodes | ms/replay | vs psi4 layout |
| --- | --- | --- | --- |
| padded, no passes | 3440 | 23.6 | 0.45x |
| padded + passes (shipped) | 3440 -> 254 | 5.9 | **1.78x** |
| ragged + passes (psi4 layout) | 3440 -> 1673 | 10.5 | 1.00x |

(water dimer/cc-pVDZ, 100 pairs, 1720 couplings, PNO counts 5..29 padded to 29. Methanol gives 1.33x.)

Note the first row: padding *on its own* is a 2.2x loss, because it does real extra flops on the padded elements.
It only pays once the pass can exploit the uniformity, and then it wins.
Bucketing pairs by PNO count would get the batching without the wasted flops, and is the obvious next refinement.

## Performance against psi4

`bench_vs_psi4.py` runs psi4's native C++ DLPNO-MP2 in a subprocess and this port in process.
Both sides screen identically, which the script checks by comparing correlation energies before reporting any timing.
It also sets psi4's `R_CONVERGENCE` to match the port's: psi4's LMP2 default is 1e-6, two orders looser, which costs it two iterations in exactly the phase being compared.

Whole calculation, ethanol, SCF excluded, 4 P-cores + 6 E-cores:

| | psi4 | this port | |
| --- | --- | --- | --- |
| cc-pVTZ, 1 thread | 2.853 | **2.456** | **1.16x faster** |
| cc-pVDZ, 1 thread | 0.452 | 0.615 | 1.4x |
| cc-pVTZ, 10 threads | 0.889 | 1.280 | 1.4x |
| cc-pVDZ, 10 threads | 0.146 | 0.470 | 3.2x |

**Single threaded the port is ahead of the C++ on the larger basis; threaded it is behind, and by a widening margin as the problem gets smaller.**

The LMP2 iteration itself is at parity everywhere, threaded or not: 0.1157 vs 0.1141 s at cc-pVTZ on one thread, 0.0404 vs 0.0405 threaded.
That is the phase the ComputeGraph and the batching cover, and it is not where the threaded deficit lives.

### Parallelizing the setup phases

psi4 runs its setup under `#pragma omp parallel for` over pairs.
The port cannot simply do the same from Python: driving those loops from a `ThreadPoolExecutor` returns silently wrong numbers against the OpenMP-built OpenBLAS that conda resolves by default, because that build indexes its internal scratch by `omp_get_thread_num()`, which is 0 on every caller-created thread.
See [BLAS threading](../../docs/sphinx/building/blas_threading.rst).

The way that *is* safe is to hand the independent work to the graph and run it under the OpenMP executor, so the parallelism is a real OpenMP team and each node stays serial underneath:

* `precompute_fits` captures every distinct domain's fitting solve into one graph. There are only 23 of them at ethanol/cc-pVTZ, but they were 31% of the phase, because the solve is per *domain* and carries `naocc * npao` right-hand sides.
* `pno_transform` is three stages rather than one loop, so the two eigendecompositions per pair batch into two graphs, with the data-dependent truncation decision in between. Issued one at a time these got *worse* with threads (219 ms serial against 329 ms on ten threads), each small `syev` fighting the BLAS's own threading.

At cc-pVTZ on ten threads PNO Transform went 0.779 -> 0.438 s against psi4's 0.275, so 2.7x behind became 1.6x.
PNO Overlaps was already parallel inside `gemm_batch` and moved only 0.150 -> 0.141.
What is left of the threaded gap is spread across PNO Overlaps (3.1x), the dense `(Q|mn)` build (2.3x) and the host-side gathers, which are not parallel at all.

A note on the energies at cc-pVTZ: the two agree to 2.3e-8 rather than the ~1e-13 seen at cc-pVDZ.
Every input to the comparison matches psi4 exactly - PAO and auxiliary domains (average, min *and* max, per LMO and per pair), pairs screened, PNOs per pair (54/5/92), and the PNO truncation correction to 1.4e-10.
What is left is the linear-dependence tie-break: cc-pVTZ puts PAO-domain overlap eigenvalues near `S_CUT`, and retaining four more vectors (at `s_cut` 1e-9) moves the correlation energy by 1.6e-7, so a single vector is worth ~4e-8.
The comparison tolerance is therefore scaled to 1% of the PNO truncation correction rather than fixed, since a genuine domain disagreement is orders of magnitude larger.
Getting there took, in order of size:

1. **psi4 clamps the process-wide OpenMP thread count to 1 when imported.** Every einsums BLAS and `gemm_batch` call in the process is then serialized and `OMP_NUM_THREADS` has no effect. `psi4.set_num_threads(n)` sets it for both. Nothing warns.
2. **Per-domain work was being redone per pair.** The orthocanonical PAO basis and the local fit's metric factorization depend only on the domain, and with screening off there is one domain: 182 of 364 eigendecompositions and all 91 linear solves computed the same answer repeatedly. Memoized by domain identity, PNO Transform went 0.220s -> 0.065s at cc-pVDZ.
3. **`GEMMBatching` requires bit-identical `alpha`**, and the residual carries a different Fock prefactor per coupling. Folding `sqrt(|f|)` into the (constant) overlaps leaves alpha at exactly +/-1.
4. **`cg.batched_gemm` emits the fused node directly** instead of emitting one node per contraction for the pass to collapse: 8215 captured nodes -> 335.
5. **Two quadratics in capture** (storage-alias linking per registration, and a linear scan to answer "is this object registered?") reduced capture+optimize from 0.419s to 0.082s.
6. **Cross-slot accumulators and PNO-count bucketing**, below.

### Why the two knobs interact

### Why the two knobs interact

A pair's couplings all accumulate into its residual, so they serialize: with one accumulator the batch is capped at *one coupling per pair per dependency level*, and bucketing then makes things worse by splitting the pairs further.
Spreading a pair's couplings over `G` accumulators (coupling `c` goes to accumulator `c % G` at level `c // G`) cuts the levels by `G` and multiplies the batch by `G`.
The partials are folded back by `G` rank-3 `axpby` nodes per bucket, a count that does not grow with the molecule.
Level 0 assigns rather than accumulates, so the accumulators never need re-zeroing.

Ethanol/cc-pVDZ, whole-iteration ms:

| buckets | G=1 | G=8 | G=32 | |
| --- | --- | --- | --- | --- |
| 1 | 46.8 | 47.3 | 49.0 | 1 thread |
| 4 | 30.6 | **24.1** | 24.0 | 1 thread |
| 6 | 21.9 | **21.6** | 22.6 | 1 thread |
| 1 | 14.6 | **13.9** | 15.8 | 10 threads |
| 4 | 23.7 | **14.5** | 14.6 | 10 threads |
| 6 | 26.0 | **16.1** | 16.4 | 10 threads |

The accumulators are what make bucketing viable when threaded: at B=4 they take the 10-thread iteration from 23.7 ms to 14.5.
Beyond `G ≈ 8` there is nothing left to win, and the extra partials start costing memory traffic.

The right bucket count depends on system size and thread count, which is why it is a knob rather than a constant.
Single threaded, flops dominate and more buckets is monotonically better.
Threaded, the batches have to stay large enough to fill the cores: at cc-pVDZ (169 pairs) B=1 wins, while at cc-pVTZ the same molecule prefers B=4 by 2.3x.
Defaults are `n_buckets=4`, `n_accumulators=8`.

Graph capture and optimization is excluded from the per-iteration figures and reported separately: ~0.4 s for ethanol, paid once, with no psi4 equivalent.
At ten iterations it is the larger half of the port's LMP2 wall time, so it matters in practice even though it does not belong in a per-iteration number.

## Gaps found in Einsums

**Gather and scatter by index list — FIXED.**
Domain restriction is the fundamental operation in DLPNO, and einsums had no primitive for "take these rows and these columns", so `sparse.submatrix_*` went through the tensor's numpy view and every domain extraction ran eagerly on the host, uncapturable.
`cg::gather` and `cg::scatter` now cover it, and `sparse.submatrix` / `sparse.scatter_into` are thin wrappers over them.
Note `gather` has no whole-axis wildcard: an empty index list selects nothing, because an empty domain silently becoming the full axis is a wrong answer rather than an error.

The remaining numpy in the package is a useful map of what else is missing, in rough order of what this workload would pay for.
Bookkeeping numpy (integer index maps like `i_j_to_ij`, scalars, and the psi4 buffers arriving through the bridge) is excluded — that is interop, not a gap.

Check before assuming something is missing: several of these already exist, and the real gap is narrower than "einsums cannot do X".
`linalg.abs`, `scale`, `direct_product` and `element_transform` all capture, as do the pointer-writer `sum`, `max`, `dot`, `trace` and `norm` (the returning forms throw during capture by design).

1. **`pow` does not capture.** It exists — `linalg.pow(A, alpha, cutoff)`, in place — but throws inside a capture, so `sqrt` as `pow(A, 0.5)` is only available eagerly. Making it capture-aware is a smaller change than a new op and would cover the DOI finish, which is `sqrt(abs(...))` over a whole matrix and whose `abs` half already captures.
2. **Reductions along an axis.** `linalg.sum(result, A)` exists and captures, but sums the *whole* tensor into a scalar. The Mulliken populations need `share_u.sum(axis=1)`, and there is no axis-wise form.
3. **Reshape.** `_fit_operands` gathers `(Q|iu)` for a domain and then reshapes to feed one `gesv` with `naocc * npao` right-hand sides. The gather captures now; the reshape does not, so the assembly still breaks the graph.
4. **Diagonal extract and set.** `np.diag` appears in the PAO normalization and the dipole centroids.
5. **Accumulating scatter.** The Mulliken populations are `np.add.at`, which is exactly scatter-with-accumulate. `cg::scatter` deliberately rejects repeated indices rather than silently picking a winner, so this is a separate op and not a relaxation of that rule.
6. **Concatenate / stack along an axis.** DIIS flattens every pair block into one vector and back; the dipole code stacks three components into an `(naocc, 3)`.
7. **Masked select.** `np.where` guards a divide-by-zero in the population split. Lowest value of the seven: it is one guarded division, not a hot path.

**In-place elementwise einsum against a lower-rank operand silently yielded zeros** (fixed).
`einsum("ab <- ab ; b", X, X, f)` was accepted by the aliasing guard, because the aliased operand's index list matches the output's, but returned zeros: both generic algorithms clear C before reading anything, so the aliased operand read back already zeroed.
The equal-rank `"ab <- ab ; ab"` was fine, which is what hid it: that shape is served by the elementwise route, which really does read each element immediately before overwriting it.
The fix snapshots any operand overlapping C in `StringDispatch.hpp`'s generic loop and in `BaseAlgebra.hpp`'s `einsum_generic_algorithm`, so only the aliased case pays a copy and the loops (and their summation order) are untouched.
Regression tests are in `ComputeGraph/tests/unit/EagerParityGaps.cpp`, covering all four generic routes on both the graph and eager paths.

**View/parent aliasing was not a graph dependency** (fixed).
A write through `R[:, :, p]` and a read or write of `R` were treated as touching unrelated tensors, so the scheduler was free to order them arbitrarily: wrong answers, no error, and not confined to parallel execution.
With the residual's phases in one graph the energy dot was scheduled at node 201 of 405 under *both* the default and sequential executors; under the parallel executors even a plain write-then-read was unordered:

```python
views = [R[:, :, p] for p in range(P)]
with cg.capture(g):
    for p in range(P):
        einsums.einsum("ab <- ac ; cb", views[p], A, B)   # write each block
    la.dot(out, R, W)                                     # read the parent
g.set_executor(cg.OpenMPExecutor())                       # or DataflowExecutor
g.execute()          # out != sum(R*W); Sequential and the default are correct here
```

`cg::view()` always set `TensorHandle::aliases`, but a view sliced *outside* a capture never goes through it: Python's capture-aware `__getitem__` falls through to the eager slice, and the view reaches the graph as an ordinary operand on first use with `aliases == 0`.
That is exactly the pattern this port needs, because a captured view must outlive the graph.
`Graph::link_alias_storage` now recovers the relationship from the registration-time data pointer and strides, in both directions (a parent is routinely registered after the views built from it), and reconstructs the view's box so disjoint slices still do not serialize.
`examples/dlpno/repro_view_aliasing.py` is the standalone reproducer, and `ComputeGraph/tests/unit/View.cpp` carries the regression test.

**`linalg.gemm` is invisible to `GEMMBatching`.**
It captures as `OpKind::Gemm`, and the pass only groups `Einsum` nodes carrying a `gemm_hint`.
Writing the obvious `linalg.gemm` instead of the equivalent `einsum` costs the whole optimization with nothing to indicate it.
Either `gemm` should carry the hint, or the pass should grow a `Gemm` case.

**Strided views cannot be captured.**
`T[:, :, j::naocc]` works eagerly but raises `only step=1 slices are supported` under capture.
That ruled out indexing the `{T_kj : k}` set directly and shaped the first (later abandoned) design around a transposed second copy of the amplitudes.

**`np.asarray` on a tensor is F-contiguous.**
So `.reshape(-1)` silently copies rather than viewing, which is easy to write by accident and hard to see: an early version of the DIIS here extrapolated into throwaway buffers and quietly degraded to unaccelerated Jacobi (48 iterations instead of 11) with no error anywhere.
`ravel(order="F")` is the view; `mp2.py` asserts it.

## Next steps

1. **The host-side gather seam.** With the fits and eigendecompositions on the graph, what is left of the setup cost is largely `np.ix_` gathers and `from_numpy` copies building each domain's operands, which no executor can parallelize because they are not captured. An einsums gather primitive would let the whole per-pair setup be one graph.
2. **Batch the PNO transform.** Screening made this worth doing: pairs used to share a single domain, so memoization removed the duplication outright, whereas now there are many distinct domains but many pairs per domain. The per-pair exchange build and semicanonical MP2 step are uniform within a domain and could go through `cg.batched_gemm` the way the residual couplings do.
3. **One graph for the whole iteration**, using a loop node, with DIIS either captured or hoisted.
4. **DLPNO-CCSD** (`ccsd.cc`), which is where the graph work gets interesting: the residuals are much larger and the intermediates are shared across pairs.
5. **DLPNO-(T)** (`triples.cc`).
