# DLPNO on the einsums ComputeGraph

A port of psi4's DLPNO module (`psi4/src/psi4/dlpno`) with the tensor algebra expressed in einsums and the iterative solvers captured as ComputeGraphs.
DLPNO is a good fit for deferred execution: it is thousands of small dense operations whose shapes and dependency pattern are fixed for a whole calculation and change only in their values from iteration to iteration, which is exactly the capture-once, replay-many shape.

The module split follows psi4's, so the two can be read side by side.

| this package | psi4 | status |
| --- | --- | --- |
| `dlpno/sparse.py` | `dlpno/sparse.cc` | ported |
| `dlpno/base.py` | `dlpno/dlpno.cc` (class `DLPNO`) | orbitals, PAOs, DF integrals, PNO transform ported; screening not yet |
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
At `T_CUT_PNO = 1e-8` the PNO statistics and truncation correction line up with psi4's:

| | psi4 DLPNO-MP2 | this port |
| --- | --- | --- |
| PNOs per pair (avg/min/max) | 18 / 9 / 19 | 18.1 / 9 / 19 |
| PNO truncation correction | -1.5693e-8 | -1.5921e-8 |
| local MP2 correlation energy | -0.204089507549 | -0.204089517862 |

(water, cc-pVDZ, no frozen core.)

The two do not agree to machine precision, and should not: psi4 additionally screens LMO pairs and shrinks the PAO and auxiliary domains, so it discards strictly more.
The port's truncation error is consequently the smaller of the two on every molecule tested, and the driver asserts that ordering rather than equality.
Once screening lands, this becomes an exact comparison.

## Design notes

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

**Write the coupling as a plain loop and let the pass batch it.**
The residual's Fock coupling is written the way psi4 writes it, one `(pair, k)` at a time, with two conditions attached:

* the einsums must be `einsum`, not `linalg.gemm`. A 2D x 2D -> 2D einsum with one link index carries the `gemm_hint` that `GEMMBatching` groups on; `linalg.gemm` captures as `OpKind::Gemm` and the pass skips it outright. This is easy to get wrong and costs the entire optimization silently.
* the blocks must be padded, per above.

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
At these sizes psi4 screens no LMO pairs (it keeps all `naocc²`, same as the port) and its average PNO counts match the port's, so the two are solving essentially the same problem.
LMP2 seconds per iteration, ethanol:

| | psi4 | this port | |
| --- | --- | --- | --- |
| cc-pVDZ, 1 thread | 0.0256 | 0.0216 | **1.19x faster** |
| cc-pVTZ, 1 thread | 0.1245 | 0.1068 | **1.17x faster** |
| cc-pVTZ, 10 threads | 0.0393 | 0.0408 | 0.96x (parity) |
| cc-pVDZ, 10 threads | 0.0098 | 0.0139 | 0.71x |

**Single threaded the port beats the C++ by ~1.2x.**
Threaded, it reaches parity at cc-pVTZ and is still behind at cc-pVDZ; the gap closes as the system grows, because the thing that limits it is batch size and batch size grows with the pair count.

Getting here from an initial 3.5x deficit took five things.

1. **psi4 clamps the process-wide OpenMP thread count to 1 when imported.** Every einsums BLAS and `gemm_batch` call in the process is then serialized and `OMP_NUM_THREADS` has no effect (a batch measured 172 GFLOP/s standalone, 46 with psi4 imported). `psi4.set_num_threads(n)` sets it for both. Nothing warns.
2. **`GEMMBatching` requires bit-identical `alpha`.** The residual carries a different Fock prefactor on every coupling, fragmenting 8112 contractions into 309 batches of 26. Since `f·S T Sᵀ = sign(f)·(√|f| S) T (√|f| S)ᵀ` and S is constant, folding `√|f|` into the overlaps leaves alpha at exactly ±1.
3. **Capture order:** all the `S T` products first, then the accumulations, so each group sits at one dependency level.
4. **Bucketing pairs by PNO count** (`n_buckets`), padding to a bucket maximum rather than the global one.
5. **Cross-slot accumulators** (`n_accumulators`), below.

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

**No gather by index list.**
Domain restriction is the fundamental operation in DLPNO, and there is no einsums primitive for "take these rows and these columns".
`sparse.submatrix_*` therefore goes through the tensor's numpy view, which means every domain extraction runs eagerly on the host and cannot be captured into a graph.
A gather/scatter op over index lists is the single most valuable thing the library could add for this workload.

**In-place elementwise einsum against a lower-rank operand silently yielded zeros** (fixed).
`einsum("ab <- ab ; b", X, X, f)` was accepted by the aliasing guard, because the aliased operand's index list matches the output's, but returned zeros: both generic algorithms clear C before reading anything, so the aliased operand read back already zeroed.
The equal-rank `"ab <- ab ; ab"` was fine, which is what hid it: that shape is served by the elementwise route, which really does read each element immediately before overwriting it.
The fix snapshots any operand overlapping C in `StringDispatch.hpp`'s generic loop and in `BaseAlgebra.hpp`'s `einsum_generic_algorithm`, so only the aliased case pays a copy and the loops (and their summation order) are untouched.
Regression tests are in `ComputeGraph/tests/unit/EagerParityGaps.cpp`, covering all four generic routes on both the graph and eager paths.

**View/parent aliasing is not a graph dependency** (open, worked around).
A write through `R[:, :, p]` and a read or write of `R` are treated as touching unrelated tensors, so the scheduler is free to order them arbitrarily.
This produces wrong answers with no error, and it is not confined to parallel execution: with the residual's three phases in one graph the energy dot was scheduled at node 201 of 405 under *both* the default and sequential executors.
It is worse under the parallel executors, where even a plain write-then-read is unordered:

```python
views = [R[:, :, p] for p in range(P)]
with cg.capture(g):
    for p in range(P):
        einsums.einsum("ab <- ac ; cb", views[p], A, B)   # write each block
    la.dot(out, R, W)                                     # read the parent
g.set_executor(cg.OpenMPExecutor())                       # or DataflowExecutor
g.execute()          # out != sum(R*W); Sequential and the default are correct here
```

`examples/dlpno/repro_view_aliasing.py` is the standalone reproducer.
This one deserves a real fix: the pair-block layout above is a natural and useful pattern, and it is a silent-wrong-answer trap today.

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

1. **Close the threaded gap at small sizes.** The port is ~1.2x ahead single threaded and at parity threaded on cc-pVTZ, but 1.4x behind on cc-pVDZ, where there is not enough work per batch to fill ten cores. Choosing `n_buckets` automatically from the pair count and thread count would get most of it.
2. **Screening**, in psi4's order: differential overlap integrals (`compute_overlap_ints`, needs a DFT grid, which psi4 exposes to Python), dipole pair energies (`compute_dipole_ints`), then `prep_sparsity` proper. This turns the comparison against psi4's DLPNO-MP2 into an exact one.
3. **One graph for the whole iteration**, using a loop node, with DIIS either captured or hoisted.
4. **DLPNO-CCSD** (`ccsd.cc`), which is where the graph work gets interesting: the residuals are much larger and the intermediates are shared across pairs.
5. **DLPNO-(T)** (`triples.cc`).
