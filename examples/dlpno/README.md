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
Every setup phase is one or a few graphs replayed under the OpenMP executor, because the parallelism worth having there is across pairs, with each pair's BLAS call left serial underneath.
`compute_pno_overlaps` builds 4056 pair-basis overlaps from 20 captured nodes: each pair's PNO transform is scattered onto the full PAO axis (which makes the domain restriction implicit and deletes a per-coupling `S_pao` gather that was 42% of the phase), and the couplings are then grouped by the bucket pair `(ij, partner)` and emitted as one `cg.batched_gemm` per group.
`pno_transform` is three graphs of ~1500, ~270 and ~1100 nodes, split where the truncation decision has to come back to the host.

Node count is a real constraint in this phase, not just flops: capture costs tens of microseconds a node and these phases run once, so a form that emits one node per coupling can spend more building the graph than the eager version spends computing.
That is what makes the bucketing pay twice - it is what lets the residual batch, and it is what makes every coupling in a group agree on the single m/n/k and leading dimension `gemm_batch` takes.

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

The dense `(Q|mn)` build is inside the timed region, and it did not used to be.
`from_psi4()` ran before the clock started, which excluded the port's single largest DF cost while psi4's figure included generating its own integrals - so the difference this section calls out as being in the port's disfavour was not actually being counted. It is 0.24 s.
Every number below therefore reads worse than the ones this file carried before, against an unchanged port.

Whole calculation, ethanol/cc-pVTZ, SCF excluded, 4 P-cores + 6 E-cores:

| | psi4 | this port | |
| --- | --- | --- | --- |
| 1 thread | 3.06 | **2.78** | **1.10x faster** |
| 10 threads | 1.02 | 1.49 | 1.5x |

**Single threaded the port is ahead of the C++ on the larger basis; threaded it is behind.**

The LMP2 iteration itself is at parity everywhere, threaded or not: 0.1214 vs 0.1242 s on one thread, 0.0438 vs 0.0404 threaded.
That is the phase the ComputeGraph and the batching cover, and it is not where the deficit lives.

Timings on this machine move by 10-15% with background load; the ratios are stable across runs, the absolute numbers less so.

### Parallelizing the setup phases

psi4 runs its setup under `#pragma omp parallel for` over pairs.
The port cannot simply do the same from Python: driving those loops from a `ThreadPoolExecutor` returns silently wrong numbers against the OpenMP-built OpenBLAS that conda resolves by default, because that build indexes its internal scratch by `omp_get_thread_num()`, which is 0 on every caller-created thread.
See [BLAS threading](../../docs/sphinx/building/blas_threading.rst).

The way that *is* safe is to hand the independent work to the graph and run it under the OpenMP executor, so the parallelism is a real OpenMP team and each node stays serial underneath.
`DLPNOBase._run` is that call, and every setup phase goes through it:

* `precompute_fits` captures every distinct domain's fitting solve into one graph. There are only 23 of them at ethanol/cc-pVTZ, but they were 31% of the phase, because the solve is per *domain* and carries `naocc * npao` right-hand sides.
* `pno_transform` captures its three stages rather than only the two eigendecompositions inside them. This is the one that matters, and not for the reason the name suggests: the phase issues ~1400 small GEMMs against ~190 gathers, and eagerly issued they run serially no matter how many threads the BLAS has, each call far too small to fill them. Captured, stage 1 scales 5.0x on ten threads and stage 3 4.4x.
* `compute_pno_overlaps` emits `cg.batched_gemm` per bucket pair instead of concatenating each pair's partners side by side. The concatenation was the phase's dominant cost and none of it was arithmetic: laying out a pair's partners re-copies every partner block once per pair coupling to it, 335 MiB of allocate-and-memcpy at ethanol/cc-pVTZ over blocks that already exist in 14 MiB of `X_pad`. It was also the one part that could not be parallelized, being numpy slice assignment on the calling thread.

Ten threads at cc-pVTZ, against psi4. The "before" column is the eager per-pair
form these phases replaced, on the same DF timing:

| phase | before | after | psi4 |
| --- | --- | --- | --- |
| PNO Transform | 0.446 (1.7x) | **0.35** (1.1x) | 0.31 |
| PNO Overlaps | 0.143 (3.0x) | **0.085** (1.5x) | 0.058 |

Single-threaded times are unchanged, which is the point: nothing got cheaper, it got issued in parallel.
Setup scaling from 1 to 10 threads went 2.0x -> 2.9x for PNO Transform and 1.7x -> 2.4x for PNO Overlaps, against psi4's 4.8x.

**The graph's execution already scales as well as psi4's OpenMP; the setup deficit is entirely the serial tax around it.**
Splitting each phase into `execute()` and everything else shows `execute` scaling 4.9x on PNO Transform against psi4's 4.8x, while a remainder of ~0.15 s across the setup is *identical* at one thread and at ten.
That remainder is graph capture, tensor allocation, and the two eager eigendecompositions the truncation decision needs - roughly 40/40/20 - and at ten threads it is 41% of the whole setup.
It is Amdahl, not a threading problem: no number of cores takes the setup below it.

The dominant gap now is elsewhere.
DF Ints is 7-8x, and 0.24 s of its 0.34 s is `mints.ao_eri` building the dense `(Q|mn)`: 98 MiB of integrals of which Schwarz screening would drop about one, because ethanol is small and compact enough that nearly every AO pair is significant.
psi4's DLPNO builder screens by *domain* instead - for each auxiliary atom, only the AOs on atoms in its extended LMO/PAO domain - and does the whole phase in 0.05 s.
That is the tier that turns O(naux nbf^2) into something linear, it is the one that gets worse with system size, and psi4 does not expose it: `DFHelper` is reachable from Python but applies J^-1/2 and `set_metric_pow` is not bound, which a method that fits per domain cannot use.

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

The kernel behind them was worth a second look, and the answer was not the expected one.
`gather`, `scatter` and `scatter_add` were each their own odometer rebuilding both offsets from scratch per element, which costs a multiply-add per axis per element and, worse, hides the shape these callers hit most: because there is no wildcard, a whole axis is spelled `range(n)`, and when that is the fastest axis and both sides step by one, the selection along it is a contiguous block that the element loop was copying a multiply-add at a time.
They now share `detail::for_each_selection_run`, which hoists the per-axis offsets into tables and collapses that case into one `copy_n`.
On the shapes this port issues: 1.7x on `S_pao[dom, dom]`, 2.0x on the fit right-hand side, and 4.5x on `X[:, :keep]`, which is the contiguous case.

This is worth stating plainly because the SIMD module looks like it should be the answer here and is not.
`simd::gather` is a strided-lane load - `base[0], base[stride], ...` for a kernel's inner loop - not an index-list gather, and the psABI rung ladder is x86-only, so on an arm64 host the dispatch collapses to a single native TU regardless.
The gathers were never the bottleneck either: `pno_transform` issues ~1400 small GEMMs against ~190 gathers.
What was actually left on the table was index arithmetic and a missed `memcpy`, and both fixes are scalar.
Even after them the kernel is 7-18x off a straight `memcpy` of the same payload, which is the honest ceiling for a scattered read.

The remaining numpy in the package is a useful map of what else is missing, in rough order of what this workload would pay for.
Bookkeeping numpy (integer index maps like `i_j_to_ij`, scalars, and the psi4 buffers arriving through the bridge) is excluded — that is interop, not a gap.

Check before assuming something is missing: several of these already exist, and the real gap is narrower than "einsums cannot do X".
`linalg.abs`, `scale`, `direct_product` and `element_transform` all capture, as do the pointer-writer `sum`, `max`, `dot`, `trace` and `norm` (the returning forms throw during capture by design).

1. **Element-wise `sqrt` — ADDED.** `cg::sqrt` is the element-wise partner to `abs`; `linalg.pow` is a *matrix* power by eigendecomposition and is not it. The DOI finish is now `abs` then `sqrt` with no host round-trip. Real-only: the square root of a negative real is a branch choice the caller should make, so negative input throws.
2. **Axis-wise reduction — ADDED.** `cg::sum_axes`, numpy's `A.sum(axis=...)`. `linalg.sum` still covers whole-tensor-to-scalar. Assigns rather than accumulates, so a replay does not add to the previous execution.
3. **Reshape — ADDED.** `cg::reshape`, with a mandatory `row_major` argument. Not a formality: einsums is column major and numpy's default is C, and picking the wrong walk transposes blocks silently instead of raising. It copies rather than aliasing.
4. **Diagonal extract — ADDED.** `cg::diagonal`, numpy's `np.diag` on a matrix; rectangular input is fine, the shorter axis wins.
5. **Accumulating scatter — ADDED.** `cg::scatter_add`, numpy's `np.add.at`. Repeated indices are *allowed* here and rejected by `scatter`: under a plain write two writes to one element depend on loop order, under an accumulation they do not.
6. **Concatenate / stack along an axis.** STILL MISSING. DIIS flattens every pair block into one vector and back; the dipole code stacks three components into an `(naocc, 3)`.
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

1. **Screen the `(Q|mn)` build.** The largest single item left, 0.24 s, and the only one that gets structurally worse with system size. Needs an integral interface psi4 does not currently offer from Python: `MintsHelper.ao_eri` is dense, `DFHelper` is Schwarz-screened but metric-locked (`set_metric_pow` is in the header, not in the bindings), and the domain-screened builder DLPNO itself uses is private to the C++ class. Binding `set_metric_pow` is one line in `export_fock.cc` and would unlock the middle tier, worth ~1.9x on this phase.
2. **Per-node capture cost.** The ~0.15 s serial setup tax is ~40% graph capture. Restructuring does *not* reach it - see below - so the lever is the per-node cost itself (~10 us here, against 1.85 us for a C++ caller) or moving the phase out of Python. `DESIGN-cpp-hybrid.md` sketches the latter.

Three attempts at that tax were measured and rejected; the patches are under `parked/` so they are not re-derived.

* **Batching `pno_transform` by domain group.** Node count 2977 -> 1075, capture -12 ms, execute +29 ms. Net loss. The padding is most of it - a group pads to its widest PNO count, and they run 5..92 - plus eight graph barriers needed because a store written through per-pair slices and then read whole is not reliably ordered.
* **`declare_tensor` for the captured scratch.** Capture -30 ms, execute +26 ms: Materialization adds ~2651 lifecycle nodes whose scheduling costs back the allocation saving. Also settles the "allocation is overhead" theory - `create_zero_tensor` is 5.30 us for a 92x92 block against a 1.57 us empty-call floor, so most of it is the memset, which no allocator change avoids.
* **Binding `cg::parallel_for`.** Not attempted after analysis. The body would be a Python callable on TaskPool workers; `execute()` has already released the GIL, so every iteration re-acquires it and the loop serializes - and it would run BLAS on `std::thread`, which the OpenMP-built OpenBLAS miscomputes. It is a C++-caller feature.

The pattern across all three: moving work from capture into the graph costs about what it saves, because per-node graph overhead is comparable to the per-operation Python overhead it displaces.

Then, unchanged from before:

3. **One graph for the whole iteration**, using a loop node, with DIIS either captured or hoisted.
4. **DLPNO-CCSD** (`ccsd.cc`), which is where the graph work gets interesting: the residuals are much larger and the intermediates are shared across pairs.
5. **DLPNO-(T)** (`triples.cc`).
