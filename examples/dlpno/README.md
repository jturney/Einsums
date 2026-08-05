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

Two scripts sweep rather than run a single geometry:

```bash
# pair screening through the separation where it decides something
python examples/dlpno/sweep_separation.py --distances 2.9 4.0 6.0 8.0 12.0
# the locality claim itself: pairs grow as n^2, kept pairs should not
python examples/dlpno/sweep_chain.py --lengths 2 3 4 5 6
```

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

**The 1e-13 agreement is a property of these geometries, not an invariant.**
`sweep_separation.py` walks the dimer's O-O distance, and at 4.0 A the two codes differ by 9.5e-08 - six orders worse than the 3.8e-14 at 2.9 A and the 8.4e-14 at 12 A.
It is not a convergence artifact: the difference is 9.451e-08 at `r_convergence` of 1e-8, 1e-10 and 1e-12 alike, unchanged to four significant figures while the iteration count goes 10, 14, 17.
It is not the basis either. cc-pVTZ reproduces it at the same separation, 5.4e-08 against 4.2e-14 at 2.9 A, which is what rules out the first guess: a PAO linear-dependence tie-break should have moved or vanished in a different PAO space.

Nor is it any of the three truncations, all of which match psi4 exactly at that geometry:

| quantity at 4.0 A | port | psi4 |
| --- | --- | --- |
| pairs kept | 100 / 100 | 100 / 100 |
| PAO domain per LMO, min / max | 24 / 38 | 24 / 38 |
| aux BFs per LMO, average | 78.4 | 78 |
| PNOs per pair, min / max | 5 / 20 | 5 / 20 |

So the two codes construct the same domains, keep the same pairs and the same PNOs, and still land 9.5e-08 apart, which puts it in the numerics inside those identical domains rather than in any truncation decision.
It sits well under the PNO truncation error at that geometry (8.5e-05), so it is not a correctness problem - but it does mean the 1e-13 figures above hold where nothing lands near a threshold, and 4.0 A is where the domains first straddle both monomers.
Unexplained; the differential-overlap values there do sit on the `t_cut_do` cutoff (0.00972 and 0.01044 against 1e-2) even though the resulting domains agree, which is the thread worth pulling next.

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

Two caveats on this section, both later than the measurements in it.
`bench_batching.py` has not run since bucketing landed - it reaches for `T_all` as a single store and that is now a list of one per bucket - so the table above is a record rather than something reproducible today.
And the lesson in it needs qualifying: leaving the batching to the optimizer is right when the alternative is guessing a batch shape, but the second half of the residual is now *one GEMM per pair* by construction, and no pass can find that, because it is a change to which contraction is written rather than to how the contractions are grouped.
See [The residual was bound by GEMM calls](#the-residual-was-bound-by-gemm-calls-not-by-arithmetic).

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

* `precompute_fits` captures every distinct domain's fitting solve into one graph. There are only 23 of them at ethanol/cc-pVTZ, but they were 31% of the phase, because the solve is per *domain* and carried `naocc * npao` right-hand sides. It carries far fewer now; see [Most of that phase was solving for right-hand sides nobody reads](#most-of-that-phase-was-solving-for-right-hand-sides-nobody-reads).
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

## What the chain shows and a fixed molecule cannot

Every profile above is a compact molecule, where each occupied orbital's domain reaches most of the system and there is effectively one domain.
`sweep_chain.py` lengthens a uniform water chain instead, and the phase that dominates changes.
Single thread, cc-pVDZ, 2.9 A spacing:

| phase | n=3 | n=5 | n=6 | share at n=6 |
| --- | --- | --- | --- | --- |
| `precompute_fits` | 0.11 s | 1.32 s | 2.47 s | 46% |
| `lmp2_iterations` | 0.32 s | 0.93 s | 1.58 s | 30% |
| `compute_pno_overlaps` | 0.04 s | 0.39 s | 0.58 s | 11% |
| `pno_transform` | 0.08 s | 0.35 s | 0.58 s | 11% |
| `compute_qia` | ~0.01 s | 0.03 s | 0.05 s | **1%** |

`precompute_fits` grew 22x from n=3 to n=6 while the whole run grew 9x, and `compute_qia` - the phase that consumes the dense three-index integrals - is 1% of it.
That contradicts the reasoning in "Next steps" below: on a compact molecule `(Q|mn)` is the largest item, but it is not the phase that gets structurally worse as the system extends.
The port also lost to psi4 somewhere around n=4 on this chain, having been 1.4x faster at n=2, so this was the phase that cost the lead.

### Most of that phase was solving for right-hand sides nobody reads

The numbers above are what the phase cost before this was found; the paragraphs that follow are what it cost afterwards, and the fix changes no computed quantity at all.

`precompute_fits` solves one set of fitting equations per distinct *pair* domain, sharing the factorization across every pair over that domain - and it does that by carrying **all `naocc`** right-hand-side blocks through the solve, so any pair over the domain can find its own.
Its only consumer, `_pair_exchange(ij, i, j)`, reads exactly **one** of them: the `i` of the pair it is building.

Whether that is nearly free or nearly all waste depends on how many pairs share a domain, and that is exactly what screening changes:

| | ethanol/cc-pVTZ | chain n=3 | n=4 | n=5 | n=6 |
| --- | --- | --- | --- | --- | --- |
| `naocc` | 13 | 15 | 20 | 25 | 30 |
| distinct pair domains | 23 | 36 | 88 | 138 | 189 |
| LMO blocks a domain is asked for, mean | 2.8 | 1.9 | 1.5 | 1.5 | 1.6 |
| solve GFLOP, all `naocc` | 9.1 | 1.6 | 9.2 | 24.1 | 45.9 |
| solve GFLOP, only what is read | 2.9 | 0.3 | 1.0 | 2.0 | 3.3 |
| gathered `(Q\|iu)` blocks | 107 MiB | 37 MiB | 183 MiB | 440 MiB | 799 MiB |
| gathered, restricted | 27 MiB | 5 MiB | 14 MiB | 26 MiB | 40 MiB |

The right-hand side carried is `naocc / 1.6` times the one read, so the discarded part grows with the system while the useful part does not.
That, not the pair-domain granularity, is where the 22x came from.
Note the ethanol column: even on a compact molecule the mean is 2.8 of 13, because screening splits its domains too.
The regime where solving for every LMO is the right thing is the *unscreened* one, where there is a single domain and every pair over it asks for a different `i`, and that is the regime the phase was written in.

Restricting the right-hand sides to the LMOs some pair will actually read (`DLPNOBase._fit_demand`) is a pure deletion of unread work - the surviving columns are the same numbers from the same factorization - and the correlation energies are unchanged to every digit printed, at every chain length and on ethanol/cc-pVTZ.
Single thread, same machine:

| | before | after | |
| --- | --- | --- | --- |
| ethanol/cc-pVTZ | 0.435 s | **0.123 s** | 3.5x |
| chain n=3 | 0.101 s | **0.021 s** | 4.8x |
| n=4 | 0.536 s | **0.057 s** | 9.4x |
| n=5 | 1.435 s | **0.111 s** | 12.9x |
| n=6 | 2.570 s | **0.170 s** | 15.1x |

`precompute_fits` is now 5.7% of the n=6 run rather than 46%, and grows 9.3x from n=3 to n=6 against the whole run's 6.3x rather than 22x against 9x.
The phase order at n=6 becomes `lmp2_iterations` 51%, `pno_transform` 20%, `compute_pno_overlaps` 20%, `precompute_fits` 6%, `compute_qia` 2%.
`sweep_chain.py` no longer shows the crossover: the port is ahead of psi4 at every length up to n=6 on one thread (3.23 s against 3.36 s at n=6, 1.69 against 2.10 at n=5), where before it fell behind at n=4.
The next section takes `lmp2_iterations` apart, which is what that leaves on top.

### The residual was bound by GEMM calls, not by arithmetic

With the fits fixed, `lmp2_iterations` is half the chain run, and it does not thread: 1.05x from one core to ten, against psi4's whole-run 2.0x.
Splitting it shows the coupling graph is 92% of an iteration and the serial Python around it (DIIS concatenate, the per-pair RMS loop) only 6%, so this is not the Amdahl story the setup phases had.

The graph issued **65,896 GEMM calls per iteration for 737 MFLOP** - 11,000 flops per call.
That is 13.0 GFLOP/s on one thread where a well-sized GEMM does 53, and 19.2 threaded where a well-sized batch does 241.
The blocks are that small because a chain's PNO buckets come out at 9/21/26/41 and most surviving pairs are distant with few PNOs: the commonest single shape is 9x9, 10,844 of the 32,948 couplings.

Two things were wrong, one in the library and one here.

**A vendor GEMM cannot be called concurrently.**
`gemm_batch` parallelized the batch with OpenMP and called OpenBLAS `dgemm` per item, and OpenBLAS serializes inside each call, so the same 4000-element 9x9 batch cost 0.44 ms on one thread, 0.78 on two and 1.20 on ten.
The OpenMP loop was not at fault: a hand-written kernel in the identical loop scales 5.4x over that range.
`libs/Einsums/BLASVendor/src/gemm_batch.cpp` now runs batches whose every dimension is at most 16 on an inline kernel, gated on there being more than one thread, because on one thread the vendor GEMM is simply better.
Worth 10% here, which is all it can be worth: only a third of the calls have every dimension small enough.
OpenBLAS's own `cblas_dgemm_batch` was measured as the alternative and is 15x slower than the per-item loop.

**The sum over a pair's couplings is a contraction, so it is one GEMM.**
`R_ij = sum_c sign_c S_c T_c S_c^T` was a loop of `S T` products, a loop of `tmp S^T` products, and an accumulator tree to keep the second loop from serializing on `R_ij`.
But `compute_pno_overlaps` already lays a pair's overlaps side by side in one `S_cat`, and if the `tmp_c` go into a `tmp_cat` with the same layout then the whole sum is `tmp_cat S_cat^T` - a single `(M, W) x (M, W)^T` GEMM per pair.
The signs move to the first half's `alpha`, because `S` appears twice in the second and a sign folded there would square away.

That collapses the second half from 32,948 GEMMs to 776, and `_choose_width_groups` batches the 776 into 16 by padding each PNO bucket's pairs into four groups of similar total coupling width (padding to the bucket's widest costs 1.42x the flops; four groups cost 1.10x).
The padding is inert rather than merely cheap: a padded column of `S_cat` is zero and so is its partner in `tmp_cat`.
The accumulators and the reduction graph are gone with it - they existed only to break the dependency chain that a contraction does not have.

One trap on the way. The two batched graphs must not be handed the default pass pipeline.
They are already emitted in the form the passes would produce, and `apply` costs 445 ms on a 32-node graph whose nodes carry ~1000 operands each - more than the entire solve replays in - while removing exactly one node of 85.
That is the same argument, and the same decision, `compute_pno_overlaps` already records.

Water chain n=6, `lmp2_iterations`, correlation energies unchanged to every printed digit at every length:

| | 1 thread | 10 threads | scaling |
| --- | --- | --- | --- |
| before | 1.54 s | 1.50 s | 1.03x |
| after | **0.97 s** | **0.72 s** | 1.35x |

The whole n=6 run goes 2.98 -> 2.33 s on one thread and 2.45 -> 1.73 s on ten.
Against psi4 on a loaded machine (medians of three, and the load is why these are ranges): one thread 2.31 s against 4.04, ten threads 1.83 against 1.99 - so the 1.56x threaded deficit at n=6 is gone, and the single-thread lead widened.

Where the chain stands after both fixes, replacing the table at the top of this section. Single thread, cc-pVDZ, 2.9 A spacing:

| phase | n=3 | n=5 | n=6 | share at n=6 | was at n=6 |
| --- | --- | --- | --- | --- | --- |
| `lmp2_iterations` | 0.20 s | 0.60 s | 0.97 s | 41% | 1.58 s (30%) |
| `pno_transform` | 0.09 s | 0.40 s | 0.65 s | 28% | 0.58 s (11%) |
| `compute_pno_overlaps` | 0.05 s | 0.22 s | 0.37 s | 16% | 0.58 s (11%) |
| `precompute_fits` | 0.02 s | 0.12 s | 0.19 s | 8% | **2.47 s (46%)** |
| `compute_qia` | 0.01 s | 0.03 s | 0.07 s | 3% | 0.05 s (1%) |

Nothing dominates any more, which is the useful part: the next thing to fix is no longer obvious from the profile, and `pno_transform` - untouched by either fix - has risen to second on share alone.

`lmp2_iterations` is still the largest phase and still scales worst.
The remaining ceiling is the *first* half, which is 63% of the replay and stays one GEMM per coupling because every coupling has its own `T`; collapsing it the same way needs the couplings grouped by partner rather than by pair, and the two groupings want incompatible layouts.
Capture is now the other half of the phase (~300 ms of 970 at n=6), most of it 32,948 pybind slices and 32 batch emissions.

### The pair-domain granularity itself, and what psi4 does

The union structure this section originally blamed is real, and is still what remains after the fix:

| n | distinct LMO domains | max `naux*npao` | distinct pair domains | max `naux*npao` |
| --- | --- | --- | --- | --- |
| 3 | 12 | 13132 | 36 | 16128 |
| 5 | 20 | 13132 | 138 | 38640 |
| 6 | 24 | 13132 | 189 | 52528 |

LMO domains behave the way locality promises: their count grows linearly with the occupied space and each one's size is *constant*, because a domain around one orbital does not grow when the chain lengthens.
Pair domains do neither. The count grows faster than the orbital count, and a distant pair's union is two disjoint blobs whose size grows with the separation.
The memoization is not at fault and was checked: distinct solves by interned identity equal distinct solves by content at every length (36/36, 138/138, 189/189), so no cache hit is being missed.

The restructure that suggests - solve per LMO domain, 24 bounded solves at n=6 against 189 growing ones, and assemble each pair from its two orbitals' fits - is now a much smaller prize, against a question that has since been answered:

* **psi4 also fits per pair domain.** `dlpno.cc:1373` builds `A_solve = submatrix_rows_and_cols(*full_metric_, lmopair_to_ribfs_[ij], lmopair_to_ribfs_[ij])` and solves it inside the pair loop, and `ccsd.cc:569`, `ccsd.cc:1506` and `triples.cc:724` do the same over pair and triplet domains. So per-LMO fitting is a change to the method, not a return to psi4's choice, and the 1e-13 agreement on compact geometries would not survive it.
* The union solve and two per-LMO solves are still *not* the same object: the metric over a union carries cross terms between the two auxiliary sets that neither single-domain metric has. Whether that difference sits below the truncation error is answerable cheaply by computing both for one pair, and remains unanswered.

Worth noting that psi4 carries one right-hand-side block per solve and pays a factorization per pair, where the port now carries the ~1.6 blocks a domain is asked for and pays one factorization per domain.
Those are the two ends of the same trade, and the port is on the better end of it at every size measured here.

## Next steps

1. **Screen the `(Q|mn)` build.** The largest single item on a compact molecule, 0.24 s - but see the chain measurements above, which show `compute_qia` at 2% of the n=6 run, so this is not the first thing to reach for. Needs an integral interface psi4 does not currently offer from Python: `MintsHelper.ao_eri` is dense, `DFHelper` is Schwarz-screened but metric-locked (`set_metric_pow` is in the header, not in the bindings), and the domain-screened builder DLPNO itself uses is private to the C++ class. Binding `set_metric_pow` is one line in `export_fock.cc` and would unlock the middle tier, worth ~1.9x on this phase.
2. **Per-node capture cost.** The ~0.15 s serial setup tax is ~40% graph capture. Restructuring does *not* reach it - see below - so the lever is the per-node cost itself (~10 us here, against 1.85 us for a C++ caller) or moving the phase out of Python. `DESIGN-cpp-hybrid.md` sketches the latter.

Three attempts at that tax were measured and rejected; the patches are under `parked/` so they are not re-derived.

* **Batching `pno_transform` by domain group.** Node count 2977 -> 1075, capture -12 ms, execute +29 ms. Net loss. The padding is most of it - a group pads to its widest PNO count, and they run 5..92 - plus eight graph barriers needed because a store written through per-pair slices and then read whole is not reliably ordered.
* **`declare_tensor` for the captured scratch.** Capture -30 ms, execute +26 ms: Materialization adds ~2651 lifecycle nodes whose scheduling costs back the allocation saving. Also settles the "allocation is overhead" theory - `create_zero_tensor` is 5.30 us for a 92x92 block against a 1.57 us empty-call floor, so most of it is the memset, which no allocator change avoids.
* **Binding `cg::parallel_for`.** Not attempted after analysis. The body would be a Python callable on TaskPool workers; `execute()` has already released the GIL, so every iteration re-acquires it and the loop serializes - and it would run BLAS on `std::thread`, which the OpenMP-built OpenBLAS miscomputes. It is a C++-caller feature.

The pattern across all three: moving work from capture into the graph costs about what it saves, because per-node graph overhead is comparable to the per-operation Python overhead it displaces.

3. **Collapse the residual's first half too.** `tmp_c = S_c T_c` is still one GEMM per coupling, and it is now 63% of the replay. The second half collapsed because a pair's couplings share nothing but the pair; the first half's operands are shared the other way round - many pairs couple through the same `T` - so it wants the couplings grouped by *partner*, stacking the `S_c` along rows into one tall GEMM per partner pair. That would take it from 32,948 GEMMs to 776 as well. The obstacle is layout: the first half's output would be grouped by partner and the second half needs it grouped by pair, and one of the two has to repack.
4. **Capture cost in the residual**, now ~300 ms of `lmp2_iterations`' 970 at n=6, most of it 32,948 pybind slices building the coupling views. The same per-node-cost problem as item 2, in a phase where it is now a third of the total.

Then, unchanged from before:

5. **One graph for the whole iteration**, using a loop node, with DIIS either captured or hoisted.
6. **DLPNO-CCSD** (`ccsd.cc`), which is where the graph work gets interesting: the residuals are much larger and the intermediates are shared across pairs.
7. **DLPNO-(T)** (`triples.cc`).
