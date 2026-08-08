# DLPNO on the einsums ComputeGraph

A port of psi4's DLPNO module (`psi4/src/psi4/dlpno`) with the tensor algebra expressed in einsums and the iterative solvers captured as ComputeGraphs.
DLPNO is a good fit for deferred execution: it is thousands of small dense operations whose shapes and dependency pattern are fixed for a whole calculation and change only in their values from iteration to iteration, which is exactly the capture-once, replay-many shape.

**Single threaded, this Python port is at parity with psi4's C++ DLPNO-MP2.**
On ethanol/cc-pVTZ it is 2.54 s against psi4's 2.87 s, and on a six-monomer water chain 2.64 s against 2.06 s, with the correlation energies agreeing to 1e-08 or better.
That is a Python program driving a tensor library beating, or landing within 30% of, a mature C++ implementation of the same method on the same machine against the same BLAS.

It is worth being precise about why, because the reason is not that Python got fast: it is that almost no arithmetic happens in Python.
Every contraction is captured into a graph once and replayed, so the per-iteration Python cost is a handful of `execute()` calls however many GEMMs they stand for.
And the layout decisions below turn what psi4 issues as thousands of individually dispatched small GEMMs into a few batched calls, which at these block sizes is worth more than the language difference.

Threaded, psi4 is 1.6-3.7x faster, and the reason is measured rather than guessed: 73% of the port's wall time is host-side work outside the graph, and that part does not thread.
See [Where the threading goes](#where-the-threading-goes).

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

Three scripts do more than run a single geometry:

```bash
# phase against phase with psi4; chain<N> builds an N-monomer water chain
python examples/dlpno/bench_vs_psi4.py --molecule chain6 --threads 10
# pair screening through the separation where it decides something
python examples/dlpno/sweep_separation.py --distances 2.9 4.0 6.0 8.0 12.0
# the locality claim itself: pairs grow as n^2, kept pairs should not
python examples/dlpno/sweep_chain.py --lengths 2 3 4 5 6
```

Note `--threads`: importing psi4 clamps the process-wide OpenMP thread count to 1, so einsums runs serial unless it is set, and `OMP_NUM_THREADS` alone will not do it.
The scripts above are not wired into CTest or pytest, since they need a psi4 install.

### Running without psi4

A converged reference can be frozen to a `.npz` fixture on a machine that has psi4, then replayed anywhere:

```bash
# once, with psi4 importable
python examples/dlpno/dump_reference.py --molecule water --basis cc-pvdz \
    --out examples/dlpno/fixtures/water-ccpvdz.npz

# thereafter, einsums only
PYTHONPATH=/path/to/Einsums/build/lib \
    python examples/dlpno/run_pno_mp2_offline.py examples/dlpno/fixtures/water-ccpvdz.npz
```

`dump_reference.py` records psi4's own DF-MP2 and DLPNO-MP2 correlation energies inside the file, so the replay checks itself against them to the same tolerances `run_pno_mp2.py` uses.
It takes the same geometries the bench scripts do (`chain<N>`, `dimer@<R>`), plus `--freeze-core`.

`run_fixtures.py` replays every fixture in `fixtures/`, which is the psi4-free suite in one command and takes about a second:

```bash
PYTHONPATH=/path/to/Einsums/build/lib python examples/dlpno/run_fixtures.py
```

Six are checked in, chosen to cover distinct paths rather than distinct molecules:

| fixture | size | what it exercises |
| --- | --- | --- |
| `water-ccpvdz` | 0.46 MiB | the baseline: nothing screened, one domain |
| `water-ccpvdz-frozencore` | 0.46 MiB | `n_core` handling, 4 active occupied instead of 5 |
| `water-ccpvtz` | 2.06 MiB | a larger basis, and PAO linear-dependence removal |
| `water-dimer-ccpvdz` | 2.80 MiB | the compact dimer: more pairs, still nothing screened |
| `water-dimer-far-ccpvdz` | 2.18 MiB | pair prescreening, which drops 50 of 100 pairs |
| `methanol-ccpvdz` | 3.01 MiB | a heteroatom: 10 distinct domains, 16 shape classes, PNO counts from 7 to 37 |

Size scales with the DFT grid, so anything much larger is better generated on demand than committed.
`PIPEK_MEZEY` fixtures are deliberately absent: psi4's own DLPNO-MP2 reference is taken with its default localization, so the truncated comparison would be against a different set of domains and fails by the truncation error rather than by anything being wrong.

This is a **smoke test, not a validation**: the reference is frozen, so it cannot catch a change anywhere upstream of `Reference` - a different localization, a different auxiliary basis, a psi4 change in the integrals.
Those still need `run_pno_mp2.py`.
What it does catch is any change in the port itself, end to end, on a machine with no quantum chemistry program installed.

A FCIDUMP would not have served: it carries MO-basis `h_pq`, `(pq|rs)` and the nuclear repulsion, while `Reference` is AO-basis throughout and also carries the auxiliary metric, the raw three-index integrals, the atom-to-basis maps the domain construction keys off, and the DFT grid the differential overlap integrals are quadratured on.
`dlpno/reference_io.py` writes all of it as flat numpy arrays with no pickling; `test_reference_io.py` round-trips it and imports neither psi4 nor einsums.

## Two backends

`compute_pno_overlaps` exists twice: in Python at `dlpno/pno_overlaps.py`, and in C++ under `cpp/`.
Both implement the same contract, both are selectable at runtime, and neither is being retired.
This is the first worked instance of the hybrid framework, so the mechanics are worth stating in full.

Build the C++ side the way an external developer would, against an *installed* Einsums rather than a build tree:

```bash
cmake --install /path/to/Einsums/build --prefix /tmp/einsums-install
cmake -S examples/dlpno/cpp -B /tmp/build-dlpno-stages -GNinja \
      -DCMAKE_PREFIX_PATH=/tmp/einsums-install
cmake --build /tmp/build-dlpno-stages
```

It is not wired into the top-level build on purpose: a stage module that only ever compiles inside the tree that produced libEinsums proves nothing about the path every external user is on, and that path is the one that breaks first.

Then select a backend per stage:

```bash
export PYTHONPATH=/path/to/Einsums/build/lib:/tmp/build-dlpno-stages
python run_fixtures.py --backend compute_pno_overlaps=cpp
```

Three scripts support this, and the reason each exists is not obvious from its name:

| script | answers |
| --- | --- |
| `run_fixtures.py` | is the method still correct against psi4's recorded energies |
| `dump_energies.py` | did a pure-Python refactor move any digit (full `repr`, diff two runs) |
| `check_backends.py` | do the two backends of a stage agree, and did the one you selected actually run |

The third question is easy to skip and the reason it matters is that the two backends agree to `0.000e+00`, which is also exactly what a backend that never executed produces.
`check_backends.py --prove` reports the name of a tensor each side returned: the Python implementation names per-class tensors by the class tuple and the C++ one by the class ordinal, so the output says which code made it.

Exact agreement rather than agreement-to-tolerance is the expected result, not a lucky one.
Both backends emit the same GEMMs in the same order into the same BLAS; only the operand-list construction differs, and that is index arithmetic with no floating-point content.
A disagreement at any tolerance would mean something reordered.

`stage_state_report.py` is the fourth and answers a different question: how many `self` fields each phase touches, which is the width of the contract it would need if promoted.
Run it before choosing what to promote rather than after.

### Everything under `cpp/` except the port is generated

`dlpno/contracts.py` and the `@stage` signature in `dlpno/stages.py` are the source of truth, including the prose: the struct documentation in `Contracts.hpp` is carried from the contracts' docstrings and `#:` field comments, and the `@param` lines from the stage docstring's `Args:` section.
Regenerate with:

```bash
python -m einsums.stages promote examples/dlpno/dlpno/stages.py \
       --out examples/dlpno/cpp --license-header devtools/LicenseHeader.txt
```

`--license-header` is what keeps the repository's own license hook from editing a generated file out from under the hash in its banner.

`src/ComputePnoOverlaps.cpp` is the one file `promote` does not write.
It is scaffolded once and then belongs to whoever ported it, and `--force` does not reach it; the regenerated files are the ones that carry a `promote-hash:` and are refused if they have been edited by hand.
Editing the C++ contract structs directly is therefore a mistake the tool will report rather than silently undo - the change belongs in `contracts.py`.

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

## Performance against psi4

`bench_vs_psi4.py` runs psi4's native C++ DLPNO-MP2 in a subprocess and this port in process, comparing phase against phase using psi4's own module timers.
Both sides get the same thread count and the same converged reference, and the SCF is excluded from both.

**Ethanol/cc-pVTZ, one thread.**
Correlation energies agree to 2.3e-08.

| phase | psi4 | this port | |
| --- | --- | --- | --- |
| Setup Orbitals | 0.003 | 0.003 | 1.1x |
| DF Ints | **0.132** | 0.333 | 2.5x |
| PNO Transform | 1.236 | **0.577** | **0.5x** |
| PNO Overlaps | 0.221 | 0.227 | 1.0x |
| LMP2 | **1.170** | 1.307 | 1.1x |
| **total** | 2.873 | **2.538** | **0.9x** |

**Water chain n=6, cc-pVDZ, one thread.**
Correlation energies agree to 9.7e-09.

| phase | psi4 | this port | |
| --- | --- | --- | --- |
| Setup Orbitals | 0.004 | **0.002** | 0.5x |
| DF Ints | **0.138** | 0.410 | 3.0x |
| PNO Transform | 0.995 | **0.789** | 0.8x |
| PNO Overlaps | **0.179** | 0.776 | 4.3x |
| LMP2 | 0.668 | **0.596** | 0.9x |
| **total** | **2.061** | 2.637 | 1.3x |

Single threaded the port wins on the compact molecule in the larger basis and loses on the extended one in the smaller basis, and the two phases that decide it are identifiable: the dense `(Q|mn)` build, and `PNO Overlaps` once there are many pairs.
The LMP2 iteration itself, which is what the graph work is really about, is 0.9-1.1x on both.

**Ten threads.**
psi4 is 1.6x ahead on ethanol and 3.7x on the chain.

| phase, chain n=6 | psi4 | this port | |
| --- | --- | --- | --- |
| DF Ints | **0.047** | 0.394 | 8.4x |
| PNO Transform | **0.202** | 0.494 | 2.4x |
| PNO Overlaps | **0.033** | 0.699 | 21x |
| LMP2 | **0.235** | 0.377 | 1.6x |
| **total** | **0.561** | 2.076 | 3.7x |

Per LMP2 iteration the two are level threaded, 0.0214 s against 0.0215.
The port gets there while doing 1.6x the coupling flops psi4 does, because each block runs on its bucket's dimension rather than its own PNO count, so per flop that actually needs doing the residual is at about 0.67x psi4's cost.

That 1.6x is not headroom, which is worth stating because the obvious reading is that it should be.
Raising `n_buckets` does cut it - to 1.15x at eight buckets and 1.07x at twelve - and single threaded that converts, 48.1 ms per iteration at four buckets down to 37.6 at twelve.
Threaded it goes the other way, 28.4 ms at four buckets up to 54.0 at twelve, because the batches stop being large enough to fill the cores.
Four is the shipped default because it is the best threaded value, and the two regimes genuinely disagree.

### Where the threading goes

Splitting every phase into graph replay and everything else, on the chain at n=6:

| | 1 thread | 10 threads | scaling |
| --- | --- | --- | --- |
| graph replay | 976 ms | 335 ms | **2.9x** |
| everything else | 1136 ms | 915 ms | **1.24x** |

**At ten threads, 73% of the run is outside the graph.**
The graph itself threads about as well as psi4's OpenMP does.
It is simply the minority of the wall time, and the majority barely moves, which is why every phase's non-replay share *rises* from one thread to ten: only the replay half shrinks.

| phase | non-replay, 1 thread | non-replay, 10 threads |
| --- | --- | --- |
| `compute_doi` | 100% | 100% (78 ms) |
| `prep_sparsity` | 100% | 100% (33 ms) |
| `compute_qia` | 100% | 100% (46 ms) |
| `precompute_fits` | 15% | 41% (24 ms) |
| `pno_transform` | 49% | **82%** (329 ms) |
| `compute_pno_overlaps` | 79% | **88%** (235 ms) |
| `lmp2_iterations` | 38% | 46% (171 ms) |

That bucket is Python loops, but not only: it is also eager numpy bookkeeping (`compute_doi` and `prep_sparsity` are entirely that), tensor allocation, graph capture, and pybind view construction.
The accurate label is host-side serial work outside the graph.

`compute_pno_overlaps` was the clearest case, and the one that has since been fixed, which is worth keeping as the worked example.
Its graph replay was 107 ms on one thread and 26 ms on ten, a clean 4.1x, and only 7-19% of the phase.
Roughly half the phase was **32,948 Python tensor-view constructions**, one per coupling, purely to hand the batched GEMM its output operands.
Serial by nature, so the phase looked like it did not thread when its parallel part was parallelizing fine.

`cg::batched_gemm_blocked` takes those destinations as offsets into one tensor instead of as a list of views, so C++ builds them.
The phase went from 690-800 ms to 254-289 ms on one thread and roughly halved on ten, and against psi4 it went from 4.3x to 1.3x serially and 21x to about 5x threaded.
It also registers one output slot rather than 32,948, and because that output is the base tensor rather than a view of it, a later read of the whole base is ordered against the write without needing a graph boundary.

This is the ceiling on the current shape: even if graph replay went to zero, the port would still take 915 ms at n=6 against psi4's 561 ms total.
No amount of optimizing what is inside the graph reaches that.
The work has to move into the graph, or out of Python, which is what `DESIGN-cpp-hybrid.md` sketches.

One gap in the table is not this and should not be attributed to it.
`DF Ints` is `from_psi4` calling psi4's *dense* `ao_eri` against psi4's own domain-screened builder: C++ on both sides, and an algorithmic difference rather than overhead.

### Two traps in measuring this at all

Both were live in this directory and both flattered the port, which is why these numbers are not the ones earlier versions of this file carried.

**Timing psi4's `energy("dlpno-mp2")` includes an SCF.**
The port is handed a converged reference and times only the correlation, so a wall clock around psi4's driver compares different work.
At ten threads that SCF is 61-84% of psi4's timed region on this chain, which silently turned a 2.4x deficit into a reported 1.1x lead.
`ref_wfn=wfn` fixes it, and `bench_vs_psi4.py` sidesteps it entirely by reading psi4's module timers.

**Leaving `from_psi4` outside the port's clock hides its largest DF cost.**
It builds the dense `(Q|mn)`, which is work psi4 does too, screened, inside the run being timed.
It does not thread and it is 0.36 s at n=6.

## Design notes

These are the decisions parity rests on.
They are mostly about what *shape* the work is issued in, not about arithmetic.

**Everything per-pair lives in one contiguous store, padded to a uniform shape.**
This is the layout decision the whole calculation turns on, and the one psi4 makes differently.
psi4 keeps each pair's block in its own `SharedMatrix` sized to that pair's PNO count, which leaves the residual a loop of individually dispatched small GEMMs over operands sharing neither shape nor allocation.
Here each per-pair quantity is one `(npno_max, npno_max, n_pairs)` tensor.

Two things follow.
Everything elementwise over pairs (the amplitude update, the antisymmetrization, the energy) becomes a single operation on the whole store rather than one per pair: the Jacobi step is 1 node and the antisymmetrization is 2, whatever the molecule.
And because every coupling GEMM has the same shape, they batch.

Padding is inert: integrals, amplitudes and overlaps are zero outside each pair's logical block and the energy denominators are one, so padded components stay zero for the life of the calculation.
It is not free, though - it is the 1.6x flop overhead noted above - which is what `n_buckets` trades against.
Single threaded, flops dominate and more buckets is monotonically better; threaded, the batches have to stay large enough to fill the cores.
That is why it is a knob rather than a constant.

**The residual is two GEMMs per pair, not thousands.**
`R_ij = sum_c sign_c S_c T_c S_c^T` reads as a loop over couplings and was written that way: 65,896 GEMM calls per iteration for 737 MFLOP, which is 11,000 flops per call.
At that size a GEMM is bound by the cost of *making the call* rather than by arithmetic - 13.0 GFLOP/s on one thread where a well-sized GEMM does 53.

Both halves collapse, for mirror-image reasons.

The second half sums over a pair's couplings, and a sum is a contraction: with the intermediate and the `S_c` both concatenated along it, the whole sum is one `(M, W) x (M, W)^T` GEMM.
32,948 calls become 776.
The partial accumulators this used to need went with it - they existed only to break a dependency chain a contraction does not have.

The first half applies each partner's `T`, and many pairs couple through the same partner, so grouping by partner rather than by pair makes it one GEMM per partner: 32,948 into 2,822.

The catch is that those two groupings form a bipartite graph and cannot both be contiguous, so the intermediate is produced by partner, read by pair, and moved between once per iteration by one permuting gather per shape class.
Two consequences of column-major storage are worth stating rather than rediscovering.
A GEMM's output blocks are contiguous only when the *shared* operand is on the left, and here that is `T`, so the products come out transposed.
And the per-coupling sign cannot ride on the batch's `alpha`, because one batch now spans many pairs; it rides on one of the two copies of `S` instead, since `S` appears on both sides of the congruence and a sign in both would square away.

Together these took `lmp2_iterations` at n=6 from 1.58 s to 0.61 s on one thread and to 0.36 s on ten.

**Emit the batch directly with `cg.batched_gemm`.**
Capture costs tens of microseconds a node and these graphs are built once, so a form that emits one node per coupling can spend more building the graph than the eager version spends computing.
Emitting the fused form directly took the ethanol capture from 8215 nodes to 335.

If you do write a per-contraction form, use `einsum` and not `linalg.gemm`: a 2D x 2D -> 2D einsum with one link index carries the `gemm_hint` that `GEMMBatching` groups on, while `linalg.gemm` captures as `OpKind::Gemm` and the pass skips it outright, silently costing the whole optimization.

The older lesson here was "make the data uniform and leave the batching to the optimizer", and it needs qualifying.
That is right when the alternative is guessing a batch shape.
It is not the whole story, because both halves of the residual are now single GEMMs *by construction*, and no pass can find that: it is a change to which contraction is written, not to how contractions are grouped.

**Do not hand a pre-batched graph to the pass pipeline.**
The coupling and residual graphs are emitted in the form the passes would produce, and `apply` costs 445 ms on a 32-node graph whose nodes carry ~1000 operands each - more than the entire solve replays in - while removing exactly one node of 85.
The passes scale with operand count, not node count.
`compute_pno_overlaps` had already made the same call for the same reason.

**Phases are separate graphs.**
The graph does not know that a write through `R_all[:, :, ij]` touches `R_all`, so mixing parent-level and view-level access to one store in a single graph lets the scheduler interleave them.
This is not hypothetical: with the residual's phases captured together the energy dot landed at node 201 of 405, summing a half-built residual and quietly returning a wrong correlation energy.
Graphs execute in the order they are replayed, so keeping each granularity in its own graph is an explicit barrier.

**Setup phases are captured too, but shaped differently.**
Each is one or a few graphs replayed under the OpenMP executor, because the parallelism worth having there is across pairs with each pair's BLAS call left serial underneath.
Driving the same loops from a Python thread pool is not safe: the OpenMP-built OpenBLAS conda resolves by default indexes internal scratch by `omp_get_thread_num()`, which is 0 on every caller-created thread, so workers silently overwrite each other.
An OpenMP parallel region is a real team with distinct thread numbers, so the same BLAS is safe underneath it.

**DIIS runs on the host.**
It is a solver detail over flattened buffers rather than tensor algebra, and with every pair in one contiguous store it needs no gathering at all, just a `ravel` view of `T_all`.
It writes back into the same tensors the graphs already reference, so the replay picks the new values up.
Folding the whole loop into one graph with a loop node is the obvious next step.

## What the chain shows and a fixed molecule cannot

Every compact geometry has each occupied orbital's domain reaching most of the system, so none of them can show the property DLPNO exists for.
`sweep_chain.py` lengthens a uniform water chain instead, and the kept fraction falls as it should:

| n | occupied | kept / total | fraction |
| --- | --- | --- | --- |
| 2 | 10 | 100 / 100 | 1.00 |
| 3 | 15 | 225 / 225 | 1.00 |
| 4 | 20 | 398 / 400 | 0.99 |
| 5 | 25 | 587 / 625 | 0.94 |
| 6 | 30 | 776 / 900 | 0.86 |

psi4 drops exactly the same pairs at every length.

The chain also found the port's worst scaling bug, which no fixed molecule would have.

**`precompute_fits` was solving for right-hand sides nobody reads.**
It solves one set of fitting equations per distinct *pair* domain, sharing the factorization across every pair over that domain, and it did that by carrying **all `naocc`** right-hand-side blocks through each solve so any pair could find its own.
Its only consumer reads exactly **one** of them.

That is nearly free when one domain serves every pair, which is the unscreened case the phase was written in.
Under screening the domains become nearly distinct per pair, and a domain is asked for 1.6 LMO blocks on average against `naocc`, which is 30 at n=6.
The discarded fraction therefore grows with the system while the useful part does not, and the phase grew 22x from n=3 to n=6 where the whole run grew 9x.

Restricting the right-hand sides to the LMOs some pair will actually read is a pure deletion of unread work, so the energies are unchanged to every digit:

| | before | after | |
| --- | --- | --- | --- |
| ethanol/cc-pVTZ | 0.435 s | **0.123 s** | 3.5x |
| chain n=6 | 2.570 s | **0.170 s** | **15x** |

**psi4 also fits per pair domain**, which was an open question worth settling before considering anything more invasive.
`dlpno.cc:1373` builds `A_solve = submatrix_rows_and_cols(*full_metric_, lmopair_to_ribfs_[ij], ...)` inside the pair loop, and `ccsd.cc:569`, `ccsd.cc:1506` and `triples.cc:724` do the same over pair and triplet domains.
So fitting per LMO domain instead would be a change to the method rather than a return to psi4's choice, and the 1e-13 agreement on compact geometries would not survive it.

## Gaps found in Einsums

This port has driven several einsums features.
Check before assuming something is missing: `linalg.abs`, `scale`, `direct_product` and `element_transform` all capture, as do the pointer-writer `sum`, `max`, `dot`, `trace` and `norm` (the returning forms throw during capture by design).

**Gather and scatter by index list - ADDED.**
Domain restriction is the fundamental operation in DLPNO, and einsums had no primitive for "take these rows and these columns", so every domain extraction ran eagerly on the host, uncapturable.
`cg::gather` and `cg::scatter` now cover it.
Note `gather` has no whole-axis wildcard: an empty index list selects nothing, because an empty domain silently becoming the full axis is a wrong answer rather than an error.

The kernel behind them was worth a second look, and the answer was not the expected one.
They share `detail::for_each_selection_run`, which hoists per-axis offsets into tables and collapses the common case - a whole axis spelled `range(n)` as the fastest axis, where the selection is a contiguous block - into one `copy_n`.
That is 1.7x to 4.5x on the shapes this port issues.
Worth stating plainly: the SIMD module looks like it should be the answer here and is not.
`simd::gather` is a strided-lane load, not an index-list gather, and the psABI rung ladder is x86-only.
What was left on the table was index arithmetic and a missed `memcpy`, and both fixes are scalar.

**A batched GEMM whose destinations are blocks of one tensor - ADDED.**
`cg::batched_gemm_blocked` takes the destinations as offsets into a base tensor instead of as a list of views.
The list form needs one tensor object per member, and building those is not free: 32,948 Python view constructions plus 32,948 slot registrations here, together about half of `compute_pno_overlaps` and none of it arithmetic.
A column block of a column-major tensor inherits the parent's leading dimension, which is what makes the description sufficient.
Worth 2.7x on that phase serially.

**`gather` can permute axes as it moves - ADDED.**
Selecting and reordering are both full passes over the result, and doing them separately means two.
`axes[k]` names the destination axis that source axis `k` lands on.
The traversal needed nothing: it already walks the destination through per-axis strides, and a permutation is a different set of them.
On the residual's repack, 66 MiB over 16 gathers: 12.7 ms to 6.9 on one thread, 5.7 to 3.3 on ten.

**`reshape_view` - ADDED.**
Reshaping went through `linalg::reshape`, which copies.
That is right when the shape cannot be addressed over the existing strides and wrong when it can, which is the common case: merging adjacent axes, or splitting one, over a tensor already contiguous across them.
`reshape_view` returns a view when each new axis lands inside a run of old axes that abut, and throws when it does not rather than quietly copying.
Without it the residual's restructure is not expressible: every operand there is a slice reinterpreted as a matrix, and `linalg.reshape` would have copied the whole intermediate twice per iteration.

Two traps it exposed, both pinned in tests.
The row-major flag is not usable for this - `infer_row_major()` reports true for any rank below 2, so the rank-1 view of a flattened column-major tensor claims to be row major and reshaping it back would transpose it - so the storage order comes off the strides instead.
And slicing behaves opposite to intuition: `A[:, 0:4, :]` still merges its first two axes, because axis 0 stays whole and the slice is therefore the parent's first 36 elements in order, while `A[0:4, :, :]` does not.

**Tiny batched GEMMs must not call the vendor - FIXED.**
`gemm_batch` parallelized the batch with OpenMP and called OpenBLAS `dgemm` per item, and OpenBLAS serializes inside each call, so the same batch got *slower* as threads were added: 4000 9x9 doubles is 0.44 ms on one thread, 0.78 on two, 1.20 on ten.
The OpenMP loop was not at fault - a hand-written kernel in the identical loop scales 5.4x over the same range.
Batches whose every dimension is at most 16 now run on an inline kernel, gated on there being more than one thread, because on a single thread the vendor GEMM is simply the better kernel.
OpenBLAS's own `cblas_dgemm_batch` was measured as the alternative and is 15x slower than the per-item loop.

**In-place elementwise einsum against a lower-rank operand silently yielded zeros - FIXED.**
`einsum("ab <- ab ; b", X, X, f)` was accepted by the aliasing guard, because the aliased operand's index list matches the output's, but returned zeros: both generic algorithms clear C before reading anything.
The equal-rank `"ab <- ab ; ab"` was fine, which is what hid it.
Regression tests in `ComputeGraph/tests/unit/EagerParityGaps.cpp`.

**View/parent aliasing was not a graph dependency - FIXED.**
A write through `R[:, :, p]` and a read of `R` were treated as touching unrelated tensors, so the scheduler could order them arbitrarily: wrong answers, no error.
`cg::view()` always set `TensorHandle::aliases`, but a view sliced *outside* a capture never goes through it, and that is exactly the pattern this port needs, because a captured view must outlive the graph.
`Graph::link_alias_storage` now recovers the relationship from the registration-time data pointer and strides.
`repro_view_aliasing.py` is the standalone reproducer; `ComputeGraph/tests/unit/View.cpp` carries the regression test.

Still missing, in rough order of what this workload would pay for:

1. **Concatenate / stack along an axis.** DIIS flattens every pair block into one vector and back; the dipole code stacks three components into an `(naocc, 3)`.
2. **Strided views under capture.** `T[:, :, j::naocc]` works eagerly but raises `only step=1 slices are supported` under capture.
3. **`linalg.gemm` is invisible to `GEMMBatching`.** It captures as `OpKind::Gemm`, and the pass only groups `Einsum` nodes carrying a `gemm_hint`.
4. **Masked select.** `np.where` guards a divide-by-zero in the population split.

One numpy trap worth repeating: `np.asarray` on a tensor is F-contiguous, so `.reshape(-1)` silently copies rather than viewing.
An early DIIS here extrapolated into throwaway buffers and quietly degraded to unaccelerated Jacobi, 48 iterations instead of 11, with no error anywhere.
`ravel(order="F")` is the view, and `mp2.py` asserts it.

## Next steps

In the order the measurements above argue for.

1. **Move the serial layer off the host.** 73% of the threaded run is outside the graph and does not thread, and that is the ceiling on everything else. Either more of the work becomes graph nodes, or the phases move to C++; `DESIGN-cpp-hybrid.md` sketches the latter.
2. **Screen the `(Q|mn)` build.** 2.5-8.4x depending on thread count, and algorithmic rather than overhead. psi4 does not expose the interface: `MintsHelper.ao_eri` is dense, `DFHelper` is Schwarz-screened but metric-locked (`set_metric_pow` is in the header, not the bindings), and the domain-screened builder DLPNO itself uses is private to the C++ class. Binding `set_metric_pow` is one line in `export_fock.cc` and would unlock the middle tier.
3. **Halve the residual's working set.** Both halves are single GEMMs at the price of holding the intermediate in two orders at once. That is the standing cost of the bipartite structure, and the thing to attack before pushing to larger systems.
4. **One graph for the whole iteration**, using a loop node, with DIIS either captured or hoisted.
5. **DLPNO-CCSD** (`ccsd.cc`), which is where the graph work gets interesting: the residuals are much larger and the intermediates are shared across pairs.
6. **DLPNO-(T)** (`triples.cc`).

`bench_batching.py` has not run since bucketing landed - it reaches for `T_all` as a single store and that is now a list of one per bucket.
Its conclusion, that padding only pays once a pass can exploit the uniformity, is superseded anyway: the residual no longer has a per-coupling form to compare against.

Three attempts at the capture cost were measured and rejected; the patches are under `parked/` so they are not re-derived.

* **Batching `pno_transform` by domain group.** Node count 2977 -> 1075, capture -12 ms, execute +29 ms. Net loss, mostly padding.
* **`declare_tensor` for the captured scratch.** Capture -30 ms, execute +26 ms: Materialization adds ~2651 lifecycle nodes whose scheduling costs back the allocation saving. It also settles the "allocation is overhead" theory - `create_zero_tensor` is 5.30 us for a 92x92 block against a 1.57 us empty-call floor, so most of it is the memset, which no allocator change avoids.
* **Binding `cg::parallel_for`.** Not attempted after analysis. The body would be a Python callable on TaskPool workers; `execute()` has already released the GIL, so every iteration re-acquires it and the loop serializes - and it would run BLAS on `std::thread`, which the OpenMP-built OpenBLAS miscomputes. It is a C++-caller feature.
