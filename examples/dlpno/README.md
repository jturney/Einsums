# DLPNO-MP2 on the einsums ComputeGraph

A port of psi4's DLPNO module (`psi4/src/psi4/dlpno`) with the tensor algebra expressed in einsums and the solvers captured as ComputeGraphs.
DLPNO-MP2 is complete and validated against psi4 two ways; DLPNO-CCSD and (T) are not started.

DLPNO fits deferred execution unusually well: it is thousands of small dense operations whose shapes and dependency pattern are fixed for a whole calculation and change only in their values, which is exactly the capture-once, replay-many shape.
Every contraction is captured into a graph once and replayed, so the per-iteration Python cost is a few `execute()` calls however many GEMMs they stand for; the LMP2 iteration itself runs as a single graph with a loop node and DIIS as its predicate.
Single threaded, the port beats psi4's C++ implementation on both benchmark geometries - 0.67x its wall time on ethanol/cc-pVTZ, 0.77x on an extended water chain - and its LMP2 iteration runs at 0.39x psi4 serially on the chain and ahead of it threaded.
Threaded it also wins ethanol (0.88x); on the chain psi4 is 1.44x ahead, and the gap is measured rather than mysterious: the dense `(Q|mn)` build (algorithmic, psi4's screened builder is not exposed), the transform's serial capture emission, and the one-time LMP2 graph build psi4 has no analogue of.
Current numbers are in [Performance against psi4](#performance-against-psi4); `bench_vs_psi4.py` reproduces them on your machine, phase against phase.

## Layout

```
dlpno/                the package; see its __init__ docstring for the module map
  base.py             orbitals, PAOs, screening, domains, DF integrals, PNO transform
  mp2.py              the local MP2 solver: one loop graph, DIIS predicate
  sparse.py           SparseMap bookkeeping, mirroring psi4's sparse.cc
  thresholds.py       psi4's truncation thresholds and presets
  reference.py        the psi4-free input contract: plain numpy buffers
  reference_io.py     that contract as an .npz fixture on disk
  integrals.py        where (Q|i u) comes from: the declared demand and the dense source
  cost.py             the machine model the bucket chooser optimizes against
  psi4_source.py      the ONE module that imports psi4; fills a Reference
  tensors.py          einsums.linalg helpers shaped like psi4's Matrix utilities
  stages.py           phases registered with einsums.stages (per-phase timing)
  contracts.py        cross-boundary contracts of the two promoted stages
  pno_overlaps.py     promoted numerics: PNO overlap matrices (Python backend)
  pno_xform.py        promoted numerics: the PNO transform (Python backend)
  molecules.py        shared test geometries
cpp/                  C++ backends of the promoted stages; generated + one port each
fixtures/             frozen references with psi4's energies recorded inside
```

Every module and phase method carries a docstring stating what it does, why it is shaped that way, and which psi4 function it mirrors; the package `__init__` maps the phase pipeline.
The loose scripts are listed under [Tools](#tools).

## Running with psi4

Needs `einsums` and `psi4` importable.
Point `PYTHONPATH` at an in-tree Einsums build and any psi4 install, and use the conda env's Python:

```bash
PYTHONPATH=/path/to/Einsums/build/lib:/path/to/psi4/stage/lib \
    python examples/dlpno/run_dlpno_mp2.py --molecule water-dimer
```

This runs the molecule twice and asserts both results: untruncated against psi4's canonical DF-MP2, and truncated (psi4's NORMAL preset) against psi4's own DLPNO-MP2.
Useful flags: `--basis`, `--localization {BOYS,PIPEK_MEZEY}`, `--t-cut-pno`, `--buckets`, `--threads`, `--no-optimize` (skip the graph passes), `--no-diis`.

Note `--threads`: importing psi4 clamps the process-wide OpenMP thread count to 1, so einsums runs serial unless it is set; `OMP_NUM_THREADS` alone will not do it.
None of the psi4-driven scripts are wired into CTest or pytest, since they need a psi4 install.

## Running without psi4

A converged reference can be frozen to an `.npz` fixture on a machine that has psi4, then replayed anywhere the library builds:

```bash
# once, with psi4 importable
python examples/dlpno/dump_reference.py --molecule water --basis cc-pvdz \
    --out examples/dlpno/fixtures/water-ccpvdz.npz

# thereafter, einsums only
PYTHONPATH=/path/to/Einsums/build/lib \
    python examples/dlpno/run_dlpno_mp2_offline.py examples/dlpno/fixtures/water-ccpvdz.npz
```

The fixture records psi4's own DF-MP2 and DLPNO-MP2 correlation energies, so a replay checks itself against them without psi4 present.
`run_fixtures.py` replays every fixture at both threshold settings - the whole psi4-free suite in one command, about a second - and exits non-zero on any disagreement:

```bash
PYTHONPATH=/path/to/Einsums/build/lib python examples/dlpno/run_fixtures.py
```

Six fixtures are checked in, chosen to cover distinct code paths rather than distinct molecules:

| fixture | what it exercises |
| --- | --- |
| `water-ccpvdz` | the baseline: nothing screened, one domain |
| `water-ccpvdz-frozencore` | `n_core` handling |
| `water-ccpvtz` | a larger basis, and PAO linear-dependence removal |
| `water-dimer-ccpvdz` | the compact dimer: more pairs, still nothing screened |
| `water-dimer-far-ccpvdz` | pair prescreening, which drops 50 of 100 pairs |
| `methanol-ccpvdz` | a heteroatom: many distinct domains and shape classes |

This is a smoke test, not a validation: the reference is frozen, so it cannot catch a change upstream of `Reference` (a different localization, a psi4 change in the integrals).
Those still need `run_dlpno_mp2.py`.

## What is validated

**Untruncated, against canonical DF-MP2.**
With PNO truncation off and full domains, local MP2 in the PAO space *is* canonical MP2 in a non-canonical basis, so the correlation energy must match psi4's DF-MP2 to machine precision.
It does, to 1e-13 or better on every molecule in the driver.
This is the check that pins the port down: the PAO construction, the local density fit, the PNO machinery, the residual, and the solver all have to be right simultaneously for it to pass.

**Truncated, against psi4's own DLPNO-MP2.**
Both sides apply the same three truncations - PNO occupation cutoff, differential-overlap PAO and auxiliary domains, dipole pair prescreening - so this is an exact comparison, and it agrees to ~1e-12 on the fixture set, with the far dimer dropping exactly the pairs psi4 drops.

One accuracy caveat worth knowing: the 1e-13 figures hold where nothing sits near a threshold.
`sweep_separation.py` walks the dimer separation and finds ~1e-7 disagreement where the domains first straddle both monomers, with identical domains, pairs and PNO counts on both sides - well under the truncation error there, but real.
The runner's assertion tolerance is scaled to the truncation correction for this reason.

## The two backends

Two stages exist twice: `compute_pno_overlaps` and `transform_pnos`, each a Python numerics module in the package and a C++ backend under `cpp/`.
Both implement the same contracts, both are selectable at runtime, and neither Python side is retired; the C++ side is regenerated from the Python contracts by the hybrid framework's promote tool:

```bash
python -m einsums.stages promote examples/dlpno/dlpno/stages.py \
       --out examples/dlpno/cpp --license-header devtools/LicenseHeader.txt
```

The port sources under `cpp/src/` are hand-written, and the build file and differential tests are scaffolded once and then yours; the regenerated files (headers, bindings) carry a `promote-hash:` and are refused if edited by hand - a contract change belongs in `contracts.py`.

Build the C++ side the way an external developer would, against an *installed* Einsums:

```bash
cmake --install /path/to/Einsums/build --prefix /tmp/einsums-install
cmake -S examples/dlpno/cpp -B /tmp/build-dlpno-stages -GNinja \
      -DCMAKE_BUILD_TYPE=Release \
      "-DCMAKE_PREFIX_PATH=/tmp/einsums-install;$CONDA_PREFIX" \
      -DPython3_EXECUTABLE=$CONDA_PREFIX/bin/python
cmake --build /tmp/build-dlpno-stages
```

It is deliberately not part of the top-level build: a stage module that only compiles inside the tree that produced libEinsums proves nothing about the path external users are on.
Then select a backend per stage:

```bash
export PYTHONPATH=/path/to/Einsums/build/lib:/tmp/build-dlpno-stages
python examples/dlpno/run_fixtures.py --backend transform_pnos=cpp
```

The two backends of a stage agree bit for bit, because both emit the same einsums operations on the same values; `check_backends.py` asserts exactly that, and `check_backends.py --prove` also proves which backend actually ran, since perfect agreement is what a silently-unselected backend produces too.

## Performance against psi4

Measured 2026-08-10 (Apple M-series, 4 performance and 6 efficiency cores, on mains power, best of three per row over runs interleaved across the four configurations) with `bench_vs_psi4.py`, which runs psi4's native C++ DLPNO-MP2 in a subprocess and this port in process, same thread count, same converged reference, SCF excluded from both.
psi4's column is the drift control rather than the machine being assumed quiet: it reproduced an earlier session's to within 3% while a Spotlight reindex held one core, which is the tolerance to read every number below at.
The port runs its hybrid configuration (`--backend compute_pno_overlaps=cpp,transform_pnos=cpp`); correlation energies agree to 2e-08 (ethanol) and 1e-08 (chain).
The three-index integrals come from `LocalQiaBuilder` (`--integrals screened`, the default), which builds only the blocks the solver declares it will read, so `DF Ints` is now the same class of quantity psi4's own `compute_qia` produces rather than a full AO build followed by a dense transform.
That makes both sides domain-restricted at `T_CUT_CLMO`/`T_CUT_CPAO` of 1e-4, which is what a like-for-like row requires; `--integrals dfhelper` selects the exact source instead, and costs what the previous table's `DF Ints` row cost.
The bucket count is the automatic one, which chose 15, 12, 9 and 7 on ethanol and 12, 8, 7 and 6 on the chain at 1, 2, 4 and 10 threads.

Every phase either side times appears below, including the ones the other side has no analogue for.
That is worth stating because it was not always so: the table used to print a fixed list of labels and silently dropped whatever was not on it, which hid the port's `Sparsity` phase (16% of an ethanol run) and psi4's `Overlap Ints` and `Dipole Ints`.
`bench_vs_psi4.py` now derives its rows from what was timed and prints the difference between the rows and the total as `other`, so a phase cannot go missing again.
The LMP2 rows split the one-time graph build (allocate, capture, optimize - a cost psi4 has no analogue of, which amortizes with iteration count) from the iterations themselves.

**Ethanol/cc-pVTZ** - the compact molecule in the larger basis.

| phase | psi4, 1 thread | port | psi4, 10 threads | port |
| --- | --- | --- | --- | --- |
| Setup Orbitals | 0.003 | 0.003 | 0.004 | **0.002** |
| Sparsity | **0.013** | 0.091 | **0.013** | 0.052 |
| DF Ints | 0.134 | **0.125** | **0.041** | 0.054 |
| PNO Transform | 1.230 | **0.539** | 0.247 | **0.179** |
| PNO Overlaps | 0.221 | **0.157** | **0.040** | 0.052 |
| LMP2 iterations | 1.165 | **0.839** | 0.343 | **0.279** |
| LMP2 graph build | - | 0.042 | - | 0.034 |
| Overlap + Dipole Ints | 0.096 | - | 0.134 | - |
| other | 0.001 | 0.007 | 0.000 | 0.004 |
| **total** | 2.873 | **1.806** | 0.830 | **0.657** |

**Water chain n=6, cc-pVDZ** - the extended system with many small pairs.

| phase | psi4, 1 thread | port | psi4, 10 threads | port |
| --- | --- | --- | --- | --- |
| Setup Orbitals | 0.004 | **0.002** | 0.005 | **0.002** |
| Sparsity | **0.013** | 0.080 | **0.012** | 0.061 |
| DF Ints | 0.138 | 0.137 | **0.046** | 0.069 |
| PNO Transform | 0.991 | **0.632** | **0.193** | 0.238 |
| PNO Overlaps | 0.191 | **0.147** | **0.032** | 0.072 |
| LMP2 iterations | 0.859 | **0.336** | 0.245 | **0.160** |
| LMP2 graph build | - | 0.123 | - | 0.076 |
| Overlap + Dipole Ints | 0.063 | - | 0.029 | - |
| other | 0.000 | 0.020 | 0.000 | 0.013 |
| **total** | 2.267 | **1.478** | **0.566** | 0.695 |

The port wins ethanol at both thread counts and the chain serially, and is 1.23x psi4 on the chain at ten threads.
One caveat on the ethanol threaded win: psi4's own `Dipole Ints` measures 0.124 s at ten threads against 0.064 s at one, so part of that margin is psi4 getting slower rather than the port getting faster.
The chain is where the work is, and its remaining 0.129 s divides as the graph build 0.076, `Sparsity` 0.049, the transform 0.045, the overlaps 0.040, `DF Ints` 0.023 and `other` 0.013, against a 0.085 s credit from the iterations and 0.029 s psi4 spends on integrals the port takes from its reference.
That gap was 0.171 s until the graph-teardown finding below, and 0.248 before the screened integral source; `DF Ints`, which once led it at 0.119, is now next to last.

Per LMP2 iteration the port is 0.40x psi4 on the chain and 0.80x on ethanol single threaded, and 0.68x (chain) to 0.91x (ethanol) threaded: the iteration engine is ahead everywhere measured, by the largest margin where the pairs are smallest and most numerous.
It was 1.03x and 1.42x threaded before the couplings and the residual both became grouped batched GEMMs, which put every shape class under one OpenMP region and took the chain from 754 batched calls per iteration to 13.
The serial iteration is about a fifth better than before the bucket chooser landed: with no OpenMP region to pay for, it pads tighter than the fixed four buckets it replaced.
The folded body replays under the default executor - an OpenMP team across its nodes would nest the batched GEMMs inside OpenBLAS's threads - so the repack's parallelism lives inside the node instead: `cg::gather` runs its outer walk on an OpenMP team when it is not already inside one, which its disjoint-by-construction writes make safe.

`DF Ints` was the largest item on the chain at 0.164 s, and closing it is the clearest case in this port of measuring the right thing before optimizing.
The phase looks like a transform, and the port's transform is written densely: two einsums over the full AO and PAO spaces, of which the second is 4.79 GFLOP at ethanol/cc-pVTZ against the first's 0.67.
Restricting that second contraction to the domains the solver will actually read is the natural optimization, and it would have been worth almost nothing.
Profiled at ten threads the two einsums together were 18 ms of the chain's 164 and 13 ms of ethanol's 117; what the rest was, was `DFHelper::initialize` building the dense AO integrals (94 ms on the chain, 65 on ethanol) plus two full copies of that buffer on its way into an einsums tensor, 80 MiB on the chain and 98 on ethanol.
Flops were the wrong currency: at ten threads a GEMM of that size is a few milliseconds and the memory traffic around it is not.
So the phase did not get cheaper by transforming less.
It got cheaper by *building* less, which is what psi4 does: a screened shell-triplet loop straight into each auxiliary atom's domain, so the dense `(Q|mn)` is never formed.

That needed a producer that can answer for scattered domains, which no psi4 entry point offered - `DFHelper::get_AO_tensor` slices `[start, stop)` slabs and cannot express them - so it is a second psi4 patch alongside the one that added `get_AO_tensor`: `LocalQiaBuilder` in `lib3index`, a standalone builder that takes per-atom LMO and PAO lists and returns one small block per auxiliary atom.
The seam it plugs into was already there: `compute_qia` declares every domain the run will read before any integral is built, so `ScreenedQiaSource` answers that declaration without a consumer changing.
What the declaration needed was the *pairing* it used to throw away - not which domains exist, but which orbitals are read against which atom - which `_aux_atom_demand` recovers from the pair list as an exact union, so every element any consumer reads is built by construction rather than by trusting psi4's own extended-map derivation.
Chain6 `DF Ints` went 0.164 to 0.071 at ten threads and 0.373 to 0.138 serially; ethanol 0.116 to 0.055 and 0.239 to 0.125.

Two things about that row are worth stating plainly rather than leaving to be inferred.
Of the 0.071 s remaining on the chain, 0.043 is the three-index work and 0.021 is parsing the RIFIT auxiliary basis-set file, which psi4 also pays but outside the timers this table reads; the actual integral build is at parity with psi4's own.
And the row got 0.010 s cheaper for a reason that is bookkeeping rather than speed: `grid_block_provider` used to build the DFT grid while assembling the reference, and that grid is a differential-overlap input, so its cost now lands in `Sparsity` where it is used - which is part of why `Sparsity` reads higher here than in the previous table.
A further 0.005 s is real: the auxiliary metric now comes from `FittingMetric` rather than `MintsHelper::ao_eri`, which is the same integrals threaded and symmetry-aware, and is the routine psi4 uses for that block.

The screened source is the first one that is not exact, and the distinction it draws is the one psi4 draws.
Its shell-pair tolerance is controlled screening - the error vanishes with the tolerance - but its coefficient tolerances restrict which basis functions enter each atom's transform, and that stays an approximation at exact arithmetic, mitigated by the Boughton-Pulay refit inside the builder.
So `screening_threshold` reports the largest of the three and means domain-restricted, not Schwarz-screened, and `--integrals dfhelper` remains available and remains exact.
At psi4's own NORMAL thresholds the whole approximation is worth 1.9e-14 Eh of the chain's correlation energy, against a PNO truncation correction eight orders of magnitude larger; with all three tolerances switched off the source reproduces the dense oracle on every declared block to 3e-15 on all six fixture geometries, which is the gate that guards the indexing.

The transform's chain share used to be 0.24 s, the largest single item in these tables, and the cause was in the planning half rather than the numerics: it issued one domain-sized eigendecomposition at a time, which threads made slower rather than faster, and batching them across the independent domains took that phase from 0.42 s to 0.28.
`Sparsity` had the same disease in its dipole prescreen and got the same fix, which is worth 0.05 s on ethanol at ten threads and nothing at all on the chain, whose `Sparsity` time is the differential-overlap integration rather than the prescreen.
Ethanol has two distinct PAO domains and never saw the transform's version of it.

The differential-overlap integration is a different disease with a different cure, and the difference is worth keeping.
`compute_doi` streams the DFT grid a block at a time - 186 blocks averaging 84 points on the chain - and used to do seven eager calls per block, so 1302 calls stood in front of 5 ms of arithmetic.
Capturing those same per-block chains into one graph was tried first, because that is what fixed the three phases above, and it moved the phase from 44 ms to 40: those phases were losing to thread contention, where reordering is the whole fix, and this one is losing to call count, where capturing a call costs what dispatching it costs.
What worked was issuing fewer, larger calls - two grouped batched GEMMs for the per-block collocations, three elementwise operations over the concatenated stream, two more grouped batches back down - seven calls in total, and 44 ms to 23.
The residue is 7 ms of psi4's own collocation, which the bridge streams serially because `compute_functions` is called from Python, and 5 ms of arithmetic.
Each block's GEMM is the GEMM it always was, so the DOI matrices, the survivor pair lists and every domain list are bit for bit what they were on all six fixtures.

Two costs are the port's own and worth naming rather than burying.
The one-time graph build scales with the shape classes, and the classes are pairs of buckets, so the chooser pays for its own padding win: on the chain it is 0.123 s serial against 0.097 s at the fixed four buckets it replaced.
The chooser does price this now (`cost.CLASS_FLOOR_ELEMENTS`), which is the only term left pulling against finer buckets since the grouped batched GEMM collapsed the call count; without it the objective simply saturates at `max_buckets` and the chain pays 1.17 s against 0.93.
What the build is actually made of is view construction, not emission: building one chain iteration body constructs 18342 views at about 1.6 us each, and the emission that consumes them is 15 ms of a 75 ms build.
Flattening each shape class once and slicing the flat form, rather than slicing and reshaping per member, took 12390 of those round trips down to 380 and the build from 0.125 s to 0.103.
The other third of the build was `end_capture`, which is the graph's hazard scan - alias resolution, effective I/O, pairwise view-box intersection - and it ran twice, once to build the Kahn adjacency and once to key the dependency lists to the sorted node order.
Since every hazard edge points forward, the sort is the identity and the second scan rederived what the first already had; running it once took `end_capture` from 28 ms to 17 (`Graph::topological_sort`).
The per-view cost was cut next, in einsums core rather than here: the recording path's executor closure shrank to a single pointer (it fits `std::function`'s inline buffer, so recording no longer heap-allocates for the lambda or copies the axis list twice), which took `cg::view_indexed` from 2.7 us to 1.9, and the executor's per-replay scratch vectors are allocated once and reused.
A correction that matters for anyone reading the older analysis: the build's 18k views are EAGER slices - `_allocate_iteration_tensors` runs before the capture and hands long-lived views in - so a batched capture-mode API does not touch them.
`cg.views` (N views of one parent in one crossing) exists now and is the right tool for capture loops, but the port records only ~800 capture-mode views per run, all in the transform, so it deliberately does not use it.

The largest single finding of that round was not in the build at all but in graph TEARDOWN, and it was invisible precisely because it ran inside whichever phase happened to drop a graph.
Every executed graph is registered for the profiler, and on destruction it serialized its complete structure to JSON into a global post-mortem cache - unconditionally, whether anything would ever read it, and the cache grew without bound.
Measured by leaking every graph so destruction fell outside the timers, that was 0.080 s of the chain's ten-thread run: 0.041 in the transform, 0.029 in LMP2.
The serialization is now gated on someone actually collecting (`--einsums:profiler-save` configured, or a viewer attached), and the cache replaces by graph name instead of appending.
A viewer that attaches mid-run still sees every live graph; what it loses is the post-mortem record of graphs that died before it connected.
The rest is the serial layer psi4 does not have: the transform's capture emission and memo-warming solves, the graph build, and the numpy bookkeeping phases - work that is constant while the replays shrink with cores.

Two lessons from taking these numbers, both now guarded.
A result crossing back from a C++ stage converts each list-valued field to a fresh Python list on every attribute access, so indexing `result.field[u]` inside a per-pair loop is quadratic in the pair count - at chain6's 403 upper pairs that was 320 ms of pure conversion hiding in the transform's finish, and it made the C++ backend look slower than Python at exactly the scale it was promoted for.
Consumers read each field into a local once (the generated bindings now say so); with that fixed, both C++ stages win on every geometry measured.
And an unoptimized stage module (empty `CMAKE_BUILD_TYPE`) measures as a severalfold regression with nothing else wrong; `einsums_add_stage_module` now defaults an empty build type to Release, but say what you mean when configuring.

## Design decisions

The rationale lives with the code it explains; these are the load-bearing ones and where to read about them.

* **Every per-pair quantity lives in contiguous stores, padded to bucketed shapes** (`base.py`: `new_pair_stores`, `_choose_buckets`).
  psi4 keeps one small matrix per pair; here elementwise work over all pairs is a single graph node and every coupling GEMM has a batchable shape.
  Padding costs elements and buckets cost batched calls, and how that trade lands is a property of the machine rather than the molecule: a call is nearly free serially and costs tens of microseconds of OpenMP team launch on ten threads.
  So the bucket count is chosen rather than fixed, against the region cost `einsums.hardware` measures at startup (`dlpno/cost.py`).
  It also prices the one-time graph build, which grows with the shape classes and is the only thing left pulling against finer buckets now that the grouped batched GEMM has collapsed the call count.
  On ethanol/cc-pVTZ it picks 15, 12, 9 and 7 buckets at 1, 2, 4 and 10 threads, and 12, 8, 7 and 6 on the chain.
  `--buckets` still pins it, which is how the calibration measurements were taken.
* **The LMP2 residual is a handful of batched GEMMs, not thousands of small ones** (`mp2.py`: `plan_pno_couplings` and the residual emission).
  Both halves of the Fock coupling are grouped - by partner on one side, by pair on the other - with one permuting gather per shape class between them.
* **The whole iteration is one graph with a loop node** (`mp2.py`: `lmp2_iterations`), convergence test and DIIS (`einsums.graph.diis`) in the loop predicate.
* **Setup phases are separate graphs replayed under the OpenMP executor** (`base.py`: `_run`), because the parallelism worth having is across independent per-pair chains - and never a Python thread pool, which silently corrupts results under the OpenMP-built OpenBLAS.
* **The three-index integrals are a request, not an array** (`integrals.py`: `Demand` and `ThreeIndexSource`).
  `compute_qia` declares every block the run will read - before any integral exists, which `prep_sparsity` running first is what makes possible - and a source satisfies that however it likes.
  The dense source satisfies it by ignoring it, which is what makes it exact and the oracle; the screened one takes the per-atom pairing and builds nothing else.
  Asking for `(Q|mn)` and slicing it is the one shape of request that can be neither screened nor threaded, so the seam belongs after the first transform rather than before it.
* **psi4 stays behind a buffer-level seam** (`reference.py`, `psi4_source.py`): everything crosses as plain numpy arrays, so neither library is built against the other.

## Tools

| script | answers |
| --- | --- |
| `run_dlpno_mp2.py` | is the port right, against psi4 DF-MP2 and DLPNO-MP2 at once |
| `run_dlpno_mp2_offline.py` | the same solver on one frozen fixture, no psi4 needed |
| `run_fixtures.py` | the psi4-free suite: every fixture, both threshold settings |
| `dump_reference.py` | freeze a new fixture (needs psi4) |
| `dump_energies.py` | did a refactor move any digit (full `repr`, diff two runs) |
| `check_backends.py` | do a stage's two backends agree, and did the selected one run |
| `check_integral_sources.py` | what each three-index source costs and gives up; `--sweep` is the exactness gate |
| `bench_vs_psi4.py` | wall-clock against psi4's C++ DLPNO-MP2, phase against phase |
| `sweep_chain.py` | the locality claim: kept-pair fraction must fall as a chain grows |
| `sweep_separation.py` | pair prescreening through the separations where it decides |
| `stage_state_report.py` | how many `self` fields each phase touches (promotion width) |
| `test_reference_io.py` | fixture format round-trips (numpy only; `python -m pytest`) |

Each script's docstring says how to run it and why it is written the way it is.
