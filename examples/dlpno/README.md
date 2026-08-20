# DLPNO-MP2, CCSD and (T) on the einsums ComputeGraph

A port of psi4's DLPNO module (`psi4/src/psi4/dlpno`) with the tensor algebra expressed in einsums and the solvers captured as ComputeGraphs.
All three methods are complete and validated against psi4 two ways: DLPNO-MP2, DLPNO-CCSD, and the triples in both their semicanonical (T0) and iterative (T) forms.

DLPNO fits deferred execution unusually well: it is thousands of small dense operations whose shapes and dependency pattern are fixed for a whole calculation and change only in their values, which is exactly the capture-once, replay-many shape.
Every contraction is captured into a graph once and replayed, so the per-iteration Python cost is a few `execute()` calls however many GEMMs they stand for; the LMP2 iteration itself runs as a single graph with a loop node and DIIS as its predicate.

The coupled-cluster layers are validated and measured, and at one thread the port now BEATS psi4's native C++ on the full DLPNO-CCSD(T) cascade on both benchmark geometries: 189.0 s against 216.3 s on ethanol/cc-pVTZ (0.87x) and 17.6 s against 18.7 s on the six-monomer water chain (0.94x).
At ten threads ethanol stands at 1.12x, with the remaining gap concentrated in the semicanonical triples (`LCCSD(T0)`, 1.5x) rather than spread across the cascade; the small chain geometry threads worse (2.3x), for the reasons its table's discussion gives.
The MP2 port's standing (measured 2026-08-10) is faster than psi4 on three of its four configurations, the exception being the chain at ten threads.
Current numbers and their provenance are in [Performance against psi4](#performance-against-psi4); `bench_vs_psi4.py` reproduces them on your machine, phase against phase.

## Requirements

* An in-tree Einsums build with the Python bindings (`-DEINSUMS_BUILD_PYTHON=ON`), on `PYTHONPATH` as `build/lib`.
* psi4, for everything that builds a reference live; the frozen-fixture path below needs no psi4 at all.
* The C++ stage backends, built and selected automatically with the main build (see [The two backends](#the-two-backends)).

Two of the three integral sources depend on psi4-side entry points that are patches carried in this project's psi4 tree, not yet in released psi4:

| source | psi4 entry point | needs |
| --- | --- | --- |
| `dense` | `MintsHelper::ao_eri` | any psi4 |
| `dfhelper` | `DFHelper::get_AO_tensor` | patched psi4 |
| `screened` | `LocalQiaBuilder` | patched psi4 |

Everything defaults so that stock psi4 works: the validation driver uses `dense`, and a missing entry point is reported as a clear error naming the patch, never as an `AttributeError`.
`screened` is the benchmark configuration, because it is what makes the integral phase comparable with psi4's own (both then build only the blocks their domains will read); `dense` and `dfhelper` are exact.

## Quick start

### With psi4

```bash
PYTHONPATH=/path/to/Einsums/build/lib:/path/to/psi4/stage/lib \
    python examples/dlpno/run_dlpno_mp2.py --molecule water-dimer
```

This runs the molecule twice and asserts both results: untruncated against psi4's canonical DF-MP2, and truncated (psi4's NORMAL preset) against psi4's own DLPNO-MP2.
Useful flags: `--basis`, `--localization {BOYS,PIPEK_MEZEY}`, `--t-cut-pno`, `--buckets`, `--threads`, `--integrals {dense,dfhelper}`, `--no-optimize` (skip the graph passes), `--no-diis`.

Note `--threads`: importing psi4 takes the process-wide OpenMP thread count over, setting it to `OMP_NUM_THREADS` if that was exported and to 1 if it was not, so einsums runs serial unless one of the two is set.
None of the psi4-driven scripts are wired into CTest or pytest, since they need a psi4 install.

### Without psi4

A converged reference can be frozen to an `.npz` fixture on a machine that has psi4, then replayed anywhere the library builds:

```bash
# once, with psi4 importable. --with-cc also records psi4's DLPNO-CCSD and
# DLPNO-CCSD(T) energies and the classification counts, at the CC branch's own
# NORMAL preset; without it the fixture carries the MP2 references only.
python examples/dlpno/dump_reference.py --molecule water --basis cc-pvdz --with-cc \
    --out examples/dlpno/fixtures/water-ccpvdz.npz

# thereafter, einsums only
PYTHONPATH=/path/to/Einsums/build/lib \
    python examples/dlpno/run_dlpno_mp2_offline.py examples/dlpno/fixtures/water-ccpvdz.npz

# every fixture, against every recorded reference
PYTHONPATH=/path/to/Einsums/build/lib python examples/dlpno/run_fixtures.py
PYTHONPATH=/path/to/Einsums/build/lib python examples/dlpno/run_fixtures.py --method 'ccsd(t)'
```

All six fixtures carry CC and (T) references, so the coupled-cluster path is checkable with no psi4 present - including the frozen-core one, which is the only fixture exercising `n_core > 0` through the triples.

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

## Layout

```
dlpno/                the package; see its __init__ docstring for the module map
  base.py             orbitals, PAOs, screening, domains, DF integrals, PNO transform
  mp2.py              DLPNO-MP2: the phase sequence and the energy accounting
  lmp2_solver.py      the local MP2 iteration: one loop graph, DIIS predicate
  layout.py           how per-pair blocks are bucketed into contiguous stores
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

## What is validated

**Untruncated, against canonical DF-MP2.**
With PNO truncation off and full domains, local MP2 in the PAO space *is* canonical MP2 in a non-canonical basis, so the correlation energy must match psi4's DF-MP2 to machine precision.
It does, to 1e-13 or better on every molecule in the driver.
This is the check that pins the port down: the PAO construction, the local density fit, the PNO machinery, the residual, and the solver all have to be right simultaneously for it to pass.

**Truncated, against psi4's own DLPNO-MP2.**
Both sides apply the same three truncations - PNO occupation cutoff, differential-overlap PAO and auxiliary domains, dipole pair prescreening - so this is an exact comparison, and it agrees to ~1e-12 on the fixture set, with the far dimer dropping exactly the pairs psi4 drops.

**Coupled cluster, against two independent oracles and then against psi4.**
The CC layers are pinned harder than the MP2 ones, because a coupled-cluster energy is a fixed point rather than a closed form and two codes reach it along different DIIS trajectories.
So the sharp check is not a converged energy at all: `canonical_ccsd.py` and `canonical_triples.py` are numpy implementations sharing no code with the port, and each is itself pinned by a SECOND implementation before anything is judged against it - spin-adapted against spin-orbital, agreeing to 1e-15 and 3e-18 respectively.
Against those, the port's CCSD residuals agree to 7.7e-12 relative at arbitrary probe amplitudes an order of magnitude past the physical ones, and its (T0) to 1.6e-16 evaluated at the port's own converged amplitudes.
Untruncated, local CCSD is canonical DF-CCSD to 7.8e-12 and the iterative (T) is canonical DF-CCSD(T) to 1e-12.
Truncated against psi4 at NORMAL, on water, both water dimers and methanol: every energy term and every classification count matches, CCSD at 2.3e-12 to 2.1e-9, (T0) at 1.5e-11 to 1.1e-10, and the full (T) at 1.1e-9 to 6.5e-9.

**One property worth knowing before reading a (T) number.**
(T0) is NOT invariant to a rotation of the occupied orbitals, where CCSD and the iterative (T) are: its denominator carries only the diagonal of the occupied Fock matrix, so in the localized basis this port works in it drops the coupling between triplets.
On water/cc-pVDZ that is 1.5e-4 Eh, five percent of the correction.
It is the approximation rather than an error, and it is why the (T0) gate is run in two occupied bases while the (T) gate needs only one.

**The screened integral source, against the dense oracle.**
`check_integral_sources.py --sweep` runs the domain-restricted source with every tolerance switched off, where it must reproduce the dense transform on every block it declares; it does, to 3e-15 across all six fixture geometries.
At psi4's own NORMAL tolerances its whole domain approximation is worth ~2e-14 Eh of the chain's correlation energy, eight orders of magnitude under the PNO truncation correction it sits beneath.

One accuracy caveat worth knowing: the 1e-13 figures hold where nothing sits near a threshold.
`sweep_separation.py` walks the dimer separation and finds ~1e-7 disagreement where the domains first straddle both monomers, with identical domains, pairs and PNO counts on both sides - well under the truncation error there, but real.
The runner's assertion tolerance is scaled to the truncation correction for this reason.

## The two backends

Three stages exist twice: `compute_pno_overlaps`, `transform_pnos` and `compute_pno_integrals`, each a Python numerics module in the package and a C++ backend under `cpp/`.
Both implement the same contracts, both are selectable at runtime, and neither Python side is retired; the C++ side is regenerated from the Python contracts by the hybrid framework's promote tool:

```bash
python -m einsums.stages promote examples/dlpno/dlpno/stages.py \
       --out examples/dlpno/cpp --license-header devtools/LicenseHeader.txt
```

The port sources under `cpp/src/` are hand-written, and the build file and differential tests are scaffolded once and then yours; the regenerated files (headers, bindings) carry a `promote-hash:` and are refused if edited by hand - a contract change belongs in `contracts.py`.

The main build compiles the C++ side automatically when `EINSUMS_BUILD_PYTHON` is on, and it still builds it the way an external developer would: an external project configured against the build tree's own exported `EinsumsConfig`, never against in-tree targets, so the external consumer path is exercised on every build rather than proven nothing about.
The module lands in `build/lib` next to the `einsums` package, and `dlpno.stages` selects the compiled backends automatically whenever `dlpno_stages` is importable - so on the standard `PYTHONPATH=build/lib` the hybrid configuration is what runs, with nothing to remember.

Selection precedence, weakest first: the autoload, then `EINSUMS_STAGE_BACKEND`, then an explicit `--backend` spec.
`EINSUMS_STAGE_BACKEND=python` is the off switch, and per-stage overrides read the other way now that cpp is the default:

```bash
python examples/dlpno/run_fixtures.py --backend transform_pnos=python
```

The by-hand path still works, against an *installed* Einsums, for a module built outside this tree:

```bash
cmake --install /path/to/Einsums/build --prefix /tmp/einsums-install
cmake -S examples/dlpno/cpp -B /tmp/build-dlpno-stages -GNinja \
      -DCMAKE_BUILD_TYPE=Release \
      "-DCMAKE_PREFIX_PATH=/tmp/einsums-install;$CONDA_PREFIX" \
      -DPython3_EXECUTABLE=$CONDA_PREFIX/bin/python
cmake --build /tmp/build-dlpno-stages
```

The two backends of a stage agree bit for bit, because both emit the same einsums operations on the same values; `check_backends.py` asserts exactly that, and `check_backends.py --prove` also proves which backend actually ran, since perfect agreement is what a silently-unselected backend produces too.

## Performance against psi4

Measured 2026-08-20 (Apple M-series, 4 performance and 6 efficiency cores, on mains power, one run per cell on an otherwise idle machine) with `bench_vs_psi4.py`, which runs psi4's native C++ DLPNO in a subprocess and this port in process, same thread count, same converged reference, SCF excluded from both.
psi4's column doubles as the in-run drift control; it has historically reproduced to about 3% on most cells, with the ten-thread ethanol cell known to spread to about 15%, and those are the tolerances to read the numbers at.
The port runs its hybrid configuration automatically - the three compiled stage backends are built with the main build and selected whenever `dlpno_stages` is importable - and the coupled-cluster rows use the dense integral source by construction, which is the `DF Ints` handicap the header describes.
Every phase either side times appears in the tables, and the difference between the rows and the total prints as `other`, so a phase cannot silently go missing.
The one-time LCCSD graph build (allocate, capture, optimize - a cost psi4 has no analogue of, which amortizes with iteration count) is printed under the total and excluded from it, because a one-time cost mixed into a steady-state comparison misprices both.

**Water chain n=6, cc-pVDZ** - the extended system in the smaller basis: 326 triplets at 18.3 PNOs per pair average.

| phase | psi4, 1 thread | port | psi4, 10 threads | port |
| --- | --- | --- | --- | --- |
| Setup Orbitals | **0.004** | 0.055 | **0.005** | 0.038 |
| Sparsity | **0.017** | 0.039 | **0.018** | 0.042 |
| DF Ints | **0.321** | 0.541 | **0.097** | 0.512 |
| Initial Prescreening | 0.539 | **0.349** | **0.095** | 0.288 |
| PNO Transform | 0.330 | **0.236** | **0.061** | 0.101 |
| PNO-LMP2 iterations | **0.440** | 0.554 | **0.173** | 0.256 |
| Compute PNOs (CCSD) | 0.174 | **0.083** | **0.033** | 0.079 |
| PNO Integrals | 2.375 | **2.163** | **0.411** | 0.481 |
| PNO Overlaps | **0.116** | 0.116 | **0.022** | 0.077 |
| LCCSD iterations | 5.484 | **4.487** | **1.438** | 2.337 |
| Triples Sparsity | **0.006** | 0.015 | **0.006** | 0.016 |
| TNO transform | 1.002 | **0.524** | **0.212** | 0.292 |
| LCCSD(T0) | **4.965** | 5.274 | **1.060** | 2.924 |
| LCCSD(T) iterations | **2.600** | 3.055 | **0.533** | 2.501 |
| Overlap + Dipole Ints | 0.060 | - | 0.028 | - |
| other | 0.276 | **0.091** | 0.105 | **0.099** |
| **total** | 18.709 | **17.581** | **4.297** | 10.044 |
| LCCSD graph build (excluded) | - | 0.407 | - | 1.037 |

Correlation energies agree to 1.6e-08 (1 thread) and 2.4e-09 (10 threads), so the two sides are solving the same problem.
Both iterative-(T) levers are ON, which is their default: the table is the port as it runs out of the box.

**Serially the port wins the chain outright, 17.6 s against 18.7 s (0.94x)**, taking the four largest phases: `LCCSD iterations` at 0.8x, `PNO Integrals` at 0.9x, `TNO transform` at 0.5x and `Initial Prescreening` at 0.6x; the triples rows are the ones still behind, `LCCSD(T0)` at 1.1x and the (T) iterations at 1.2x on 9 Jacobi passes against psi4's 6 in-place.

**At ten threads the chain is 2.3x, and the shape of that gap is smallness, not any single defect.**
At this geometry a pair carries 18 PNOs, psi4's phases shrink to tens of milliseconds, and every port phase keeps a floor the replays cannot shrink: graph submission, region entry, the per-replay bookkeeping of five-thousand-node graphs.
The three rows that matter are the triples - `LCCSD(T) iterations` at 4.7x (2.2x per pass, times 9 passes against 6) and `LCCSD(T0)` at 2.8x - and the residual at 1.6x.
The small geometry still cannot substitute for the large one: a defect that is quadratic in the PNO count is invisible here and glaring at ethanol/cc-pVTZ, so any conclusion drawn from this table needs the larger basis to confirm it.

**Ethanol/cc-pVTZ.** 316 triplets, 39.0 PNOs per pair average, and the configuration the chain cannot substitute for.

| phase | psi4, 1 thread | port | psi4, 10 threads | port |
| --- | --- | --- | --- | --- |
| Setup Orbitals | **0.004** | 0.032 | **0.004** | 0.024 |
| Sparsity | **0.018** | 0.071 | **0.015** | 0.045 |
| DF Ints | **0.454** | 0.479 | **0.112** | 0.485 |
| Initial Prescreening | 0.906 | **0.213** | 0.163 | **0.139** |
| PNO Transform | 1.413 | **0.627** | 0.253 | **0.234** |
| PNO-LMP2 iterations | 2.482 | **2.477** | **0.752** | 0.907 |
| Compute PNOs (CCSD) | 0.763 | **0.130** | 0.139 | **0.097** |
| PNO Integrals | 11.527 | **9.380** | **2.206** | 2.920 |
| PNO Overlaps | 0.550 | **0.162** | 0.100 | **0.076** |
| LCCSD iterations | 22.811 | **16.102** | **4.916** | 6.494 |
| Triples Sparsity | **0.005** | 0.017 | **0.005** | 0.025 |
| TNO transform | 9.990 | **2.411** | 2.172 | **0.803** |
| LCCSD(T0) | 80.016 | **61.359** | **19.849** | 30.450 |
| LCCSD(T) iterations | **84.893** | 95.453 | 36.786 | **33.054** |
| Overlap + Dipole Ints | 0.109 | - | 0.133 | - |
| other | 0.326 | **0.090** | 0.119 | **0.104** |
| **total** | 216.267 | **189.002** | **67.724** | 75.856 |
| LCCSD graph build (excluded) | - | 0.293 | - | 0.754 |

Correlation energies agree to 2.1e-08 and 1.3e-08. Both iterative-(T) levers are on, which is their default.

**At one thread the port beats psi4 on the full cascade by 13%: 189.0 s against 216.3 s.**
It wins eleven of the fourteen phases outright, including every heavy one: `LCCSD iterations` **0.7x**, `LCCSD(T0)` **0.8x**, `PNO Integrals` **0.8x**, `TNO transform` **0.2x**, `PNO Transform` **0.4x**.
The one substantial row behind is `LCCSD(T) iterations` at 1.1x, and even there the port is faster per pass - 11.7 s against 14.1 - on 8 Jacobi passes against psi4's 6 in-place.

**At ten threads ethanol is 1.12x, and the remaining gap is nearly one row.**
`LCCSD(T0)` at 1.5x is 10.6 s of the 8.1 s total difference - more than the whole of it, since the port WINS the biggest row, `LCCSD(T) iterations`, at 0.9x.
Behind it come the residual (`LCCSD iterations`, 1.3x, 1.6 s), `PNO Integrals` (1.3x, 0.7 s) and `DF Ints` (4.3x, 0.4 s, the dense-source handicap).
The residual's own per-iteration figure is 1.4x against psi4's, softened in the row because the port converges in 17 iterations to psi4's 19.

**Provenance.**
One run per cell rather than the earlier best-of-three, leaning on the in-run psi4 control; the four cells ran back to back on an idle machine on mains power.
psi4's ten-thread ethanol figure landed at 67.7 s, inside its historically noisy 57.8 to 67.5 s band, so that cell's ratio carries the wider tolerance noted above.

### How the gap closed, in brief

The ethanol ten-thread CCSD(T) cascade has gone 1.8x (2026-08-10), then through the 2026-08-19/20 campaign to 1.12x; the full record lives in the git history of this directory and of `libs/Einsums/ComputeGraph`.
What follows is the conclusions, because each one changed how the next problem was approached.

* **A measured plan beats a modeled one only when the measurement is clean, and it never is.**
  A chain of cost-model defects ended at a structural one: a re-plan computed from timings taken under the previous plan can only ever narrow it, because co-run contention inflates the area bound past the critical path.
  The re-plan is now a timed TRIAL - the candidate widths must beat the incumbent on the wall clock of one replay - so a polluted measurement cannot undo a good plan.
* **The PNO integral build took psi4's own shape.**
  Five staged phases with barriers became one cost-sorted parallel pass over pairs, each pair's gathers consumed cache-warm; the phase went from 4x psi4 threaded to about parity, and the compiled stage backends that carry it are now built and selected automatically.
* **The residual's heaviest term is a sandwich, and its dressed factor never needs to exist.**
  A traffic census (two thirds of the iteration's bytes were re-streams of the same data) led to `cg::grouped_sandwich`: the Eq. 93 T1-dressing and the Eq. 76 sandwich as one node, the dressed slice built per auxiliary block in cache, eight full streams of every strong pair's `(Q|ab)` reduced to one.
* **Two levers measured and rejected are recorded as firmly as the ones that shipped.**
  Chain-affinity scheduling (built, affinity went from chance to 65%, wall time unmoved - the intermediates were never the traffic) and finer PNO buckets (no reproducible signal above machine noise); the census data that killed them is what found the sandwich.
* **One-time graph builds are excluded from the compared totals**, because psi4 has no analogue and a one-time cost mixed into a steady-state comparison misprices both; the build is printed under each total.

* **`DF Ints` got cheaper by building less, not transforming less.**
  The dense transform was 18 ms of a 164 ms phase; the rest was building and copying AO integrals the domains would never read.
  `LocalQiaBuilder` (a psi4-side patch) now builds only the blocks the solver declares per auxiliary atom, taking the phase to 0.069 s, of which 0.043 is integrals and 0.021 is parsing the auxiliary basis-set file, a cost psi4 pays outside its timers.
  The integral build itself is at parity with psi4's own.
* **A graph fixes thread contention, not call count.**
  Phases losing to one-small-BLAS-call-at-a-time under threading (the transform's eigendecompositions, the dipole prescreen, the fits) were fixed by batching across their independent domains.
  The differential-overlap quadrature was losing to call count instead, and capturing it changed nothing; issuing seven grouped calls instead of 1302 took it from 44 ms to 23.
* **The LMP2 iteration is a handful of grouped batched GEMMs.**
  754 batched calls per iteration became 13, which is what moved the threaded iteration from 1.03x psi4 to 0.68x; the bucket chooser (`cost.py`) sets the padding-versus-call-count trade per machine.
* **Graph teardown used to serialize every dying graph to JSON** for the profiler's post-mortem cache, unconditionally, inside whichever phase dropped the graph - 0.080 s of the chain's ten-thread run, found by leaking graphs so destruction fell outside the timers.
  It is now gated on someone actually collecting.
* **The graph build's cost is eager view construction plus the hazard scan**, both since reduced (flatten-once slicing, a single hazard scan per sort, a 2.7-to-1.9 us recording path in einsums core).
  A note for anyone extending this: those 18k build views are eager slices made before capture begins, so capture-side batching (`cg.views`) does not apply to them; the port records only ~800 capture-mode views per run.

## Design decisions

The rationale lives with the code it explains; these are the load-bearing ones and where to read about them.

* **Every per-pair quantity lives in contiguous stores, padded to bucketed shapes** (`base.py`: `new_pair_stores`, `_choose_buckets`).
  psi4 keeps one small matrix per pair; here elementwise work over all pairs is a single graph node and every coupling GEMM has a batchable shape.
  Padding costs elements and buckets cost batched calls, and how that trade lands is a property of the machine rather than the molecule, so the bucket count is chosen at runtime against the OpenMP region cost `einsums.hardware` measures at startup (`dlpno/cost.py`); `--buckets` pins it.
* **The LMP2 residual is a handful of batched GEMMs, not thousands of small ones** (`mp2.py`: `plan_pno_couplings` and the residual emission).
  Both halves of the Fock coupling are grouped - by partner on one side, by pair on the other - with one permuting gather per shape class between them.
* **The whole iteration is one graph with a loop node** (`mp2.py`: `lmp2_iterations`), convergence test and DIIS (`einsums.graph.diis`) in the loop predicate.
* **Setup phases are separate graphs replayed under the OpenMP executor** (`base.py`: `_run`), because the parallelism worth having is across independent per-pair chains - and never a Python thread pool, which silently corrupts results under the OpenMP-built OpenBLAS.
* **Solver graphs are planned per node instead** (`base.py`: `plan_widths`), because their parallelism is not all of one kind.
  `Graph.plan_threads()` chooses a thread width for every node from a cost model, and a re-plan computed from a replay's measured timings is adopted only if a timed trial replay beats the cold plan on the wall clock - estimates from the two worlds are not comparable, so the clock referees.
  `DataflowExecutor` guarantees each node's kernel exactly its planned width while the process-wide width budget keeps their sum on the machine.
  That replaces an all-or-nothing split: under the OpenMP executor a node alone on its execution level ran unwrapped and could thread its own contraction while a node sharing a level could not, so a structural accident decided whether a fat GEMM got the machine, and the (T) phase carried a calibrated cutoff on the plan's mean TNO count to work around it.
  A graph whose nodes are all too small to widen is left on the OpenMP executor, which is the right answer for the four one-node batched graphs of `lccsd_t0.py` and for the coupled-cluster residual at small pair sizes.
* **The three-index integrals are a request, not an array** (`integrals.py`: `Demand` and `ThreeIndexSource`).
  `compute_qia` declares every block the run will read - before any integral exists, which `prep_sparsity` running first is what makes possible - and a source satisfies that however it likes.
  The dense source satisfies it by ignoring it, which is what makes it exact and the oracle; the screened one takes the per-atom pairing and builds nothing else.
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
