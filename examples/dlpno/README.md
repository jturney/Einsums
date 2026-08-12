# DLPNO-MP2, CCSD and (T) on the einsums ComputeGraph

A port of psi4's DLPNO module (`psi4/src/psi4/dlpno`) with the tensor algebra expressed in einsums and the solvers captured as ComputeGraphs.
All three methods are complete and validated against psi4 two ways: DLPNO-MP2, DLPNO-CCSD, and the triples in both their semicanonical (T0) and iterative (T) forms.

DLPNO fits deferred execution unusually well: it is thousands of small dense operations whose shapes and dependency pattern are fixed for a whole calculation and change only in their values, which is exactly the capture-once, replay-many shape.
Every contraction is captured into a graph once and replayed, so the per-iteration Python cost is a few `execute()` calls however many GEMMs they stand for; the LMP2 iteration itself runs as a single graph with a loop node and DIIS as its predicate.

The coupled-cluster layers are validated and now measured: at one thread the full DLPNO-CCSD(T) cascade on ethanol/cc-pVTZ runs in 212.3 s against psi4's 218.6 s.
Threaded they are still behind, by 1.8x at ten cores on the same molecule, and the gap is down to two phases: the coupled-cluster residual, which replays serially until its shared scratch is privatized, and the PNO integral build, which psi4 threads over pairs and this port drives from Python.

Against psi4's native C++ implementation, the MP2 port is faster on three of the four benchmark configurations: 0.65x psi4's wall time on a six-monomer water chain and 0.63x on ethanol/cc-pVTZ single threaded, and 0.79x on ethanol at ten threads.
The one configuration psi4 wins is the chain at ten threads, at 1.23x, and the gap is measured rather than mysterious: it is serial setup work (graph construction, capture emission) that stays constant while the replays shrink with cores.
Current numbers and their provenance are in [Performance against psi4](#performance-against-psi4); `bench_vs_psi4.py` reproduces them on your machine, phase against phase.

## Requirements

* An in-tree Einsums build with the Python bindings (`-DEINSUMS_BUILD_PYTHON=ON`), on `PYTHONPATH` as `build/lib`.
* psi4, for everything that builds a reference live; the frozen-fixture path below needs no psi4 at all.
* Optionally, the C++ stage backends, built separately (see [The two backends](#the-two-backends)).

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

Note `--threads`: importing psi4 clamps the process-wide OpenMP thread count to 1, so einsums runs serial unless it is set; `OMP_NUM_THREADS` alone will not do it.
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
psi4's column doubles as the drift control: it reproduces across sessions to within about 3%, which is the tolerance to read every number below at.
The port runs its hybrid configuration (`--backend compute_pno_overlaps=cpp,transform_pnos=cpp`) with the screened integral source, so both sides build only the three-index blocks their domains will read; correlation energies agree to 2e-08 (ethanol) and 1e-08 (chain).
Every phase either side times appears in the tables, including the ones the other side has no analogue for, and the difference between the rows and the total prints as `other`, so a phase cannot silently go missing.
The LMP2 rows split the one-time graph build (allocate, capture, optimize - a cost psi4 has no analogue of, which amortizes with iteration count) from the iterations themselves.

**Ethanol/cc-pVTZ** - the compact molecule in the larger basis.

| phase | psi4, 1 thread | port | psi4, 10 threads | port |
| --- | --- | --- | --- | --- |
| Setup Orbitals | **0.004** | 0.054 | **0.005** | 0.034 |
| Sparsity | **0.017** | 0.039 | **0.017** | 0.042 |
| DF Ints | **0.305** | 0.519 | **0.093** | 0.484 |
| Initial Prescreening | 0.542 | **0.341** | **0.102** | 0.271 |
| PNO Transform | 0.330 | **0.275** | **0.065** | 0.155 |
| PNO-LMP2 iterations | **0.438** | 0.576 | **0.155** | 0.269 |
| Compute PNOs (CCSD) | 0.176 | **0.086** | **0.033** | 0.075 |
| PNO Integrals | **2.239** | 2.279 | **0.408** | 1.880 |
| PNO Overlaps | **0.114** | 0.121 | **0.021** | 0.068 |
| LCCSD iterations | **5.349** | 8.645 | **1.423** | 9.994 |
| LCCSD graph build | - | 2.261 | - | 2.064 |
| Triples Sparsity | **0.006** | 0.019 | **0.006** | 0.016 |
| TNO transform | 1.007 | **0.518** | **0.186** | 0.269 |
| LCCSD(T0) | **4.872** | 6.313 | **0.941** | 4.264 |
| LCCSD(T) iterations | **2.555** | 4.763 | **0.531** | 3.033 |
| Overlap + Dipole Ints | 0.063 | - | 0.027 | - |
| other | 0.269 | **0.162** | **0.097** | 0.160 |
| **total** | **18.286** | 26.970 | **4.110** | 23.078 |

Correlation energies agree to 1.6e-08 (1 thread) and 2.0e-08 (10 threads), so the two sides are solving the same problem.
Both iterative-(T) levers are ON, which is their default: the table is the port as it runs out of the box.

**The port is 1.5x psi4 serially and 5.6x at ten threads, and the difference between those two numbers is the whole story: the port barely gains from cores.**
Its total moves 26.970 to 23.078 (1.2x) while psi4's goes 18.286 to 4.110, a 4.4x speedup.

That is not uniform across the port, and the split follows the executor exactly.
What gains and what does not follows the executor exactly.
The setup phases replay under an OpenMP executor over their independent per-pair chains (`base.py::_run`): `PNO Transform` 0.275 to 0.155, `TNO transform` 0.518 to 0.269.
So does `LCCSD(T0)`, which replays its per-chunk graphs the same way, 6.313 to 4.264, and `LCCSD(T) iterations`, which threads over triplets, 4.763 to 3.033.
`LCCSD iterations` is the one solver phase still on the serial executor - 8.645 to 9.994, slightly WORSE with ten cores - and it cannot take one until its scratch is privatized: `lccsd.py` shares a single rank-3 buffer across every pair, so an executor would serialize the chains it was meant to overlap.

That row getting slower rather than merely flat is worth keeping in view. A serial replay should be indifferent to the thread count, so something in it pays for cores it does not use, which is the signature of the Eq. 84c defect found earlier - one badly shaped operand that cost a factor of two at ten threads and nothing at one. The iterative (T) showed the same symptom until it got an executor, which overtook rather than explained it.

**This geometry is not sufficient on its own, and that is the most useful thing in the table.**
A defect worth a factor of TWELVE on the iterative (T) at ethanol/cc-pVTZ - three `n_tno^3` scratch blocks allocated per coupling instead of per triplet, all captured and so live for the whole iteration - is invisible here: that row measured 10.205 s before the fix and 9.850 s after, which is inside the run-to-run spread.
The reason is size. At chain n=6 a triplet carries about 22 TNOs, so ninety of those blocks is roughly 7.6 MB and stays in cache; at ethanol/cc-pVTZ it is 52 to 85 TNOs and the same ninety blocks are 100 to 440 MB per triplet.
So a benchmark restricted to the small geometry will report a phase as healthy while it is quadratically sick, and any conclusion drawn from this table about where the time goes needs the larger basis to confirm it.

Two more things worth reading out of the rows rather than the total.
`LCCSD(T) iterations` is 1.9x serially and 5.7x threaded, and what remains at this geometry is work count rather than engine speed: the port takes **9 passes where psi4 takes 6**.
`PNO Integrals` is at parity serially (1.0x) and 4.6x threaded, which is the promotion signature exactly: a per-pair chain psi4 threads over pairs and the port drives from Python.

**Ethanol/cc-pVTZ.** 316 triplets, 51.9 TNOs each, and the configuration the chain cannot substitute for.

| phase | psi4, 1 thread | port | psi4, 10 threads | port |
| --- | --- | --- | --- | --- |
| Setup Orbitals | **0.003** | 0.031 | **0.004** | 0.025 |
| Sparsity | **0.016** | 0.073 | **0.016** | 0.043 |
| DF Ints | 0.437 | **0.479** | **0.116** | 0.462 |
| Initial Prescreening | 0.823 | **0.212** | 0.167 | **0.138** |
| PNO Transform | 1.300 | **0.734** | **0.257** | 0.369 |
| PNO-LMP2 iterations | 2.473 | **2.494** | **0.730** | 0.878 |
| Compute PNOs (CCSD) | 0.761 | **0.132** | 0.148 | **0.089** |
| PNO Integrals | 10.531 | **10.264** | **2.271** | 8.234 |
| PNO Overlaps | 0.503 | **0.161** | 0.105 | **0.072** |
| LCCSD iterations | 22.243 | **21.012** | **5.246** | 22.108 |
| LCCSD graph build | - | 1.615 | - | 1.556 |
| Triples Sparsity | **0.005** | 0.020 | **0.005** | 0.022 |
| TNO transform | 9.966 | **2.410** | 2.243 | **0.818** |
| LCCSD(T0) | 82.729 | **63.826** | **19.737** | 33.170 |
| LCCSD(T) iterations | **86.434** | 108.683 | **35.057** | 48.519 |
| Overlap + Dipole Ints | 0.096 | - | 0.135 | - |
| other | 0.296 | **0.154** | **0.117** | 0.167 |
| **total** | 218.616 | **212.298** | **66.354** | 116.672 |

Correlation energies agree to 2.1e-08 and 1.5e-08. Both iterative-(T) levers are on, which is their default.

**At one thread the port BEATS psi4 on the full DLPNO-CCSD(T) cascade: 212.3 s against 218.6 s.**
That is the first configuration where the whole coupled-cluster path, triples included, is faster than psi4's native C++, and it is not carried by one lucky row.
The port wins nine of the sixteen phases outright, including every phase it was written to be good at: `LCCSD iterations` **0.9x**, `LCCSD(T0)` **0.8x**, `TNO transform` **0.2x**, `Compute PNOs (CCSD)` **0.2x**, `PNO Overlaps` **0.3x**, `Initial Prescreening` **0.3x**, and `PNO Integrals` - psi4's second-largest phase - at **1.0x**.
`LCCSD(T) iterations` is the one substantial row still behind at 1.3x, and even there the port is faster PER PASS than psi4: 13.6 s against 14.4 s, on 8 passes against psi4's 6.

**At ten threads the port is 1.8x, and the remaining gap is two rows.**
`LCCSD iterations` at 4.2x and `PNO Integrals` at 3.6x are together 30 s of the 50 s difference.
The first is the coupled-cluster residual, the one solver phase still replaying serially, and it stays that way until its shared rank-3 scratch is privatized.
The second is a per-pair chain psi4 threads over pairs and the port drives from Python, which is the promotion signature and the clearest remaining target.
The iterative (T) is no longer among them: it threads over triplets and sits at 1.4x.

**Provenance, and one caveat worth stating plainly.**
These four cells were taken behind a gate that waits for background CPU to fall below 70% of one core, after Spotlight indexing was seen at up to 257%; the level at each cell's start is recorded, and the ethanol one-thread cell began at 163% - tolerable for a one-thread comparison on ten cores, which is the discipline the design notes record, but not ideal.
More importantly, **psi4's ten-thread ethanol figure is not reproducing well today**: 57.8 s, 60.6 s, 66.4 s and 67.5 s across four sessions, a 15% spread against the roughly 3% the MP2 tables above rely on. The port's own figure is stable at 116 to 124 s, and the two most recent readings agree on 1.8x, so that is the number given - but the ten-thread ethanol ratio should be read at about 15%, not 3%.

### How the gap closed, in brief

The chain's ten-thread gap has gone 0.248 s (before the screened integral source), 0.171 (before the graph-teardown gate), 0.129 now; the full record lives in the git history of this directory and of `libs/Einsums/ComputeGraph`.
What follows is the conclusions, because each one changed how the next problem was approached.

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
