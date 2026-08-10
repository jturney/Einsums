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

Measured 2026-08-10 (Apple M-series, 4 performance and 6 efficiency cores, on mains power, load under 2, best of three interleaved runs) with `bench_vs_psi4.py`, which runs psi4's native C++ DLPNO-MP2 in a subprocess and this port in process, same thread count, same converged reference, SCF excluded from both.
The port runs its hybrid configuration (`--backend compute_pno_overlaps=cpp,transform_pnos=cpp`); correlation energies agree to 2e-08 (ethanol) and 1e-08 (chain).
The three-index integrals come from `DFHelper::get_AO_tensor`, so `DF Ints` is threaded rather than the flat, unthreaded `ao_eri` build it used to be.
The bucket count is the automatic one, which chose 15, 12, 9 and 7 on ethanol and 12, 8, 7 and 6 on the chain at 1, 2, 4 and 10 threads.

Every phase either side times appears below, including the ones the other side has no analogue for.
That is worth stating because it was not always so: the table used to print a fixed list of labels and silently dropped whatever was not on it, which hid the port's `Sparsity` phase (16% of an ethanol run) and psi4's `Overlap Ints` and `Dipole Ints`.
`bench_vs_psi4.py` now derives its rows from what was timed and prints the difference between the rows and the total as `other`, so a phase cannot go missing again.
The LMP2 rows split the one-time graph build (allocate, capture, optimize - a cost psi4 has no analogue of, which amortizes with iteration count) from the iterations themselves.

**Ethanol/cc-pVTZ** - the compact molecule in the larger basis.

| phase | psi4, 1 thread | port | psi4, 10 threads | port |
| --- | --- | --- | --- | --- |
| Setup Orbitals | 0.003 | 0.003 | 0.004 | **0.002** |
| Sparsity | **0.012** | 0.088 | **0.013** | 0.050 |
| DF Ints | **0.131** | 0.239 | **0.042** | 0.116 |
| PNO Transform | 1.224 | **0.542** | 0.243 | **0.189** |
| PNO Overlaps | 0.220 | **0.158** | **0.043** | 0.052 |
| LMP2 iterations | 1.171 | **0.840** | 0.347 | **0.284** |
| LMP2 graph build | - | 0.042 | - | 0.033 |
| Overlap + Dipole Ints | 0.096 | - | 0.138 | - |
| other | -0.001 | 0.012 | -0.002 | 0.007 |
| **total** | 2.862 | **1.930** | 0.841 | **0.743** |

**Water chain n=6, cc-pVDZ** - the extended system with many small pairs.

| phase | psi4, 1 thread | port | psi4, 10 threads | port |
| --- | --- | --- | --- | --- |
| Setup Orbitals | 0.004 | **0.002** | 0.005 | **0.002** |
| Sparsity | **0.013** | 0.061 | **0.012** | 0.051 |
| DF Ints | **0.139** | 0.373 | **0.045** | 0.164 |
| PNO Transform | 0.990 | **0.650** | **0.188** | 0.256 |
| PNO Overlaps | 0.190 | **0.152** | **0.033** | 0.075 |
| LMP2 iterations | 0.863 | **0.336** | 0.248 | **0.162** |
| LMP2 graph build | - | 0.123 | - | 0.076 |
| Overlap + Dipole Ints | 0.064 | - | 0.029 | - |
| other | 0.000 | 0.050 | -0.000 | 0.030 |
| **total** | 2.270 | **1.752** | **0.568** | 0.816 |

The port wins ethanol at both thread counts and the chain serially, and is 1.44x psi4 on the chain at ten threads.
One caveat on the ethanol threaded win: psi4's own `Dipole Ints` measures 0.128 s at ten threads against 0.064 s at one, so part of that margin is psi4 getting slower rather than the port getting faster.
The chain is where the work is, and its remaining 0.248 s divides as `DF Ints` 0.119, the graph build 0.076, the transform 0.068, `Sparsity` 0.039, the overlaps 0.042 and `other` 0.030, against a 0.086 s credit from the iterations and 0.029 s psi4 spends on integrals the port takes from its reference.

Per LMP2 iteration the port is 0.39x psi4 on the chain and 0.80x on ethanol single threaded, and 0.65x (chain) to 0.91x (ethanol) threaded: the iteration engine is ahead everywhere measured, by the largest margin where the pairs are smallest and most numerous.
It was 1.03x and 1.42x threaded before the couplings and the residual both became grouped batched GEMMs, which put every shape class under one OpenMP region and took the chain from 754 batched calls per iteration to 13.
The serial iteration is about a fifth better than before the bucket chooser landed: with no OpenMP region to pay for, it pads tighter than the fixed four buckets it replaced.
The folded body replays under the default executor - an OpenMP team across its nodes would nest the batched GEMMs inside OpenBLAS's threads - so the repack's parallelism lives inside the node instead: `cg::gather` runs its outer walk on an OpenMP team when it is not already inside one, which its disjoint-by-construction writes make safe.

`DF Ints` is the largest remaining item on the chain, and it is worth being precise about what it is made of, because the obvious reading is wrong.
The phase looks like a transform, and the port's transform is written densely: two einsums over the full AO and PAO spaces, of which the second is 4.79 GFLOP at ethanol/cc-pVTZ against the first's 0.67.
Restricting that second contraction to the domains the solver will actually read is the natural optimization, and it is not worth doing.
Profiled at ten threads the two einsums together are 18 ms of the chain's 164 and 13 ms of ethanol's 117; what the rest is, is `DFHelper::initialize` building the dense AO integrals (94 ms on the chain, 65 on ethanol) plus two full copies of that buffer on its way into an einsums tensor, 80 MiB on the chain and 98 on ethanol.
Flops were the wrong currency: at ten threads a GEMM of that size is a few milliseconds and the memory traffic around it is not.
So this phase does not get cheaper by transforming less.
It gets cheaper by *building* less, which is what psi4 does: a screened shell-triplet loop straight into each auxiliary atom's domain, so the dense `(Q|mn)` is never formed.
Reaching psi4 here needs a producer that can answer for scattered domains; `DFHelper::get_AO_tensor` slices `[start, stop)` slabs and cannot express them, so it needs a second psi4 patch alongside the one that added `get_AO_tensor`.
`dlpno/integrals.py` already has the seam for it: `compute_qia` declares every domain the run will read before any integral is built, and a screened source can answer that declaration without a consumer changing.
One cheaper lever exists and is deliberately not taken: a Schwarz cutoff of 1e-12 takes the chain's AO build from 106 ms to 76 ms and perturbs `(Q|mn)` by 1.6e-12, but `DFHelperSource` reports `screening_threshold == 0.0` and the untruncated fixtures rest on that being literally true.

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
Cutting the per-view cost itself is the next lever and is not a Python-side one: of the 2.8 us a capture-mode slice costs, 1.9 is inside `cg::view_indexed` recording the node, and it is roughly ten heap allocations rather than any single hot spot.
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
| `bench_vs_psi4.py` | wall-clock against psi4's C++ DLPNO-MP2, phase against phase |
| `sweep_chain.py` | the locality claim: kept-pair fraction must fall as a chain grows |
| `sweep_separation.py` | pair prescreening through the separations where it decides |
| `stage_state_report.py` | how many `self` fields each phase touches (promotion width) |
| `test_reference_io.py` | fixture format round-trips (numpy only; `python -m pytest`) |

Each script's docstring says how to run it and why it is written the way it is.
