# psi4 → Einsums bridge checks

Reference scripts that validate psi4's `MintsHelper` returning symmetrized
(SO-basis) integrals as `einsums::TiledRuntimeTensor`. They are **not** wired
into CTest or pytest — they require a psi4 build linked against Einsums, which
the Einsums test suite does not provide. Run them by hand.

| Script | Checks |
| --- | --- |
| `so_one_electron_tiled.py` | `MintsHelper.so_overlap_tiled()` / `so_kinetic_tiled()` / `so_potential_tiled()` reproduce the per-irrep blocks of the existing symmetry-blocked `so_*()` matrices. |
| `so_two_electron_eri_tiled.py` | `MintsHelper.so_eri_tiled()` (rank-4 SO ERIs, legacy `TwoBodySOInt` functor path): C1 matches `ao_eri()` exactly; C2v tiles are all symmetry-allowed irrep quadruples with 8-fold permutational symmetry. |
| `ao_eri_dense.py` | `MintsHelper.ao_eri_einsums()` (dense rank-4 AO ERIs via the performant shell-batched `compute_shell` primitive): matches `ao_eri()` and obeys 8-fold permutational symmetry. |
| `df_three_index.py` | `DFTensor.Qso_einsums()` / `Qov_einsums()` / `Qvv_einsums()` (density-fitted 3-index `(Q\|pq)` as dense rank-3 `RuntimeTensor` — DF has no symmetry): exactly match the `Qso()`/`Qov()`/`Qvv()` matrices reshaped, and reconstruct the AO ERIs within DF error. |
| `df_mp2_energy.py` | End-to-end, the way DF-MP2 is actually run, with **every tensor op in einsums** (no numpy in the compute): pair-driven over `i≤j`, one `nvir×nvir` GEMM per pair (`einsum`), never forming the O(o²v²) `(ia\|jb)`. Uses `einsum` / `permute` / `axpby` / `outer_sum` / `element_transform` / `direct_product` / `dot`. Matches psi4's DF-MP2 to machine precision. Pass `--profile [FILE]` for an einsums profile report at exit. |
| `df_mp2_graph.py` | The **deferred / ComputeGraph** path running the **same memory-optimal pair-driven algorithm** as `df_mp2_energy.py` (no O(o²v²) intermediate): the per-pair work is recorded with `with cg.capture(g):` (~10 nodes/pair), the default passes run (`PassManager.populate_default` + `g.apply`), then `g.execute()`. Matches psi4's DF-MP2 to machine precision. Pass `--show-passes` for a per-pass `modified` table + the node execution order before/after optimization. |

## Running

Both need `einsums` and `psi4` importable. With an in-tree Einsums build and a
psi4 built against it (see the bridge notes), point `PYTHONPATH` at both and use
the conda env's Python:

```bash
PYTHONPATH=/path/to/Einsums/build/lib:/path/to/psi4/cmake-build-debug/stage/lib \
  python examples/psi4-bridge/so_one_electron_tiled.py
PYTHONPATH=/path/to/Einsums/build/lib:/path/to/psi4/cmake-build-debug/stage/lib \
  python examples/psi4-bridge/so_two_electron_eri_tiled.py
```

`import einsums` is enough — it eagerly loads `einsums._core`, registering the
`TiledRuntimeTensor` pybind types so psi4 can hand one back. The Einsums runtime
itself still initializes lazily on first compute use.
