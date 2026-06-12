# psi4 → Einsums bridge checks

Reference scripts that validate psi4's `MintsHelper` returning symmetrized
(SO-basis) integrals as `einsums::TiledRuntimeTensor`. They are not wired
into CTest or pytest, since they require a psi4 build linked against Einsums,
which the Einsums test suite does not provide. Run them by hand.

| Script | Checks |
| --- | --- |
| `matrix_alias.py` | `Matrix.to_einsums_tiled()` exposes a symmetry-blocked psi4 `Matrix` as a `TiledRuntimeTensor` that aliases the irrep blocks zero-copy, mapping block h to tile (h, h^symmetry). Storage is shared both ways: writes through the Matrix or the tensor are visible to both. This is the zero-copy bridge toward an Einsums-tensor backend for `Matrix`. |
| `so_one_electron_tiled.py` | Uses `Matrix.to_einsums_tiled()` on the SO overlap/kinetic/potential matrices. Each tile equals the per-irrep block, and mutating the matrix shows through the tile, verifying the alias on real integrals. |
| `so_two_electron_eri_tiled.py` | `MintsHelper.so_eri_tiled()` returns rank-4 SO ERIs via the legacy `TwoBodySOInt` functor path. C1 matches `ao_eri()` exactly; C2v tiles are all symmetry-allowed irrep quadruples with 8-fold permutational symmetry. |
| `ao_eri_dense.py` | `MintsHelper.ao_eri_einsums()` returns dense rank-4 AO ERIs via the shell-batched `compute_shell` primitive. It matches `ao_eri()` and obeys 8-fold permutational symmetry. |
| `df_three_index.py` | `DFTensor.Qso_einsums()`, `Qov_einsums()`, and `Qvv_einsums()` return the density-fitted 3-index `(Q\|pq)` as a dense rank-3 `RuntimeTensor`, since DF has no symmetry. They exactly match the `Qso()`/`Qov()`/`Qvv()` matrices reshaped, and reconstruct the AO ERIs within DF error. |
| `df_mp2_energy.py` | End-to-end DF-MP2 the way it is actually run, with every tensor op in einsums and no numpy in the compute. The algorithm is pair-driven over `i≤j` with one `nvir×nvir` GEMM per pair via `einsum`, never forming the O(o²v²) `(ia\|jb)`. It uses `einsum`, `permute`, `axpby`, `outer_sum`, `element_transform`, `direct_product`, and `dot`. Matches psi4's DF-MP2 to machine precision. Pass `--profile [FILE]` for an einsums profile report at exit. |
| `df_mp2_graph.py` | The deferred ComputeGraph path. It runs the same memory-optimal pair-driven algorithm as `df_mp2_energy.py`, with no O(o²v²) intermediate. The per-pair work is recorded with `with cg.capture(g):` at roughly 10 nodes per pair, the default passes run via `PassManager.populate_default` and `g.apply`, then `g.execute()`. Matches psi4's DF-MP2 to machine precision. Pass `--show-passes` for a per-pass `modified` table plus the node execution order before and after optimization. |

## Running

Both need `einsums` and `psi4` importable. With an in-tree Einsums build and a
psi4 built against it, as described in the bridge notes, point `PYTHONPATH` at
both and use the conda env's Python:

```bash
PYTHONPATH=/path/to/Einsums/build/lib:/path/to/psi4/cmake-build-debug/stage/lib \
  python examples/psi4-bridge/so_one_electron_tiled.py
PYTHONPATH=/path/to/Einsums/build/lib:/path/to/psi4/cmake-build-debug/stage/lib \
  python examples/psi4-bridge/so_two_electron_eri_tiled.py
```

`import einsums` is enough: it eagerly loads `einsums._core`, registering the
`TiledRuntimeTensor` pybind types so psi4 can hand one back. The Einsums runtime
itself still initializes lazily on first compute use.
