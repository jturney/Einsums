# psi4 → Einsums bridge checks

Reference scripts that validate the buffer-level psi4 → Einsums bridge.
psi4's `MintsHelper`/`DFTensor` return only plain psi4 types (Matrices and per-irrep-block numpy buffers plus index metadata), and `einsums.interop.psi4` assembles them into einsums tensors on the Python side.
Neither library links the other, so either can be rebuilt freely with no ABI skew.
The scripts are not wired into CTest or pytest, since they require a psi4 install; run them by hand.
(The interop adapter itself is unit-tested psi4-free in `libs/Einsums/Python/tests/unit/test_interop_psi4_python.py`.)

| Script | Checks |
| --- | --- |
| `matrix_interop.py` | `einsums.interop.psi4.from_matrix` copies a symmetry-blocked psi4 `Matrix` into a rank-2 `TiledRuntimeTensor`, mapping block h to tile (h, h^symmetry). Verifies values and that the tensor owns its storage (writes through the Matrix do not alias through). |
| `so_one_electron_tiled.py` | `from_matrix` on the SO overlap/kinetic/potential matrices. Each tile equals the per-irrep block, with independent storage, on real integrals. |
| `so_two_electron_eri_tiled.py` | `MintsHelper.so_eri_blocked()` returns rank-4 SO ERIs as `([hp,hq,hr,hs], Matrix)` block pairs and `interop.so_eri` assembles the tiled tensor. C1 matches `ao_eri()` exactly; C2v tiles are all symmetry-allowed irrep quadruples with 8-fold permutational symmetry. |
| `ao_eri_dense.py` | `interop.dense` reshapes psi4's `ao_eri()` chemists'-pair matrix into dense rank-4 AO ERIs. It matches the reshaped reference and obeys 8-fold permutational symmetry. |
| `df_three_index.py` | `interop.df_tensor` over `DFTensor.Qso()`/`Qov()`/`Qvv()` yields the density-fitted 3-index `(Q\|pq)` as a dense rank-3 `RuntimeTensor`, since DF has no symmetry. Exactly matches the matrices reshaped, and reconstructs the AO ERIs within DF error. |
| `cc_half_transform.py` | `MintsHelper.mo_bra_half_transform` (integral-direct, no N^4 AO tensor) crosses as a flat `(n1*n2, nbf*nbf)` Matrix; `interop.mo_bra_half_transform` reshapes to rank 4. Validated against numpy, then the ComputeGraph finishes the ket transform into the five exact CC blocks. |
| `df_mp2_energy.py` | End-to-end DF-MP2 the way it is actually run, with every tensor op in einsums and no numpy in the compute. The algorithm is pair-driven over `i≤j` with one `nvir×nvir` GEMM per pair via `einsum`, never forming the O(o²v²) `(ia\|jb)`. It uses `einsum`, `permute`, `axpby`, `outer_sum`, `element_transform`, `direct_product`, and `dot`. Matches psi4's DF-MP2 to machine precision. Pass `--profile [FILE]` for an einsums profile report at exit. |
| `df_mp2_graph.py` | The deferred ComputeGraph path. It runs the same memory-optimal pair-driven algorithm as `df_mp2_energy.py`, with no O(o²v²) intermediate. The per-pair work is recorded with `with cg.capture(g):` at roughly 10 nodes per pair, the default passes run via `PassManager.populate_default` and `g.apply`, then `g.execute()`. Matches psi4's DF-MP2 to machine precision. Pass `--show-passes` for a per-pass `modified` table plus the node execution order before and after optimization. |

## Running

All scripts need `einsums` and `psi4` importable.
With an in-tree Einsums build and any psi4 install (no Einsums linkage required), point `PYTHONPATH` at both and use the conda env's Python:

```bash
PYTHONPATH=/path/to/Einsums/build/lib:/path/to/psi4/cmake-build-debug/stage/lib \
  python examples/psi4-bridge/so_one_electron_tiled.py
PYTHONPATH=/path/to/Einsums/build/lib:/path/to/psi4/cmake-build-debug/stage/lib \
  python examples/psi4-bridge/so_two_electron_eri_tiled.py
```
