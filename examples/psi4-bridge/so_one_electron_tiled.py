#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

"""Runtime check: MintsHelper.so_*_tiled() (einsums TiledRuntimeTensor) matches
the existing symmetry-blocked so_*() Matrix, block for block.

Run with einsums + psi4 on PYTHONPATH. Importing einsums first registers the
TiledRuntimeTensorD pybind type in the shared registry so psi4 can return it.
"""
import numpy as np
import einsums          # loads einsums._core (registers the pybind types) so
                        # psi4 can hand back a TiledRuntimeTensor; the runtime
                        # itself initializes lazily on first compute use.
import psi4

psi4.core.set_output_file("/tmp/psi4_so_tiled.out", False)

mol = psi4.geometry(
    """
O
H 1 0.96
H 1 0.96 2 104.5
symmetry c2v
"""
)
basis = psi4.core.BasisSet.build(mol, "ORBITAL", "STO-3G")
mints = psi4.core.MintsHelper(basis)

cases = [
    ("so_overlap", mints.so_overlap, mints.so_overlap_tiled),
    ("so_kinetic", mints.so_kinetic, mints.so_kinetic_tiled),
    ("so_potential", mints.so_potential, mints.so_potential_tiled),
]

all_ok = True
for name, mat_fn, tiled_fn in cases:
    S = mat_fn()
    St = tiled_fn()
    print(f"{name}: type={type(St).__name__} rank={St.rank()} dims={list(St.dims())} "
          f"nirrep={S.nirrep()} filled_tiles={St.num_filled_tiles()}")
    for h in range(S.nirrep()):
        blk = np.asarray(S.nph[h])                 # per-irrep block (rows x cols)
        if blk.size == 0 or blk.shape[0] == 0:
            continue
        assert St.has_tile([h, h]), f"{name}: missing tile ({h},{h})"
        tile = np.asarray(St.tile_view([h, h]))    # einsums view over that tile
        if tile.shape != blk.shape or not np.allclose(tile, blk):
            all_ok = False
            print(f"  MISMATCH irrep {h}: block {blk.shape} vs tile {tile.shape}")
        else:
            print(f"  irrep {h}: match {blk.shape}")

print("ALL TILED 1e INTEGRALS MATCH" if all_ok else "FAILURES DETECTED")
assert all_ok
