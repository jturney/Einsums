#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

"""Runtime check: Matrix.to_einsums_tiled() exposes a symmetry-blocked psi4
Matrix as an einsums TiledRuntimeTensor that ALIASES the irrep blocks (zero-copy
— block h maps to tile (h, h^symmetry)).

Verifies two things for the SO overlap/kinetic/potential matrices:
  1. each tile equals the corresponding per-irrep block, and
  2. the tile and the Matrix share storage — mutating the Matrix block shows
     through the tile (no copy was made).

Run with einsums + psi4 on PYTHONPATH. Importing einsums first registers the
TiledRuntimeTensorD pybind type so psi4 can hand one back.
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
    ("so_overlap", mints.so_overlap),
    ("so_kinetic", mints.so_kinetic),
    ("so_potential", mints.so_potential),
]

all_ok = True
for name, mat_fn in cases:
    S = mat_fn()                       # symmetry-blocked Matrix
    St = S.to_einsums_tiled()          # zero-copy alias: block h -> tile (h, h^symmetry)
    print(f"{name}: type={type(St).__name__} rank={St.rank()} dims={list(St.dims())} "
          f"nirrep={S.nirrep()} filled_tiles={St.num_filled_tiles()}")
    for h in range(S.nirrep()):
        blk = np.asarray(S.nph[h])                 # per-irrep block (rows x cols)
        if blk.size == 0 or blk.shape[0] == 0:
            continue
        assert St.has_tile([h, h]), f"{name}: missing tile ({h},{h})"
        tile = np.asarray(St.tile_view([h, h]))    # einsums view over that tile
        match = tile.shape == blk.shape and np.allclose(tile, blk)

        # Zero-copy proof: mutate the Matrix block, see it through the tile.
        sv = np.asarray(S.nph[h])
        orig = sv[0, 0]
        sv[0, 0] = orig + 1.0
        aliased = np.asarray(St.tile_view([h, h]))[0, 0] == orig + 1.0
        sv[0, 0] = orig                            # restore

        if match and aliased:
            print(f"  irrep {h}: match + zero-copy aliased {blk.shape}")
        else:
            all_ok = False
            print(f"  MISMATCH irrep {h}: match={match} aliased={aliased}")

print("ALL TILED 1e INTEGRALS MATCH (zero-copy alias)" if all_ok else "FAILURES DETECTED")
assert all_ok
