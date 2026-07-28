#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

"""Runtime check: einsums.interop.psi4.from_matrix copies a symmetry-blocked
psi4 Matrix into an einsums TiledRuntimeTensor, mapping the per-irrep block h
to tile (h, h^symmetry).

Verifies two things for the SO overlap/kinetic/potential matrices:
  1. each tile equals the corresponding per-irrep block, and
  2. the tensor owns its storage: mutating the Matrix block afterwards does
     NOT show through the tile (the bridge exchanges buffers, not aliases).
"""
import numpy as np
from einsums.interop import psi4 as interop
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
    St = interop.from_matrix(S)        # copy: block h -> tile (h, h^symmetry)
    print(f"{name}: type={type(St).__name__} rank={St.rank()} dims={list(St.dims())} "
          f"nirrep={S.nirrep()} filled_tiles={St.num_filled_tiles()}")
    for h in range(S.nirrep()):
        blk = np.asarray(S.nph[h])                 # per-irrep block (rows x cols)
        if blk.size == 0 or blk.shape[0] == 0:
            continue
        assert St.has_tile([h, h]), f"{name}: missing tile ({h},{h})"
        tile = np.asarray(St.tile_view([h, h]))    # einsums view over that tile
        match = tile.shape == blk.shape and np.allclose(tile, blk)

        # Copy proof: mutate the Matrix block; the tile must NOT change.
        sv = np.asarray(S.nph[h])
        orig = sv[0, 0]
        sv[0, 0] = orig + 1.0
        independent = np.asarray(St.tile_view([h, h]))[0, 0] == orig
        sv[0, 0] = orig                            # restore

        if match and independent:
            print(f"  irrep {h}: match + independent storage {blk.shape}")
        else:
            all_ok = False
            print(f"  MISMATCH irrep {h}: match={match} independent={independent}")

print("ALL TILED 1e INTEGRALS MATCH (copied via interop)" if all_ok else "FAILURES DETECTED")
assert all_ok
