#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

"""Matrix.to_einsums_tiled(): a psi4 Matrix as a zero-copy einsums tensor.

A symmetry-blocked psi4 Matrix is exposed as a rank-2 einsums TiledRuntimeTensor
that ALIASES the irrep blocks (block h <-> tile (h, h^symmetry)) — no copy. The
tensor and the Matrix share storage, so writes through either side are visible
to both. This is the zero-copy bridge toward making an Einsums tensor the
storage backend for Matrix/Vector (vs. the copy-based MintsHelper.so_*_tiled()).

Run (Einsums build + psi4 stage on PYTHONPATH, conda-env Python)::

    PYTHONPATH=/Users/jturney/Code/Einsums/Einsums/build/lib:/Users/jturney/Code/psi4/cmake-build-debug/stage/lib \
        /Users/jturney/miniconda3/envs/einsums-dev/bin/python \
        /Users/jturney/Code/Einsums/Einsums/examples/psi4-bridge/matrix_alias.py
"""
import numpy as np
import einsums  # registers the pybind types so psi4 can hand back a tensor
import psi4

psi4.core.set_output_file("/tmp/psi4_matrix_alias.out", False)
mol = psi4.geometry("O\nH 1 0.96\nH 1 0.96 2 104.5\nsymmetry c2v\n")
basis = psi4.core.BasisSet.build(mol, "ORBITAL", "STO-3G")
mints = psi4.core.MintsHelper(basis)

S = mints.so_overlap()        # symmetry-blocked Matrix
T = S.to_einsums_tiled()      # zero-copy alias; keep_alive ties S's lifetime to T
print(f"{type(T).__name__}  rank={T.rank()}  dims={list(T.dims())}  "
      f"nirrep={S.nirrep()}  filled_tiles={T.num_filled_tiles()}")

for h in range(S.nirrep()):
    blk = np.asarray(S.nph[h])
    if blk.size == 0:
        continue
    tile = np.asarray(T.tile_view([h, h]))
    assert tile.shape == blk.shape and np.allclose(tile, blk), f"irrep {h}: value mismatch"

    # Shared storage, both directions:
    #  (a) write through the Matrix  -> the tile sees it
    sv = np.asarray(S.nph[h])
    sv[0, 0] += 2.5
    assert np.asarray(T.tile_view([h, h]))[0, 0] == sv[0, 0], f"irrep {h}: Matrix->tile not shared"
    #  (b) write through the tensor  -> the Matrix sees it
    np.asarray(T.tile_view([h, h]))[0, 0] -= 2.5   # undo, via the tensor
    assert np.asarray(S.nph[h])[0, 0] == blk[0, 0], f"irrep {h}: tile->Matrix not shared"

    print(f"  irrep {h}: tile == block {blk.shape}, storage shared both ways")

print("Matrix.to_einsums_tiled() aliases the Matrix (zero-copy, bidirectional) — OK")
