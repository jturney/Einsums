#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

"""einsums.interop.psi4.from_matrix: a psi4 Matrix as an einsums tiled tensor.

A symmetry-blocked psi4 Matrix crosses the bridge as its per-irrep numpy
blocks (``.nph``) plus ``nirrep()``/``symmetry()``; ``from_matrix`` copies
block h into tile (h, h^symmetry) of a rank-2 TiledRuntimeTensor. The two
libraries exchange only buffers — neither links the other, so either side can
be rebuilt freely (this replaced the old ABI-coupled ``Matrix.to_einsums_tiled``
zero-copy alias).

Run with the Einsums build and psi4 stage on PYTHONPATH, using the conda-env
Python::

    PYTHONPATH=/Users/jturney/Code/Einsums/Einsums/build/lib:/Users/jturney/Code/psi4/cmake-build-debug/stage/lib \
        /Users/jturney/miniconda3/envs/einsums-dev/bin/python \
        /Users/jturney/Code/Einsums/Einsums/examples/psi4-bridge/matrix_interop.py
"""
import numpy as np
from einsums.interop import psi4 as interop
import psi4

psi4.core.set_output_file("/tmp/psi4_matrix_interop.out", False)
mol = psi4.geometry("O\nH 1 0.96\nH 1 0.96 2 104.5\nsymmetry c2v\n")
basis = psi4.core.BasisSet.build(mol, "ORBITAL", "STO-3G")
mints = psi4.core.MintsHelper(basis)

S = mints.so_overlap()          # symmetry-blocked Matrix
T = interop.from_matrix(S)      # per-irrep blocks copied into tiles
print(f"{type(T).__name__}  rank={T.rank()}  dims={list(T.dims())}  "
      f"nirrep={S.nirrep()}  filled_tiles={T.num_filled_tiles()}")

for h in range(S.nirrep()):
    blk = np.asarray(S.nph[h])
    if blk.size == 0:
        continue
    tile = np.asarray(T.tile_view([h, h]))
    assert tile.shape == blk.shape and np.allclose(tile, blk), f"irrep {h}: value mismatch"

    # Copy semantics: the tensor owns its storage, so a write through the
    # Matrix must NOT show through the tensor.
    before = float(tile[0, 0])
    np.asarray(S.nph[h])[0, 0] += 2.5
    after = float(np.asarray(T.tile_view([h, h]))[0, 0])
    np.asarray(S.nph[h])[0, 0] -= 2.5
    assert before == after, f"irrep {h}: from_matrix must copy, not alias"

    print(f"  irrep {h}: tile == block {blk.shape}, independent storage")

print("interop.from_matrix copies the Matrix blocks into einsums tiles — OK")
