#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

"""Validate einsums.interop.psi4.dense over psi4's ao_eri(): AO two-electron
integrals (pq|rs), chemists' notation, as a dense rank-4 einsums RuntimeTensor.

psi4 hands over its plain (nbf*nbf, nbf*nbf) chemists'-pair matrix and the
interop adapter reshapes the buffer; no einsums types cross the psi4 ABI.
AO integrals carry no point-group symmetry, so a dense rank-4 tensor (not a
tiled one) is the natural representation.
"""
import numpy as np
from einsums.interop import psi4 as interop
import psi4

psi4.core.set_output_file("/tmp/psi4_ao_eri.out", False)

mol = psi4.geometry("O\nH 1 0.96\nH 1 0.96 2 104.5\nsymmetry c1\n")
basis = psi4.core.BasisSet.build(mol, "ORBITAL", "STO-3G")
nbf = basis.nbf()
mints = psi4.core.MintsHelper(basis)

ref = np.asarray(mints.ao_eri()).reshape(nbf, nbf, nbf, nbf)  # (pq|rs)
T = interop.dense(mints.ao_eri(), (nbf, nbf, nbf, nbf), name="AO ERI")
got = np.asarray(T)
print(f"interop.dense(ao_eri): type={type(T).__name__} shape={got.shape}")
assert got.shape == (nbf, nbf, nbf, nbf)
assert np.allclose(got, ref), f"AO ERI mismatch (max|Δ|={np.abs(got-ref).max():.3e})"
print(f"  matches ao_eri()  (max|Δ|={np.abs(got-ref).max():.2e})")

# 8-fold permutational symmetry of (pq|rs)
for perm in [(1, 0, 2, 3), (0, 1, 3, 2), (2, 3, 0, 1), (3, 2, 1, 0)]:
    assert np.allclose(got, got.transpose(perm)), f"fails permutation {perm}"
print("  obeys 8-fold permutational symmetry")
print("AO ERI DENSE CHECK PASSES")
