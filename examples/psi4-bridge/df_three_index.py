#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

"""Validate einsums.interop.psi4.df_tensor over DFTensor's matrix accessors:
the density-fitted 3-index integrals (Q|pq) as dense rank-3 RuntimeTensors.

DF integrals carry no point-group symmetry, so a plain dense tensor rather than a
tiled one is the natural representation. Two checks:
  1. Bridge correctness: df_tensor(Qso()/Qov()/Qvv()) exactly equals the psi4
     matrices reshaped to rank 3.
  2. Physics: the fitted B = J^{-1/2}(Q|pq) reconstructs the AO ERIs within the
     density-fitting error, i.e. einsum('Qpq,Qrs->pqrs', B, B) ~ ao_eri().
"""
import numpy as np
from einsums.interop import psi4 as interop
import psi4

psi4.core.set_output_file("/tmp/psi4_df.out", False)
psi4.set_options({"basis": "cc-pvdz", "scf_type": "pk", "e_convergence": 1e-9})

mol = psi4.geometry("O\nH 1 0.96\nH 1 0.96 2 104.5\nsymmetry c1\n")
e, wfn = psi4.energy("scf", return_wfn=True)

primary = wfn.basisset()
aux = psi4.core.BasisSet.build(mol, "DF_BASIS_MP2", "", "RIFIT", primary.name())
C = wfn.Ca()
nocc = wfn.nalpha()
nmo = wfn.nmo()
nvir = nmo - nocc
nbf = primary.nbf()
naux = aux.nbf()
print(f"nbf={nbf} nmo={nmo} nocc={nocc} nvir={nvir} naux={naux}")

dft = psi4.core.DFTensor(primary, aux, C, nocc, nvir)

# ---- 1) bridge correctness: einsums tensor == matrix accessor reshaped -----
cases = [
    ("Qso", dft.Qso, (naux, nbf, nbf)),
    ("Qov", dft.Qov, (naux, nocc, nvir)),
    ("Qvv", dft.Qvv, (naux, nvir, nvir)),
]
for name, mat_fn, shape in cases:
    ref = np.asarray(mat_fn()).reshape(shape)
    got = np.asarray(interop.df_tensor(mat_fn(), shape[1], shape[2], name=name))
    assert got.shape == shape, f"{name}: {got.shape} != {shape}"
    assert np.allclose(got, ref), f"df_tensor({name}) != {name} (max|Δ|={np.abs(got-ref).max():.3e})"
    print(f"  df_tensor({name}()) matches  shape={shape}")

# ---- 2) physics: DF reconstruction of the AO ERIs --------------------------
mints = psi4.core.MintsHelper(primary)
ao = np.asarray(mints.ao_eri()).reshape(nbf, nbf, nbf, nbf)
B = np.asarray(interop.df_tensor(dft.Qso(), nbf, nbf))   # J^{-1/2} (Q|pq)
eri_df = np.einsum("Qpq,Qrs->pqrs", B, B)
err = np.abs(eri_df - ao).max()
print(f"  DF reconstruction max|(pq|rs)_DF - (pq|rs)| = {err:.2e}")
assert err < 1e-1, "DF reconstruction error unexpectedly large"

print("ALL DF 3-INDEX CHECKS PASS")
