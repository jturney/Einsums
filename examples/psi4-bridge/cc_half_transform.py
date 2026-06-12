#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

"""Validate the integral-direct half-transform bridge against the in-core blocks.

`MintsHelper.mo_bra_half_transform_einsums(C1, C2)` returns the first-half (bra)
MO transform of the AO ERIs, computed integral-direct so the N^4 AO tensor is
never formed:

    (pq|λσ) = Σ_{μν} C1[μ,p] C2[ν,q] (μν|λσ)        dense (n1, n2, nbf, nbf)

Two calls produce half-transforms that feed all five exact CC blocks once the
ComputeGraph finishes the ket (3rd/4th quarter) transform. This is the same
back-half as cc_integrals_incore.py, just fed a half-transform instead of
ao_eri:

    HT_oo = bra_half(C_o, C_o)  ->  oooo / ooov / oovv
    HT_ov = bra_half(C_o, C_v)  ->  ovov / ovvv

This checks two things: the bridge half-transforms equal their numpy reference,
and the graph-finished blocks equal numpy, which transitively validates against
the in-core oracle.

Run with the Einsums build and psi4 stage on PYTHONPATH, using the conda-env
Python::

    PYTHONPATH=/Users/jturney/Code/Einsums/Einsums/build/lib:/Users/jturney/Code/psi4/cmake-build-debug/stage/lib \
        /Users/jturney/miniconda3/envs/einsums-dev/bin/python \
        /Users/jturney/Code/Einsums/Einsums/examples/psi4-bridge/cc_half_transform.py
"""
import numpy as np
import psi4
import einsums
import einsums._core            # force pybind type registration (bug-916: _core is lazy)
import einsums.graph as cg

psi4.core.set_output_file("/tmp/psi4_cc_half_transform.out", False)
psi4.set_options({"basis": "cc-pvdz", "scf_type": "pk"})
mol = psi4.geometry("O\nH 1 0.96\nH 1 0.96 2 104.5\nsymmetry c1\n")
e_scf, wfn = psi4.energy("scf", return_wfn=True)

mints = psi4.core.MintsHelper(wfn.basisset())
nbf  = wfn.nso()
nocc = wfn.nalpha()
nvir = nbf - nocc
print(f"nbf={nbf} nocc={nocc} nvir={nvir}")

C = np.asarray(wfn.Ca())
Co_np = np.ascontiguousarray(C[:, :nocc])
Cv_np = np.ascontiguousarray(C[:, nocc:])
Co_psi = psi4.core.Matrix.from_array(Co_np)
Cv_psi = psi4.core.Matrix.from_array(Cv_np)

# ---- the bridge: integral-direct bra half-transforms (no N^4 AO tensor) ----
HT_oo = mints.mo_bra_half_transform_einsums(Co_psi, Co_psi)   # (ij|λσ)  (o, o, nbf, nbf)
HT_ov = mints.mo_bra_half_transform_einsums(Co_psi, Cv_psi)   # (ia|λσ)  (o, v, nbf, nbf)

# ---- (1) half-transforms vs numpy reference ----
Iao = np.asarray(mints.ao_eri_einsums())   # (μν|λσ); only to build the reference
ref_HToo = np.einsum("mi,nj,mnls->ijls", Co_np, Co_np, Iao, optimize=True)
ref_HTov = np.einsum("mi,na,mnls->ials", Co_np, Cv_np, Iao, optimize=True)
for name, got, ref in [("HT_oo (ij|λσ)", HT_oo, ref_HToo), ("HT_ov (ia|λσ)", HT_ov, ref_HTov)]:
    d = float(np.max(np.abs(np.asarray(got) - ref)))
    print(f"  half-transform {name}: max|Δ| vs numpy = {d:.2e}  {'OK' if d < 1e-11 else 'FAIL'}")
    assert d < 1e-11, f"{name} half-transform disagrees with numpy"

# ---- (2) graph finishes the ket transform -> the five blocks ----
def to_tensor(name, arr):
    t = einsums.create_zero_tensor(name, list(arr.shape), dtype="float64")
    np.asarray(t)[:] = arr
    return t
C_o = to_tensor("C_o", Co_np)
C_v = to_tensor("C_v", Cv_np)
def zt(name, dims):
    return einsums.create_zero_tensor(name, dims, dtype="float64")

ijkS = zt("ijkS", [nocc, nocc, nocc, nbf])     # shared by oooo & ooov
ijaS = zt("ijaS", [nocc, nocc, nvir, nbf])
iajS = zt("iajS", [nocc, nvir, nocc, nbf])
iabS = zt("iabS", [nocc, nvir, nvir, nbf])
oooo = zt("oooo", [nocc, nocc, nocc, nocc])
ooov = zt("ooov", [nocc, nocc, nocc, nvir])
oovv = zt("oovv", [nocc, nocc, nvir, nvir])
ovov = zt("ovov", [nocc, nvir, nocc, nvir])
ovvv = zt("ovvv", [nocc, nvir, nvir, nvir])

g = cg.Graph("cc_ket_finish")
with cg.capture(g):
    # from HT_oo = (ij|λσ)
    einsums.einsum("i,j,k,sig <- lam,k ; i,j,lam,sig", ijkS, C_o, HT_oo)   # (ij|kσ) shared
    einsums.einsum("i,j,k,l   <- sig,l ; i,j,k,sig",   oooo, C_o, ijkS)
    einsums.einsum("i,j,k,a   <- sig,a ; i,j,k,sig",   ooov, C_v, ijkS)
    einsums.einsum("i,j,a,sig <- lam,a ; i,j,lam,sig", ijaS, C_v, HT_oo)
    einsums.einsum("i,j,a,b   <- sig,b ; i,j,a,sig",   oovv, C_v, ijaS)
    # from HT_ov = (ia|λσ)
    einsums.einsum("i,a,j,sig <- lam,j ; i,a,lam,sig", iajS, C_o, HT_ov)
    einsums.einsum("i,a,j,b   <- sig,b ; i,a,j,sig",   ovov, C_v, iajS)
    einsums.einsum("i,a,b,sig <- lam,b ; i,a,lam,sig", iabS, C_v, HT_ov)
    einsums.einsum("i,a,b,c   <- sig,c ; i,a,b,sig",   ovvv, C_v, iabS)

pm = cg.PassManager()
pm.populate_default()
g.apply(pm)
g.execute()

Co, Cv = Co_np, Cv_np
refs = {
    "oooo": np.einsum("pi,qj,pqrs,rk,sl->ijkl", Co, Co, Iao, Co, Co, optimize=True),
    "ooov": np.einsum("pi,qj,pqrs,rk,sa->ijka", Co, Co, Iao, Co, Cv, optimize=True),
    "oovv": np.einsum("pi,qj,pqrs,ra,sb->ijab", Co, Co, Iao, Cv, Cv, optimize=True),
    "ovov": np.einsum("pi,qa,pqrs,rj,sb->iajb", Co, Cv, Iao, Co, Cv, optimize=True),
    "ovvv": np.einsum("pi,qa,pqrs,rb,sc->iabc", Co, Cv, Iao, Cv, Cv, optimize=True),
}
got = {"oooo": oooo, "ooov": ooov, "oovv": oovv, "ovov": ovov, "ovvv": ovvv}
all_ok = True
for nm in refs:
    d = float(np.max(np.abs(np.asarray(got[nm]) - refs[nm])))
    ok = d < 1e-11
    all_ok &= ok
    print(f"  block {nm}: max|Δ| vs numpy = {d:.2e}  {'OK' if ok else 'FAIL'}")
assert all_ok, "a half-transform-derived block disagrees with numpy"
print("half-transform bridge -> ComputeGraph ket-finish: all five blocks MATCH numpy")
