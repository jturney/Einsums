#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

"""In-core conventional AO->MO transform of the exact CC integral blocks.

Builds the four "exact" coupled-cluster integral classes that are kept as
conventional, non-DF integrals, namely ``oooo``, ``ooov``, ``oovv``, ``ovov``,
as a single einsums ComputeGraph capture, starting from the dense AO ERIs
``MintsHelper.ao_eri_einsums()``.

Integral strategy (see project memory project_native_cc_integral_strategy):
DF introduces error in the final energy, so DF is reserved for ``vvvv``, plus
``ovvv`` only under memory pressure; the cheap blocks here stay exact. All
four exact blocks have >=2 occupied indices, so contracting the large AO
indices against occupied C-columns first collapses nbf->o early. The peak
intermediate is ``o.nbf^3`` after the 1st quarter and ``o^2.nbf^2`` after the
2nd, and everything downstream is small. ``vvvv`` and ``ovvv`` are deliberately
absent: vvvv is a DF-direct sub-graph and ovvv is the exact-vs-DF toggle.

This is the in-core validation path: it forms the dense nbf^4 AO tensor and
does all four quarter-transforms as graph einsums. It is good to a few hundred
basis functions, and is meant to check native CC against psi4 before the
scaling-path half-transform producer exists.

Transform tree (chemist (pq|rs); contract over the bracketed AO index):

    (uv|ls)
      +- u->i --> (iv|ls)              SHARED first quarter (occ on index 0)
            +- v->j --> (ij|ls) = H1    feeds oooo / ooov / oovv
            |     +- l->k -> (ij|ks) --+- s->l -> (ij|kl)  oooo
            |     |                     +- s->a -> (ij|ka)  ooov
            |     +- l->a -> (ij|as) ---- s->b -> (ij|ab)  oovv
            +- l->j --> (iv|js) = H2    feeds ovov
                  +- v->a -> (ia|js) --- s->b -> (ia|jb)  ovov

Important: intermediates are pre-shared on purpose. ``(iv|ls)`` is built once
and consumed by both H1 (v->j) and H2 (l->j), and ``(ij|ks)`` feeds both oooo
and ooov. We do not write each block as an independent chain and rely on the CSE
pass to dedup, because CSE is currently unsound when a surviving node consumes
a folded duplicate's output. The executor runs a lambda that baked in the
captured tensor refs (Node.hpp), so CSE's TensorId redirect is silently
ignored at execution and the consumer reads a now-never-written, zero buffer.
See buglog bug-999. Pre-sharing sidesteps CSE entirely and is verified correct
under all default passes.

Run with the Einsums build and psi4 stage on PYTHONPATH, using the conda-env
Python::

    PYTHONPATH=/Users/jturney/Code/Einsums/Einsums/build/lib:/Users/jturney/Code/psi4/cmake-build-debug/stage/lib \
        /Users/jturney/miniconda3/envs/einsums-dev/bin/python \
        /Users/jturney/Code/Einsums/Einsums/examples/psi4-bridge/cc_integrals_incore.py
"""
import numpy as np
import psi4
import einsums
import einsums._core            # force pybind type registration (bug-916: _core is lazy)
import einsums.graph as cg      # the graph.py shell (capture/PassManager); NOT
                                # `from einsums import graph` (that lacks capture).

psi4.core.set_output_file("/tmp/psi4_cc_integrals.out", False)
psi4.set_options({"basis": "cc-pvdz", "scf_type": "pk"})
mol = psi4.geometry("O\nH 1 0.96\nH 1 0.96 2 104.5\nsymmetry c1\n")
e_scf, wfn = psi4.energy("scf", return_wfn=True)

mints = psi4.core.MintsHelper(wfn.basisset())
nbf  = wfn.nso()
nocc = wfn.nalpha()              # restricted reference; frozen core omitted for the demo
nvir = nbf - nocc
print(f"nbf={nbf} nocc={nocc} nvir={nvir}")

# ---- ingest: AO ERIs as an Einsums tensor (the only "integral generation") ----
I_ao = mints.ao_eri_einsums()    # dense RuntimeTensorD, chemist (mu,nu,lam,sig)

# ---- ingest: MO coefficient blocks via numpy (C1 -> plain dense tensors) ----
# np.asarray(create_zero_tensor(...)) yields a WRITABLE view of the buffer
# (verified; same pattern as df_mp2_graph.py). ascontiguousarray guards against
# a non-contiguous psi4 sub-block.
C = np.asarray(wfn.Ca())
def to_tensor(name, arr):
    t = einsums.create_zero_tensor(name, list(arr.shape), dtype="float64")
    np.asarray(t)[:] = arr
    return t
C_o = to_tensor("C_o", np.ascontiguousarray(C[:, :nocc]))   # (mu, i)
C_v = to_tensor("C_v", np.ascontiguousarray(C[:, nocc:]))   # (mu, a)

def zt(name, dims):
    return einsums.create_zero_tensor(name, dims, dtype="float64")

# Shared intermediates (built once) + outputs.
iNLS = zt("iNLS", [nocc, nbf, nbf, nbf])    # (iv|ls)  shared first quarter
H1   = zt("H1",   [nocc, nocc, nbf, nbf])   # (ij|ls)
H2   = zt("H2",   [nocc, nbf, nocc, nbf])   # (iv|js)
ijkS = zt("ijkS", [nocc, nocc, nocc, nbf])  # (ij|ks)  shared by oooo & ooov
ijaS = zt("ijaS", [nocc, nocc, nvir, nbf])  # (ij|as)
iajS = zt("iajS", [nocc, nvir, nocc, nbf])  # (ia|js)
oooo = zt("oooo", [nocc, nocc, nocc, nocc])
ooov = zt("ooov", [nocc, nocc, nocc, nvir])
oovv = zt("oovv", [nocc, nocc, nvir, nvir])
ovov = zt("ovov", [nocc, nvir, nocc, nvir])

g = cg.Graph("cc_exact_blocks")
with cg.capture(g):
    # shared first quarter: (iv|ls) = C_ui (uv|ls)
    einsums.einsum("i,nu,lam,sig <- mu,i ; mu,nu,lam,sig", iNLS, C_o, I_ao)

    # H1 = (ij|ls) = C_vj (iv|ls)   -> oooo / ooov / oovv
    einsums.einsum("i,j,lam,sig <- nu,j ; i,nu,lam,sig", H1, C_o, iNLS)
    einsums.einsum("i,j,k,sig <- lam,k ; i,j,lam,sig", ijkS, C_o, H1)   # (ij|ks) shared
    einsums.einsum("i,j,k,l   <- sig,l ; i,j,k,sig",   oooo, C_o, ijkS)
    einsums.einsum("i,j,k,a   <- sig,a ; i,j,k,sig",   ooov, C_v, ijkS)
    einsums.einsum("i,j,a,sig <- lam,a ; i,j,lam,sig", ijaS, C_v, H1)
    einsums.einsum("i,j,a,b   <- sig,b ; i,j,a,sig",   oovv, C_v, ijaS)

    # H2 = (iv|js) = C_lj (iv|ls)   -> ovov   (reuses the shared iNLS)
    einsums.einsum("i,nu,j,sig <- lam,j ; i,nu,lam,sig", H2, C_o, iNLS)
    einsums.einsum("i,a,j,sig <- nu,a ; i,nu,j,sig", iajS, C_v, H2)
    einsums.einsum("i,a,j,b   <- sig,b ; i,a,j,sig", ovov, C_v, iajS)

print(f"captured '{g.name}': {g.num_nodes()} nodes, {g.num_tensors()} tensors")

# Default passes are safe here because the graph has no redundancy to fold
# (intermediates are pre-shared). Reorder/MemoryPlanning/Inplace only reschedule.
pm = cg.PassManager()
pm.populate_default()
before = g.num_nodes()
modified = g.apply(pm)
print(f"optimization: {pm.size} passes, modified={modified}, nodes {before} -> {g.num_nodes()}")

g.execute()

# ---- validate against a numpy reference ----
Iao = np.asarray(I_ao)
Co, Cv = C[:, :nocc], C[:, nocc:]
refs = {
    "oooo": np.einsum("pi,qj,pqrs,rk,sl->ijkl", Co, Co, Iao, Co, Co, optimize=True),
    "ooov": np.einsum("pi,qj,pqrs,rk,sa->ijka", Co, Co, Iao, Co, Cv, optimize=True),
    "oovv": np.einsum("pi,qj,pqrs,ra,sb->ijab", Co, Co, Iao, Cv, Cv, optimize=True),
    "ovov": np.einsum("pi,qa,pqrs,rj,sb->iajb", Co, Cv, Iao, Co, Cv, optimize=True),
}
got = {"oooo": oooo, "ooov": ooov, "oovv": oovv, "ovov": ovov}
all_ok = True
for nm in refs:
    d = float(np.max(np.abs(np.asarray(got[nm]) - refs[nm])))
    ok = d < 1e-11
    all_ok &= ok
    print(f"  {nm}: max|Δ| vs numpy = {d:.2e}  {'OK' if ok else 'FAIL'}")
assert all_ok, "a transformed block disagrees with the numpy reference"
print("exact CC blocks (oooo/ooov/oovv/ovov) via ComputeGraph MATCH numpy")
