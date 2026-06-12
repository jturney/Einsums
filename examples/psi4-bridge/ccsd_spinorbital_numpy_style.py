#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

"""Spin-orbital CCSD in einsums (eager, numpy-style).

Term-by-term translation of the validated oracle (ccsd_spinorbital_oracle.py),
so each einsums.einsum call maps to a numpy einsum already checked against psi4.
This is the first native CCSD in einsums; the closed-shell spin-adapted + hybrid
DF-v4 production version (df_ccsd_*) builds on the machinery proven here.

Tensor algebra is numpy-style: contractions via einsums.einsum ("OUT <- A ; B"),
combination via the +/-/* operators, amplitude denominators via the '/' operator
(direct_division), and the P(ij)/P(ab) antisymmetrizers via permute + subtract.

Validated: cc-pVDZ water, conventional SCF/CCSD. Matches psi4 to ~1e-11:
    spin-orbital einsums CCSD corr = -0.2134804971
    psi4 conv CCSD corr            = -0.2134804971   (== the oracle)

Note: this surfaced two einsum bugs fixed along the way (see buglog
bug-1000/1001). PackedGemm's rank_of returned the runtime-rank sentinel for
RuntimeTensors, aborting on rank4xrank4->rank2 contractions; and string_gemm
ignored the requested output index order, mishandling transposed-output GEMMs
like "ia <- ma ; mi".

Run with the Einsums build and psi4 stage on PYTHONPATH, using the conda-env
Python::

    PYTHONPATH=/Users/jturney/Code/Einsums/Einsums/build/lib:/Users/jturney/Code/psi4/cmake-build-debug/stage/lib \
        /Users/jturney/miniconda3/envs/einsums-dev/bin/python \
        examples/psi4-bridge/ccsd_spinorbital_numpy_style.py
"""
import numpy as np
import psi4
import einsums
import einsums._core
from einsums import linalg as la

psi4.core.set_output_file("/tmp/psi4_ccsd_conv.out", False)
psi4.set_options({"basis": "cc-pvdz", "scf_type": "pk", "freeze_core": "false",
                  "e_convergence": 1e-10, "d_convergence": 1e-10, "cc_type": "conv", "r_convergence": 1e-9})
mol = psi4.geometry("O\nH 1 0.96\nH 1 0.96 2 104.5\nsymmetry c1\n")
e_scf, wfn = psi4.energy("ccsd", return_wfn=True)
ref_ccsd = psi4.variable("CCSD CORRELATION ENERGY")

nbf, nocc = wfn.nso(), wfn.nalpha()
C = np.asarray(wfn.Ca()); eps = np.asarray(wfn.epsilon_a())
mints = psi4.core.MintsHelper(wfn.basisset())
Iao = np.asarray(mints.ao_eri())
mo = np.einsum("mp,nr,lq,os,mnlo->pqrs", C, C, C, C, Iao, optimize=True)

# spin-block to interleaved spin orbitals, then antisymmetrize to <PQ||RS>
n = nbf
G = np.zeros((n, 2, n, 2, n, 2, n, 2))
for s1 in (0, 1):
    for s2 in (0, 1):
        G[:, s1, :, s2, :, s1, :, s2] = mo
nso = 2 * n
gnp = G.reshape(nso, nso, nso, nso); gnp = gnp - gnp.transpose(0, 1, 3, 2)
eps_so = np.repeat(eps, 2)
no = 2 * nocc; nv = nso - no
o, vv = slice(0, no), slice(no, nso)
eo, ev = eps_so[o], eps_so[vv]

# --- ingest spin-orbital integral slices + denominators as einsums tensors ---
def t(name, a): return einsums.asarray(np.ascontiguousarray(a), name=name)
g_oovv = t("oovv", gnp[o, o, vv, vv]); g_vovv = t("vovv", gnp[vv, o, vv, vv])
g_ooov = t("ooov", gnp[o, o, o, vv]); g_oooo = t("oooo", gnp[o, o, o, o])
g_vvvv = t("vvvv", gnp[vv, vv, vv, vv]); g_ovvo = t("ovvo", gnp[o, vv, vv, o])
g_ovvv = t("ovvv", gnp[o, vv, vv, vv]); g_oovo = t("oovo", gnp[o, o, vv, o])
g_ovov = t("ovov", gnp[o, vv, o, vv]); g_vvvo = t("vvvo", gnp[vv, vv, vv, o])
g_ovoo = t("ovoo", gnp[o, vv, o, o])
Dia = t("Dia", eo[:, None] - ev[None, :])
Dijab = t("Dijab", eo[:, None, None, None] + eo[None, :, None, None]
          - ev[None, None, :, None] - ev[None, None, None, :])

t1 = einsums.zeros((no, nv), name="t1")
t2 = t("t2", gnp[o, o, vv, vv]) / Dijab          # MP2 guess via '/' operator

# helpers -------------------------------------------------------------------
def E(spec, A, B, shape, name="x", pf=1.0):
    out = einsums.create_zero_tensor(name, list(shape), dtype="float64")
    einsums.einsum(spec, out, A, B, c_pf=0.0, ab_pf=pf)
    return out

def perm(spec, x, shape, name="p"):
    out = einsums.create_zero_tensor(name, list(shape), dtype="float64")
    einsums.permute(spec, out, x)
    return out

def Pij(x, sh): return x - perm("j,i,a,b <- i,j,a,b", x, sh, "Pij")   # axes (0,1)
def Pab(x, sh): return x - perm("i,j,b,a <- i,j,a,b", x, sh, "Pab")   # axes (2,3)

def energy(t1, t2):
    e = 0.25 * float(la.dot(g_oovv, t2))
    X = E("ia <- ijab ; jb", g_oovv, t1, (no, nv), "Xen")   # sum_jb <ij||ab> t1_jb
    return e + 0.5 * float(la.dot(t1, X))

OOOO, OVVO, SH2 = (no, no, no, no), (no, nv, nv, no), (no, no, nv, nv)
VVVV = (nv, nv, nv, nv)

e_old = energy(t1, t2)
print(f"MP2 (spin-orbital einsums) = {e_old:.10f}")
for it in range(100):
    # tau, tau~ (antisymmetrized t1 outer products)
    tt = E("i,j,a,b <- i,a ; j,b", t1, t1, SH2, "tt")            # t1_ia t1_jb
    tt = tt - perm("i,j,b,a <- i,j,a,b", tt, SH2, "ttp")          # - t1_ib t1_ja
    tau = t2 + tt
    taut = t2 + 0.5 * tt

    Fae = E("ae <- mf ; amef", t1, g_vovv, (nv, nv), "Fae") \
        - E("ae <- mnaf ; mnef", taut, g_oovv, (nv, nv), "Fae2", pf=0.5)
    Fmi = E("mi <- ne ; mnie", t1, g_ooov, (no, no), "Fmi") \
        + E("mi <- inef ; mnef", taut, g_oovv, (no, no), "Fmi2", pf=0.5)
    Fme = E("me <- nf ; mnef", t1, g_oovv, (no, nv), "Fme")

    wt = E("mnij <- je ; mnie", t1, g_ooov, OOOO, "wt")
    Wmnij = g_oooo + (wt - perm("m,n,j,i <- m,n,i,j", wt, OOOO, "wtp")) \
        + E("mnij <- ijef ; mnef", tau, g_oovv, OOOO, "Wmnij2", pf=0.25)
    wt = E("abef <- mb ; amef", t1, g_vovv, VVVV, "wt2")
    Wabef = g_vvvv - (wt - perm("b,a,e,f <- a,b,e,f", wt, VVVV, "wtp2")) \
        + E("abef <- mnab ; mnef", tau, g_oovv, VVVV, "Wabef2", pf=0.25)
    Wmbej = g_ovvo + E("mbej <- jf ; mbef", t1, g_ovvv, OVVO, "Wmbej1") \
        - E("mbej <- nb ; mnej", t1, g_oovo, OVVO, "Wmbej2")
    tw = t2 * 0.5 + E("j,n,f,b <- j,f ; n,b", t1, t1, SH2, "jnfb")
    Wmbej = Wmbej - E("mbej <- jnfb ; mnef", tw, g_oovv, OVVO, "Wmbej3")

    # T1
    t1n = E("ia <- ie ; ae", t1, Fae, (no, nv), "t1a") \
        - E("ia <- ma ; mi", t1, Fmi, (no, nv), "t1b") \
        + E("ia <- imae ; me", t2, Fme, (no, nv), "t1c") \
        - E("ia <- nf ; naif", t1, g_ovov, (no, nv), "t1d") \
        - E("ia <- imef ; maef", t2, g_ovvv, (no, nv), "t1e", pf=0.5) \
        - E("ia <- mnae ; nmei", t2, g_oovo, (no, nv), "t1f", pf=0.5)
    t1n = t1n / Dia

    # T2
    t2n = g_oovv * 1.0          # fresh copy of <ij||ab>
    be = Fae - E("be <- mb ; me", t1, Fme, (nv, nv), "be05", pf=0.5)
    t2n = t2n + Pab(E("ijab <- ijae ; be", t2, be, SH2, "t2ab"), SH2)
    mj = Fmi + E("mj <- je ; me", t1, Fme, (no, no), "mj05", pf=0.5)
    t2n = t2n - Pij(E("ijab <- imab ; mj", t2, mj, SH2, "t2ij"), SH2)
    t2n = t2n + E("ijab <- mnab ; mnij", tau, Wmnij, SH2, "t2mn", pf=0.5)
    t2n = t2n + E("ijab <- ijef ; abef", tau, Wabef, SH2, "t2ef", pf=0.5)
    ring = E("ijab <- imae ; mbej", t2, Wmbej, SH2, "ring1")
    tmp = E("abej <- ma ; mbej", t1, g_ovvo, (nv, nv, nv, no), "tmpabej")
    ring = ring - E("ijab <- ie ; abej", t1, tmp, SH2, "ring2")
    t2n = t2n + Pij(Pab(ring, SH2), SH2)
    t2n = t2n + Pij(E("ijab <- ie ; abej", t1, g_vvvo, SH2, "t2ie"), SH2)
    t2n = t2n - Pab(E("ijab <- ma ; mbij", t1, g_ovoo, SH2, "t2ma"), SH2)
    t2n = t2n / Dijab

    t1, t2 = t1n, t2n
    e_new = energy(t1, t2)
    if abs(e_new - e_old) < 1e-11:
        print(f"converged in {it+1} iters")
        break
    e_old = e_new

print(f"spin-orbital einsums CCSD corr = {e_new:.10f}")
print(f"psi4 conv CCSD corr            = {ref_ccsd:.10f}")
print(f"difference                     = {abs(e_new - ref_ccsd):.2e}")
assert abs(e_new - ref_ccsd) < 1e-7, "einsums CCSD disagrees with psi4"
print("spin-orbital CCSD (eager einsums, numpy-style) MATCHES psi4")
