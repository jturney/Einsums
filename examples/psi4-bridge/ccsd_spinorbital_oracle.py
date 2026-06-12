#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

"""Spin-orbital CCSD reference oracle (pure numpy) for the native DF-CCSD build.

This is not an einsums example; it is the trusted reference that the einsums
DF-CCSD versions (df_ccsd_numpy_style.py / df_ccsd_graph_numpy_style.py) are
validated against. The spin-orbital Stanton-Gauss-Watts-Bartlett (1991) equations
carry no spin-adaptation factors, so they are unambiguous to transcribe, and the
closed-shell spin-adapted einsums code targets the energy this produces.

Validated: cc-pVDZ water, conventional SCF, all-electron. Matches psi4
conventional CCSD to ~1e-11:
    spin-orbital CCSD corr = -0.2134804971
    psi4 conv CCSD corr    = -0.2134804971   (MP2 = -0.2041547996, == psi4)

Gotcha baked into the equations below, which cost a 2e-6 ghost while MP2 stayed
exact: the permutation operators P(ij)/P(ab) act on different axes per
intermediate. For the T2 residual T2[i,j,a,b], P(ij) is axes (0,1) and P(ab) is
axes (2,3), which is what P_ij/P_ab below do. But in Wmnij[m,n,i,j], P(ij) is
axes (2,3); in Wabef[a,b,e,f], P(ab) is axes (0,1). Those two are spelled out
inline.

Run with the psi4 stage on PYTHONPATH, using the conda-env Python; no einsums
needed::

    PYTHONPATH=/Users/jturney/Code/psi4/cmake-build-debug/stage/lib \
        /Users/jturney/miniconda3/envs/einsums-dev/bin/python \
        examples/psi4-bridge/ccsd_spinorbital_oracle.py
"""
import numpy as np
import psi4

psi4.core.set_output_file("/tmp/psi4_ccsd_conv.out", False)
psi4.set_options({"basis": "cc-pvdz", "scf_type": "pk", "freeze_core": "false",
                  "e_convergence": 1e-10, "d_convergence": 1e-10, "cc_type": "conv", "r_convergence": 1e-9})
mol = psi4.geometry("O\nH 1 0.96\nH 1 0.96 2 104.5\nsymmetry c1\n")
e_scf, wfn = psi4.energy("ccsd", return_wfn=True)
ref_ccsd = psi4.variable("CCSD CORRELATION ENERGY")

nbf, nocc = wfn.nso(), wfn.nalpha()
C = np.asarray(wfn.Ca())
eps = np.asarray(wfn.epsilon_a())
mints = psi4.core.MintsHelper(wfn.basisset())
Iao = np.asarray(mints.ao_eri())  # chemist (μν|λσ)

# spatial MO, physicist <pq|rs> = (pr|qs)
mo = np.einsum("mp,nr,lq,os,mnlo->pqrs", C, C, C, C, Iao, optimize=True)

# interleaved spin orbitals: index 2p+σ -> spatial p, spin σ. <PQ|RS> nonzero when
# spin(P)=spin(R) and spin(Q)=spin(S); antisymmetrize to <PQ||RS>.
n = nbf
G = np.zeros((n, 2, n, 2, n, 2, n, 2))
for s1 in (0, 1):
    for s2 in (0, 1):
        G[:, s1, :, s2, :, s1, :, s2] = mo
nso = 2 * n
g = G.reshape(nso, nso, nso, nso)
g = g - g.transpose(0, 1, 3, 2)        # <PQ||RS>
eps_so = np.repeat(eps, 2)

no = 2 * nocc
o, v = slice(0, no), slice(no, nso)
eo, ev = eps_so[o], eps_so[v]
Dia = eo[:, None] - ev[None, :]
Dijab = (eo[:, None, None, None] + eo[None, :, None, None]
         - ev[None, None, :, None] - ev[None, None, None, :])

oovv = g[o, o, v, v]
t1 = np.zeros((no, nso - no))
t2 = oovv / Dijab

def P_ij(x): return x - x.transpose(1, 0, 2, 3)   # T2[i,j,a,b]: P(ij) on axes (0,1)
def P_ab(x): return x - x.transpose(0, 1, 3, 2)   # T2[i,j,a,b]: P(ab) on axes (2,3)

def energy(t1, t2):
    return 0.25 * np.einsum("ijab,ijab->", oovv, t2, optimize=True) \
        + 0.5 * np.einsum("ijab,ia,jb->", oovv, t1, t1, optimize=True)

e_old = energy(t1, t2)
print(f"MP2 (spin-orbital) = {e_old:.10f}")
for it in range(100):
    tau_t = t2 + 0.5 * (np.einsum("ia,jb->ijab", t1, t1) - np.einsum("ib,ja->ijab", t1, t1))
    tau = t2 + np.einsum("ia,jb->ijab", t1, t1) - np.einsum("ib,ja->ijab", t1, t1)

    Fae = np.einsum("mf,amef->ae", t1, g[v, o, v, v], optimize=True) \
        - 0.5 * np.einsum("mnaf,mnef->ae", tau_t, oovv, optimize=True)
    Fmi = np.einsum("ne,mnie->mi", t1, g[o, o, o, v], optimize=True) \
        + 0.5 * np.einsum("inef,mnef->mi", tau_t, oovv, optimize=True)
    Fme = np.einsum("nf,mnef->me", t1, oovv, optimize=True)

    # NB: P(ij) in Wmnij[m,n,i,j] acts on axes (2,3); P(ab) in Wabef[a,b,e,f] on axes (0,1).
    wt = np.einsum("je,mnie->mnij", t1, g[o, o, o, v], optimize=True)
    Wmnij = g[o, o, o, o] + (wt - wt.transpose(0, 1, 3, 2)) \
        + 0.25 * np.einsum("ijef,mnef->mnij", tau, oovv, optimize=True)
    wt = np.einsum("mb,amef->abef", t1, g[v, o, v, v], optimize=True)
    Wabef = g[v, v, v, v] - (wt - wt.transpose(1, 0, 2, 3)) \
        + 0.25 * np.einsum("mnab,mnef->abef", tau, oovv, optimize=True)
    Wmbej = g[o, v, v, o] + np.einsum("jf,mbef->mbej", t1, g[o, v, v, v], optimize=True) \
        - np.einsum("nb,mnej->mbej", t1, g[o, o, v, o], optimize=True) \
        - np.einsum("jnfb,mnef->mbej", (0.5 * t2 + np.einsum("jf,nb->jnfb", t1, t1)), oovv, optimize=True)

    # T1
    t1n = np.einsum("ie,ae->ia", t1, Fae, optimize=True) \
        - np.einsum("ma,mi->ia", t1, Fmi, optimize=True) \
        + np.einsum("imae,me->ia", t2, Fme, optimize=True) \
        - np.einsum("nf,naif->ia", t1, g[o, v, o, v], optimize=True) \
        - 0.5 * np.einsum("imef,maef->ia", t2, g[o, v, v, v], optimize=True) \
        - 0.5 * np.einsum("mnae,nmei->ia", t2, g[o, o, v, o], optimize=True)
    t1n /= Dia

    # T2
    t2n = oovv.copy()
    tmp = Fae - 0.5 * np.einsum("mb,me->be", t1, Fme, optimize=True)
    t2n += P_ab(np.einsum("ijae,be->ijab", t2, tmp, optimize=True))
    tmp = Fmi + 0.5 * np.einsum("je,me->mj", t1, Fme, optimize=True)
    t2n -= P_ij(np.einsum("imab,mj->ijab", t2, tmp, optimize=True))
    t2n += 0.5 * np.einsum("mnab,mnij->ijab", tau, Wmnij, optimize=True)
    t2n += 0.5 * np.einsum("ijef,abef->ijab", tau, Wabef, optimize=True)
    tmp = np.einsum("imae,mbej->ijab", t2, Wmbej, optimize=True) \
        - np.einsum("ie,ma,mbej->ijab", t1, t1, g[o, v, v, o], optimize=True)
    t2n += P_ij(P_ab(tmp))
    t2n += P_ij(np.einsum("ie,abej->ijab", t1, g[v, v, v, o], optimize=True))
    t2n -= P_ab(np.einsum("ma,mbij->ijab", t1, g[o, v, o, o], optimize=True))
    t2n /= Dijab

    t1, t2 = t1n, t2n
    e_new = energy(t1, t2)
    if abs(e_new - e_old) < 1e-11:
        print(f"converged in {it+1} iters")
        break
    e_old = e_new

print(f"spin-orbital CCSD corr = {e_new:.10f}")
print(f"psi4 conv CCSD corr    = {ref_ccsd:.10f}")
print(f"difference             = {abs(e_new - ref_ccsd):.2e}")
