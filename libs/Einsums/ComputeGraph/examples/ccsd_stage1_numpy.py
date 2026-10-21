# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""CCSD tutorial, stage 1: closed-shell DF-CCSD in NumPy.

The reference every later stage is checked against. The equations are the spin-adapted
closed-shell ones of psi4numpy's RHF-CCSD (Crawford's ccenergy formulation), with every
two-electron integral taken from the density-fitted three-index tensor.
"""

from __future__ import annotations

import functools
import time

import numpy as np

from ccsd_problem import PSI4_DF_CCSD, PSI4_DF_CCSD_TOLERANCE, PSI4_DF_MP2, REFERENCE_CCSD, load

# [doc:integrals]
# Without optimize, np.einsum runs a contraction as one loop nest; with it, NumPy hands each pair
# of operands to BLAS, which is what a reference worth timing against does.
einsum = functools.partial(np.einsum, optimize=True)

p = load()
no, nv = p.nocc, p.nvir
o, v = slice(0, no), slice(no, no + nv)
B = {"o": p.B[:, o, :], "v": p.B[:, v, :]}


def g(blocks):
    """The block <pq|rs> = (pr|qs) = sum_Q B[Q,p,r] B[Q,q,s], named like "oovv"."""
    P, Q, R, S = blocks
    left = B[P][:, :, v if R == "v" else o]
    right = B[Q][:, :, v if S == "v" else o]
    return einsum("Xpr,Xqs->pqrs", left, right)


oovv, ooov, oovo, oooo = g("oovv"), g("ooov"), g("oovo"), g("oooo")
ovvo, ovov, ovvv, vvvo, ovoo = g("ovvo"), g("ovov"), g("ovvv"), g("vvvo"), g("ovoo")
Bvv = B["v"][:, :, v]  # the v^4 block <ab|ef> is never formed; the ladder reads its factors
# [doc:integrals-end]

# [doc:amplitudes]
eo, ev = p.eps_occ, p.eps_vir
Dia = eo[:, None] - ev[None, :]
Dijab = eo[:, None, None, None] + eo[None, :, None, None] - ev[None, None, :, None] - ev[None, None, None, :]

t1 = np.zeros((no, nv))
t2 = oovv / Dijab


def energy(t1, t2):
    tau = t2 + einsum("ia,jb->ijab", t1, t1)
    return 2.0 * einsum("ijab,ijab->", tau, oovv) - einsum("ijab,ijba->", tau, oovv)


e_mp2 = energy(t1, t2)
# [doc:amplitudes-end]
print(f"MP2  correlation energy = {e_mp2:.10f}   psi4 DF-MP2 = {PSI4_DF_MP2:.10f}")
assert abs(e_mp2 - PSI4_DF_MP2) < 1e-9


# [doc:iteration]
def sym(x):
    """x[i,j,a,b] + x[j,i,b,a]: the closed-shell residual is symmetric under swapping electrons."""
    return x + x.transpose(1, 0, 3, 2)


def iterate(t1, t2):
    tau = t2 + einsum("ia,jb->ijab", t1, t1)
    taut = t2 + 0.5 * einsum("ia,jb->ijab", t1, t1)

    # [doc:fae]
    Fae = np.diag(ev) + 2.0 * einsum("mf,mafe->ae", t1, ovvv) - einsum("mf,maef->ae", t1, ovvv)
    Fae -= 2.0 * einsum("mnaf,mnef->ae", taut, oovv) - einsum("mnaf,mnfe->ae", taut, oovv)
    # [doc:fae-end]

    Fmi = np.diag(eo) + 2.0 * einsum("ne,mnie->mi", t1, ooov) - einsum("ne,mnei->mi", t1, oovo)
    Fmi += 2.0 * einsum("inef,mnef->mi", taut, oovv) - einsum("inef,mnfe->mi", taut, oovv)

    Fme = 2.0 * einsum("nf,mnef->me", t1, oovv) - einsum("nf,mnfe->me", t1, oovv)

    Wmnij = oooo + einsum("je,mnie->mnij", t1, ooov) + einsum("ie,mnej->mnij", t1, oovo)
    Wmnij += einsum("ijef,mnef->mnij", tau, oovv)

    jnfb = 0.5 * t2 + einsum("jf,nb->jnfb", t1, t1)
    Wmbej = ovvo + einsum("jf,mbef->mbej", t1, ovvv) - einsum("nb,mnej->mbej", t1, oovo)
    Wmbej -= einsum("jnfb,mnef->mbej", jnfb, oovv)
    Wmbej += einsum("njfb,mnef->mbej", t2, oovv) - 0.5 * einsum("njfb,mnfe->mbej", t2, oovv)

    Wmbje = -ovov - einsum("jf,mbfe->mbje", t1, ovvv) + einsum("nb,mnje->mbje", t1, ooov)
    Wmbje += einsum("jnfb,mnfe->mbje", jnfb, oovv)

    Zmbij = einsum("mbef,ijef->mbij", ovvv, tau)

    r1 = einsum("ie,ae->ia", t1, Fae) - einsum("ma,mi->ia", t1, Fmi)
    r1 += 2.0 * einsum("imae,me->ia", t2, Fme) - einsum("imea,me->ia", t2, Fme)
    r1 += 2.0 * einsum("nf,nafi->ia", t1, ovvo) - einsum("nf,naif->ia", t1, ovov)
    r1 += 2.0 * einsum("mief,maef->ia", t2, ovvv) - einsum("mife,maef->ia", t2, ovvv)
    r1 -= 2.0 * einsum("mnae,nmei->ia", t2, oovo) - einsum("mnae,nmie->ia", t2, ooov)

    # [doc:r2]
    # The particle ladder, from the factors of <ab|ef> at every iteration. Written as one einsum
    # over the three operands, NumPy's path search would refuse both ways of splitting it, since it
    # caps an intermediate at the size of the largest input or output, and run one seven-index loop.
    abef = einsum("Qae,Qbf->abef", Bvv, Bvv)
    r2 = oovv + einsum("mnab,mnij->ijab", tau, Wmnij) + einsum("ijef,abef->ijab", tau, abef)
    r2 += sym(einsum("ijae,be->ijab", t2, Fae - 0.5 * einsum("mb,me->be", t1, Fme)))
    r2 -= sym(einsum("imab,mj->ijab", t2, Fmi + 0.5 * einsum("je,me->mj", t1, Fme)))
    r2 -= sym(einsum("ma,mbij->ijab", t1, Zmbij))
    r2 += sym(einsum("imae,mbej->ijab", t2 - t2.transpose(0, 1, 3, 2), Wmbej))
    r2 += sym(einsum("imae,mbej->ijab", t2, Wmbej) + einsum("imae,mbje->ijab", t2, Wmbje))
    r2 += sym(einsum("mjae,mbie->ijab", t2, Wmbje))
    r2 -= sym(einsum("ie,ma,mbej->ijab", t1, t1, ovvo))
    r2 -= sym(einsum("ie,mb,maje->ijab", t1, t1, ovov))
    r2 += sym(einsum("ie,abej->ijab", t1, vvvo)) - sym(einsum("ma,mbij->ijab", t1, ovoo))
    # [doc:r2-end]

    # The diagonal Fock terms in Fae and Fmi contribute -t D to the residual, so adding r / D
    # cancels the old amplitudes and leaves the new ones.
    return t1 + r1 / Dia, t2 + r2 / Dijab
# [doc:iteration-end]


start, e_old = time.perf_counter(), e_mp2
for iteration in range(1, 101):
    t1, t2 = iterate(t1, t2)
    e_ccsd = energy(t1, t2)
    if abs(e_ccsd - e_old) < 1e-11:
        break
    e_old = e_ccsd
elapsed = time.perf_counter() - start

print(f"CCSD correlation energy = {e_ccsd:.10f}   psi4 DF-CCSD = {PSI4_DF_CCSD:.10f}   ({iteration} iterations, {1000 * elapsed / iteration:.2f} ms each)")
assert abs(e_ccsd - PSI4_DF_CCSD) < PSI4_DF_CCSD_TOLERANCE
assert abs(e_ccsd - REFERENCE_CCSD) < 1e-12
