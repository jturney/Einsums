#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

"""Closed-shell spin-adapted RHF-CCSD reference oracle (pure numpy).

A faithful reproduction of psi4numpy Coupled-Cluster/RHF/helper_ccenergy.py
(D. Crawford's ccenergy formulation). This is not an einsums example; it is the
equation reference that the closed-shell einsums DF-CCSD (df_ccsd_*) is validated
against, term-for-term. v4 is exact here, and it matches psi4 conventional CCSD
to ~1e-11 for cc-pVDZ water: closed-shell CCSD corr = -0.2134804971.

Conventions, matching psi4numpy:
  * MO physicist <pq|rs> = mo_eri.swapaxes(1,2), built here from ao_eri + C.
  * Bare Fock blocks kept: canonical RHF -> F_vv=diag(ev), F_oo=diag(eo), F_ov=0.
    The update t += r/D works because the diagonal Fock in Fae/Fmi contributes
    -t·D, which cancels t and leaves t_new = (off-diagonal terms)/D.
  * Closed-shell symmetrizer sym(r) = r_ijab + r_jiba (swapaxes(0,1).swapaxes(2,3)).
  * Wabef is done via the Zmbij auxiliary plus the explicit vvvv term.

Run with the psi4 stage on PYTHONPATH, using the conda-env Python::

    PYTHONPATH=/Users/jturney/Code/psi4/cmake-build-debug/stage/lib \
        /Users/jturney/miniconda3/envs/einsums-dev/bin/python \
        examples/psi4-bridge/ccsd_rhf_oracle.py
"""
import numpy as np
import psi4

psi4.core.set_output_file("/tmp/psi4_ccsd_conv.out", False)
psi4.set_options({"basis": "cc-pvdz", "scf_type": "pk", "freeze_core": "false",
                  "e_convergence": 1e-10, "d_convergence": 1e-10, "cc_type": "conv", "r_convergence": 1e-9})
mol = psi4.geometry("O\nH 1 0.96\nH 1 0.96 2 104.5\nsymmetry c1\n")
e_scf, wfn = psi4.energy("ccsd", return_wfn=True)
ref = psi4.variable("CCSD CORRELATION ENERGY")

nbf, nocc = wfn.nso(), wfn.nalpha()
C = np.asarray(wfn.Ca()); eps = np.asarray(wfn.epsilon_a())
Iao = np.asarray(psi4.core.MintsHelper(wfn.basisset()).ao_eri())
MO = np.einsum("mp,nr,lq,os,mnlo->pqrs", C, C, C, C, Iao, optimize=True)  # <pq|rs>
o, v = slice(0, nocc), slice(nocc, nbf)
eo, ev = eps[:nocc], eps[nocc:]

Fov = np.zeros((nocc, nbf - nocc)); Fvv = np.diag(ev); Foo = np.diag(eo)
Dia = eo[:, None] - ev[None, :]
Dijab = eo[:, None, None, None] + eo[None, :, None, None] - ev[None, None, :, None] - ev[None, None, None, :]

def MOb(s):
    sl = {'o': o, 'v': v}
    return MO[sl[s[0]], sl[s[1]], sl[s[2]], sl[s[3]]]

def nd(spec, A, B, pf=1.0): return pf * np.einsum(spec, A, B, optimize=True)
def sym(t): return t + t.swapaxes(0, 1).swapaxes(2, 3)

t1 = np.zeros((nocc, nbf - nocc))
t2 = MOb('oovv') / Dijab

def tau():  return t2 + np.einsum('ia,jb->ijab', t1, t1)
def taut(): return t2 + 0.5 * np.einsum('ia,jb->ijab', t1, t1)

def energy():
    tt = tau()
    return (2.0 * np.einsum('ia,ia->', Fov, t1)
            + 2.0 * np.einsum('ijab,ijab->', tt, MOb('oovv'))
            - 1.0 * np.einsum('ijab,ijba->', tt, MOb('oovv')))

e_old = energy()
print(f"MP2 (closed-shell) = {e_old:.10f}")
for it in range(100):
    Fae = Fvv.copy()
    Fae -= nd('me,ma->ae', Fov, t1, 0.5)
    Fae += nd('mf,mafe->ae', t1, MOb('ovvv'), 2.0)
    Fae += nd('mf,maef->ae', t1, MOb('ovvv'), -1.0)
    Fae -= nd('mnaf,mnef->ae', taut(), MOb('oovv'), 2.0)
    Fae -= nd('mnaf,mnfe->ae', taut(), MOb('oovv'), -1.0)

    Fmi = Foo.copy()
    Fmi += nd('ie,me->mi', t1, Fov, 0.5)
    Fmi += nd('ne,mnie->mi', t1, MOb('ooov'), 2.0)
    Fmi += nd('ne,mnei->mi', t1, MOb('oovo'), -1.0)
    Fmi += nd('inef,mnef->mi', taut(), MOb('oovv'), 2.0)
    Fmi += nd('inef,mnfe->mi', taut(), MOb('oovv'), -1.0)

    Fme = Fov.copy()
    Fme += nd('nf,mnef->me', t1, MOb('oovv'), 2.0)
    Fme += nd('nf,mnfe->me', t1, MOb('oovv'), -1.0)

    Wmnij = MOb('oooo').copy()
    Wmnij += nd('je,mnie->mnij', t1, MOb('ooov'))
    Wmnij += nd('ie,mnej->mnij', t1, MOb('oovo'))
    Wmnij += nd('ijef,mnef->mnij', tau(), MOb('oovv'))

    Wmbej = MOb('ovvo').copy()
    Wmbej += nd('jf,mbef->mbej', t1, MOb('ovvv'))
    Wmbej -= nd('nb,mnej->mbej', t1, MOb('oovo'))
    tmp = 0.5 * t2 + np.einsum('jf,nb->jnfb', t1, t1)
    Wmbej -= nd('jnfb,mnef->mbej', tmp, MOb('oovv'))
    Wmbej += nd('njfb,mnef->mbej', t2, MOb('oovv'), 1.0)
    Wmbej += nd('njfb,mnfe->mbej', t2, MOb('oovv'), -0.5)

    Wmbje = -1.0 * MOb('ovov').copy()
    Wmbje -= nd('jf,mbfe->mbje', t1, MOb('ovvv'))
    Wmbje += nd('nb,mnje->mbje', t1, MOb('ooov'))
    tmp = 0.5 * t2 + np.einsum('jf,nb->jnfb', t1, t1)
    Wmbje += nd('jnfb,mnfe->mbje', tmp, MOb('oovv'))

    Zmbij = nd('mbef,ijef->mbij', MOb('ovvv'), tau())

    # T1
    r1 = Fov.copy()
    r1 += nd('ie,ae->ia', t1, Fae)
    r1 -= nd('ma,mi->ia', t1, Fmi)
    r1 += nd('imae,me->ia', t2, Fme, 2.0)
    r1 += nd('imea,me->ia', t2, Fme, -1.0)
    r1 += nd('nf,nafi->ia', t1, MOb('ovvo'), 2.0)
    r1 += nd('nf,naif->ia', t1, MOb('ovov'), -1.0)
    r1 += nd('mief,maef->ia', t2, MOb('ovvv'), 2.0)
    r1 += nd('mife,maef->ia', t2, MOb('ovvv'), -1.0)
    r1 -= nd('mnae,nmei->ia', t2, MOb('oovo'), 2.0)
    r1 -= nd('mnae,nmie->ia', t2, MOb('ooov'), -1.0)

    # T2
    r2 = MOb('oovv').copy()
    r2 += sym(nd('ijae,be->ijab', t2, Fae))
    r2 -= sym(nd('ijae,be->ijab', t2, nd('mb,me->be', t1, Fme), 0.5))
    r2 -= sym(nd('imab,mj->ijab', t2, Fmi))
    r2 -= sym(nd('imab,jm->ijab', t2, nd('je,me->jm', t1, Fme), 0.5))
    r2 += nd('mnab,mnij->ijab', tau(), Wmnij)
    r2 += nd('ijef,abef->ijab', tau(), MOb('vvvv'))
    r2 -= sym(nd('ma,mbij->ijab', t1, Zmbij))
    r2 += sym(nd('imae,mbej->ijab', t2, Wmbej) + nd('imea,mbej->ijab', t2, Wmbej, -1.0))
    r2 += sym(nd('imae,mbej->ijab', t2, Wmbej) + nd('imae,mbje->ijab', t2, Wmbje))
    r2 += sym(nd('mjae,mbie->ijab', t2, Wmbje))
    r2 -= sym(nd('imea,mbej->ijab', np.einsum('ie,ma->imea', t1, t1), MOb('ovvo')))
    r2 -= sym(nd('imeb,maje->ijab', np.einsum('ie,mb->imeb', t1, t1), MOb('ovov')))
    r2 += sym(nd('ie,abej->ijab', t1, MOb('vvvo')))
    r2 -= sym(nd('ma,mbij->ijab', t1, MOb('ovoo')))

    t1 = t1 + r1 / Dia
    t2 = t2 + r2 / Dijab
    e_new = energy()
    if abs(e_new - e_old) < 1e-11:
        print(f"converged in {it+1} iters"); break
    e_old = e_new

print(f"closed-shell CCSD corr = {e_new:.10f}")
print(f"psi4 conv CCSD corr    = {ref:.10f}")
print(f"difference             = {abs(e_new - ref):.2e}")
assert abs(e_new - ref) < 1e-7
print("closed-shell spin-adapted RHF-CCSD (numpy reference) MATCHES psi4")
