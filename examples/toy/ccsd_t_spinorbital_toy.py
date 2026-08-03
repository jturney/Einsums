#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

"""Toy spin-orbital CCSD(T) - the perturbative triples correction on the ComputeGraph.

The companion to examples/tiled/ccsd_spinorbital_tiled_toy.py: the same synthetic
integrals, the same Stanton-Gauss-Watts-Bartlett spin-orbital CCSD iteration, the
same numpy oracle discipline, with the (T) correction bolted on the end. It needs
no psi4 and no bridge, so it runs anywhere einsums does.

The triples equations are Raghavachari's (T) in the spin-orbital form of
Crawford's programming project 6::

    Xc[ijkabc] = sum_e t2[jkae] <ei||bc> - sum_m t2[imbc] <ma||jk>
    Xd[ijkabc] = t1[ia] <jk||bc>
    W = P(i/jk) P(a/bc) Xc            V = P(i/jk) P(a/bc) Xd
    E(T) = 1/36 sum_{ijkabc} W (W + V) / D

with P(i/jk) f = f(ijk) - f(jik) - f(kji). Both the CCSD equations and this (T)
expression were checked against psi4's CCSD(T) on water/STO-3G before this file
was written: CCSD correlation to 8e-13 Eh, (T) to 9e-15 Eh.

WHY THIS FILE EXISTS. Triples is the first workload in the repo where the
antisymmetrizers, not the contractions, are the bill. Every P(i/jk)P(a/bc) is nine
permuted accumulations of a rank-6 tensor, and at this toy size the three
contractions that build Xc and Xd cost about 6 ms while the eighteen permuted
accumulations that antisymmetrize them cost about 30 ms. That ratio barely improves
with system size - the connected term is o^3 v^4 of arithmetic against 9 o^3 v^3 of
permute traffic, so flops per moved element grow only like v/9 - which makes the
P-operator the obvious next target for an optimization pass.

To make the headroom measurable rather than asserted, the correction is written
two ways and both are validated against the oracle:

    naive   both W and V antisymmetrized, E = 1/36 sum W (W + V) / D
    folded  only W antisymmetrized,       E = 1/4  sum (W/D) (Xc + Xd)

The folded form is exact, not an approximation. Because D is invariant under
P(i/jk)P(a/bc) and W is antisymmetric under it, summing W against any one of the
nine permuted copies of a tensor gives the same value up to the permutation's
sign, so the nine terms of V collapse into 9 * (the unpermuted term) and 9/36
becomes 1/4. A pass that recognized "this antisymmetrized tensor is only ever
contracted against another antisymmetrized tensor" could do that rewrite itself.
The naive-minus-folded time is what such a pass would be worth here.

Each formulation runs twice, with and without the default pass manager, so the
table also shows what today's passes recover on a flat rank-6 body. As of writing
that is nothing: the pipeline's only edits are Materialization, FreeInsertion and
MemoryPlanning, and replay time is unchanged. SymmetrizedAccumulation is the pass
that looks closest to relevant, and it declines every antisymmetrizer permute with
"permute accumulates into its output (beta != 0)" - correctly, since the P-operator
is already written in the accumulating form that pass rewrites toward. Its
documented Level 2 (one fused sweep instead of one sweep per permuted term) is what
this workload actually wants: nine terms that read the same source and accumulate
into the same destination move 852 MB here, where a fused kernel would move 98 MB.
Pass ``--explain`` for the full report.

Run with the Einsums build on PYTHONPATH, using the conda-env Python::

    PYTHONPATH=/Users/jturney/Code/Einsums/Einsums/build/lib \
        /Users/jturney/miniconda3/envs/einsums-dev/bin/python \
        examples/toy/ccsd_t_spinorbital_toy.py --einsums:profile:disable
"""
import argparse
import time

import numpy as np

import einsums
import einsums.graph as cg   # the graph.py shell (capture/default_pass_manager); NOT
                             # `from einsums import graph`, which resolves to the bare
                             # _core.graph submodule and lacks the `capture` helper.
from einsums import linalg as la

_argp = argparse.ArgumentParser(description="Toy spin-orbital CCSD(T) in einsums.")
_argp.add_argument(
    "--explain", action="store_true",
    help="run the (T) pipeline at pass verbosity 2 and print the pass report, "
         "including which passes declined the antisymmetrizer and why",
)
_args, _ = _argp.parse_known_args()

# ── Toy system: 13 spatial orbitals (5 occ), 26 spin orbitals ────────────────
NSPATIAL, NOCC_SP = 13, 5
NO, NV = 2 * NOCC_SP, 2 * (NSPATIAL - NOCC_SP)          # 10 occupied, 16 virtual
GSCALE = 0.02                                           # keeps the toy safely convergent

rng = np.random.default_rng(2026)

# Random chemist-like AO integrals with full 8-fold symmetry, then the same
# spatial -> interleaved-spin-orbital -> antisymmetrized route the oracle uses.
I = rng.standard_normal((NSPATIAL,) * 4)
I = I + I.transpose(1, 0, 2, 3)
I = I + I.transpose(0, 1, 3, 2)
I = I + I.transpose(2, 3, 0, 1)
mo = GSCALE * I.transpose(0, 2, 1, 3)                   # physicist <pq|rs> = (pr|qs)

G6 = np.zeros((NSPATIAL, 2, NSPATIAL, 2, NSPATIAL, 2, NSPATIAL, 2))
for s1 in (0, 1):
    for s2 in (0, 1):
        G6[:, s1, :, s2, :, s1, :, s2] = mo
nso = 2 * NSPATIAL
g_np = G6.reshape(nso, nso, nso, nso)
g_np = g_np - g_np.transpose(0, 1, 3, 2)                # <PQ||RS>

eps_sp = np.concatenate([-(4.0 + rng.uniform(0, 1, NOCC_SP)), 4.0 + rng.uniform(0, 1, NSPATIAL - NOCC_SP)])
eps_so = np.repeat(eps_sp, 2)
o, v = slice(0, NO), slice(NO, nso)
eo, ev = eps_so[o], eps_so[v]
Dia_np = eo[:, None] - ev[None, :]
Dijab_np = (eo[:, None, None, None] + eo[None, :, None, None]
            - ev[None, None, :, None] - ev[None, None, None, :])
Dijkabc_np = (eo[:, None, None, None, None, None] + eo[None, :, None, None, None, None]
              + eo[None, None, :, None, None, None] - ev[None, None, None, :, None, None]
              - ev[None, None, None, None, :, None] - ev[None, None, None, None, None, :])

# Every integral block the iteration and the correction read, as its own tensor.
BLOCKS = {"oovv": g_np[o, o, v, v], "vovv": g_np[v, o, v, v], "ooov": g_np[o, o, o, v],
          "oooo": g_np[o, o, o, o], "vvvv": g_np[v, v, v, v], "ovvo": g_np[o, v, v, o],
          "ovvv": g_np[o, v, v, v], "oovo": g_np[o, o, v, o], "ovov": g_np[o, v, o, v],
          "vvvo": g_np[v, v, v, o], "ovoo": g_np[o, v, o, o]}

# The nine (permutation spec, sign) pairs of P(i/jk) P(a/bc). Every one of them is
# an involution - a swap in the occupied triple times a swap in the virtual triple -
# so the permute output spec reads the same forwards and backwards.
PIJK_PABC = [("i,j,k,a,b,c", +1.0), ("j,i,k,a,b,c", -1.0), ("k,j,i,a,b,c", -1.0),
             ("i,j,k,b,a,c", -1.0), ("i,j,k,c,b,a", -1.0), ("j,i,k,b,a,c", +1.0),
             ("j,i,k,c,b,a", +1.0), ("k,j,i,b,a,c", +1.0), ("k,j,i,c,b,a", +1.0)]


# ── numpy oracle: the SGWB equations, then (T) ───────────────────────────────
def ccsd_oracle():
    oovv = BLOCKS["oovv"]
    P_ij = lambda x: x - x.transpose(1, 0, 2, 3)
    P_ab = lambda x: x - x.transpose(0, 1, 3, 2)
    t1 = np.zeros((NO, NV))
    t2 = oovv / Dijab_np
    energy = lambda: 0.25 * np.einsum("ijab,ijab->", oovv, t2) + 0.5 * np.einsum("ijab,ia,jb->", oovv, t1, t1)
    e_old = energy()
    for it in range(200):
        tau_t = t2 + 0.5 * (np.einsum("ia,jb->ijab", t1, t1) - np.einsum("ib,ja->ijab", t1, t1))
        tau = t2 + np.einsum("ia,jb->ijab", t1, t1) - np.einsum("ib,ja->ijab", t1, t1)
        Fae = np.einsum("mf,amef->ae", t1, BLOCKS["vovv"]) - 0.5 * np.einsum("mnaf,mnef->ae", tau_t, oovv)
        Fmi = np.einsum("ne,mnie->mi", t1, BLOCKS["ooov"]) + 0.5 * np.einsum("inef,mnef->mi", tau_t, oovv)
        Fme = np.einsum("nf,mnef->me", t1, oovv)
        wt = np.einsum("je,mnie->mnij", t1, BLOCKS["ooov"])
        Wmnij = BLOCKS["oooo"] + (wt - wt.transpose(0, 1, 3, 2)) + 0.25 * np.einsum("ijef,mnef->mnij", tau, oovv)
        wt = np.einsum("mb,amef->abef", t1, BLOCKS["vovv"])
        Wabef = BLOCKS["vvvv"] - (wt - wt.transpose(1, 0, 2, 3)) + 0.25 * np.einsum("mnab,mnef->abef", tau, oovv)
        Wmbej = BLOCKS["ovvo"] + np.einsum("jf,mbef->mbej", t1, BLOCKS["ovvv"]) \
            - np.einsum("nb,mnej->mbej", t1, BLOCKS["oovo"]) \
            - np.einsum("jnfb,mnef->mbej", 0.5 * t2 + np.einsum("jf,nb->jnfb", t1, t1), oovv)
        t1n = np.einsum("ie,ae->ia", t1, Fae) - np.einsum("ma,mi->ia", t1, Fmi) \
            + np.einsum("imae,me->ia", t2, Fme) - np.einsum("nf,naif->ia", t1, BLOCKS["ovov"]) \
            - 0.5 * np.einsum("imef,maef->ia", t2, BLOCKS["ovvv"]) - 0.5 * np.einsum("mnae,nmei->ia", t2, BLOCKS["oovo"])
        t1n /= Dia_np
        t2n = oovv.copy()
        t2n += P_ab(np.einsum("ijae,be->ijab", t2, Fae - 0.5 * np.einsum("mb,me->be", t1, Fme)))
        t2n -= P_ij(np.einsum("imab,mj->ijab", t2, Fmi + 0.5 * np.einsum("je,me->mj", t1, Fme)))
        t2n += 0.5 * np.einsum("mnab,mnij->ijab", tau, Wmnij)
        t2n += 0.5 * np.einsum("ijef,abef->ijab", tau, Wabef)
        t2n += P_ij(P_ab(np.einsum("imae,mbej->ijab", t2, Wmbej) - np.einsum("ie,ma,mbej->ijab", t1, t1, BLOCKS["ovvo"])))
        t2n += P_ij(np.einsum("ie,abej->ijab", t1, BLOCKS["vvvo"]))
        t2n -= P_ab(np.einsum("ma,mbij->ijab", t1, BLOCKS["ovoo"]))
        t2n /= Dijab_np
        t1, t2 = t1n, t2n
        e_new = energy()
        if abs(e_new - e_old) < 1e-11:
            return e_new, t1, t2, it + 1
        e_old = e_new
    raise RuntimeError("oracle did not converge")


def triples_oracle(t1, t2):
    def antisym(X):
        Y = X - X.transpose(1, 0, 2, 3, 4, 5) - X.transpose(2, 1, 0, 3, 4, 5)
        return Y - Y.transpose(0, 1, 2, 4, 3, 5) - Y.transpose(0, 1, 2, 5, 4, 3)

    Xc = (np.einsum("jkae,eibc->ijkabc", t2, BLOCKS["vovv"])
          - np.einsum("imbc,majk->ijkabc", t2, BLOCKS["ovoo"]))
    Xd = np.einsum("ia,jkbc->ijkabc", t1, BLOCKS["oovv"])
    W = antisym(Xc)
    V = antisym(Xd)
    naive = (1.0 / 36.0) * np.sum(W * (W + V) / Dijkabc_np)
    folded = 0.25 * np.sum((W / Dijkabc_np) * (Xc + Xd))
    return naive, folded


# ── einsums helpers ──────────────────────────────────────────────────────────
def make(name, arr):
    return einsums.asarray(np.ascontiguousarray(arr), name=name)


def ein(spec, out, A, B, pf=1.0, acc=False):
    einsums.einsum(spec, out, A, B, c_pf=(1.0 if acc else 0.0), ab_pf=pf)


def antisymmetrize(dst, src):
    """dst <- P(i/jk) P(a/bc) src, as one overwrite plus eight permuted accumulations."""
    for n, (spec, sign) in enumerate(PIJK_PABC):
        if n == 0:
            la.axpby(sign, src, 0.0, dst)
        else:
            einsums.permute(f"{spec} <- i,j,k,a,b,c", dst, src, c_pf=1.0, a_pf=sign)


# ── CCSD: capture the SGWB iteration once, replay under DIIS ─────────────────
def run_ccsd():
    G = {nm: make(nm, arr) for nm, arr in BLOCKS.items()}
    Dia = make("Dia", Dia_np)
    Dijab = make("Dijab", Dijab_np)
    t1 = einsums.zeros((NO, NV), dtype="float64")
    t2 = make("t2", BLOCKS["oovv"] / Dijab_np)          # MP2 start
    s1 = einsums.zeros((NO, NV), dtype="float64")       # DIIS step tensors: host-readable
    s2 = einsums.zeros((NO, NO, NV, NV), dtype="float64")
    Ecorr = einsums.zeros((1,), dtype="float64")
    e_part = einsums.zeros((1,), dtype="float64")

    dims = {"o": NO, "v": NV}
    gr = cg.Graph("ccsd")
    S = {nm: gr.declare_zero_tensor(nm, [dims[a] for a in ax], dtype="float64", intermediate=True)
         for nm, ax in [
             ("tau", "oovv"), ("taut", "oovv"), ("ot", "oovv"), ("Fae", "vv"), ("Fmi", "oo"),
             ("Fme", "ov"), ("Wmnij", "oooo"), ("Wabef", "vvvv"), ("Wmbej", "ovvo"), ("jnfb", "oovv"),
             ("wt4o", "oooo"), ("wt4v", "vvvv"), ("be", "vv"), ("mj", "oo"), ("imea", "oovv"),
             ("r1", "ov"), ("r2", "oovv"), ("tmp", "oovv"), ("Xe", "ov")]}

    e_prev = [1e9]
    iters = [0]

    def cont(it):
        iters[0] = it + 1
        e = float(np.asarray(Ecorr)[0])
        d = abs(e - e_prev[0])
        e_prev[0] = e
        return (d > 1e-10) and (it < 59)

    body = gr.add_loop("ccsd_iter", 60, cg.diis([(t1, s1), (t2, s2)]).wrap(cont))
    with cg.capture(body):
        # tau / taut: t2 + (1, 1/2) * (t1(x)t1 antisymmetrized in ab)
        ein("i,j,a,b <- i,a ; j,b", S["ot"], t1, t1)
        for dst, w in (("tau", 1.0), ("taut", 0.5)):
            la.axpby(1.0, t2, 0.0, S[dst])
            la.axpby(w, S["ot"], 1.0, S[dst])
            einsums.permute("i,j,b,a <- i,j,a,b", S[dst], S["ot"], c_pf=1.0, a_pf=-w)

        ein("a,e <- m,f ; a,m,e,f", S["Fae"], t1, G["vovv"])
        ein("a,e <- m,n,a,f ; m,n,e,f", S["Fae"], S["taut"], G["oovv"], -0.5, True)
        ein("m,i <- n,e ; m,n,i,e", S["Fmi"], t1, G["ooov"])
        ein("m,i <- i,n,e,f ; m,n,e,f", S["Fmi"], S["taut"], G["oovv"], 0.5, True)
        ein("m,e <- n,f ; m,n,e,f", S["Fme"], t1, G["oovv"])

        # Wmnij: P(ij) acts on axes (2,3) of Wmnij[m,n,i,j]
        ein("m,n,i,j <- j,e ; m,n,i,e", S["wt4o"], t1, G["ooov"])
        la.axpby(1.0, G["oooo"], 0.0, S["Wmnij"])
        la.axpby(1.0, S["wt4o"], 1.0, S["Wmnij"])
        einsums.permute("m,n,j,i <- m,n,i,j", S["Wmnij"], S["wt4o"], c_pf=1.0, a_pf=-1.0)
        ein("m,n,i,j <- i,j,e,f ; m,n,e,f", S["Wmnij"], S["tau"], G["oovv"], 0.25, True)

        # Wabef: P(ab) acts on axes (0,1) of Wabef[a,b,e,f]
        ein("a,b,e,f <- m,b ; a,m,e,f", S["wt4v"], t1, G["vovv"])
        la.axpby(1.0, G["vvvv"], 0.0, S["Wabef"])
        la.axpby(-1.0, S["wt4v"], 1.0, S["Wabef"])
        einsums.permute("b,a,e,f <- a,b,e,f", S["Wabef"], S["wt4v"], c_pf=1.0, a_pf=1.0)
        ein("a,b,e,f <- m,n,a,b ; m,n,e,f", S["Wabef"], S["tau"], G["oovv"], 0.25, True)

        ein("j,n,f,b <- j,f ; n,b", S["jnfb"], t1, t1)
        la.axpby(0.5, t2, 1.0, S["jnfb"])
        la.axpby(1.0, G["ovvo"], 0.0, S["Wmbej"])
        ein("m,b,e,j <- j,f ; m,b,e,f", S["Wmbej"], t1, G["ovvv"], 1.0, True)
        ein("m,b,e,j <- n,b ; m,n,e,j", S["Wmbej"], t1, G["oovo"], -1.0, True)
        ein("m,b,e,j <- j,n,f,b ; m,n,e,f", S["Wmbej"], S["jnfb"], G["oovv"], -1.0, True)

        # T1 residual -> r1
        ein("i,a <- i,e ; a,e", S["r1"], t1, S["Fae"])
        ein("i,a <- m,a ; m,i", S["r1"], t1, S["Fmi"], -1.0, True)
        ein("i,a <- i,m,a,e ; m,e", S["r1"], t2, S["Fme"], 1.0, True)
        ein("i,a <- n,f ; n,a,i,f", S["r1"], t1, G["ovov"], -1.0, True)
        ein("i,a <- i,m,e,f ; m,a,e,f", S["r1"], t2, G["ovvv"], -0.5, True)
        ein("i,a <- m,n,a,e ; n,m,e,i", S["r1"], t2, G["oovo"], -0.5, True)

        # T2 residual -> r2
        la.axpby(1.0, G["oovv"], 0.0, S["r2"])
        ein("b,e <- m,b ; m,e", S["be"], t1, S["Fme"])
        la.axpby(-0.5, S["be"], 1.0, S["Fae"])           # Fae - 1/2 t1.Fme, consumed only below
        ein("i,j,a,b <- i,j,a,e ; b,e", S["tmp"], t2, S["Fae"])
        la.axpby(1.0, S["tmp"], 1.0, S["r2"])
        einsums.permute("i,j,b,a <- i,j,a,b", S["r2"], S["tmp"], c_pf=1.0, a_pf=-1.0)
        ein("m,j <- j,e ; m,e", S["mj"], t1, S["Fme"])
        la.axpby(0.5, S["mj"], 1.0, S["Fmi"])            # Fmi + 1/2 t1.Fme
        ein("i,j,a,b <- i,m,a,b ; m,j", S["tmp"], t2, S["Fmi"])
        la.axpby(-1.0, S["tmp"], 1.0, S["r2"])
        einsums.permute("j,i,a,b <- i,j,a,b", S["r2"], S["tmp"], c_pf=1.0, a_pf=1.0)
        ein("i,j,a,b <- m,n,a,b ; m,n,i,j", S["r2"], S["tau"], S["Wmnij"], 0.5, True)
        ein("i,j,a,b <- i,j,e,f ; a,b,e,f", S["r2"], S["tau"], S["Wabef"], 0.5, True)
        # ring: P(ij)P(ab)[ t2.Wmbej - (t1(x)t1).<mb||ej> ]
        ein("i,j,a,b <- i,m,a,e ; m,b,e,j", S["tmp"], t2, S["Wmbej"])
        ein("i,m,e,a <- i,e ; m,a", S["imea"], t1, t1)
        ein("i,j,a,b <- i,m,e,a ; m,b,e,j", S["tmp"], S["imea"], G["ovvo"], -1.0, True)
        la.axpby(1.0, S["tmp"], 1.0, S["r2"])
        einsums.permute("j,i,a,b <- i,j,a,b", S["r2"], S["tmp"], c_pf=1.0, a_pf=-1.0)
        einsums.permute("i,j,b,a <- i,j,a,b", S["r2"], S["tmp"], c_pf=1.0, a_pf=-1.0)
        einsums.permute("j,i,b,a <- i,j,a,b", S["r2"], S["tmp"], c_pf=1.0, a_pf=1.0)
        ein("i,j,a,b <- i,e ; a,b,e,j", S["tmp"], t1, G["vvvo"])
        la.axpby(1.0, S["tmp"], 1.0, S["r2"])
        einsums.permute("j,i,a,b <- i,j,a,b", S["r2"], S["tmp"], c_pf=1.0, a_pf=-1.0)
        ein("i,j,a,b <- m,a ; m,b,i,j", S["tmp"], t1, G["ovoo"])
        la.axpby(-1.0, S["tmp"], 1.0, S["r2"])
        einsums.permute("i,j,b,a <- i,j,a,b", S["r2"], S["tmp"], c_pf=1.0, a_pf=1.0)

        # updates: s = new_t - t (the DIIS step), then t += s
        la.direct_division(1.0, S["r1"], Dia, 0.0, s1)
        la.axpby(-1.0, t1, 1.0, s1)
        la.axpby(1.0, s1, 1.0, t1)
        la.direct_division(1.0, S["r2"], Dijab, 0.0, s2)
        la.axpby(-1.0, t2, 1.0, s2)
        la.axpby(1.0, s2, 1.0, t2)

        # E = 1/4 <ij||ab> t2 + 1/2 <ij||ab> t1 t1  (from the UPDATED amplitudes)
        la.dot(Ecorr, G["oovv"], t2)
        la.scale(0.25, Ecorr)
        ein("j,b <- i,j,a,b ; i,a", S["Xe"], G["oovv"], t1)
        la.dot(e_part, S["Xe"], t1)
        la.axpby(0.5, e_part, 1.0, Ecorr)

    gr.apply(cg.default_pass_manager())
    gr.execute()
    return float(np.asarray(Ecorr)[0]), t1, t2, iters[0], body.num_nodes()


# ── (T): capture the correction once, replay it ──────────────────────────────
def run_triples(t1, t2, folded, optimize, replays=3):
    G = {nm: make(nm, BLOCKS[nm]) for nm in ("oovv", "vovv", "ovoo")}
    D6 = make("Dijkabc", Dijkabc_np)
    Et = einsums.zeros((1,), dtype="float64")

    six = [NO, NO, NO, NV, NV, NV]
    gr = cg.Graph("triples_folded" if folded else "triples_naive")
    names = ["Xc", "Xd", "W", "Wd"] + ([] if folded else ["V"])
    # Graph-owned scratch is deferred until the Materialization pass allocates it, and
    # Materialization is not separable from the pipeline through the Python bindings, so
    # the unoptimized baseline hands the graph eagerly allocated scratch instead.
    if optimize:
        S = {nm: gr.declare_zero_tensor(nm, six, dtype="float64", intermediate=True) for nm in names}
    else:
        S = {nm: einsums.zeros(tuple(six), dtype="float64") for nm in names}

    with cg.capture(gr):
        # Xc = sum_e t2[jkae] <ei||bc> - sum_m t2[imbc] <ma||jk>
        ein("i,j,k,a,b,c <- j,k,a,e ; e,i,b,c", S["Xc"], t2, G["vovv"])
        ein("i,j,k,a,b,c <- i,m,b,c ; m,a,j,k", S["Xc"], t2, G["ovoo"], -1.0, True)
        # Xd = t1[ia] <jk||bc>
        ein("i,j,k,a,b,c <- i,a ; j,k,b,c", S["Xd"], t1, G["oovv"])

        antisymmetrize(S["W"], S["Xc"])
        la.direct_division(1.0, S["W"], D6, 0.0, S["Wd"])       # Wd = W / D

        if folded:
            la.axpby(1.0, S["Xd"], 1.0, S["Xc"])                # Xc := Xc + Xd
            la.dot(Et, S["Wd"], S["Xc"])
            la.scale(0.25, Et)
        else:
            antisymmetrize(S["V"], S["Xd"])
            la.axpby(1.0, S["W"], 1.0, S["V"])                  # V := W + V
            la.dot(Et, S["Wd"], S["V"])
            la.scale(1.0 / 36.0, Et)

    explain = None
    t0 = time.perf_counter()
    if optimize:
        pm = cg.default_pass_manager()
        if _args.explain:
            pm.set_verbosity(2)
        gr.apply(pm)
        explain = pm.explain()
    opt_ms = (time.perf_counter() - t0) * 1e3

    best, energies = None, []
    for _ in range(replays):
        t0 = time.perf_counter()
        gr.execute()
        dt = time.perf_counter() - t0
        best = dt if best is None else min(best, dt)
        energies.append(float(np.asarray(Et)[0]))
    assert max(abs(e - energies[0]) for e in energies) == 0.0, "replay is not idempotent"

    return {"label": ("folded" if folded else "naive") + ("" if optimize else " (no passes)"),
            "energy": energies[0], "nodes": gr.num_nodes(), "opt_ms": opt_ms, "ms": best * 1e3,
            "explain": explain}


# ── run ──────────────────────────────────────────────────────────────────────
e_ccsd_ref, t1_ref, t2_ref, it_ref = ccsd_oracle()
e_t_ref, e_t_folded_ref = triples_oracle(t1_ref, t2_ref)
print(f"numpy oracle    : E(CCSD) = {e_ccsd_ref:.12f}   ({it_ref} plain iterations)")
print(f"                  E(T)    = {e_t_ref:.12f}   folded form agrees to {abs(e_t_folded_ref - e_t_ref):.1e}")

e_ccsd, t1, t2, iters, ccsd_nodes = run_ccsd()
print(f"einsums CCSD    : E(CCSD) = {e_ccsd:.12f}   err = {abs(e_ccsd - e_ccsd_ref):.2e}   "
      f"{iters} DIIS iterations, {ccsd_nodes} body nodes")
assert abs(e_ccsd - e_ccsd_ref) < 1e-9, "CCSD disagrees with the oracle"
assert np.abs(np.asarray(t1) - t1_ref).max() < 1e-9, "t1 disagrees with the oracle"
assert np.abs(np.asarray(t2) - t2_ref).max() < 1e-9, "t2 disagrees with the oracle"

results = []
for folded in (False, True):
    for optimize in (False, True):
        r = run_triples(t1, t2, folded=folded, optimize=optimize)
        results.append(r)
        err = abs(r["energy"] - e_t_ref)
        print(f"einsums (T)     : E(T)    = {r['energy']:.12f}   err = {err:.2e}   [{r['label']}]")
        assert err < 1e-10, f"{r['label']} disagrees with the oracle"
        if _args.explain and r["explain"]:
            print(f"\n--- pass report for the {r['label']} correction ---")
            print(r["explain"])

print()
print(f"total CCSD(T)   : {e_ccsd + e_t_ref:.12f}")
print()
print(f"{'(T) formulation':<22} {'nodes':>7} {'optimize ms':>12} {'ms / evaluation':>16}")
for r in results:
    print(f"{r['label']:<22} {r['nodes']:>7} {r['opt_ms']:>12.1f} {r['ms']:>16.2f}")

naive_ms = next(r["ms"] for r in results if r["label"] == "naive")
folded_ms = next(r["ms"] for r in results if r["label"] == "folded")
mb = NO ** 3 * NV ** 3 * 8 / 1e6
print()
print(f"rank-6 tensors are {mb:.1f} MB each. One P(i/jk)P(a/bc) is an overwriting axpby "
      f"(2 sweeps) plus eight accumulating permutes (3 sweeps each: read source, read and "
      f"write destination), so {26 * mb:.0f} MB moves where a single fused sweep over the "
      f"nine terms would move {3 * mb:.0f} MB.")
print(f"dropping the disconnected antisymmetrizer costs {naive_ms - folded_ms:.1f} ms of the "
      f"{naive_ms:.1f} ms naive evaluation ({100 * (naive_ms - folded_ms) / naive_ms:.0f}%) - "
      f"the headroom for a pass that folds P-operators into their consumer.")
print("toy spin-orbital CCSD(T) (synthetic integrals, SGWB + Raghavachari equations) OK")
