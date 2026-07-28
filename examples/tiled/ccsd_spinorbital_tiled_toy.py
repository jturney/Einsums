#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

"""Toy spin-orbital CCSD on TiledRuntimeTensor - the tiled workflow set, end to end.

Synthetic integrals (random, with the physical symmetries and a safe gap), so it
runs anywhere einsums does: no psi4, no bridge. The equations are the
Stanton-Gauss-Watts-Bartlett spin-orbital CCSD transcribed from
examples/psi4-bridge/ccsd_spinorbital_oracle.py, and every einsums run is
validated against a numpy oracle of those same equations.

The point of this file is that ONE iteration body - einsum, permute (the
P(ij)/P(ab) antisymmetrizers), axpby, direct_division, dot, graph-owned scratch,
and cg.diis - runs unchanged across three tensor configurations:

    dense          RuntimeTensor + declare_zero_tensor scratch
    tiled-1        TiledRuntimeTensor with ONE tile per axis
    tiled-blocked  TiledRuntimeTensor with occ blocks [4,2,2,2], vir [4,4,4,4]

Only the tensor factories differ. That is the "two front doors, one engine"
arrangement made executable: the op surface is uniform (enforced by
Tests.Unit.Modules.ComputeGraph.TiledParityPython), dense remains the canonical
storage/IR underneath (TiledExpansion lowers the blocked runs onto per-tile
dense nodes ahead of the optimizer), and the dense-vs-tiled-1 timing column IS
the measured cost of the tiled machinery at one tile - the number the
"could tiled subsume the dense interface?" question turns on.

The captured loop uses the graph idiom throughout: scratch is graph-owned and
DEFERRED (declare_zero_tensor / declare_zero_tiled_tensor), amplitudes update
in place, cg.diis extrapolates between replays through the loop predicate, and
a convergence scalar drives the predicate. Per-tile-block work via cg.tile_view
is not needed at this size (the v^4 ladder is tiny); it becomes the blocking
tool when a real system outgrows whole-tensor contractions.

Known state of the tiled-blocked column: it currently measures the PER-TILE
engine, not densified replay. TiledExpansion declines to densify this body
because the tiled permute and tiled dot nodes are opaque to it (no expandable
descriptor), and its correctness rule excludes every tensor an inexpandable
node touches - which cascades through the whole iteration. Teaching
TiledExpansion to expand Permute and Dot is the recorded follow-up; when it
lands, the blocked column should drop toward the dense one at this size, the
way the original residual-only toy did (140 densified nodes, ~4 ms replay).

Run with the Einsums build on PYTHONPATH, using the conda-env Python::

    PYTHONPATH=/Users/jturney/Code/Einsums/Einsums/build/lib \
        /Users/jturney/miniconda3/envs/einsums-dev/bin/python \
        examples/tiled/ccsd_spinorbital_tiled_toy.py --einsums:profile:disable
"""
import time

import numpy as np

import einsums
import einsums.graph as cg
from einsums import linalg as la

# ── Toy system: 13 spatial orbitals (5 occ), 26 spin orbitals ────────────────
NSPATIAL, NOCC_SP = 13, 5
NO, NV = 2 * NOCC_SP, 2 * (NSPATIAL - NOCC_SP)          # 10 occupied, 16 virtual
OCC_BLOCKS, VIR_BLOCKS = [4, 2, 2, 2], [4, 4, 4, 4]     # the classic toy blocking
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

# Every integral block the iteration reads, as its own tensor.
BLOCKS = {"oovv": g_np[o, o, v, v], "vovv": g_np[v, o, v, v], "ooov": g_np[o, o, o, v],
          "oooo": g_np[o, o, o, o], "vvvv": g_np[v, v, v, v], "ovvo": g_np[o, v, v, o],
          "ovvv": g_np[o, v, v, v], "oovo": g_np[o, o, v, o], "ovov": g_np[o, v, o, v],
          "vvvo": g_np[v, v, v, o], "ovoo": g_np[o, v, o, o]}
AXES = {"o": None, "v": None}  # filled per configuration below


# ── numpy oracle (the SGWB equations, verbatim from the checked-in oracle) ───
def oracle():
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
            return e_new, it + 1
        e_old = e_new
    raise RuntimeError("oracle did not converge")


# ── tensor factories per configuration ───────────────────────────────────────
def dense_factories():
    def make(name, arr):
        return einsums.asarray(np.ascontiguousarray(arr), name=name)

    def zeros(name, axes):
        return einsums.zeros(tuple(sum(AXES[a]) for a in axes), dtype="float64")

    def scratch(graph, name, axes):
        return graph.declare_zero_tensor(name, [sum(AXES[a]) for a in axes], dtype="float64", intermediate=True)

    return make, zeros, scratch


def tiled_factories():
    def grid_for(axes):
        return [list(AXES[a]) for a in axes]

    def make(name, arr):
        axes = _axes_of(arr.shape)
        t = einsums.TiledRuntimeTensorD(name, grid_for(axes))
        off, sz = t.tile_offsets(), t.tile_sizes()
        import itertools
        for coord in itertools.product(*[range(len(s)) for s in sz]):
            t.add_tile(list(coord))
        t.materialize()
        for coord in itertools.product(*[range(len(s)) for s in sz]):
            slc = tuple(slice(off[ax][coord[ax]], off[ax][coord[ax]] + sz[ax][coord[ax]]) for ax in range(len(sz)))
            np.asarray(t.tile_view(list(coord)))[...] = arr[slc]
        return t

    def zeros(name, axes):
        t = einsums.TiledRuntimeTensorD(name, grid_for(axes))
        import itertools
        for coord in itertools.product(*[range(len(s)) for s in t.tile_sizes()]):
            t.add_tile(list(coord))
        t.materialize()
        return t

    def scratch(graph, name, axes):
        return graph.declare_zero_tiled_tensor(name, grid_for(axes), dtype="float64", intermediate=True)

    return make, zeros, scratch


def _axes_of(shape):
    return tuple("o" if d == NO else "v" for d in shape)


def _gather(t):
    """Dense numpy copy of a dense or tiled einsums tensor."""
    if not hasattr(t, "tile_sizes"):
        return np.asarray(t).copy()
    import itertools
    off, sz = t.tile_offsets(), t.tile_sizes()
    M = np.zeros(tuple(sum(s) for s in sz))
    for coord in itertools.product(*[range(len(s)) for s in sz]):
        if t.has_tile(list(coord)):
            slc = tuple(slice(off[ax][coord[ax]], off[ax][coord[ax]] + sz[ax][coord[ax]]) for ax in range(len(sz)))
            M[slc] = np.asarray(t.tile_view(list(coord)))
    return M


# ── one CCSD run: capture the SGWB iteration once, replay under DIIS ─────────
def run_ccsd(make, zeros, scratch, label):
    G = {nm: make(nm, arr) for nm, arr in BLOCKS.items()}
    Dia = make("Dia", Dia_np)
    Dijab = make("Dijab", Dijab_np)
    t1 = zeros("t1", "ov")
    t2 = make("t2", BLOCKS["oovv"] / Dijab_np)          # MP2 start
    s1 = zeros("s1", "ov")                              # DIIS step tensors: host-readable
    s2 = zeros("s2", "oovv")
    Ecorr = einsums.zeros((1,), dtype="float64")
    e_part = einsums.zeros((1,), dtype="float64")

    gr = cg.Graph(f"ccsd_{label}")
    S = {nm: scratch(gr, nm, ax) for nm, ax in [
        ("tau", "oovv"), ("taut", "oovv"), ("ot", "oovv"), ("Fae", "vv"), ("Fmi", "oo"),
        ("Fme", "ov"), ("Wmnij", "oooo"), ("Wabef", "vvvv"), ("Wmbej", "ovvo"), ("jnfb", "oovv"),
        ("wt4o", "oooo"), ("wt4v", "vvvv"), ("be", "vv"), ("mj", "oo"), ("imea", "oovv"),
        ("r1", "ov"), ("r2", "oovv"), ("tmp", "oovv"), ("Xe", "ov")]}

    def ein(spec, out, A, B, pf=1.0, acc=False):
        einsums.einsum(spec, out, A, B, c_pf=(1.0 if acc else 0.0), ab_pf=pf)

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

    t0 = time.perf_counter()
    gr.apply(cg.default_pass_manager())
    t_opt = time.perf_counter() - t0
    nodes = body.num_nodes()

    t0 = time.perf_counter()
    gr.execute()
    t_exec = time.perf_counter() - t0

    return {"label": label, "energy": float(np.asarray(Ecorr)[0]), "iters": iters[0],
            "nodes": nodes, "opt_ms": t_opt * 1e3, "ms_per_iter": t_exec * 1e3 / max(iters[0], 1),
            "t1": _gather(t1), "t2": _gather(t2)}


# ── run all three configurations against the oracle ──────────────────────────
e_ref, it_ref = oracle()
print(f"numpy oracle           : E = {e_ref:.12f}   ({it_ref} plain iterations)")

results = []
for label, axes in (("dense", {"o": [NO], "v": [NV]}),
                    ("tiled-1", {"o": [NO], "v": [NV]}),
                    ("tiled-blocked", {"o": OCC_BLOCKS, "v": VIR_BLOCKS})):
    AXES.update(axes)
    make, zeros, scratch = dense_factories() if label == "dense" else tiled_factories()
    r = run_ccsd(make, zeros, scratch, label)
    results.append(r)
    err = abs(r["energy"] - e_ref)
    print(f"{r['label']:<22} : E = {r['energy']:.12f}   err vs oracle = {err:.2e}   "
          f"{r['iters']} DIIS iterations")
    assert err < 1e-9, f"{label} disagrees with the oracle"

print()
print(f"{'config':<15} {'body nodes':>10} {'optimize ms':>12} {'ms / iteration':>15}")
for r in results:
    print(f"{r['label']:<15} {r['nodes']:>10} {r['opt_ms']:>12.1f} {r['ms_per_iter']:>15.2f}")
d, t1r = results[0], results[1]
print(f"\ntiled machinery at ONE tile costs {t1r['ms_per_iter'] / d['ms_per_iter']:.2f}x "
      f"the dense replay ({t1r['ms_per_iter']:.2f} vs {d['ms_per_iter']:.2f} ms/iter) - "
      f"the number the 'tiled as the generalized interface' question turns on.")
print("toy tiled CCSD (synthetic integrals, SGWB spin-orbital equations, cg.diis) OK")
