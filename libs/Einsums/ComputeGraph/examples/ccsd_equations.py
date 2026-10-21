# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""The closed-shell DF-CCSD equations, written once with Einsums for every later stage.

Stage 1's NumPy, term for term. Every call here runs eagerly outside a capture and records a node
inside one, so the stages after this one change what surrounds iterate(), never iterate() itself.

Three choices make that possible, and each is what NumPy code has to give up to be captured:

* Every tensor comes from new(name, shape), which the stage supplies. Eagerly it is an ordinary
  tensor; captured it is a graph intermediate, which the optimizer is free to free, reuse or
  hold a block at a time.
* Each intermediate's first term overwrites it (assign) and the rest add to it (add). A graph replays
  the same calls on the same tensors, so a sum that started from a zero filled once at creation
  would keep the previous replay's value.
* The amplitudes are updated in place. Rebinding a Python name, t1 = t1 + r1 / Dia, would change
  nothing a replay reads.
"""

from __future__ import annotations

import time

import numpy as np

import einsums
import einsums.graph as cg
from einsums import linalg as la


# [doc:integrals]
def integrals(p):
    """The integral blocks and denominators, formed once, eagerly.

    <pq|rs> = (pr|qs) = sum_Q B[Q,p,r] B[Q,q,s]. The v^4 block is not formed: the one term that
    reads it takes the three-index factors instead.
    """
    no, nv = p.nocc, p.nvir
    o, v = slice(0, no), slice(no, no + nv)
    span = {"o": o, "v": v}
    ints = {"Bvv": einsums.asarray(np.ascontiguousarray(p.B[:, v, v]), name="Bvv")}

    for blocks in ("oovv", "ooov", "oovo", "oooo", "ovvo", "ovov", "ovvv", "vvvo", "ovoo"):
        P, Q, R, S = blocks
        left = einsums.asarray(np.ascontiguousarray(p.B[:, span[P], span[R]]), name=f"B{P}{R}")
        right = einsums.asarray(np.ascontiguousarray(p.B[:, span[Q], span[S]]), name=f"B{Q}{S}")
        out = einsums.zeros([left.shape[1], right.shape[1], left.shape[2], right.shape[2]], name=blocks)
        einsums.einsum("p,q,r,s <- X,p,r ; X,q,s", out, left, right)
        ints[blocks] = out

    eo, ev = p.eps_occ, p.eps_vir
    ints["Foo"] = einsums.asarray(np.diag(eo), name="Foo")
    ints["Fvv"] = einsums.asarray(np.diag(ev), name="Fvv")
    ints["Dia"] = einsums.asarray(eo[:, None] - ev[None, :], name="Dia")
    ints["Dijab"] = einsums.asarray(
        eo[:, None, None, None] + eo[None, :, None, None] - ev[None, None, :, None] - ev[None, None, None, :], name="Dijab"
    )
    return ints
# [doc:integrals-end]


# [doc:spaces]
def register_spaces():
    """The three index spaces, with extents for the family of molecules the program is for."""
    registry = cg.global_space_registry()
    for name, symbol, extent, dim in (("occ", "o", 50.0, "nocc"), ("vir", "v", 450.0, "nvir"), ("aux", "Q", 1500.0, "naux")):
        if registry.find(name) is None:
            registry.register_space(cg.index_space(name, symbol, extent, cg.GrowthClass.linear(), dim))


def annotate(t1, t2, ints):
    """Say what every input's axes range over. Called inside the capture."""
    space = {"o": "occ", "v": "vir", "Q": "aux"}
    blocks = {"t1": "ov", "t2": "oovv", "Bvv": "Qvv", "Foo": "oo", "Fvv": "vv", "Dia": "ov", "Dijab": "oovv"}
    for tensor, letters in ((t1, blocks["t1"]), (t2, blocks["t2"])):
        cg.annotate(tensor, tuple(space[x] for x in letters))
    for name, tensor in ints.items():
        cg.annotate(tensor, tuple(space[x] for x in blocks.get(name, name)))
# [doc:spaces-end]


# [doc:helpers]
def assign(out, spec, A, B, pf=1.0):
    """out = pf * A B."""
    einsums.einsum(spec, out, A, B, c_pf=0.0, ab_pf=pf)


def add(out, spec, A, B, pf=1.0):
    """out += pf * A B."""
    einsums.einsum(spec, out, A, B, c_pf=1.0, ab_pf=pf)
# [doc:helpers-end]


# [doc:energy]
def energy(e, t1, t2, ints, new):
    """e[0] = 2 tau.<ij|ab> - tau.<ij|ba>, with tau = t2 + t1 t1."""
    tau = new("tau_e", t2.shape)
    assign(tau, "i,j,a,b <- i,a ; j,b", t1, t1)
    la.axpby(1.0, t2, 1.0, tau)
    assign(e, " <- i,j,a,b ; i,j,a,b", tau, ints["oovv"], 2.0)
    add(e, " <- i,j,a,b ; i,j,b,a", tau, ints["oovv"], -1.0)
# [doc:energy-end]


# [doc:iterate]
def iterate(t1, t2, ints, new):
    """One CCSD iteration: t1 and t2 are overwritten with the next amplitudes."""
    no, nv = t1.shape
    OV, OOVV = (no, nv), (no, no, nv, nv)
    g = ints

    tau, taut = new("tau", OOVV), new("taut", OOVV)
    assign(tau, "i,j,a,b <- i,a ; j,b", t1, t1)
    la.axpby(1.0, t2, 1.0, tau)
    assign(taut, "i,j,a,b <- i,a ; j,b", t1, t1, 0.5)
    la.axpby(1.0, t2, 1.0, taut)

    # [doc:fae]
    Fae = new("Fae", (nv, nv))
    la.axpby(1.0, g["Fvv"], 0.0, Fae)
    add(Fae, "a,e <- m,f ; m,a,f,e", t1, g["ovvv"], 2.0)
    add(Fae, "a,e <- m,f ; m,a,e,f", t1, g["ovvv"], -1.0)
    add(Fae, "a,e <- m,n,a,f ; m,n,e,f", taut, g["oovv"], -2.0)
    add(Fae, "a,e <- m,n,a,f ; m,n,f,e", taut, g["oovv"], 1.0)
    # [doc:fae-end]

    Fmi = new("Fmi", (no, no))
    la.axpby(1.0, g["Foo"], 0.0, Fmi)
    add(Fmi, "m,i <- n,e ; m,n,i,e", t1, g["ooov"], 2.0)
    add(Fmi, "m,i <- n,e ; m,n,e,i", t1, g["oovo"], -1.0)
    add(Fmi, "m,i <- i,n,e,f ; m,n,e,f", taut, g["oovv"], 2.0)
    add(Fmi, "m,i <- i,n,e,f ; m,n,f,e", taut, g["oovv"], -1.0)

    Fme = new("Fme", OV)
    assign(Fme, "m,e <- n,f ; m,n,e,f", t1, g["oovv"], 2.0)
    add(Fme, "m,e <- n,f ; m,n,f,e", t1, g["oovv"], -1.0)

    Wmnij = new("Wmnij", (no, no, no, no))
    la.axpby(1.0, g["oooo"], 0.0, Wmnij)
    add(Wmnij, "m,n,i,j <- j,e ; m,n,i,e", t1, g["ooov"])
    add(Wmnij, "m,n,i,j <- i,e ; m,n,e,j", t1, g["oovo"])
    add(Wmnij, "m,n,i,j <- i,j,e,f ; m,n,e,f", tau, g["oovv"])

    jnfb = new("jnfb", OOVV)
    assign(jnfb, "j,n,f,b <- j,f ; n,b", t1, t1)
    la.axpby(0.5, t2, 1.0, jnfb)

    Wmbej = new("Wmbej", (no, nv, nv, no))
    la.axpby(1.0, g["ovvo"], 0.0, Wmbej)
    add(Wmbej, "m,b,e,j <- j,f ; m,b,e,f", t1, g["ovvv"])
    add(Wmbej, "m,b,e,j <- n,b ; m,n,e,j", t1, g["oovo"], -1.0)
    add(Wmbej, "m,b,e,j <- j,n,f,b ; m,n,e,f", jnfb, g["oovv"], -1.0)
    add(Wmbej, "m,b,e,j <- n,j,f,b ; m,n,e,f", t2, g["oovv"])
    add(Wmbej, "m,b,e,j <- n,j,f,b ; m,n,f,e", t2, g["oovv"], -0.5)

    Wmbje = new("Wmbje", (no, nv, no, nv))
    la.axpby(-1.0, g["ovov"], 0.0, Wmbje)
    add(Wmbje, "m,b,j,e <- j,f ; m,b,f,e", t1, g["ovvv"], -1.0)
    add(Wmbje, "m,b,j,e <- n,b ; m,n,j,e", t1, g["ooov"])
    add(Wmbje, "m,b,j,e <- j,n,f,b ; m,n,f,e", jnfb, g["oovv"])

    Zmbij = new("Zmbij", (no, nv, no, no))
    assign(Zmbij, "m,b,i,j <- m,b,e,f ; i,j,e,f", g["ovvv"], tau)

    r1 = new("r1", OV)
    assign(r1, "i,a <- i,e ; a,e", t1, Fae)
    add(r1, "i,a <- m,a ; m,i", t1, Fmi, -1.0)
    add(r1, "i,a <- i,m,a,e ; m,e", t2, Fme, 2.0)
    add(r1, "i,a <- i,m,e,a ; m,e", t2, Fme, -1.0)
    add(r1, "i,a <- n,f ; n,a,f,i", t1, g["ovvo"], 2.0)
    add(r1, "i,a <- n,f ; n,a,i,f", t1, g["ovov"], -1.0)
    add(r1, "i,a <- m,i,e,f ; m,a,e,f", t2, g["ovvv"], 2.0)
    add(r1, "i,a <- m,i,f,e ; m,a,e,f", t2, g["ovvv"], -1.0)
    add(r1, "i,a <- m,n,a,e ; n,m,e,i", t2, g["oovo"], -2.0)
    add(r1, "i,a <- m,n,a,e ; n,m,i,e", t2, g["ooov"], 1.0)

    # Every term the closed-shell residual symmetrizes, collected in X, and then
    # r2 = <ij|ab> + tau.W + the ladder + X[i,j,a,b] + X[j,i,b,a].
    Fbe, Fmj = new("Fbe", (nv, nv)), new("Fmj", (no, no))
    la.axpby(1.0, Fae, 0.0, Fbe)
    add(Fbe, "b,e <- m,b ; m,e", t1, Fme, -0.5)
    la.axpby(1.0, Fmi, 0.0, Fmj)
    add(Fmj, "m,j <- j,e ; m,e", t1, Fme, 0.5)

    X = new("X", OOVV)
    assign(X, "i,j,a,b <- i,j,a,e ; b,e", t2, Fbe)
    add(X, "i,j,a,b <- i,m,a,b ; m,j", t2, Fmj, -1.0)
    add(X, "i,j,a,b <- m,a ; m,b,i,j", t1, Zmbij, -1.0)
    add(X, "i,j,a,b <- i,m,a,e ; m,b,e,j", t2, Wmbej, 2.0)
    add(X, "i,j,a,b <- i,m,e,a ; m,b,e,j", t2, Wmbej, -1.0)
    add(X, "i,j,a,b <- i,m,a,e ; m,b,j,e", t2, Wmbje)
    add(X, "i,j,a,b <- m,j,a,e ; m,b,i,e", t2, Wmbje)
    add(X, "i,j,a,b <- i,e ; a,b,e,j", t1, g["vvvo"])
    add(X, "i,j,a,b <- m,a ; m,b,i,j", t1, g["ovoo"], -1.0)
    tt = new("t1t1", OOVV)
    assign(tt, "i,m,e,a <- i,e ; m,a", t1, t1)
    add(X, "i,j,a,b <- i,m,e,a ; m,b,e,j", tt, g["ovvo"], -1.0)
    assign(tt, "i,m,e,b <- i,e ; m,b", t1, t1)
    add(X, "i,j,a,b <- i,m,e,b ; m,a,j,e", tt, g["ovov"], -1.0)

    # [doc:r2]
    r2 = new("r2", OOVV)
    la.axpby(1.0, g["oovv"], 0.0, r2)
    add(r2, "i,j,a,b <- m,n,a,b ; m,n,i,j", tau, Wmnij)
    # The particle ladder, sum_ef tau[i,j,e,f] <ab|ef>, with <ab|ef> = sum_Q B[Q,a,e] B[Q,b,f].
    einsums.einsum("i,j,a,b <- i,j,e,f ; Q,a,e ; Q,b,f", r2, tau, g["Bvv"], g["Bvv"], c_pf=1.0)
    la.axpby(1.0, X, 1.0, r2)
    einsums.permute("j,i,b,a <- i,j,a,b", r2, X, c_pf=1.0)
    # [doc:r2-end]

    # [doc:update]
    # The diagonal Fock terms in Fae and Fmi put -t D in the residual, so t + r / D is the update.
    d1, d2 = new("d1", OV), new("d2", OOVV)
    la.direct_division(1.0, r1, g["Dia"], 0.0, d1)
    la.direct_division(1.0, r2, g["Dijab"], 0.0, d2)
    la.axpby(1.0, d1, 1.0, t1)
    la.axpby(1.0, d2, 1.0, t2)
    # [doc:update-end]
# [doc:iterate-end]


# [doc:converge]
def converge(step, current_energy, threshold=1e-11):
    """Call step() until current_energy() changes by less than threshold.

    Returns the number of iterations and the milliseconds each took.
    """
    start, previous = time.perf_counter(), current_energy()
    for iteration in range(1, 101):
        step()
        if abs(current_energy() - previous) < threshold:
            break
        previous = current_energy()
    return iteration, 1000.0 * (time.perf_counter() - start) / iteration
# [doc:converge-end]
