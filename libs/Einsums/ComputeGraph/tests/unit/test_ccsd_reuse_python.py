# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""A CCSD iteration, captured once and replayed on a DIFFERENT problem.

This is the proving ground for cross-problem reuse, at the scale the feature exists
for. A three-node chain shows the mechanism; a CCSD iteration shows it holding up
across 62 nodes, 17 interface tensors and 19 graph-owned deferred intermediates whose
extents all have to be re-derived from the operands.

The assertion is bitwise equality, not closeness: a graph captured at one size and
bound to another runs the same kernels over the same values in the same order as a
fresh capture at that size, so anything short of identical is a defect rather than
arithmetic. That makes this a correctness gate rather than a demonstration, and it
fails loudly if a later pass starts pinning an intermediate's extent numerically.

The equations are the Stanton-Gauss-Watts-Bartlett spin-orbital CCSD residual,
transcribed from examples/tiled/ccsd_spinorbital_tiled_toy.py, which validates them
against a numpy oracle. Here the integrals are synthetic: this file is about whether
the SHAPE of the computation survives a round trip, and the physics is checked there.

The body is flat rather than a captured loop, deliberately. A loop's predicate is a
host closure and the IR cannot carry one, so the saveable unit is the iteration body,
which is what a caller would replay under their own convergence control anyway.
"""

from __future__ import annotations


import os
import tempfile

import numpy as np

import einsums
import einsums.graph as cg
from einsums import linalg as la

TAG = "ccsd"
OCC_SYM, VIR_SYM = f"{TAG}_no", f"{TAG}_nv"

# Every integral block a CCSD iteration reads, by its occ/virt axis pattern.
BLOCKS = {
    "oovv": "oovv", "vovv": "vovv", "ooov": "ooov", "oooo": "oooo",
    "vvvv": "vvvv", "ovvo": "ovvo", "ovvv": "ovvv", "oovo": "oovo",
    "ovov": "ovov", "vvvo": "vvvo", "ovoo": "ovoo",
}

# Graph-owned scratch, by axis pattern. Every one of these is DEFERRED and has to
# re-derive its extents from the operands when a bind moves the problem.
SCRATCH = {
    "tau": "oovv", "taut": "oovv", "ot": "oovv", "Fae": "vv", "Fmi": "oo",
    "Fme": "ov", "Wmnij": "oooo", "Wabef": "vvvv", "Wmbej": "ovvo",
    "jnfb": "oovv", "wt4o": "oooo", "wt4v": "vvvv", "be": "vv", "mj": "oo",
    "imea": "oovv", "r1": "ov", "r2": "oovv", "tmp": "oovv", "Xe": "ov",
}


def register_spaces():
    reg = cg.global_space_registry()
    occ = reg.register_space(cg.index_space(f"{TAG}_occ", "o", 10.0, cg.GrowthClass.linear(), OCC_SYM))
    vir = reg.register_space(cg.index_space(f"{TAG}_virt", "v", 16.0, cg.GrowthClass.linear(), VIR_SYM))
    return occ, vir


def dims_of(pattern, no, nv):
    return [no if c == "o" else nv for c in pattern]


def spaces_of(pattern, occ, vir):
    return [occ if c == "o" else vir for c in pattern]


def symbols_of(pattern):
    return [OCC_SYM if c == "o" else VIR_SYM for c in pattern]


def make_inputs(no, nv, seed):
    """Synthetic integrals and amplitudes: deterministic, right shapes, no psi4."""
    rng = np.random.default_rng(seed)
    data = {}
    for name, pattern in BLOCKS.items():
        shape = dims_of(pattern, no, nv)
        t = einsums.create_zero_tensor(name, shape)
        np.asarray(t)[...] = 0.02 * rng.standard_normal(shape)
        data[name] = t
    for name, pattern in (("t1", "ov"), ("t2", "oovv"), ("Dia", "ov"), ("Dijab", "oovv")):
        shape = dims_of(pattern, no, nv)
        t = einsums.create_zero_tensor(name, shape)
        arr = rng.standard_normal(shape)
        if name.startswith("D"):
            arr = -(4.0 + np.abs(arr))  # denominators, safely away from zero
        else:
            arr = 0.05 * arr
        np.asarray(t)[...] = arr
        data[name] = t
    for name, pattern in (("s1", "ov"), ("s2", "oovv")):
        data[name] = einsums.create_zero_tensor(name, dims_of(pattern, no, nv))
    data["Ecorr"] = einsums.zeros((1,), dtype="float64")
    data["e_part"] = einsums.zeros((1,), dtype="float64")
    return data


def build(no, nv, occ, vir, data, label):
    """One captured CCSD iteration: the SGWB equations, no loop.

    Flat rather than a captured loop on purpose. A loop's predicate is a host closure,
    which the IR cannot carry, so the saveable unit here is the iteration body.
    """
    g = cg.Graph(f"ccsd_{label}")

    # Interface: annotate every caller-owned operand with its spaces AND its dim symbols.
    # The symbols are what let a bind move the problem; the spaces are what the cost model
    # and the cross-space checker read.
    for name, pattern in list(BLOCKS.items()) + [("t1", "ov"), ("t2", "oovv"),
                                                 ("Dia", "ov"), ("Dijab", "oovv"),
                                                 ("s1", "ov"), ("s2", "oovv")]:
        g.annotate_spaces(data[name], spaces_of(pattern, occ, vir))
        g.annotate_dims(data[name], symbols_of(pattern))

    # Scratch: one call each, shaped in what the axes MEAN.
    S = {
        name: g.declare_zero_tensor_over(name, [cg.SpaceDim(s) for s in spaces_of(pattern, occ, vir)], True)
        for name, pattern in SCRATCH.items()
    }
    G, t1, t2 = data, data["t1"], data["t2"]

    def ein(spec, out, A, B, pf=1.0, acc=False):
        einsums.einsum(spec, out, A, B, c_pf=(1.0 if acc else 0.0), ab_pf=pf)

    with cg.capture(g):
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

        ein("m,n,i,j <- j,e ; m,n,i,e", S["wt4o"], t1, G["ooov"])
        la.axpby(1.0, G["oooo"], 0.0, S["Wmnij"])
        la.axpby(1.0, S["wt4o"], 1.0, S["Wmnij"])
        einsums.permute("m,n,j,i <- m,n,i,j", S["Wmnij"], S["wt4o"], c_pf=1.0, a_pf=-1.0)
        ein("m,n,i,j <- i,j,e,f ; m,n,e,f", S["Wmnij"], S["tau"], G["oovv"], 0.25, True)

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

        ein("i,a <- i,e ; a,e", S["r1"], t1, S["Fae"])
        ein("i,a <- m,a ; m,i", S["r1"], t1, S["Fmi"], -1.0, True)
        ein("i,a <- i,m,a,e ; m,e", S["r1"], t2, S["Fme"], 1.0, True)
        ein("i,a <- n,f ; n,a,i,f", S["r1"], t1, G["ovov"], -1.0, True)
        ein("i,a <- i,m,e,f ; m,a,e,f", S["r1"], t2, G["ovvv"], -0.5, True)
        ein("i,a <- m,n,a,e ; n,m,e,i", S["r1"], t2, G["oovo"], -0.5, True)

        la.axpby(1.0, G["oovv"], 0.0, S["r2"])
        ein("b,e <- m,b ; m,e", S["be"], t1, S["Fme"])
        la.axpby(-0.5, S["be"], 1.0, S["Fae"])
        ein("i,j,a,b <- i,j,a,e ; b,e", S["tmp"], t2, S["Fae"])
        la.axpby(1.0, S["tmp"], 1.0, S["r2"])
        einsums.permute("i,j,b,a <- i,j,a,b", S["r2"], S["tmp"], c_pf=1.0, a_pf=-1.0)
        ein("m,j <- j,e ; m,e", S["mj"], t1, S["Fme"])
        la.axpby(0.5, S["mj"], 1.0, S["Fmi"])
        ein("i,j,a,b <- i,m,a,b ; m,j", S["tmp"], t2, S["Fmi"])
        la.axpby(-1.0, S["tmp"], 1.0, S["r2"])
        einsums.permute("j,i,a,b <- i,j,a,b", S["r2"], S["tmp"], c_pf=1.0, a_pf=1.0)
        ein("i,j,a,b <- m,n,a,b ; m,n,i,j", S["r2"], S["tau"], S["Wmnij"], 0.5, True)
        ein("i,j,a,b <- i,j,e,f ; a,b,e,f", S["r2"], S["tau"], S["Wabef"], 0.5, True)

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

        la.direct_division(1.0, S["r1"], G["Dia"], 0.0, data["s1"])
        la.direct_division(1.0, S["r2"], G["Dijab"], 0.0, data["s2"])

    return g


def interface(data):
    names = list(BLOCKS) + ["t1", "t2", "Dia", "Dijab", "s1", "s2"]
    return {n: data[n] for n in names}


def test_a_ccsd_iteration_captured_once_replays_on_another_problem(tmp_path):
    occ, vir = register_spaces()

    # Capture at one geometry.
    small = make_inputs(4, 6, seed=2026)
    captured = build(4, 6, occ, vir, small, "small")
    assert captured.num_nodes() > 50, "the body should be a real CCSD iteration, not a fragment"

    path = str(tmp_path / "ccsd.eig")
    cg.save_graph(captured, path)

    # Load it with no addresses in it, and bind a DIFFERENT problem by manifest name.
    big = make_inputs(6, 8, seed=99)
    loaded = cg.load_graph(path)
    assert loaded.num_nodes() == captured.num_nodes()
    cg.bind(loaded, interface(big))
    loaded.optimize()
    loaded.execute()

    # The reference: capture the same iteration at the new size, same inputs.
    ref = make_inputs(6, 8, seed=99)
    fresh = build(6, 8, occ, vir, ref, "fresh")
    fresh.optimize()
    fresh.execute()

    for name in ("s1", "s2"):
        assert np.array_equal(np.asarray(big[name]), np.asarray(ref[name])), (
            f"{name} differs between the replayed graph and a fresh capture at the same size"
        )


def test_the_saved_form_carries_the_scratch_as_deferred(tmp_path):
    """The bit that makes the above possible, asserted on its own.

    A graph-owned intermediate has to come back DEFERRED. Loaded materialized it would
    still execute, and would silently refuse the extent-changing bind, which is how this
    went unnoticed: nothing about a materialized intermediate looks wrong until a bind
    tries to move it.
    """
    occ, vir = register_spaces()
    data = make_inputs(4, 6, seed=5)
    graph = build(4, 6, occ, vir, data, "alloc")

    path = str(tmp_path / "ccsd_alloc.eig")
    cg.save_graph(graph, path)

    import json

    saved = json.load(open(path))
    scratch = [t for t in saved["tensors"] if t["name"] in SCRATCH]
    assert scratch, "the scratch tensors should be in the saved tensors section"
    assert all(t["alloc"] == "deferred" for t in scratch), (
        "every graph-owned intermediate must save as deferred"
    )
