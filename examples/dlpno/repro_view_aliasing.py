#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------
"""Standalone reproducer: writes through a view are not ordered against the parent.

A captured op that writes ``R[:, :, p]`` and one that reads or writes ``R`` are
treated as touching unrelated tensors, so the scheduler may run them in either
order. Both cases below give wrong answers with no error raised.

Needs only einsums (no psi4)::

    PYTHONPATH=/path/to/Einsums/build/lib python examples/dlpno/repro_view_aliasing.py

Found while building the DLPNO port, where per-pair blocks live in one
contiguous store and are updated through views. See the "Gaps found in Einsums"
section of examples/dlpno/README.md.
"""

import numpy as np

import einsums
import einsums.graph as cg
from einsums import linalg as la


def tensor(name, shape, fill=None):
    t = einsums.create_zero_tensor(name, list(shape), dtype="float64")
    if fill is not None:
        np.asarray(t)[...] = fill
    return t


def case_parallel_executor(P=100, m=8):
    """Write every block through a view, then read the parent."""
    rng = np.random.default_rng(0)
    print(f"\ncase 1: {P} view writes then one parent read ({m}x{m} blocks)")
    for label, executor in [("default", None),
                            ("Sequential", cg.SequentialExecutor),
                            ("OpenMP", cg.OpenMPExecutor),
                            ("Dataflow", cg.DataflowExecutor)]:
        R = tensor("R", (m, m, P))
        A = tensor("A", (m, m), rng.random((m, m)))
        B = tensor("B", (m, m), rng.random((m, m)))
        W = tensor("W", (m, m, P), np.ones((m, m, P)))
        out = tensor("out", [1])
        views = [R[:, :, p] for p in range(P)]  # must outlive the graph

        g = cg.Graph("view writes, parent read")
        with cg.capture(g):
            for p in range(P):
                einsums.einsum("ab <- ac ; cb", views[p], A, B)
            la.dot(out, R, W)
        if executor is not None:
            g.set_executor(executor())
        g.execute()

        got = float(np.asarray(out)[0])
        want = float(np.sum(np.asarray(R) * np.asarray(W)))
        ok = abs(got - want) <= 1e-10 * max(1.0, abs(want))
        print(f"    {label:11} dot(parent) = {got:14.6f}   true = {want:14.6f}   "
              f"{'ok' if ok else 'WRONG'}")


def case_scheduling(P=100, m=8, couplings=8):
    """The shape the DLPNO residual has: parent write, many view RMWs, parent read.

    Reproduces without any parallel executor: the parent read is *scheduled*
    among the view updates, so it sums a partially built result.
    """
    rng = np.random.default_rng(1)
    print(f"\ncase 2: parent write, {P * couplings} view accumulations, parent read")
    K = tensor("K", (m, m, P), rng.random((m, m, P)))
    D = tensor("D", (m, m, P), np.ones((m, m, P)))
    T = tensor("T", (m, m, P), rng.random((m, m, P)))
    R = tensor("R", (m, m, P))
    W = tensor("W", (m, m, P), np.ones((m, m, P)))
    out = tensor("out", [1])
    A = tensor("A", (m, m), rng.random((m, m)))
    S = [tensor(f"S{c}", (m, m), rng.random((m, m))) for c in range(couplings)]
    tmp = [[tensor(f"t{p}_{c}", (m, m)) for c in range(couplings)] for p in range(P)]
    views = [R[:, :, p] for p in range(P)]

    g = cg.Graph("parent write, view RMW, parent read")
    with cg.capture(g):
        la.direct_product(1.0, D, T, 0.0, R)          # parent write
        la.axpby(1.0, K, 1.0, R)                      # parent write
        for p in range(P):
            for c in range(couplings):
                einsums.einsum("ad <- ac ; cd", tmp[p][c], S[c], A)
                einsums.einsum("ab <- ad ; bd", views[p], tmp[p][c], S[c],
                               c_pf=1.0, ab_pf=1.0)   # view read-modify-write
        la.dot(out, R, W)                             # parent read
    g.execute()

    got = float(np.asarray(out)[0])
    want = float(np.sum(np.asarray(R) * np.asarray(W)))
    ok = abs(got - want) <= 1e-10 * max(1.0, abs(want))
    print(f"    default     dot(parent) = {got:14.6f}   true = {want:14.6f}   "
          f"{'ok' if ok else 'WRONG'}")

    import json
    kinds = [n["kind"] for n in json.loads(g.to_json())["nodes"]]
    where = [i for i, k in enumerate(kinds) if k == "Dot"]
    print(f"    Dot scheduled at node {where} of {len(kinds)} "
          f"(captured last, so it should be at the end)")


if __name__ == "__main__":
    case_parallel_executor()
    case_scheduling()
