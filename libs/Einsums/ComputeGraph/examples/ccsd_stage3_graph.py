# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""CCSD tutorial, stage 3: one iteration captured once and replayed.

The same iterate() and energy() as stage 2, recorded into a graph instead of run. The loop stays
in Python and replays the graph; t1 and t2 are the caller's tensors, which the graph reads and
updates in place on every replay.
"""

from __future__ import annotations

import einsums
import einsums.graph as cg
from ccsd_equations import converge, energy, integrals, iterate
from ccsd_problem import REFERENCE_CCSD, load

p = load()
ints = integrals(p)

t1 = einsums.zeros([p.nocc, p.nvir], name="t1")
t2 = einsums.zeros([p.nocc, p.nocc, p.nvir, p.nvir], name="t2")
einsums.linalg.direct_division(1.0, ints["oovv"], ints["Dijab"], 0.0, t2)
e = einsums.zeros([1], name="e")


# [doc:capture]
def new(name, shape):
    return einsums.zeros(list(shape), name=name)


graph = cg.Graph("ccsd")
with cg.capture(graph):
    iterate(t1, t2, ints, new)
    energy(e, t1, t2, ints, new)
graph.optimize()

iterations, ms = converge(graph.execute, lambda: float(e[0]))
# [doc:capture-end]

print(f"CCSD correlation energy = {float(e[0]):.10f}   ({iterations} iterations, {ms:.2f} ms each)")
assert abs(float(e[0]) - REFERENCE_CCSD) < 1e-10
