# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""CCSD tutorial, stage 4: telling the optimizer what the program is.

The same capture as stage 3 with two declarations added. Every tensor iterate() makes is the
graph's scratch rather than a result, so the memory passes may free it, share its memory or hold
it a block at a time. And every input says what its axes range over, so the passes that weigh
one form of a contraction against another judge them for the family of molecules, not for water.
"""

from __future__ import annotations

import einsums
import einsums.graph as cg
from ccsd_equations import annotate, converge, energy, integrals, iterate, register_spaces
from ccsd_problem import REFERENCE_CCSD, load

p = load()
ints = integrals(p)

t1 = einsums.zeros([p.nocc, p.nvir], name="t1")
t2 = einsums.zeros([p.nocc, p.nocc, p.nvir, p.nvir], name="t2")
einsums.linalg.direct_division(1.0, ints["oovv"], ints["Dijab"], 0.0, t2)
e = einsums.zeros([1], name="e")

# [doc:capture]
register_spaces()
graph = cg.Graph("ccsd")


def new(name, shape):
    return graph.scratch(name, list(shape), "float64")


with cg.capture(graph):
    annotate(t1, t2, ints)
    iterate(t1, t2, ints, new)
    energy(e, t1, t2, ints, new)
graph.optimize()
# [doc:capture-end]

# [doc:report]
print(graph.explain)
# [doc:report-end]

iterations, ms = converge(graph.execute, lambda: float(e[0]))

print(f"CCSD correlation energy = {float(e[0]):.10f}   ({iterations} iterations, {ms:.2f} ms each)")
assert abs(float(e[0]) - REFERENCE_CCSD) < 1e-10
