# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""CCSD tutorial, stage 5: running in less memory than the program names.

Stage 4 with a memory budget. The particle ladder's v^4 integral is 63 MB here, and the budget
is 64 MB, so it cannot be held whole beside everything else: it is built and used a block at a
time, and the program is otherwise unchanged. At the molecules CCSD is run on, v^4 is hundreds of
gigabytes and this is the only way the program runs at all.
"""

from __future__ import annotations

import einsums.rc

# [doc:budget]
# The runtime reads the budget when it starts, so it is set before einsums is imported.
einsums.rc.max_memory = "64MB"
# [doc:budget-end]

import einsums  # noqa: E402
import einsums.graph as cg  # noqa: E402
from ccsd_equations import annotate, converge, energy, integrals, iterate, register_spaces  # noqa: E402
from ccsd_problem import REFERENCE_CCSD, load  # noqa: E402

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
for decision in graph.decisions:
    if decision.code == "axis-tiling.sliced":
        print(decision.text)
# [doc:report-end]

iterations, ms = converge(graph.execute, lambda: float(e[0]))

print(f"CCSD correlation energy = {float(e[0]):.10f}   ({iterations} iterations, {ms:.2f} ms each)")
assert abs(float(e[0]) - REFERENCE_CCSD) < 1e-10
