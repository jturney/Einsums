# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""CCSD tutorial, stage 7: declaring the symmetry the amplitudes have.

Stage 4 with one more declaration. The closed-shell amplitude is unchanged when the two electrons
trade places, t2[i,j,a,b] = t2[j,i,b,a]. Declared, it lets the optimizer prove the particle ladder
has the same symmetry and compute only the blocks on and above the diagonal, copying each one
across to the block it mirrors.
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

# [doc:symmetry]
t2.set_symmetry(einsums.SymmetryDescriptor([einsums.SymmetryOp.group_swap([0, 2], [1, 3])]))
# [doc:symmetry-end]

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
    if decision.code in ("axis-tiling.triangle", "axis-tiling.sliced"):
        print(decision.text)
# [doc:report-end]

iterations, ms = converge(graph.execute, lambda: float(e[0]))

print(f"CCSD correlation energy = {float(e[0]):.10f}   ({iterations} iterations, {ms:.2f} ms each)")
assert abs(float(e[0]) - REFERENCE_CCSD) < 1e-10
