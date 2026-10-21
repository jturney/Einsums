# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""CCSD tutorial, stage 2: the same equations with Einsums, run eagerly.

ccsd_equations.py is stage 1 transcribed. Each call runs as it is reached, which is the
reference semantics every captured version below has to reproduce.
"""

from __future__ import annotations

import einsums
from ccsd_equations import converge, energy, integrals, iterate
from ccsd_problem import REFERENCE_CCSD, load

p = load()
ints = integrals(p)


# [doc:run]
def new(name, shape):
    return einsums.zeros(list(shape), name=name)


t1 = new("t1", (p.nocc, p.nvir))
t2 = new("t2", (p.nocc, p.nocc, p.nvir, p.nvir))
einsums.linalg.direct_division(1.0, ints["oovv"], ints["Dijab"], 0.0, t2)  # the MP2 guess
e = new("e", (1,))


def step():
    iterate(t1, t2, ints, new)
    energy(e, t1, t2, ints, new)


iterations, ms = converge(step, lambda: float(e[0]))
# [doc:run-end]

print(f"CCSD correlation energy = {float(e[0]):.10f}   ({iterations} iterations, {ms:.2f} ms each)")
assert abs(float(e[0]) - REFERENCE_CCSD) < 1e-10
