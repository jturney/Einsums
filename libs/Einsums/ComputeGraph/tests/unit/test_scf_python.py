# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""Unit test: the captured-graph SCF example must reproduce reference HF energies.

This promotes ``examples/scf_simulation.py`` from a runnable demo to a real
correctness test. It's a whole-pipeline guard: a graph-optimization pass that
silently corrupts a captured loop body (e.g. CSE merging two distinct
mutable-output ops, or Reorder violating a loop-carried anti-dependency) makes
the SCF diverge, which small synthetic per-pass tests can miss but these
physical reference energies catch immediately.


"""

from __future__ import annotations

import os
import sys

import pytest

# The example modules live in ../../examples relative to this test file.
_EXAMPLES_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "examples"))
if _EXAMPLES_DIR not in sys.path:
    sys.path.insert(0, _EXAMPLES_DIR)

import scf_simulation as scf  # noqa: E402


# (system factory, expected total HF energy). Reference values: H2 from
# Szabo & Ostlund; HeH+ from the captured-graph reference run.
SCF_CASES = [
    pytest.param(scf._h2_sto3g, -1.1167529403, id="H2/STO-3G"),
    pytest.param(scf._heh_plus_sto3g, -2.8605882865, id="HeH+/STO-3G"),
]


@pytest.mark.parametrize("system_factory,expected", SCF_CASES)
def test_scf_total_energy(system_factory, expected):
    e_total = scf.run_scf(system_factory())
    assert e_total == pytest.approx(expected, abs=1e-6)
