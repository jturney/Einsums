# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""Unit test: the captured-graph MP2 example must reproduce reference energies.

Promotes ``examples/mp2_simulation.py`` to a real correctness test. Exercises
the captured-graph SCF (a loop) followed by the captured-graph MP2 correction
(a flat AO→MO transform + Δ build + contraction), asserting both the HF and
MP2 total energies against reference values, a whole-pipeline guard against
optimization passes corrupting either graph.


"""

from __future__ import annotations

import os
import sys

import pytest

_EXAMPLES_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "examples"))
if _EXAMPLES_DIR not in sys.path:
    sys.path.insert(0, _EXAMPLES_DIR)

import mp2_simulation as mp2  # noqa: E402


# (system factory, expected E_HF, expected E_MP2 total).
MP2_CASES = [
    pytest.param(mp2._h2_sto3g, -1.1167529403, -1.1299033127, id="H2/STO-3G"),
    pytest.param(mp2._heh_plus_sto3g, -2.8605882865, -2.8751351018, id="HeH+/STO-3G"),
]


@pytest.mark.parametrize("system_factory,expected_hf,expected_mp2", MP2_CASES)
def test_mp2_energies(system_factory, expected_hf, expected_mp2):
    system = system_factory()
    scf_result = mp2.run_scf(system)
    assert scf_result.E_total == pytest.approx(expected_hf, abs=1e-6)

    e_corr = mp2.run_mp2(system, scf_result)
    e_mp2 = scf_result.E_total + e_corr
    assert e_mp2 == pytest.approx(expected_mp2, abs=1e-6)
    # MP2 correlation energy is negative (stabilizing).
    assert e_corr < 0.0
