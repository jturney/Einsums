# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""One-to-one Python mirror of Pass_SymmetryPropagation.cpp.

The C++ tests assert ``C.has_symmetry()`` and inspect ``C.symmetry()->ops[0]``
to verify the descriptor written onto the graph-owned intermediate. Those
symmetry-introspection APIs aren't bound to Python yet, so the mirror
verifies only the side that *is* bound: ``pass.num_inferred``.
"""

from __future__ import annotations

import numpy as np
import pytest

import einsums
import einsums.graph as cg
from einsums.testing import assert_close


def _run(pass_obj, g):
    pm = cg.PassManager()
    pm.add(pass_obj)
    return pm.run(g)


def test_symprop_ata_produces_symmetric():
    A = einsums.create_random_tensor("A", [5, 5])

    g = cg.Graph("ata")
    C = g.create_zero_tensor("C", [5, 5], dtype="float64")
    with cg.capture(g):
        einsums.einsum("ij <- ki ; kj", C, A, A)

    pass_inst = cg.SymmetryPropagation()
    _run(pass_inst, g)
    assert pass_inst.num_inferred == 1


def test_symprop_aat_also_symmetric():
    A = einsums.create_random_tensor("A", [4, 7])

    g = cg.Graph("aat")
    C = g.create_zero_tensor("C", [4, 4], dtype="float64")
    with cg.capture(g):
        einsums.einsum("ij <- ik ; jk", C, A, A)

    pass_inst = cg.SymmetryPropagation()
    _run(pass_inst, g)
    assert pass_inst.num_inferred == 1


def test_symprop_ab_different_tensors_does_not_infer():
    A = einsums.create_random_tensor("A", [4, 4])
    B = einsums.create_random_tensor("B", [4, 4])

    g = cg.Graph("ab")
    C = g.create_zero_tensor("C", [4, 4], dtype="float64")
    with cg.capture(g):
        einsums.einsum("ij <- ik ; kj", C, A, B)

    pass_inst = cg.SymmetryPropagation()
    _run(pass_inst, g)
    assert pass_inst.num_inferred == 0


def test_symprop_does_not_mutate_user_owned_tensors():
    A = einsums.create_random_tensor("A", [4, 4])
    C = einsums.create_zero_tensor("C", [4, 4])

    g = cg.Graph("user_owned")
    with cg.capture(g):
        einsums.einsum("ij <- ki ; kj", C, A, A)

    pass_inst = cg.SymmetryPropagation()
    _run(pass_inst, g)
    assert pass_inst.num_inferred == 0


def test_symprop_inferred_tag_survives_re_run():
    """Running the pass twice does not double-count."""
    A = einsums.create_random_tensor("A", [5, 5])

    g = cg.Graph("rerun")
    C = g.create_zero_tensor("C", [5, 5], dtype="float64")
    with cg.capture(g):
        einsums.einsum("ij <- ki ; kj", C, A, A)

    pass1 = cg.SymmetryPropagation()
    _run(pass1, g)
    assert pass1.num_inferred == 1

    pass2 = cg.SymmetryPropagation()
    _run(pass2, g)
    assert pass2.num_inferred == 0


@pytest.mark.skip(
    reason="SymmetryDescriptor / Tensor.set_symmetry / Tensor.symmetry are not bound to Python."
)
def test_symprop_permute_of_symmetric_stays_symmetric():
    """C++ test calls A.set_symmetry(...) and symmetrize(A) — neither is exposed."""


def test_symprop_permute_of_general_does_not_infer():
    A = einsums.create_random_tensor("A", [4, 4])  # no descriptor → nothing to propagate

    g = cg.Graph("permute_general")
    T = g.create_zero_tensor("T", [4, 4], dtype="float64")
    with cg.capture(g):
        einsums.permute("ji <- ij", T, A, c_pf=0.0, a_pf=1.0)

    pass_inst = cg.SymmetryPropagation()
    _run(pass_inst, g)
    assert pass_inst.num_inferred == 0


def test_symprop_inferred_C_executes_correctly_via_gemm_dispatch():
    """End-to-end: SymmetryPropagation tags C symmetric; execute still produces correct result."""
    A = einsums.create_random_tensor("A", [6, 6])
    B = einsums.create_random_tensor("B", [6, 6])
    D = einsums.create_zero_tensor("D", [6, 6])

    g = cg.Graph("e2e")
    C = g.create_zero_tensor("C", [6, 6], dtype="float64")
    with cg.capture(g):
        einsums.einsum("ij <- ki ; kj", C, A, A)  # C = AᵀA → symmetric
        einsums.einsum("ik <- ij ; jk", D, C, B)

    pass_inst = cg.SymmetryPropagation()
    _run(pass_inst, g)
    assert pass_inst.num_inferred == 1

    g.execute()

    C_ref = np.asarray(A).T @ np.asarray(A)
    D_ref = C_ref @ np.asarray(B)
    assert_close(C, C_ref)
    assert_close(D, D_ref)
