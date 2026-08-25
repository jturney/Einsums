# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""Python mirror of Pass_SpacePropagation.cpp.

The C++ tests own the semantics: which rules fire, what an intermediate's slots
end up holding, and the single-writer soundness guard. This file asserts the
bound half - that the pass constructs, runs through a ``PassManager``, and
reports ``num_inferred`` - plus the round trip through ``tensor_spaces`` for the
graph-owned intermediates it annotated, which is the one place Python can see
the result rather than the count.
"""

from __future__ import annotations

import pytest

import einsums
import einsums.graph as cg


@pytest.fixture()
def reg():
    """A private registry holding occ < virt < aux.

    Private rather than global: every test in a run shares the process-wide
    registry, and a case that asserts on what is registered cannot be written
    against a table someone else also writes to.
    """
    registry = cg.SpaceRegistry()
    occ = registry.register_space(cg.index_space("occ", "o", 4.0))
    virt = registry.register_space(cg.index_space("virt", "v", 8.0))
    aux = registry.register_space(cg.index_space("aux", "x", 16.0))
    registry.declare_less(occ, virt)
    registry.declare_less(virt, aux)
    return registry


def _run(pass_obj, g):
    pm = cg.PassManager()
    pm.add(pass_obj)
    return pm.run(g)


def test_a_contraction_chain_resolves_in_one_sweep(reg):
    A = einsums.create_random_tensor("A", [4, 8])  # occ x virt
    B = einsums.create_random_tensor("B", [8, 16])  # virt x aux
    E = einsums.create_random_tensor("E", [16, 4])  # aux x occ

    g = cg.Graph("chain")
    g.set_space_registry(reg)
    C = g.create_zero_tensor("C", [4, 16], dtype="float64")
    D = g.create_zero_tensor("D", [4, 4], dtype="float64")
    with cg.capture(g):
        einsums.einsum("ix <- ia ; ax", C, A, B)
        einsums.einsum("ij <- ix ; xj", D, C, E)

    cg.annotate(A, ("occ", "virt"), graph=g)
    cg.annotate(B, ("virt", "aux"), graph=g)
    cg.annotate(E, ("aux", "occ"), graph=g)

    prop = cg.SpacePropagation()
    _run(prop, g)

    # Topological order makes one sweep a fixpoint: C is annotated before D is
    # examined.
    assert prop.num_inferred == 2
    assert g.tensor_spaces(C) == [reg.find("occ"), reg.find("aux")]
    assert g.tensor_spaces(D) == [reg.find("occ"), reg.find("occ")]

    # The inputs are untouched: propagation flows producer to output, never
    # back onto an operand.
    assert g.tensor_spaces(A) == [reg.find("occ"), reg.find("virt")]


def test_a_permute_reorders_the_spaces_with_the_axes(reg):
    A = einsums.create_random_tensor("A", [4, 8])  # occ x virt

    g = cg.Graph("permute")
    g.set_space_registry(reg)
    T = g.create_zero_tensor("T", [8, 4], dtype="float64")
    with cg.capture(g):
        einsums.permute("ji <- ij", T, A, c_pf=0.0, a_pf=1.0)

    cg.annotate(A, ("occ", "virt"), graph=g)

    prop = cg.SpacePropagation()
    _run(prop, g)

    assert prop.num_inferred == 1
    assert g.tensor_spaces(T) == [reg.find("virt"), reg.find("occ")]


def test_a_user_owned_output_is_never_annotated(reg):
    A = einsums.create_random_tensor("A", [4, 8])
    B = einsums.create_random_tensor("B", [8, 16])
    C = einsums.create_zero_tensor("C", [4, 16])  # user-owned, not graph-owned

    g = cg.Graph("user_owned")
    g.set_space_registry(reg)
    with cg.capture(g):
        einsums.einsum("ix <- ia ; ax", C, A, B)

    cg.annotate(A, ("occ", "virt"), graph=g)
    cg.annotate(B, ("virt", "aux"), graph=g)

    prop = cg.SpacePropagation()
    _run(prop, g)

    assert prop.num_inferred == 0
    assert g.tensor_spaces(C) == []


def test_an_unannotated_program_infers_nothing(reg):
    A = einsums.create_random_tensor("A", [4, 8])
    B = einsums.create_random_tensor("B", [8, 16])

    g = cg.Graph("unannotated")
    g.set_space_registry(reg)
    C = g.create_zero_tensor("C", [4, 16], dtype="float64")
    with cg.capture(g):
        einsums.einsum("ix <- ia ; ax", C, A, B)

    prop = cg.SpacePropagation()
    _run(prop, g)

    assert prop.num_inferred == 0
    assert g.tensor_spaces(C) == []


def test_a_second_run_infers_nothing_new(reg):
    A = einsums.create_random_tensor("A", [4, 8])
    B = einsums.create_random_tensor("B", [8, 16])

    g = cg.Graph("rerun")
    g.set_space_registry(reg)
    C = g.create_zero_tensor("C", [4, 16], dtype="float64")
    with cg.capture(g):
        einsums.einsum("ix <- ia ; ax", C, A, B)

    cg.annotate(A, ("occ", "virt"), graph=g)
    cg.annotate(B, ("virt", "aux"), graph=g)

    first = cg.SpacePropagation()
    _run(first, g)
    assert first.num_inferred == 1

    second = cg.SpacePropagation()
    _run(second, g)
    assert second.num_inferred == 0


def test_the_pass_reports_no_modification_and_the_graph_still_executes(reg):
    import numpy as np

    from einsums.testing import assert_close

    A = einsums.create_random_tensor("A", [4, 8])
    B = einsums.create_random_tensor("B", [8, 16])

    g = cg.Graph("executes")
    g.set_space_registry(reg)
    C = g.create_zero_tensor("C", [4, 16], dtype="float64")
    with cg.capture(g):
        einsums.einsum("ix <- ia ; ax", C, A, B)

    cg.annotate(A, ("occ", "virt"), graph=g)
    cg.annotate(B, ("virt", "aux"), graph=g)

    prop = cg.SpacePropagation()
    assert _run(prop, g) is False  # an analysis pass never claims a modification

    g.execute()
    assert_close(C, np.asarray(A) @ np.asarray(B))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
