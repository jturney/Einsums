# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""Python mirror of SpaceRegistry.cpp and SpaceAnnotation.cpp.

The C++ tests own the semantics of the registry and of capture-time letter
binding. This file asserts the half that is BOUND: that a registry can be built
and queried from Python, that ``cg.annotate`` resolves space names and reaches
``tensor_spaces``, and that the capture-time conflict a wrong annotation causes
arrives in Python as an exception rather than as a wrong answer.
"""

from __future__ import annotations

import pytest

import einsums
import einsums.graph as cg


def _chemistry_registry():
    """A private registry holding occ < virt < aux, plus a pno inside virt.

    Private on purpose: the process-global registry is shared with every other
    test in the run, and a case that asserts on ``size`` or on a rejected
    declaration cannot be written against a table someone else also writes to.
    """
    reg = cg.SpaceRegistry()
    occ = reg.register_space(cg.index_space("occ", "o", 4.0))
    virt = reg.register_space(cg.index_space("virt", "v", 8.0))
    aux = reg.register_space(cg.index_space("aux", "x", 16.0))
    pno = reg.register_space(cg.index_space("pno", "p", 2.0))
    reg.declare_less(occ, virt)
    reg.declare_less(virt, aux)
    reg.declare_contained(pno, virt)
    reg.declare_disjoint(occ, virt)
    return reg


# ── The registry ────────────────────────────────────────────────────────────


def test_register_and_look_up_by_name():
    reg = cg.SpaceRegistry()
    occ = reg.register_space(cg.index_space("occ", "o", 4.0))

    assert reg.size == 1
    assert reg.find("occ") == occ
    assert reg.find("nope") is None

    space = reg.space(occ)
    assert space.name == "occ"
    assert space.scale_symbol == "o"
    assert space.typical_extent == 4.0
    assert space.growth.exponent == cg.GrowthClass.linear().exponent


def test_register_is_idempotent_for_an_identical_declaration():
    reg = cg.SpaceRegistry()
    first = reg.register_space(cg.index_space("occ", "o", 4.0))
    again = reg.register_space(cg.index_space("occ", "o", 4.0))

    assert first == again
    assert reg.size == 1


def test_register_rejects_a_conflicting_redeclaration():
    reg = cg.SpaceRegistry()
    reg.register_space(cg.index_space("occ", "o", 4.0))

    with pytest.raises(ValueError, match="different content"):
        reg.register_space(cg.index_space("occ", "q", 4.0))


def test_growth_classes():
    reg = cg.SpaceRegistry()
    grid = reg.register_space(
        cg.index_space("grid", "g", 0.0, cg.GrowthClass.power(3.0))
    )
    frozen = reg.register_space(
        cg.index_space("frozen", "f", 0.0, cg.GrowthClass.constant())
    )

    assert reg.space(grid).growth.exponent == 3.0
    assert reg.space(frozen).growth.exponent == 0.0


def test_scale_order_is_transitive_and_answers_three_ways():
    reg = _chemistry_registry()
    occ, virt, aux = reg.find("occ"), reg.find("virt"), reg.find("aux")
    pno = reg.find("pno")

    assert reg.is_less(occ, virt) == cg.Tristate.Yes
    assert reg.is_less(occ, aux) == cg.Tristate.Yes  # through virt
    assert reg.is_less(aux, occ) == cg.Tristate.No
    assert reg.is_less(occ, occ) == cg.Tristate.No
    assert reg.is_less(pno, aux) == cg.Tristate.Unknown  # nothing declared


def test_containment_and_disjointness():
    reg = _chemistry_registry()
    occ, virt, aux = reg.find("occ"), reg.find("virt"), reg.find("aux")
    pno = reg.find("pno")

    assert reg.is_contained(pno, virt) == cg.Tristate.Yes
    assert reg.is_contained(pno, pno) == cg.Tristate.Yes  # reflexive
    assert reg.is_contained(virt, pno) == cg.Tristate.No
    assert reg.is_contained(pno, aux) == cg.Tristate.Unknown

    assert reg.is_disjoint(occ, virt) == cg.Tristate.Yes
    assert reg.is_disjoint(virt, occ) == cg.Tristate.Yes  # symmetric
    # pno is inside virt, and virt is disjoint from occ, so pno is too.
    assert reg.is_disjoint(pno, occ) == cg.Tristate.Yes
    assert reg.is_disjoint(occ, aux) == cg.Tristate.Unknown


def test_inconsistent_declarations_are_rejected():
    reg = cg.SpaceRegistry()
    occ = reg.register_space(cg.index_space("occ", "o"))
    virt = reg.register_space(cg.index_space("virt", "v"))

    reg.declare_less(occ, virt)
    with pytest.raises(ValueError, match="contradicts"):
        reg.declare_less(virt, occ)

    reg.declare_disjoint(occ, virt)
    with pytest.raises(ValueError, match="disjoint"):
        reg.declare_contained(occ, virt)


def test_ids_enumerate_in_registration_order():
    reg = _chemistry_registry()
    names = [reg.space(i).name for i in reg.ids]
    assert names == ["occ", "virt", "aux", "pno"]


def test_the_global_registry_is_one_object():
    first = cg.global_space_registry()
    second = cg.global_space_registry()

    # Registering through one is visible through the other, which is the whole
    # point of the accessor: one home per process, not one per binary.
    first.register_space(cg.index_space("test_spaces_python_probe", "z"))
    assert second.find("test_spaces_python_probe") is not None


# ── Annotation ──────────────────────────────────────────────────────────────


def test_annotate_by_name_round_trips_through_tensor_spaces():
    reg = _chemistry_registry()
    A = einsums.create_random_tensor("A", [4, 8])

    g = cg.Graph("annotate")
    g.set_space_registry(reg)
    cg.annotate(A, ("occ", "virt"), graph=g)

    assert g.tensor_spaces(A) == [reg.find("occ"), reg.find("virt")]


def test_annotate_accepts_ids_as_well_as_names():
    reg = _chemistry_registry()
    A = einsums.create_random_tensor("A", [4, 8])

    g = cg.Graph("annotate_ids")
    g.set_space_registry(reg)
    cg.annotate(A, [reg.find("occ"), reg.find("virt")], graph=g)

    assert g.tensor_spaces(A) == [reg.find("occ"), reg.find("virt")]


def test_annotate_inside_capture_finds_the_graph_itself():
    reg = _chemistry_registry()
    A = einsums.create_random_tensor("A", [4, 8])
    B = einsums.create_random_tensor("B", [8, 4])

    g = cg.Graph("annotate_in_capture")
    g.set_space_registry(reg)
    C = g.create_zero_tensor("C", [4, 4], dtype="float64")
    with cg.capture(g):
        cg.annotate(A, ("occ", "virt"))
        cg.annotate(B, ("virt", "occ"))
        einsums.einsum("ij <- ia ; aj", C, A, B)

    assert g.tensor_spaces(A) == [reg.find("occ"), reg.find("virt")]


def test_annotate_outside_a_graph_is_an_error():
    A = einsums.create_random_tensor("A", [4, 8])
    with pytest.raises(RuntimeError, match="needs a graph"):
        cg.annotate(A, ("occ", "virt"))


def test_annotate_rejects_an_unregistered_name():
    reg = _chemistry_registry()
    A = einsums.create_random_tensor("A", [4, 8])

    g = cg.Graph("bad_name")
    g.set_space_registry(reg)
    with pytest.raises(KeyError, match="no index space named"):
        cg.annotate(A, ("occ", "nonesuch"), graph=g)


def test_annotate_rejects_a_wrong_rank():
    reg = _chemistry_registry()
    A = einsums.create_random_tensor("A", [4, 8])

    g = cg.Graph("bad_rank")
    g.set_space_registry(reg)
    with pytest.raises(ValueError):
        cg.annotate(A, ("occ",), graph=g)


def test_an_unannotated_tensor_reports_no_spaces():
    A = einsums.create_random_tensor("A", [4, 8])
    B = einsums.create_random_tensor("B", [8, 4])

    g = cg.Graph("unannotated")
    C = g.create_zero_tensor("C", [4, 4], dtype="float64")
    with cg.capture(g):
        einsums.einsum("ij <- ia ; aj", C, A, B)

    assert g.tensor_spaces(A) == []


def test_a_letter_over_two_spaces_is_a_capture_error():
    reg = _chemistry_registry()
    A = einsums.create_random_tensor("A", [4, 8])
    B = einsums.create_random_tensor("B", [8, 4])

    g = cg.Graph("conflict")
    g.set_space_registry(reg)
    C = g.create_zero_tensor("C", [4, 4], dtype="float64")

    # 'a' links A's second slot to B's first, and the two disagree about it.
    cg.annotate(A, ("occ", "occ"), graph=g)
    cg.annotate(B, ("virt", "occ"), graph=g)

    with pytest.raises(ValueError, match="'a'"):
        with cg.capture(g):
            einsums.einsum("ij <- ia ; aj", C, A, B)


def test_annotate_returns_the_tensor():
    reg = _chemistry_registry()
    g = cg.Graph("returns")
    g.set_space_registry(reg)
    A = einsums.create_random_tensor("A", [4, 8])

    assert cg.annotate(A, ("occ", "virt"), graph=g) is A


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
