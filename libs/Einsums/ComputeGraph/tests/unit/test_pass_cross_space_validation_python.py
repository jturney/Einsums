# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""Python mirror of Pass_CrossSpaceValidation.cpp.

The C++ tests own the verdict rules - disjoint is an error, containment is a
note, an undeclared relation is a warning, and anything resting on an inference
drops a level. This file asserts the bound half: that the findings list reaches
Python as value objects with their severity, letter, space names and message
intact, that the severity counts agree with it, and that ``report_string``
carries the same numbers.
"""

from __future__ import annotations

import pytest

import einsums
import einsums.graph as cg


@pytest.fixture()
def reg():
    """occ and virt share no element; a pno domain lives inside virt.

    Both are declarations: the registry infers nothing, so what it can prove is
    exactly what was written down.
    """
    registry = cg.SpaceRegistry()
    occ = registry.register_space(cg.index_space("occ", "o", 2.0))
    virt = registry.register_space(cg.index_space("virt", "v", 3.0))
    registry.register_space(cg.index_space("aux", "x", 4.0))
    pno = registry.register_space(cg.index_space("pno", "p", 2.0))
    registry.declare_disjoint(occ, virt)
    registry.declare_contained(pno, virt)
    return registry


def _run(pass_obj, g):
    pm = cg.PassManager()
    pm.add(pass_obj)
    return pm.run(g)


def test_a_consistently_annotated_graph_reports_nothing(reg):
    A = einsums.create_random_tensor("A", [2, 3])
    B = einsums.create_random_tensor("B", [3, 4])

    g = cg.Graph("clean")
    g.set_space_registry(reg)
    C = g.create_zero_tensor("C", [2, 4], dtype="float64")
    with cg.capture(g):
        einsums.einsum("ix <- ia ; ax", C, A, B)

    cg.annotate(A, ("occ", "virt"), graph=g)
    cg.annotate(B, ("virt", "aux"), graph=g)

    check = cg.CrossSpaceValidation()
    assert _run(check, g) is False  # a diagnostic pass never claims a modification

    assert list(check.findings) == []
    assert check.num_errors == 0
    assert check.num_warnings == 0
    assert check.num_notes == 0
    assert "no cross-space conflicts found" in check.report_string()


def test_a_letter_binding_two_disjoint_spaces_is_an_error(reg):
    A = einsums.create_random_tensor("A", [2, 3])
    B = einsums.create_random_tensor("B", [3, 4])

    g = cg.Graph("disjoint")
    g.set_space_registry(reg)
    C = g.create_zero_tensor("C", [2, 4], dtype="float64")
    with cg.capture(g):
        einsums.einsum("ix <- ia ; ax", C, A, B)

    # Annotated AFTER capture, which is what capture's own conflict check cannot
    # see: 'a' is a virtual slot on A and an occupied slot on B, and the two are
    # declared disjoint.
    cg.annotate(A, ("occ", "virt"), graph=g)
    cg.annotate(B, ("occ", "aux"), graph=g)

    check = cg.CrossSpaceValidation()
    _run(check, g)

    findings = list(check.findings)
    assert len(findings) == 1
    finding = findings[0]
    assert finding.severity == cg.CrossSpaceSeverity.Error
    assert finding.letter == "a"
    assert finding.first_space_name == "virt"
    assert finding.second_space_name == "occ"
    assert finding.first_tensor_name == "A"
    assert finding.second_tensor_name == "B"
    assert finding.first_operand == "A"
    assert finding.second_operand == "B"
    assert finding.rests_on_inferred is False
    assert "'a'" in finding.message
    assert "disjoint" in finding.message

    assert check.num_errors == 1
    assert check.num_warnings == 0
    assert check.num_notes == 0
    assert "1 error(s), 0 warning(s), 0 note(s)" in check.report_string()
    assert finding.message in check.report_string()


def test_a_contained_space_is_a_note_not_a_mistake(reg):
    A = einsums.create_random_tensor("A", [2, 3])
    B = einsums.create_random_tensor("B", [3, 4])

    g = cg.Graph("contained")
    g.set_space_registry(reg)
    C = g.create_zero_tensor("C", [2, 4], dtype="float64")
    with cg.capture(g):
        einsums.einsum("ix <- ia ; ax", C, A, B)

    # 'a' is a pno slot on A and a virt slot on B: a restriction of the parent
    # space to a subspace, which the design treats as legitimate.
    cg.annotate(A, ("occ", "pno"), graph=g)
    cg.annotate(B, ("virt", "aux"), graph=g)

    check = cg.CrossSpaceValidation()
    _run(check, g)

    findings = list(check.findings)
    assert len(findings) == 1
    assert findings[0].severity == cg.CrossSpaceSeverity.Note
    assert check.num_notes == 1
    assert check.num_errors == 0


def test_an_undeclared_relation_is_only_a_warning(reg):
    A = einsums.create_random_tensor("A", [2, 3])
    B = einsums.create_random_tensor("B", [3, 4])

    g = cg.Graph("unknown")
    g.set_space_registry(reg)
    C = g.create_zero_tensor("C", [2, 4], dtype="float64")
    with cg.capture(g):
        einsums.einsum("ix <- ia ; ax", C, A, B)

    # 'a' meets aux on A and virt on B, and nothing declared relates the two.
    cg.annotate(A, ("occ", "aux"), graph=g)
    cg.annotate(B, ("virt", "aux"), graph=g)

    check = cg.CrossSpaceValidation()
    _run(check, g)

    findings = list(check.findings)
    assert len(findings) == 1
    assert findings[0].severity == cg.CrossSpaceSeverity.Warning
    assert check.num_warnings == 1


def test_a_verdict_resting_on_an_inferred_annotation_is_downgraded(reg):
    A = einsums.create_random_tensor("A", [2, 3])  # occ x virt
    B = einsums.create_random_tensor("B", [3, 4])  # virt x aux
    E = einsums.create_random_tensor("E", [4, 5])  # declared occ x virt

    g = cg.Graph("inferred")
    g.set_space_registry(reg)
    C = g.create_zero_tensor("C", [2, 4], dtype="float64")
    D = g.create_zero_tensor("D", [2, 5], dtype="float64")
    with cg.capture(g):
        einsums.einsum("ix <- ia ; ax", C, A, B)
        einsums.einsum("iy <- ix ; xy", D, C, E)

    cg.annotate(A, ("occ", "virt"), graph=g)
    cg.annotate(B, ("virt", "aux"), graph=g)
    cg.annotate(E, ("occ", "virt"), graph=g)

    # SpacePropagation infers C as (occ, aux). The second contraction binds 'x'
    # to that inferred aux against E's declared occ, so the warning drops to a
    # note and says so.
    check = cg.CrossSpaceValidation()
    pm = cg.PassManager()
    pm.add(cg.SpacePropagation())
    pm.add(check)
    pm.run(g)

    findings = list(check.findings)
    assert len(findings) == 1
    assert findings[0].letter == "x"
    assert findings[0].rests_on_inferred is True
    assert findings[0].severity == cg.CrossSpaceSeverity.Note
    assert "INFERRED" in findings[0].message
    assert check.num_notes == 1
    assert check.num_warnings == 0


def test_an_unannotated_graph_finds_nothing(reg):
    A = einsums.create_random_tensor("A", [2, 3])
    B = einsums.create_random_tensor("B", [3, 2])

    g = cg.Graph("unannotated")
    g.set_space_registry(reg)
    C = g.create_zero_tensor("C", [2, 2], dtype="float64")
    with cg.capture(g):
        einsums.einsum("ij <- ia ; aj", C, A, B)

    check = cg.CrossSpaceValidation()
    _run(check, g)

    assert list(check.findings) == []
    assert check.num_errors == 0


def test_a_finding_outlives_the_pass_and_the_graph(reg):
    A = einsums.create_random_tensor("A", [2, 3])
    B = einsums.create_random_tensor("B", [3, 4])

    g = cg.Graph("outlives")
    g.set_space_registry(reg)
    C = g.create_zero_tensor("C", [2, 4], dtype="float64")
    with cg.capture(g):
        einsums.einsum("ix <- ia ; ax", C, A, B)

    cg.annotate(A, ("occ", "virt"), graph=g)
    cg.annotate(B, ("occ", "aux"), graph=g)

    check = cg.CrossSpaceValidation()
    _run(check, g)
    kept = list(check.findings)

    del check, g, C
    assert kept[0].message  # a finding is a value type, not a view of the pass


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
