# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------
"""The Python examples from the manual, run as written.

Its companion ``check_doc_claims.py`` settles whether a documented name exists. This file settles
whether documented *behaviour* is real, which is the half a name check cannot reach: the manual
claimed for some time that "All Einsums functions exposed to Python can also consume NumPy
arrays", and every one of them raises ``TypeError`` on a raw array. No spelling was wrong, so
nothing but running the code could have caught it.
"""

import numpy as np
import pytest

import einsums
import einsums.graph as cg
from einsums import linalg


@pytest.fixture
def ab():
    """A deterministic operand pair, so value assertions mean something."""
    a = einsums.create_random_tensor("A", [6, 6])
    b = einsums.create_random_tensor("B", [6, 6])
    return a, b


# ── howto/from_numpy.rst ────────────────────────────────────────────────────


def test_einsum_matches_numpy(ab):
    a, b = ab
    c = einsums.zeros([6, 6])
    einsums.einsum("ik;kj->ij", c, a, b)
    assert np.allclose(np.asarray(c), np.asarray(a) @ np.asarray(b))


def test_spec_separator_is_a_semicolon(ab):
    """NumPy's comma is rejected. The page says so; this keeps it true."""
    a, b = ab
    c = einsums.zeros([6, 6])
    with pytest.raises(ValueError):
        einsums.einsum("ik,kj->ij", c, a, b)


def test_arrow_left_spelling_is_equivalent(ab):
    a, b = ab
    lhs, rhs = einsums.zeros([6, 6]), einsums.zeros([6, 6])
    einsums.einsum("ik;kj->ij", lhs, a, b)
    einsums.einsum("ij <- ik ; kj", rhs, a, b)
    assert np.allclose(np.asarray(lhs), np.asarray(rhs))


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(lambda a, b, c: einsums.einsum("ik;kj->ij", c, a, b), id="einsum"),
        pytest.param(lambda a, b, c: einsums.permute("ij->ji", c, a), id="permute"),
        pytest.param(lambda a, b, c: linalg.dot(a, b), id="dot"),
        pytest.param(lambda a, b, c: einsums.sum_of_squares(a), id="sum_of_squares"),
    ],
)
def test_operations_reject_raw_numpy_arrays(call):
    """The correction to a claim that stood in the manual: they do not accept them.

    If this ever starts passing raw arrays, the porting page is the thing to update.
    """
    a, b, c = np.random.rand(4, 4), np.random.rand(4, 4), np.zeros((4, 4))
    with pytest.raises(TypeError):
        call(a, b, c)


def test_conversion_in_copies_and_conversion_out_aliases():
    """The asymmetry the porting page warns about, which is the one that causes real bugs."""
    source = np.random.rand(4, 4)
    tensor = einsums.asarray(source)

    source[0, 0] = 777.0
    assert np.asarray(tensor)[0, 0] != 777.0, "einsums.asarray should copy"

    view = np.asarray(tensor)
    view[1, 1] = -5.0
    assert np.asarray(tensor)[1, 1] == -5.0, "np.asarray should alias"


def test_tensors_are_column_major():
    a = np.ascontiguousarray(np.random.rand(3, 4))
    back = np.asarray(einsums.asarray(a))
    assert back.flags["F_CONTIGUOUS"]
    assert np.allclose(back, a)


@pytest.mark.parametrize(
    "fn,ref",
    [
        (lambda: einsums.zeros([3, 3]), np.zeros((3, 3))),
        (lambda: einsums.ones([3, 3]), np.ones((3, 3))),
        (lambda: einsums.eye(3), np.eye(3)),
    ],
)
def test_constructors_match_numpy(fn, ref):
    assert np.allclose(np.asarray(fn()), ref)


def test_operators_match_numpy(ab):
    a, b = ab
    an, bn = np.asarray(a), np.asarray(b)
    assert np.allclose(np.asarray(a + b), an + bn)
    assert np.allclose(np.asarray(a @ b), an @ bn)


def test_linalg_needs_a_from_import():
    """``import einsums.linalg`` fails because linalg is extension-only, with no module file."""
    with pytest.raises(ModuleNotFoundError):
        __import__("einsums.linalg")
    from einsums import linalg as _ok  # noqa: F401


# ── howto/views.rst ─────────────────────────────────────────────────────────


def test_slicing_returns_a_view_that_writes_through():
    a = einsums.create_random_tensor("A", [6, 6])
    block = a[0:3, 0:3]
    block[0, 0] = 999.0
    assert a[0, 0] == 999.0


def test_orbital_blocks_have_the_documented_shapes():
    nocc, nmo = 2, 5
    f = einsums.zeros([nmo, nmo])
    assert np.asarray(f[:nocc, :nocc]).shape == (nocc, nocc)
    assert np.asarray(f[:nocc, nocc:]).shape == (nocc, nmo - nocc)
    assert np.asarray(f[nocc:, nocc:]).shape == (nmo - nocc, nmo - nocc)


# ── howto/graphs.rst ────────────────────────────────────────────────────────


def test_declared_result_survives_optimize(ab):
    a, b = ab
    reference = np.asarray(a) @ np.asarray(b)

    g = cg.Graph("demo")
    c = g.create_tensor("C", [6, 6], intermediate=False)
    with cg.capture(g):
        einsums.einsum("ik;kj->ij", c, a, b)
    g.optimize()
    g.execute()

    assert np.allclose(np.asarray(c), reference)


def test_scratch_result_is_pruned_and_reported(ab):
    """The trap the page documents, pinned in both directions."""
    a, b = ab
    g = cg.Graph("scratch_result")
    c = g.create_tensor("C", [6, 6])  # scratch by default
    with cg.capture(g):
        einsums.einsum("ik;kj->ij", c, a, b)
    g.optimize()
    g.execute()

    assert np.allclose(np.asarray(c), 0.0)
    assert "'C'" in g.explain
    assert "intermediate=false" in g.explain


def test_graph_explain_is_a_property_and_pass_manager_explain_is_a_method(ab):
    """Documented because the inconsistency trips people; asserted so the doc stays right."""
    a, b = ab
    g = cg.Graph("explain_shape")
    c = g.create_tensor("C", [6, 6], intermediate=False)
    with cg.capture(g):
        einsums.einsum("ik;kj->ij", c, a, b)
    g.optimize()

    assert isinstance(g.explain, str)
    assert callable(cg.default_pass_manager().explain)


def test_returning_forms_throw_during_capture(ab):
    a, b = ab
    g = cg.Graph("returning_form")
    with pytest.raises(RuntimeError):
        with cg.capture(g):
            linalg.dot(a, b)
