#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------
"""The index-space annotations, and the scaling they let the graph derive.

No psi4. These run against a saved fixture, like the rest of the psi4-free
suite.

The point of the second half is narrow and worth stating: :mod:`dlpno.integrals`
carries a hand-written argument about why the LMO index is transformed first,
in terms of ``naux * nbf^2 * npao`` against ``naux * nbf^2 * naocc``. Once the
buffers carry index-space annotations, ``ScalingAnalysis`` derives that same
comparison from the graph. This test asserts it derives exactly what the comment
claims - it is a correctness test about the annotations, not a benchmark.

    PYTHONPATH=/path/to/Einsums/build/lib python -m pytest examples/dlpno/test_spaces.py
"""

import os
import sys

import numpy as np
import pytest

import einsums.graph as cg

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dlpno import integrals
from dlpno import spaces as sp
from dlpno.mp2 import DLPNOMP2
from dlpno.reference_io import load_reference
from dlpno.thresholds import Thresholds

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
SMALLEST = os.path.join(FIXTURES, "water-ccpvdz.npz")


@pytest.fixture(scope="module")
def orbitals():
    """The LMO and PAO coefficient matrices from the smallest fixture."""
    reference, _ = load_reference(SMALLEST)
    mp2 = DLPNOMP2(reference, Thresholds.untruncated(), verbose=False)
    mp2.setup_orbitals()
    return reference, mp2


@pytest.fixture()
def registry():
    """A private registry carrying the package's declarations.

    Private rather than the process-wide one so that the relation assertions
    below describe what :func:`dlpno.spaces.register` declares, not whatever
    else a test session has registered.
    """
    return sp.register(cg.SpaceRegistry())


# ── The declarations ────────────────────────────────────────────────────────


def test_the_four_spaces_are_registered(registry):
    names = {registry.space(i).name for i in registry.ids}
    assert names == {sp.AO, sp.AUX, sp.LMO, sp.PAO}

    # The scale symbols are what a cost polynomial is written in, and they are
    # the letters this port's own comments use for the extents.
    symbols = {registry.space(i).name: registry.space(i).scale_symbol
               for i in registry.ids}
    assert symbols == {sp.AO: "n", sp.AUX: "Q", sp.LMO: "i", sp.PAO: "u"}


def test_register_is_idempotent(registry):
    before = registry.size
    sp.register(registry)
    sp.register(registry)
    assert registry.size == before


def test_the_declared_scale_order(registry):
    ao, aux = registry.find(sp.AO), registry.find(sp.AUX)
    lmo, pao = registry.find(sp.LMO), registry.find(sp.PAO)

    assert registry.is_less(lmo, ao) == cg.Tristate.Yes
    assert registry.is_less(lmo, pao) == cg.Tristate.Yes
    assert registry.is_less(ao, aux) == cg.Tristate.Yes
    assert registry.is_less(pao, aux) == cg.Tristate.Yes
    # Transitive: naocc < npao < naux.
    assert registry.is_less(lmo, aux) == cg.Tristate.Yes

    # Not declared, and correctly so: there are equally many AOs and PAOs.
    assert registry.is_less(ao, pao) == cg.Tristate.Unknown
    assert registry.is_less(pao, ao) == cg.Tristate.Unknown


def test_lmos_and_paos_share_nothing(registry):
    lmo, pao = registry.find(sp.LMO), registry.find(sp.PAO)
    assert registry.is_disjoint(lmo, pao) == cg.Tristate.Yes

    # Containment was deliberately left undeclared: a PAO being a combination
    # of AOs is a statement about the spanned subspace, not about index sets.
    ao = registry.find(sp.AO)
    assert registry.is_contained(pao, ao) == cg.Tristate.Unknown
    assert registry.is_contained(lmo, ao) == cg.Tristate.Unknown


def test_annotate_outside_a_capture_is_a_no_op(orbitals):
    _, mp2 = orbitals
    # This is what the eager setup path in dlpno.base does, and it must neither
    # raise nor need a graph.
    assert sp.annotate(mp2.C_lmo, sp.C_LMO) is mp2.C_lmo


# ── The scaling the annotations buy ─────────────────────────────────────────


def _capture_transform(orbitals, kinds):
    """Capture ``DenseSource.build`` for ``kinds`` and analyse its scaling.

    The annotations under test are the ones :mod:`dlpno.integrals` already
    carries; capturing the transform is what turns them from documentation into
    something the cost model reads.
    """
    reference, mp2 = orbitals
    source = integrals.DenseSource(reference.eri_3index)
    source.declare(integrals.Spaces(C_lmo=mp2.C_lmo, C_pao=mp2.C_pao),
                   integrals.Demand(kinds=kinds))

    graph = cg.Graph("dlpno three-index transform")
    with cg.capture(graph):
        source.build()

    scaling = cg.ScalingAnalysis()
    pm = cg.PassManager()
    pm.add(scaling)
    pm.run(graph)
    return graph, source, scaling


def test_the_lmo_first_transform_costs_what_the_comment_says(orbitals):
    """``naux * nbf^2 * naocc`` for the half transform, derived not asserted."""
    _, _, scaling = _capture_transform(orbitals, ("q_ia",))

    assert scaling.num_analyzed == 2
    assert scaling.num_unannotated_nodes == 0

    flops = scaling.node_flops()
    # half:  "niQ <- nmQ ; mi", loop over n, m, i, Q  ->  naux * nbf^2 * naocc
    assert "2*Q*n^2*i" in flops
    # q_ia:  "Qiu <- niQ ; nu", loop over Q, i, u, n  ->  naux * nbf * naocc * npao
    assert "2*Q*n*i*u" in flops

    # And the half transform is the small one the comment advertises: naux by
    # naocc by nbf. It is read once and written once, so it is the ``2*Q*n*i``
    # term of the traffic. Traffic rather than ``intermediate_sizes_str``,
    # because every buffer this transform allocates is caller-owned - the source
    # hands them out through its accessors - and ScalingAnalysis sizes only the
    # intermediates a graph owns.
    assert "2*Q*n*i" in scaling.total_traffic_str()
    assert scaling.intermediate_names() == []


def test_the_pao_first_transform_is_the_dominant_one(orbitals):
    """The comment's whole claim, mechanically.

    Transforming the PAO index first makes the dominant contraction
    ``naux * nbf^2 * npao`` where the LMO-first order makes it
    ``naux * nbf^2 * naocc``. Both orders appear in one build when the
    coupled-cluster kinds are declared, so one report carries both, and the
    rate-limiting verdict picks the PAO-first half transform because naocc is
    declared below npao.

    The LMO-first build on its own does NOT name its half transform as the
    rate-limiting node, and that is the right answer rather than a gap: with no
    relation declared between the AO and PAO spaces - there are equally many of
    each - ``naux * nbf^2 * naocc`` and ``naux * nbf * naocc * npao`` are the
    same size, which is the very fact the comment leans on when it says the
    PAOs span the whole AO basis.
    """
    _, _, scaling = _capture_transform(orbitals, ("q_ia", "q_ab"))

    assert scaling.num_analyzed == 4
    assert scaling.num_unannotated_nodes == 0

    by_label = dict(zip(scaling.node_labels(), scaling.node_flops()))
    flops = set(by_label.values())
    assert "2*Q*n^2*i" in flops  # LMO-first half transform
    assert "2*Q*n^2*u" in flops  # PAO-first half transform, the expensive one

    limiting = scaling.rate_limiting_labels()
    assert len(limiting) == 1
    assert by_label[limiting[0]] == "2*Q*n^2*u"

    # The half transform the PAO-first order leaves behind is the size of the AO
    # integrals themselves, which is the other half of the comment's argument:
    # ``2*Q*n*u`` of traffic against ``2*Q*n*i``, a ratio of npao to naocc.
    traffic = scaling.total_traffic_str()
    assert "2*Q*n*u" in traffic
    assert "2*Q*n*i" in traffic

    report = scaling.report_string()
    assert "2*Q*n^2*u" in report
    assert "rate-limiting" in report


def test_the_annotated_capture_computes_the_same_integrals(orbitals):
    """Additive annotation: the numbers are the ones the eager path produces."""
    reference, mp2 = orbitals

    eager = integrals.DenseSource(reference.eri_3index)
    eager.declare(integrals.Spaces(C_lmo=mp2.C_lmo, C_pao=mp2.C_pao),
                  integrals.Demand(kinds=("q_ia",)))
    eager.build()

    graph, captured, _ = _capture_transform(orbitals, ("q_ia",))
    graph.execute()

    assert np.allclose(np.asarray(captured.q_ia(), copy=False),
                       np.asarray(eager.q_ia(), copy=False))


def test_a_cross_space_check_of_the_transform_is_clean(orbitals):
    """The transform binds no letter across two spaces.

    Worth asserting rather than assuming: this is the pass whose whole job is
    to catch an ``lmo`` slot contracted against a ``pao`` one, and the two are
    declared disjoint here, so a wrong index in this file would be an error
    rather than a silent wrong energy.
    """
    graph, _, _ = _capture_transform(orbitals, ("q_ia", "q_ab"))

    check = cg.CrossSpaceValidation()
    pm = cg.PassManager()
    pm.add(check)
    pm.run(graph)

    assert check.num_errors == 0, check.report_string()
    assert check.num_warnings == 0, check.report_string()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
