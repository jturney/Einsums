#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------
"""AutoDF against the density fitting this example already does by hand.

Milestone E's last item: ship the factorization provider end to end and check it
against the DF machinery here. The check is not what it first sounds like. This
example never HOLDS a four-index integral for a pass to factorize; psi4 hands it
``(Q|mn)`` and a metric, and every contraction is written against the three-index
form already. What AutoDF produces is what this code writes by hand.

So the validation runs the other way round. Build the four-index tensor the naive
form would use, tag it, let ``FactorizationPass`` fit it from the SAME fixture
buffers, and require the answer to match the hand-written fit.

The two fits are not the same factorization, which is what makes agreeing
meaningful. This example solves ``J x = (Q|iu)``, psi4's robust fit, and
contracts the solution against the raw three-index tensor: ``R^T J^-1 R``.
``MetricFitFactorization`` forms ``B = J^-1/2 R`` and contracts ``B^T B``. Both
are ``R^T J^-1 R`` and neither computes it the way the other does, so a match is
evidence about the provider rather than a tautology.

No psi4: the fixture carries ``eri_3index`` and ``metric``, which are exactly the
provider's two inputs.

    PYTHONPATH=/path/to/Einsums/build/lib python -m pytest examples/dlpno/test_autodf.py
"""

import os
import sys

import numpy as np
import pytest

import einsums
import einsums.graph as cg
import einsums._core.graph as _G

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
SMALLEST = os.path.join(FIXTURES, "water-ccpvdz.npz")


def _fixture():
    if not os.path.exists(SMALLEST):
        pytest.skip(f"fixture not present: {SMALLEST}")
    z = np.load(SMALLEST, allow_pickle=True)
    return np.asarray(z["eri_3index"], dtype=np.float64), np.asarray(z["metric"], dtype=np.float64)


def _hand_written_fit(three, metric):
    """The four-index integral this example's DF machinery implies.

    ``base.py`` solves for ``J^-1 (Q|iu)`` and contracts it against the raw
    three-index block, so the tensor its contractions behave as though they were
    reading is ``R^T J^-1 R``. Written densely here because the point is the
    VALUE, not the pair-blocked way the example reaches it.
    """
    fit = np.linalg.solve(metric, three.reshape(three.shape[0], -1))
    return np.einsum("Qx,Qy->xy", fit, three.reshape(three.shape[0], -1)).reshape(
        three.shape[1], three.shape[2], three.shape[1], three.shape[2])


def test_metric_fit_reproduces_the_hand_written_fit():
    """The provider's symmetric fit equals the example's robust fit.

    Checked on the tensor itself rather than through a contraction, so a failure
    points at the fit and not at whatever consumed it.
    """
    three, metric = _fixture()
    naux = three.shape[0]

    values, vectors = np.linalg.eigh(metric)
    inv_sqrt = vectors @ np.diag(np.where(values > 1e-10, 1.0 / np.sqrt(np.abs(values)), 0.0)) @ vectors.T
    b = np.einsum("PQ,Qmn->Pmn", inv_sqrt, three)
    symmetric = np.einsum("Pmn,Ppq->mnpq", b, b)

    assert np.allclose(symmetric, _hand_written_fit(three, metric), atol=1e-9, rtol=1e-9), (
        "B^T B and R^T J^-1 R disagree; the provider's fit is not this example's fit")
    assert naux == metric.shape[0]


def test_autodf_factorizes_the_dense_integral_and_agrees():
    """The whole loop, on this example's own buffers.

    Tag the dense integral, register the provider on that tag, run the pass, and
    require the contraction to come out where the hand-written fit puts it.
    """
    three, metric = _fixture()
    nbf = three.shape[1]

    dense = _hand_written_fit(three, metric)
    rng = np.random.default_rng(20260901)
    operand = rng.standard_normal((nbf, nbf))

    M = einsums.asarray(dense)
    T = einsums.asarray(operand)
    C = einsums.zeros((nbf, nbf), dtype="float64")

    g = cg.Graph("dlpno_autodf")
    with cg.capture(g):
        einsums.einsum("m,n,p,q ; p,q -> m,n", C, M, T)
    g.annotate_tag(M, _G.ProvenanceTag.make("eri"))

    registry = _G.FactorizationRegistry()
    registry.add(_G.MetricFitFactorization(
        "eri", einsums.asarray(three), einsums.asarray(metric), 0.0))

    factorization = _G.FactorizationPass(registry)
    pm = cg.PassManager()
    pm.add(factorization)

    assert g.apply(pm), (
        "the pass declined. At this fixture's shapes the split is cheaper, so a "
        "decline means a refusal fired rather than the cost model speaking")
    assert factorization.num_factorized == 1
    assert [r.pass_name for r in g.approximations()] == ["MetricFit"]

    g.apply(cg.default_pass_manager())
    g.execute()

    expected = np.einsum("mnpq,pq->mn", dense, operand)
    # Loose against the dense contraction on purpose: the factorized form sums
    # over the auxiliary index instead of the two orbital ones, which is the
    # re-associating tier doing what it says, at this basis over 84 auxiliaries.
    assert np.allclose(np.asarray(C), expected, atol=1e-9, rtol=1e-9), (
        f"max deviation {np.max(np.abs(np.asarray(C) - expected)):.3g}")


def test_the_example_never_holds_a_four_index_integral():
    """Why the check above runs backwards, pinned so it is not re-derived.

    Every contraction here is written against the three-index form, so there is
    nothing for the pass to tag in the example as it stands. If a four-index
    integral ever appears, this fails and the validation above can be pointed at
    the real thing instead of a reconstruction.
    """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import glob
    import re

    four_index_specs = []
    for path in glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "dlpno", "*.py")):
        with open(path, encoding="utf-8") as handle:
            for spec in re.findall(r'einsum\("([^"]+)"', handle.read()):
                target = spec.split("<-")[0]
                if len(re.sub(r"[^A-Za-z]", "", target)) == 4:
                    four_index_specs.append((os.path.basename(path), spec))

    assert not four_index_specs, (
        "a four-index target appeared; AutoDF may now have something real to "
        f"factorize here: {four_index_specs}")
