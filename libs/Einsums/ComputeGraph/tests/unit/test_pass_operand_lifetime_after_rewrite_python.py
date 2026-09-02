# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""A rewritten graph still honours capture's operand-lifetime contract.

Capture adopts an operand's storage into a stand-in the graph keeps alive, so
that the caller's wrapper MAY be destroyed before ``execute()``. That is the
documented contract, with a worked example, and an unrewritten graph honoured
it. A rewritten one did not.

``DistributiveFactoring`` folds ``A*B1 + A*B2`` into ``A*(B1+B2)``, which
removes the capture-built einsums and emits its own axpy chain. Those
pass-built executors resolved their operands by id at replay and dereferenced
``TensorHandle::tensor_ptr``, which names the caller's WRAPPER rather than the
stand-in, so they read freed memory and segfaulted. The fix routes the dispatch
helpers through ``TensorHandle::live_ptr``.

These build inside a function and drop the references on purpose. That is the
shape ordinary code has, and it is what nothing else in the suite exercised.
"""

from __future__ import annotations

import gc

import numpy as np
import pytest

import einsums
import einsums.graph as cg
from einsums.testing import assert_close

#: Big enough that DistributiveFactoring judges the rewrite profitable. It
#: declines around n=32, and a declined rewrite tests nothing here.
_N = 128


def _one(pass_obj):
    pm = cg.PassManager()
    pm.add(pass_obj)
    return pm


#: What DistributiveFactoring emits for a two-member group, as a node count: the Alloc for the
#: intermediate, the zero, one axpy per member, and the single contraction. Asserted because
#: ``num_groups == 1`` says the pass FIRED and says nothing about what it built, and every way
#: this rewrite can go wrong leaves that counter at one.
_EXPECTED_NODES_AFTER = 5


def _diagnose(actual, a, b1, b2):
    """Name the arithmetic the graph actually performed, when it is not the right one.

    A wrong answer here is not a rounding question, it is a missing or misordered term, and each
    candidate below points at a different mechanism: ``a@b2`` means b1's axpy contributed nothing,
    ``2*a@b1`` means one operand was read twice, and so on. Without this the failure reports 16384
    mismatched elements and a reader has to reconstruct the intended forms by hand to learn
    anything from it.
    """
    candidates = {
        "a@b1 + a@b2 (correct)": a @ b1 + a @ b2,
        "a@b2 alone (b1's axpy contributed nothing)": a @ b2,
        "a@b1 alone (b2's axpy contributed nothing)": a @ b1,
        "2*a@b1 (b1 accumulated twice)": 2 * (a @ b1),
        "2*a@b2 (b2 accumulated twice)": 2 * (a @ b2),
        "a@(b1+b2) with no scaling": a @ (b1 + b2),
    }
    hits = [name for name, want in candidates.items() if np.allclose(np.asarray(actual), want)]
    return f"the graph computed {hits[0]}" if hits else "the graph computed none of the expected forms"


@pytest.mark.parametrize("keep", ["none", "shared", "nonshared"])
def test_factored_graph_survives_dropped_operands(keep):
    """The operands the rewrite consumed may go out of scope before execute.

    Parametrized over which wrapper survives because the two roles failed
    differently: the shared operand feeds the surviving contraction and was
    always fine, while the non-shared ones feed the pass-built axpys and were
    the ones read after free.
    """
    rng = np.random.default_rng(20260901)
    a = rng.standard_normal((_N, _N))
    b1 = rng.standard_normal((_N, _N))
    b2 = rng.standard_normal((_N, _N))

    out = einsums.zeros((_N, _N), dtype="float64")
    g = cg.Graph("operand_lifetime")
    kept = []

    def build():
        A, B1, B2 = einsums.asarray(a), einsums.asarray(b1), einsums.asarray(b2)
        with cg.capture(g):
            einsums.einsum("ij <- ik ; kj", out, A, B1, c_pf=1.0, ab_pf=1.0)
            einsums.einsum("ij <- ik ; kj", out, A, B2, c_pf=1.0, ab_pf=1.0)
        if keep == "shared":
            kept.append(A)
        elif keep == "nonshared":
            kept.extend((B1, B2))

    build()
    gc.collect()

    nodes_before = g.num_nodes()
    factoring = cg.DistributiveFactoring()
    assert g.apply(_one(factoring)), "the rewrite did not fire; this test then proves nothing"
    assert factoring.num_groups == 1

    # The SHAPE of the rewrite, not just that it happened. A missing axpy and a misordered one
    # both leave num_groups at 1 and both produce a wrong answer, and only this tells them apart.
    assert nodes_before == 2, f"expected the two captured contractions, got {nodes_before}"
    assert g.num_nodes() == _EXPECTED_NODES_AFTER, (
        f"the rewrite emitted {g.num_nodes()} nodes, expected {_EXPECTED_NODES_AFTER} "
        f"(alloc, zero, one axpy per member, one contraction); a missing axpy lands here")

    g.execute()
    # Names the arithmetic that actually happened, for the gross-mismatch case. A
    # rounding-level difference falls through to assert_close, which owns the tolerance.
    assert np.allclose(np.asarray(out), a @ b1 + a @ b2), _diagnose(out, a, b1, b2)
    assert_close(out, a @ b1 + a @ b2)


def test_unrewritten_graph_survives_dropped_operands():
    """The same contract without any pass, which always held.

    Here so a regression in adoption itself is told apart from a regression in
    what a pass does to it: if both cases fail, the stand-in is broken; if only
    the rewritten one fails, a pass is reading the identity pointer again.
    """
    rng = np.random.default_rng(20260902)
    a = rng.standard_normal((16, 16))
    b = rng.standard_normal((16, 16))

    out = einsums.zeros((16, 16), dtype="float64")
    g = cg.Graph("operand_lifetime_plain")

    def build():
        A, B = einsums.asarray(a), einsums.asarray(b)
        with cg.capture(g):
            einsums.einsum("ij <- ik ; kj", out, A, B, c_pf=0.0, ab_pf=1.0)

    build()
    gc.collect()

    g.execute()
    assert_close(out, a @ b)
