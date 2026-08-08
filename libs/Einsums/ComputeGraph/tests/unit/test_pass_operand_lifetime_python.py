#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

"""A rewritten graph must survive its operands being freed, exactly as an unrewritten one does.

Capture adopts an operand's storage into a stand-in the graph keeps alive
(``TensorHandle::owner``), which is what lets the caller destroy its own wrapper
before ``execute()``. ``TensorHandle::tensor_ptr`` deliberately keeps naming the
CALLER's wrapper: it is the handle's identity, not its lifetime.

Several passes rewrite nodes into ``OpKind::Custom`` closures that resolve a
tensor at execution time. A closure over ``tensor_ptr`` opts back out of that
guarantee and reads freed memory. It is silent when it does not crash: LCCF's
combined operand came back zero, so the fold produced zero and reported success.
``Graph::live_tensor_ptr`` is the resolution those closures must use.

Every test here is the same shape, and the shape is the point:

1. build a graph the pass fires on,
2. drop every Python reference to the operands and collect,
3. execute and compare.

The no-pass control in each is M0 acceptance test 1 restated, and it has to stay
green: without it a red test cannot distinguish "the pass broke it" from
"dropping operands was never supported".

A regression here can segfault rather than assert, which takes the whole file
with it. That is the honest signal for a use-after-free and is preferable to a
wrong number.
"""

import gc

import numpy as np
import pytest

import einsums
import einsums.graph as cg
from einsums import _core

_g = _core.graph


def _drop(handles):
    """Release the caller's references and make sure the objects are really gone."""
    handles.clear()
    gc.collect()


# ----------------------------------------------------------------------
# LinearCombinationContractionFolding: the folded operand is a user tensor
# ----------------------------------------------------------------------
def _build_2j_minus_k():
    n = 6
    rng = np.random.default_rng(11)
    a_np = rng.standard_normal((n, n, n, n))
    d_np = rng.standard_normal((n, n))
    ref = 2.0 * np.einsum("ijkl,kl->ij", a_np, d_np) - np.einsum("ikjl,kl->ij", a_np, d_np)

    A, D = einsums.asarray(a_np), einsums.asarray(d_np)
    C = einsums.zeros((n, n), dtype="float64")
    g = cg.Graph("2j-k")
    with cg.capture(g):
        einsums.einsum("i,j <- i,j,k,l ; k,l", C, A, D, c_pf=0.0, ab_pf=2.0)
        einsums.einsum("i,j <- i,k,j,l ; k,l", C, A, D, c_pf=1.0, ab_pf=-1.0)
    return g, C, ref, [A, D]


@pytest.mark.parametrize("with_pass", [False, True], ids=["control", "lccf"])
def test_lccf_survives_freed_operands(with_pass):
    g, C, ref, operands = _build_2j_minus_k()
    if with_pass:
        pm = _g.PassManager()
        pm.add(_g.LinearCombinationContractionFolding())
        assert g.apply(pm), "the fold must fire, or this test proves nothing"
    _drop(operands)
    g.execute()
    np.testing.assert_allclose(np.asarray(C), ref, rtol=1e-12, atol=1e-12)


# ----------------------------------------------------------------------
# StreamContractionFusion: the shared stream operand is a user tensor
# ----------------------------------------------------------------------
def _build_stream_jk():
    # 40^4 = 2.56M elements, above the pass's stream threshold. At the n=6 used
    # elsewhere in this file the pass declines and the test skips, which proves
    # nothing about the fix; the C++ mirror (Pass_StreamContractionFusion.cpp)
    # documents the same constant for the same reason.
    n = 40
    rng = np.random.default_rng(5)
    tei_np = rng.standard_normal((n, n, n, n))
    d_np = rng.standard_normal((n, n))
    j_ref = 2.0 * np.einsum("ijkl,kl->ij", tei_np, d_np)
    k_ref = -1.0 * np.einsum("ikjl,kl->ij", tei_np, d_np)

    TEI, D = einsums.asarray(tei_np), einsums.asarray(d_np)
    J = einsums.zeros((n, n), dtype="float64")
    K = einsums.zeros((n, n), dtype="float64")
    g = cg.Graph("stream_jk")
    with cg.capture(g):
        einsums.einsum("i,j <- i,j,k,l ; k,l", J, TEI, D, c_pf=0.0, ab_pf=2.0)
        einsums.einsum("i,j <- i,k,j,l ; k,l", K, TEI, D, c_pf=0.0, ab_pf=-1.0)
    return g, (J, K), (j_ref, k_ref), [TEI, D]


@pytest.mark.parametrize("with_pass", [False, True], ids=["control", "stream_fusion"])
def test_stream_contraction_fusion_survives_freed_operands(with_pass):
    g, (J, K), (j_ref, k_ref), operands = _build_stream_jk()
    if with_pass:
        pm = _g.PassManager()
        pm.add(_g.StreamContractionFusion())
        # assert, not skip: a skip here would report green for a fix nothing ran.
        assert g.apply(pm), "the fusion must fire, or this test proves nothing"
    _drop(operands)
    g.execute()
    np.testing.assert_allclose(np.asarray(J), j_ref, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(K), k_ref, rtol=1e-12, atol=1e-12)


# ----------------------------------------------------------------------
# SymmetrizedAccumulation: both the destination and the source are user tensors
# ----------------------------------------------------------------------
def _build_symacc():
    o, v, s = 2, 3, 0.5
    rng = np.random.default_rng(0)
    a_np = rng.standard_normal((o, v))
    b_np = rng.standard_normal((o, v))
    tmp_np = np.einsum("ia,jb->ijab", a_np, b_np)
    ref = s * (tmp_np + np.transpose(tmp_np, (1, 0, 3, 2)))

    A, B = einsums.asarray(a_np), einsums.asarray(b_np)
    r2 = einsums.zeros((o, o, v, v), dtype="float64")
    tmp = einsums.zeros((o, o, v, v), dtype="float64")
    tmpP = einsums.zeros((o, o, v, v), dtype="float64")
    g = cg.Graph("symacc")
    with cg.capture(g):
        einsums.einsum("i,j,a,b <- i,a ; j,b", tmp, A, B)
        einsums.linalg.axpby(s, tmp, 1.0, r2)
        einsums.permute("j,i,b,a <- i,j,a,b", tmpP, tmp)
        einsums.linalg.axpby(s, tmpP, 1.0, r2)
    # r2 is the result and is kept; everything else is droppable.
    return g, r2, ref, [A, B, tmp, tmpP]


@pytest.mark.parametrize("with_pass", [False, True], ids=["control", "symacc"])
def test_symmetrized_accumulation_survives_freed_operands(with_pass):
    g, r2, ref, operands = _build_symacc()
    if with_pass:
        pm = _g.PassManager()
        pm.add(_g.SymmetrizedAccumulation())
        assert g.apply(pm), "the rewrite must fire, or this test proves nothing"
    _drop(operands)
    g.execute()
    np.testing.assert_allclose(np.asarray(r2), ref, rtol=1e-12, atol=1e-12)


# ----------------------------------------------------------------------
# The whole default pipeline, which is what a real caller applies
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "builder,pick_ref",
    [
        (_build_2j_minus_k, lambda out, ref: (np.asarray(out), ref)),
        (_build_symacc, lambda out, ref: (np.asarray(out), ref)),
    ],
    ids=["2j-k", "symacc"],
)
def test_default_pipeline_survives_freed_operands(builder, pick_ref):
    g, out, ref, operands = builder()
    g.apply(cg.default_pass_manager())
    _drop(operands)
    g.execute()
    got, want = pick_ref(out, ref)
    np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-10)
