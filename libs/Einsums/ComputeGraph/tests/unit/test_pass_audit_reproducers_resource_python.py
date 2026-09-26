# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""Reproducers for defects a read-only audit of the resource and approximation passes claimed.

Each case here is a claim that was driven end to end from Python (capture, the
named pass alone and inside a public pipeline, execute) and came out wrong
against numpy. Each was fixed and now stays as the guard for the defect its
docstring names. A fix may take the form of the pass declining the input it
used to mishandle, so no case asserts that its pass fired: each asserts only
that the result matches numpy whether the pass rewrote the graph or not.

The GPU cases need a device that GPUPlacement actually offloads to. On Apple
Silicon that is MPS, which runs float32 only and shares memory with the host,
so the device kernels read and write the host buffers directly. Without a live
device they skip, since a guard that never reaches the device proves nothing.

Claims that did not reproduce, or that this build cannot reach from Python, are
recorded in the audit report rather than here: the distributed passes return
early on a single-rank mock world, and the mixed-dtype Laplace numerator has no
Python spelling.
"""

from __future__ import annotations

import os

os.environ.setdefault("EINSUMS_PASS_VERIFY", "1")

import numpy as np
import pytest

import einsums
import einsums.graph as cg
import einsums._core.graph as _G

from _fuzz_diff_common import _apply_one_pass
import _resource_pass_motifs as R


def _tensor(name, array):
    """A caller-owned tensor holding @p array."""
    out = einsums.create_zero_tensor(name, list(array.shape), dtype=str(array.dtype))
    np.asarray(out)[...] = array
    return out


def _gpu_offloads():
    """Whether GPUPlacement offloads a float32 GEMM in this build, on this machine."""
    if "GPUPlacement" not in R.default_pipeline_pass_names():
        return False
    n = 256
    A = einsums.create_zero_tensor("probe_A", [n, n], dtype="float32")
    B = einsums.create_zero_tensor("probe_B", [n, n], dtype="float32")
    C = einsums.create_zero_tensor("probe_C", [n, n], dtype="float32")
    g = cg.Graph("gpu_probe")
    with cg.capture(g):
        einsums.einsum("ij <- ik ; kj", C, A, B)
    return bool(_apply_one_pass(g, "GPUPlacement"))


_GPU_LIVE = _gpu_offloads()
needs_gpu = pytest.mark.skipif(not _GPU_LIVE, reason="no GPU offload in this build or on this machine")


def _optimize(g, pipeline):
    """Run one of the pipelines a reproducer is pinned under."""
    if pipeline == "GPUPlacement":
        # Declining is a correct outcome: the fix keeps these nodes on the host.
        _apply_one_pass(g, "GPUPlacement")
    elif pipeline == "resource":
        g.apply(cg.resource_pass_manager())
    elif pipeline == "O2":
        g.optimize(einsums._core.OptLevel.O2)
    else:
        raise AssertionError(pipeline)


# ──────────────────────────────────────────────────────────────────────────
# GPU dispatch
# ──────────────────────────────────────────────────────────────────────────


@needs_gpu
@pytest.mark.parametrize("pipeline", ["GPUPlacement", "resource"])
def test_gpu_gemm_drops_permutation_operator(pipeline):
    """``C = P(i/j) A B`` computes the antisymmetrized product, on the device or off it.

    Defends against GPUPlacement's ``einsum_is_dispatchable`` rejecting an
    operator only on its strided-batch branch, so that a plain two-target,
    one-link einsum was admitted whatever its operators, and ``try_gpu_gemm``
    (GpuDispatch.cpp), which never read ``lists.operators``, ran the node as a
    bare GEMM and lost the antisymmetrizer. The default O2 pipeline was not hit
    only because AntisymmetrizerExpansion removes the operator before placement;
    ``resource_pass_manager`` on a captured graph was. The measured max abs
    error was about 67, the size of ``(A B)^T`` itself. GPUPlacement now
    declines the node, which is a correct outcome.
    """
    n = 256
    rng = np.random.default_rng(1)
    a = rng.standard_normal((n, n)).astype(np.float32)
    b = rng.standard_normal((n, n)).astype(np.float32)
    A, B = _tensor("A", a), _tensor("B", b)
    C = _tensor("C", np.zeros((n, n), np.float32))

    g = cg.Graph("gpu_gemm_p")
    with cg.capture(g):
        einsums.einsum("ij <- P(i/j) ik ; kj", C, A, B)
    _optimize(g, pipeline)
    g.execute()

    ab = a.astype(np.float64) @ b.astype(np.float64)
    want = ab - ab.T
    err = float(np.max(np.abs(np.asarray(C) - want)))
    assert err <= 1e-4 * float(np.max(np.abs(want))), f"max abs error {err}"


@needs_gpu
@pytest.mark.parametrize("pipeline", ["GPUPlacement", "resource", "O2"])
@pytest.mark.parametrize("op", ["scale", "axpy", "axpby"])
def test_gpu_blas1_on_strided_view_corrupts_parent(op, pipeline):
    """Scale / axpy / axpby through a column-major sub-block view touch only the view.

    Defends against ``try_gpu_scale`` and ``try_gpu_axpy`` (GpuDispatch.cpp)
    taking the element count from ``total_bytes()`` of the view and calling
    ``scal`` / ``axpy`` with increment 1 from the view's data pointer. A view of
    rows 8..135 of a 256 x 256 column-major parent has a column stride of 256,
    so the sweep covered 128 x 128 CONTIGUOUS elements: half of the view was
    never touched and the same number of parent elements outside the view were
    overwritten, because GPUPlacement admitted every Scale and Axpby node with
    no stride check. The default O2 pipeline reached it on this machine, with a
    measured max abs error of about 4 (scale) and 16 (axpy, axpby). GPUPlacement
    now declines a strided view, which is a correct outcome.
    """
    n = 256
    rng = np.random.default_rng(0)
    m0 = rng.standard_normal((n, n)).astype(np.float32)
    x0 = rng.standard_normal((n // 2, n // 2)).astype(np.float32)
    M, X = _tensor("M", m0), _tensor("X", x0)
    rows, cols = (8, 8 + n // 2), (4, 4 + n // 2)

    g = cg.Graph(f"gpu_view_{op}")
    with cg.capture(g):
        V = cg.view(M, [rows, cols])
        if op == "scale":
            einsums.linalg.scale(2.0, V)
        elif op == "axpy":
            einsums.linalg.axpy(3.0, X, V)
        else:
            einsums.linalg.axpby(3.0, X, 0.5, V)
    _optimize(g, pipeline)
    g.execute()

    want = m0.astype(np.float64)
    block = (slice(*rows), slice(*cols))
    if op == "scale":
        want[block] *= 2.0
    elif op == "axpy":
        want[block] += 3.0 * x0
    else:
        want[block] = 3.0 * x0 + 0.5 * want[block]
    err = float(np.max(np.abs(np.asarray(M) - want)))
    assert err <= 1e-5, f"max abs error {err}"


# ──────────────────────────────────────────────────────────────────────────
# Interference checks that compare raw tensor ids
# ──────────────────────────────────────────────────────────────────────────


def _finish(g, follow_up):
    """What runs after the lossy pass: storage only, or the whole default pipeline."""
    if follow_up == "default":
        g.apply(cg.default_pass_manager())
    else:
        pm = cg.PassManager()
        pm.add(_G.Materialization())
        g.apply(pm)


@pytest.mark.parametrize("follow_up", ["materialize", "default"])
def test_factorization_flattens_across_a_view_write(follow_up):
    """``X = A B``; write ``A[:, :4]`` through a view; ``C = M X`` with M tagged.

    Defends against FactorizationPass folding X's definition into the tagged
    contraction so that the rewrite read A where C stands, after the write. The
    guard meant to stop this (FactorizationPass.cpp, the ``interference`` loop
    over ``outer_pieces``) compared each intervening statement's target to the
    piece's raw TensorId; the view has its own id, so the write was not seen.
    The same program writing A directly is declined with "a statement between
    an intermediate's definition and its use rewrites one of its operands". The
    view is created ahead of the definition so that its (non-raisable) node
    does not split the region. The metric fit is exact, so any error is the
    defect; the measured max abs error was about 50. Whether the pass declines
    or factorizes, C must match numpy.
    """
    naux, n = 3, 8
    rng = np.random.default_rng(20260901)
    three = rng.standard_normal((naux, n, n))
    metric = rng.standard_normal((naux, naux))
    metric = metric @ metric.T + naux * np.eye(naux)
    dense = np.einsum("pmn,pq,qab->mnab", three, np.linalg.inv(metric), three)
    a0 = rng.standard_normal((n, n))
    b0 = rng.standard_normal((n, n))
    y0 = rng.standard_normal((n, n))
    z0 = rng.standard_normal((n, 4))

    M = einsums.asarray(dense)
    A, B = einsums.asarray(a0.copy()), einsums.asarray(b0)
    Y, Z = einsums.asarray(y0), einsums.asarray(z0)
    C = einsums.zeros((n, n), dtype="float64")

    g = cg.Graph("factorization_view_write")
    X = g.scratch("X", [n, n], "float64")
    with cg.capture(g):
        V = cg.view(A, [(-1, -1), (0, 4)])
        einsums.einsum("p,q <- p,k ; k,q", X, A, B)
        einsums.einsum("i,j <- i,k ; k,j", V, Y, Z)
        einsums.einsum("m,n <- m,n,p,q ; p,q", C, M, X)
    g.annotate_tag(M, _G.ProvenanceTag.make("eri"))

    registry = _G.FactorizationRegistry()
    registry.add(_G.MetricFitFactorization("eri", einsums.asarray(three), einsums.asarray(metric), 0.0))
    factorization = _G.FactorizationPass(registry)
    pm = cg.PassManager()
    pm.add(factorization)
    g.apply(pm)
    _finish(g, follow_up)
    g.execute()

    want = np.einsum("mnpq,pq->mn", dense, a0 @ b0)
    err = float(np.max(np.abs(np.asarray(C) - want)))
    assert err <= 1e-10 * float(np.max(np.abs(want))), f"max abs error {err}"


@pytest.mark.parametrize("follow_up", ["materialize", "default"])
def test_laplace_transform_across_a_view_write(follow_up):
    """``N = A B``; write ``A[:, :2]`` through a view; ``P = N * D`` with D tagged.

    Defends against LaplaceTransform replacing the direct product with scalings
    of A and B placed where the product stands, after the write. Its guard
    (LaplaceTransform.cpp, the ``interference`` loop between the numerator's
    formation and its use) compared ``other.target`` to each operand's raw
    TensorId, so the view's write was missed; the measured max abs error was
    about 5.7. Writing A directly is declined with "another statement rewrites
    an operand of the numerator between its formation and its use", and the
    view's write now is too, which is a correct outcome. The quadrature error
    is held to 1e-8, so the bound below is loose by orders of magnitude whether
    the pass declines or transforms.
    """
    nocc, nvir, nlnk = 4, 5, 3
    eo = np.array([-0.9 + 0.15 * i for i in range(nocc)])
    ev = np.array([0.05 + 0.5 * a for a in range(nvir)])
    d = 1.0 / (ev[None, :] - eo[:, None])
    rng = np.random.default_rng(20260904)
    a0 = rng.standard_normal((nocc, nlnk))
    b0 = rng.standard_normal((nlnk, nvir))
    y0 = rng.standard_normal((nocc, nocc))
    z0 = rng.standard_normal((nocc, 2))

    A, B, D = einsums.asarray(a0.copy()), einsums.asarray(b0), einsums.asarray(d)
    Y, Z = einsums.asarray(y0), einsums.asarray(z0)
    P = einsums.zeros((nocc, nvir), dtype="float64")

    g = cg.Graph("laplace_view_write")
    numerator = g.scratch("numerator", [nocc, nvir], "float64")
    with cg.capture(g):
        V = cg.view(A, [(-1, -1), (0, 2)])
        einsums.einsum("i,k ; k,a -> i,a", numerator, A, B)
        einsums.einsum("i,j <- i,k ; k,j", V, Y, Z)
        einsums.linalg.direct_product(1.0, numerator, D, 0.0, P)
    cg.annotate(D, tag={"name": "laplace_denominator", "axis0": "eps_o", "sign0": "-", "axis1": "eps_v", "sign1": "+"},
                graph=g)

    laplace = cg.LaplaceTransform()
    laplace.set_epsilon(1e-8)
    laplace.add_energy("eps_o", einsums.asarray(eo))
    laplace.add_energy("eps_v", einsums.asarray(ev))
    pm = cg.PassManager()
    pm.add(laplace)
    g.apply(pm)
    _finish(g, follow_up)
    g.execute()

    want = (a0 @ b0) * d
    err = float(np.max(np.abs(np.asarray(P) - want)))
    assert err <= 1e-5 * float(np.max(np.abs(want))), f"max abs error {err}"
