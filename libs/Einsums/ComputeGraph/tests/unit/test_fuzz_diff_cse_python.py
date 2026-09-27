# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""Redundant-subexpression (CSE) diamonds: fixed regressions + randomized fuzz.

Split out of the former monolithic test_fuzz_differential_python.py; the
shared harness lives in _fuzz_diff_common.py."""

from __future__ import annotations

import numpy as np
import pytest

import einsums
import einsums._core.graph as _G
import einsums.graph as cg
from _fuzz_diff_common import *  # shared fuzz/differential harness


# ──────────────────────────────────────────────────────────────────────────
# Redundant-subexpression (CSE) shapes.
#
# The random generator above reuses a fixed tensor pool, so nearly every buffer
# has multiple writers, and CSE's single-writer guard rejects those, so CSE
# almost never fires and its redirect path went unexercised (this is exactly how
# the bug below escaped 7000+ fuzz cases). These programs deliberately build a
# write-once duplicate computation whose result is consumed by a surviving
# node, which forces CSE to eliminate the duplicate producer and redirect the
# consumer. The bug (fixed): the executor resolves operands through a captured
# TensorSlot, not Node::inputs, so CSE's metadata redirect was invisible at run
# time and the consumer read the eliminated duplicate's (stale) buffer.
# ──────────────────────────────────────────────────────────────────────────

def _check_cse(prog, m, v, t, label, dead_m=(), dead_t=(), dtype="float64"):
    """Differential check tailored to CSE's contract.

    CSE's contract (matching the original C++ unit tests, which only assert the
    survivor): when it eliminates a duplicate producer, the survivor holds the
    value and consumers are redirected to it, but the eliminated duplicate's
    own buffer is intentionally left unwritten; it is not preserved. So we:

      * require RAW (no passes) to match the oracle on every buffer, which proves
        the program is well-formed and the duplicate really was computed; and
      * require OPTIMIZED to match on every buffer except the eliminated
        duplicates (``dead_m``/``dead_t``). The consumer of the duplicate is the
        real subject: it is not dead, so it must still match, which only holds
        if the consumer reads the survivor's data via the slot redirect rather
        than the duplicate's stale buffer (the bug this guards against).
    """
    om, ov, ot = _oracle(prog, m, v, t, dtype=dtype)
    if not _usable(om, ov, ot, cap=_DTYPE_CAP[dtype]):
        pytest.skip("oracle overflowed — numerically degenerate program")

    rm, rv, rt = _run_program(prog, m, v, t, f"{label}_raw", optimize=False)
    pm_, pv, pt = _run_program(prog, m, v, t, f"{label}_opt", optimize=True)

    _assert_pools((rm, rv, rt), (om, ov, ot), prog, f"{label}-RAW", dtype=dtype)
    rtol, atol = _DTYPE_TOL[dtype]

    for idx in range(len(om)):
        if idx in dead_m:
            continue
        assert np.allclose(pm_[idx], om[idx], rtol=rtol, atol=atol), \
            f"{label}-OPTIMIZED disagrees on m{idx} (dtype={dtype}) (a pass miscompiled)\nprog={prog!r}\ngot=\n{pm_[idx]}\noracle=\n{om[idx]}"
    for idx in range(len(ot)):
        if idx in dead_t:
            continue
        assert np.allclose(pt[idx], ot[idx], rtol=rtol, atol=atol), \
            f"{label}-OPTIMIZED disagrees on t{idx} (dtype={dtype}) (a pass miscompiled)\nprog={prog!r}\ngot=\n{pt[idx]}\noracle=\n{ot[idx]}"


#: The duplicated producers' prefactor: complex on a complex pool, so CSE has to
#: compare the whole scalar, and 1 on a real one.
_DUP = ("cx", 1.0, 0.25)


def test_regression_cse_diamond_einsum(dtype):
    # buf2 = A@B ; buf3 = A@B (duplicate) ; buf5 = buf3 @ E.
    # CSE folds buf3's producer and redirects the consumer to buf2. buf5 (the
    # consumer) must match; buf3's own storage is not preserved (dead_m={3}).
    prog = [
        ("einsum", _SQ, _DUP, 0, 1, 0.0, 2),
        ("einsum", _SQ, _DUP, 0, 1, 0.0, 3),  # duplicate of the line above
        ("gemm", 1.0, 3, 4, 0.0, 5),          # consumes the eliminated duplicate
    ]
    _check_cse(prog, _sq_pool(np.random.default_rng(11), 6, dtype=dtype), [], [], "cse_diamond_einsum", dead_m={3},
               dtype=dtype)


def test_regression_cse_diamond_perm(dtype):
    # Permute (transpose) as the duplicated op; perm is CSE-eligible only when
    # its beta is 0 (pure overwrite), which is the case here.
    prog = [
        ("perm", _DUP, 0.0, 0, 1),   # buf1 = A^T
        ("perm", _DUP, 0.0, 0, 2),   # buf2 = A^T (duplicate)
        ("gemm", 1.0, 2, 3, 0.0, 4), # consumes the eliminated duplicate
    ]
    _check_cse(prog, _sq_pool(np.random.default_rng(12), 5, dtype=dtype), [], [], "cse_diamond_perm", dead_m={2},
               dtype=dtype)


def test_regression_cse_diamond_batched_einsum(dtype):
    # Rank-3 BatchedGemm duplicate + a batched consumer reading the eliminated one.
    spec = "ijb <- ikb ; kjb"
    prog = [
        ("beinsum", spec, _DUP, 0, 1, 0.0, 2),
        ("beinsum", spec, _DUP, 0, 1, 0.0, 3),  # duplicate
        ("beinsum", spec, 1.0, 3, 1, 0.0, 4),   # consumes the eliminated duplicate
    ]
    rng = np.random.default_rng(13)
    t = [typed_array(rng.standard_normal((3, 3, 2)), dtype, rng) for _ in range(5)]
    _check_cse(prog, [], [], t, "cse_diamond_beinsum", dead_t={3}, dtype=dtype)


def test_regression_cse_two_level_chain(dtype):
    # Mirrors the original (CC integral) failure: two identical *2-op* chains,
    # and a surviving consumer of the second chain's end. CSE folds both stages
    # (chained redirect), so the consumer must follow through to chain A's tail.
    prog = [
        ("einsum", _SQ, 1.0, 0, 1, 0.0, 2),  # X_a = A@B
        ("einsum", _SQ, 1.0, 2, 1, 0.0, 3),  # Y_a = X_a@B
        ("einsum", _SQ, 1.0, 0, 1, 0.0, 4),  # X_b = A@B   (dup of X_a)
        ("einsum", _SQ, 1.0, 4, 1, 0.0, 5),  # Y_b = X_b@B (dup of Y_a)
        ("gemm", 1.0, 5, 6, 0.0, 7),         # D = Y_b @ E (reads the eliminated Y_b)
    ]
    # Both duplicate intermediates (X_b=4, Y_b=5) are folded away; D=7 must match.
    _check_cse(prog, _sq_pool(np.random.default_rng(14), 8, dtype=dtype), [], [], "cse_two_level_chain", dead_m={4, 5},
               dtype=dtype)


@pytest.mark.parametrize("seed", fuzz_seeds(200))
def test_fuzz_cse_redundant(seed):
    """Randomized diamonds: a write-once duplicate (einsum/perm) plus a randomly
    chosen surviving consumer, over a square pool so all shapes are compatible.
    Forces CSE to fire with a live consumer of the folded node, the path the
    pool-reuse generator can't reach because its buffers are multi-writer. The
    consumer D must read the survivor C1 via the slot redirect, not the stale C2."""
    rng = np.random.default_rng(110_000 + seed)
    dtype = seed_dtype(seed)
    pool = _sq_pool(rng, 8, dtype=dtype)
    idx = list(range(8))
    rng.shuffle(idx)
    A, B, C1, C2, E, D = idx[:6]

    # Duplicated, pure-overwrite producer (cpf/beta = 0 → CSE-eligible), both
    # halves with one prefactor: complex on a complex pool half the time.
    side = np.random.default_rng((110_000, seed))
    ab = ("cx", 1.0, _scalar(side) or 0.5) if side.random() < 0.5 else 1.0
    if rng.random() < 0.5:
        spec = _EINSUM_SPECS[int(rng.integers(0, len(_EINSUM_SPECS)))]
        dup = [("einsum", spec, ab, A, B, 0.0, C1), ("einsum", spec, ab, A, B, 0.0, C2)]
    else:
        dup = [("perm", ab, 0.0, A, C1), ("perm", ab, 0.0, A, C2)]

    # A surviving consumer that READS the eliminated duplicate C2.
    consumer_roll = int(rng.integers(0, 3))
    if consumer_roll == 0:
        consumer = ("gemm", 1.0, C2, E, 0.0, D)         # D = C2 @ E
    elif consumer_roll == 1:
        consumer = ("perm", 1.0, 0.0, C2, D)            # D = C2^T
    else:
        consumer = ("einsum", _SQ, 1.0, C2, E, 0.0, D)  # D = C2 @ E via einsum

    prog = dup + [consumer]
    _check_cse(prog, pool, [], [], f"cse_redundant{seed}", dead_m={C2}, dtype=dtype)


# ──────────────────────────────────────────────────────────────────────────
# Diamonds over views.
#
# The duplicates read their operands through views of larger matrices, and the
# duplicate's output is graph scratch, so the merge is reachable (the pool
# diamonds above write caller tensors, which Guard C keeps). Each view is made
# once and read by both duplicates, as a caller slicing ``X[a:b]`` once and
# using it twice would; the xeinsum references above make a new view per
# statement, whose distinct ids never meet in one bucket. What varies is the
# alias shape each guard exists for: an operand block rescaled through its view
# between the two products (Guard A), a reader that reaches the duplicate's
# output through a view of it rather than by its id, and a duplicate that
# writes a view of its output instead of the whole tensor. Each program is
# compared against numpy after CSE alone and after the default pipeline.
# ──────────────────────────────────────────────────────────────────────────


def _view_diamond(seed):
    """Build and check one view diamond; True when CSE alone merged it."""
    rng = np.random.default_rng(118_000 + seed)
    # The dtype cycles with the seed; the imaginary parts come from a stream
    # of their own, so each seed draws the diamond it always drew.
    dtype = seed_dtype(seed)
    side = np.random.default_rng((118_000, seed))
    n = int(rng.choice(DIMS[1:]))
    la = einsums.linalg

    def operand(name):
        """A block of a larger matrix (sliced eagerly or captured), or a whole matrix."""
        roll = rng.random()
        if roll < 0.35:
            return ("whole", typed_array(rng.standard_normal((n, n)), dtype, side), None)
        rows, cols = n + int(rng.integers(0, 3)), n + int(rng.integers(0, 3))
        r0, c0 = int(rng.integers(0, rows - n + 1)), int(rng.integers(0, cols - n + 1))
        box = (slice(r0, r0 + n), slice(c0, c0 + n))
        return ("eager" if roll < 0.7 else "captured", typed_array(rng.standard_normal((rows, cols)), dtype, side), box)

    ops = [operand("A"), operand("B")]
    e = typed_array(rng.standard_normal((n, n)), dtype, side)
    rescale = _scalar(rng) if ops[0][0] != "whole" and rng.random() < 0.2 else None
    if rescale is not None:
        rescale = _dtype_scalar(("cx", rescale, _scalar(side) or 0.5), dtype)
    write_view = rng.random() < 0.15
    read_view = rng.random() < 0.25

    # numpy: the program as written.
    arrays = [p.copy() for _, p, _ in ops]
    blocks = [a if box is None else a[box] for (_, _, box), a in zip(ops, arrays)]
    c1 = blocks[0] @ blocks[1]
    if rescale is not None:
        blocks[0][...] *= rescale
    c2 = blocks[0] @ blocks[1]
    expected = [a.copy() for a in arrays] + [c1 @ e, c2 @ e]

    def run(how):
        held = []
        graph = cg.Graph(f"cse_view{seed}_{how}")
        parents = [einsums.asarray(p.copy()) for _, p, _ in ops]
        E, F, D = einsums.asarray(e), einsums.zeros([n, n], dtype=dtype), einsums.zeros([n, n], dtype=dtype)
        C1 = graph.create_zero_tensor("C1", [n, n], dtype=dtype)
        C2 = graph.create_zero_tensor("C2", [n, n], dtype=dtype)
        views = [None if kind == "captured" else P if box is None else P[box[0].start:box[0].stop, box[1].start:box[1].stop]
                 for (kind, _, box), P in zip(ops, parents)]
        out2 = C2[0:n, 0:n] if write_view else C2
        read2 = C2[0:n, 0:n] if read_view else C2
        with cg.capture(graph):
            for i, ((kind, _, box), P) in enumerate(zip(ops, parents)):
                if kind == "captured":
                    views[i] = cg.view(P, [(box[0].start, box[0].stop), (box[1].start, box[1].stop)])
            A, B = views
            einsums.einsum("ij <- ik ; kj", C1, A, B)
            if rescale is not None:
                la.scale(rescale, A)
            einsums.einsum("ij <- ik ; kj", out2, A, B)
            einsums.einsum("ij <- ik ; kj", F, C1, E)
            einsums.einsum("ij <- ik ; kj", D, read2, E)
        held += [views, out2, read2]
        merged = False
        if how == "alone":
            cse = _G.CSE()
            pm = cg.PassManager()
            pm.add(cse)
            graph.apply(pm)
            merged = cse.num_eliminated > 0
        else:
            graph.apply(cg.default_pass_manager())
        graph.execute()
        got = [np.asarray(P) for P in parents] + [np.asarray(F), np.asarray(D)]
        rtol, atol = _DTYPE_TOL[dtype]
        for idx, (g, w) in enumerate(zip(got, expected)):
            assert np.allclose(g, w, rtol=rtol, atol=atol), (
                f"CSE-VIEW-{how} seed={seed} dtype={dtype} output {idx}: ops={[k for k, _, _ in ops]} rescale={rescale} "
                f"write_view={write_view} read_view={read_view}\ngot=\n{g}\nwant=\n{w}")
        return merged

    merged = run("alone")
    run("default")
    return merged


@pytest.mark.parametrize("seed", fuzz_seeds(200))
def test_fuzz_cse_view_diamonds(seed):
    _view_diamond(seed)


def test_cse_view_diamonds_merge():
    """The view diamonds are not vacuous: CSE merges a fair share of them, in every dtype.

    About half of the corpus is mergeable (the operand block stable, the
    duplicate writing its whole output, the reader naming it); the floor sits
    well under that, so it catches a guard that started refusing reads through
    views rather than drift. The dtype cycles with the seed, and the floor is
    held per dtype, since a guard that refused one element type would leave the
    overall share above it.
    """
    seeds = range(80)
    merged = {dt: 0 for dt in ALL_DTYPES}
    drawn = {dt: 0 for dt in ALL_DTYPES}
    for seed in seeds:
        drawn[seed_dtype(seed)] += 1
        merged[seed_dtype(seed)] += _view_diamond(seed)
    low = {dt: f"{merged[dt]} of {drawn[dt]}" for dt in ALL_DTYPES if merged[dt] < 0.3 * drawn[dt]}
    assert not low, f"CSE merged too few view diamonds: {low}"
