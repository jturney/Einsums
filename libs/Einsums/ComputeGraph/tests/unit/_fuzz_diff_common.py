# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""Shared harness for the ComputeGraph differential / fuzz shards.

The differential idea, the tensor pool, the program representation, the
numpy oracle, the ComputeGraph builder and the trial drivers all live here;
each ``test_fuzz_diff_*_python.py`` shard imports this module and contributes
one slice of the test surface, so ctest can schedule the slices across cores.

Original harness documentation follows.
Differential / fuzz harness for ComputeGraph + the optimization passes.

The idea is simple and brutal: generate a random program over a pool of
tensors, then run it three ways and demand they all agree:

  1. numpy oracle: a pure-numpy interpreter of the program.
  2. raw graph: the program replayed into a ``cg.Graph`` and executed
     with no optimization passes.
  3. optimized graph: the same program, then ``default_pass_manager()``
     applied before execution.

If (raw == oracle) but (optimized != oracle), a pass miscompiled the graph.
If (raw != oracle), the executor itself disagrees with numpy. Either way the
seed and the offending program are printed so the failure reproduces.

The tensor pool is deliberately diverse:
  * matrices over every (r, c) with r, c ∈ {2, 3, 4} (several copies each),
  * vectors of each length,
  * rank-3 tensors over {2, 3}^3 for batched contractions.

The generator picks shape-compatible operands for each op, so contractions
exercise non-square M/N/K and batched (rank-3) gemms.

The op set stresses the read-modify-write hazards that have broken passes
before, across the BLAS levels, the einsum/permute path, batched gemm, and
views (sub-block aliases stress the scheduler's alias resolution,
since a write through a view must be seen as a write to the parent):

  * ``scale`` / ``axpy`` / ``axpby``: level-1 in-place / accumulate.
  * ``gemm``: ``C = a*A@B + b*C`` (mixed M/N/K, overwrite or accumulate).
  * ``einsum``: three contraction patterns over mixed shapes, with random
    conj_a/conj_b (native conjugation: a no-op on real dtypes, the
    real test on complex; also exercises the passes' conj guards).
  * ``beinsum``: rank-3 batched einsum ``ijb<-ikb;kjb`` (BatchedGemm path), likewise
    with random conjugation flags.
  * ``perm``: transpose ``C = a*A^T + c*C``.
  * ``symm``: symmetric double multiply ``C = B^T A B``.
  * ``gemv`` / ``ger``: matrix×vector and rank-1 update.
  * ``vscale`` / ``vaxpy``: scale / axpy applied through a view of a
    matrix sub-block; mixed with full-matrix writes to the same
    tensor to exercise alias-aware scheduling.

plus control flow:

  * ``loop``: run a body sub-program a fixed number of times.
  * ``cond``: generation-time coin flip selecting then/else branch.

Conditionals use a coin flip rather than a data-dependent predicate on purpose:
a predicate near its threshold would flip differently under fp noise between the
oracle and the executor, making the whole branch diverge and the test flaky.
Data-dependent branching is covered by the dedicated SCF/MP2 tests.
"""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pytest

import einsums
import einsums.graph as cg
import einsums._core.graph as _G  # pass classes / Workspace, re-exported for shards
from _sanitizer_scaling import fuzz_seeds  # seed-count scaling under sanitizers

# ──────────────────────────────────────────────────────────────────────────
# Pool layout, fixed so the generator and the per-trial seed arrays agree.
# ──────────────────────────────────────────────────────────────────────────

# Dimension 1 is included on purpose: degenerate extents (K=1 rank-1-like
# contractions, M=1 row / N=1 col gemms, length-1 vectors, 1×1 transpose/symm)
# stress the stride / leading-dimension / packing logic that fixed dims ≥ 2
# never reach. numpy oracles them cleanly, so any divergence, or any ASan/UBSan
# trip on a zero-stride or single-element buffer, is a real finding.
DIMS = (1, 2, 3, 4)
R3_DIMS = (1, 2, 3)
COPIES = 3  # copies of each matrix shape / vector length / rank-3 shape

# Differential tolerances and a magnitude cap, per dtype. These are *looser*
# than einsums.testing.tolerance_for (which assumes the same computation): here
# we compare a numpy oracle against BLAS and against a reordered optimized graph,
# so single precision needs more slack. Real miscompiles differ by O(1), well
# outside these, so generous tolerances avoid fp false positives without hiding
# bugs. The cap skips programs whose values grow past where the dtype keeps
# enough absolute resolution for the tolerance to be meaningful.
_DTYPE_TOL = {
    "float32": (1e-3, 1e-3),
    "complex64": (1e-3, 1e-3),
    "float64": (1e-5, 1e-5),
    "complex128": (1e-5, 1e-5),
}
_DTYPE_CAP = {"float32": 1e3, "complex64": 1e3, "float64": 1e8, "complex128": 1e8}

# Defaults used by the non-dtype-parametrized modes (which run in float64).
RTOL = 1e-5
ATOL = 1e-5

MAT_SHAPES = [(r, c) for r in DIMS for c in DIMS for _ in range(COPIES)]
VEC_LENS = [d for d in DIMS for _ in range(COPIES)]
R3_SHAPES = [(a, b, c) for a in R3_DIMS for b in R3_DIMS for c in R3_DIMS for _ in range(COPIES)]

MAT_BY_SHAPE: dict[tuple[int, int], list[int]] = {}
for _idx, _sh in enumerate(MAT_SHAPES):
    MAT_BY_SHAPE.setdefault(_sh, []).append(_idx)
VEC_BY_LEN: dict[int, list[int]] = {}
for _idx, _len in enumerate(VEC_LENS):
    VEC_BY_LEN.setdefault(_len, []).append(_idx)
R3_BY_SHAPE: dict[tuple[int, int, int], list[int]] = {}
for _idx, _sh in enumerate(R3_SHAPES):
    R3_BY_SHAPE.setdefault(_sh, []).append(_idx)

# einsum (rank-2) contraction patterns and numpy equivalents + operand-shape rule.
EINSUM_PATTERNS = {
    "ij <- ik ; kj": (lambda A, B: A @ B, lambda i, k, j: ((i, k), (k, j), (i, j))),
    "ij <- ki ; kj": (lambda A, B: A.T @ B, lambda i, k, j: ((k, i), (k, j), (i, j))),
    "ij <- ik ; jk": (lambda A, B: A @ B.T, lambda i, k, j: ((i, k), (j, k), (i, j))),
}
_EINSUM_SPECS = list(EINSUM_PATTERNS.keys())

# rank-3 batched einsum patterns (batch index b is the trailing axis).
BEINSUM_PATTERNS = {
    "ijb <- ikb ; kjb": (lambda A, B: np.einsum("ikb,kjb->ijb", A, B),
                         lambda i, k, j, b: ((i, k, b), (k, j, b), (i, j, b))),
    "ijb <- kib ; kjb": (lambda A, B: np.einsum("kib,kjb->ijb", A, B),
                         lambda i, k, j, b: ((k, i, b), (k, j, b), (i, j, b))),
}
_BEINSUM_SPECS = list(BEINSUM_PATTERNS.keys())

# "linear-combination" einsum: a vector contracted against a rank-3 tensor read
# with permuted (i,j) axes. This is exactly the shape
# LinearCombinationContractionFolding folds (the CCSD 2J-K idiom): two terms into
# the same matrix output, sharing the vector operand, reading the SAME rank-3
# tensor with transposed trailing axes. The rank-3 operand must be (k, n, n) so
# both axis orders yield the same (n, n) output. Only the opt-in arm emits this
# opcode; the random program generator never does.
LEINSUM_PATTERNS = {
    # key: (numpy fn(vec, r3) -> matrix, cg spec string)
    "kij": (lambda a, b: np.einsum("k,kij->ij", a, b), "i,j <- k ; k,i,j"),
    "kji": (lambda a, b: np.einsum("k,kji->ij", a, b), "i,j <- k ; k,j,i"),
}

# Unary element-wise transforms. Each works on a numpy array (oracle) and on a
# scalar (the C++ executor calls it per element), and is bounded so values stay
# fp-comparable across loops.
ETRANSFORM_FNS = [
    lambda x: -x,
    lambda x: 0.5 * x + 0.25,
    lambda x: 0.8 * x,
]


# ──────────────────────────────────────────────────────────────────────────
# Program representation (operands are pool indices)
#
#   ("scale",  a, x)               m[x]  *= a
#   ("axpy",   a, x, y)            m[y]  += a*m[x]
#   ("axpby",  a, x, b, y)         m[y]   = a*m[x] + b*m[y]
#   ("gemm",   a, A, B, b, C)      m[C]   = a*(A@B) + b*C
#   ("einsum", spec, ab, A, B, cpf, C)
#   ("beinsum",spec, ab, A, B, cpf, C)   on the rank-3 pool t
#   ("perm",   a, cpf, A, C)       m[C]   = a*m[A]^T + cpf*m[C]
#   ("symm",   A, B, C)            m[C]   = B^T @ A @ B
#   ("gemv",   a, A, x, b, y)      v[y]   = a*(m[A]@v[x]) + b*v[y]
#   ("ger",    a, x, y, A)         m[A]  += a*outer(v[x], v[y])
#   ("vscale", a, M, r0, r1, c0, c1)        m[M][r0:r1, c0:c1] *= a
#   ("vaxpy",  a, src, M, r0, r1, c0, c1)   m[M][r0:r1, c0:c1] += a*m[src]
#   ("loop",   n, body) / ("cond", flag, then, els)
#
# Opt-in (``rich_views=True`` on the generators only; existing shards draw
# neither, so their corpora are untouched):
#
#   ("vvscale", a, M, r0, r1, c0, c1, ir0, ir1, ic0, ic1)
#         a sub-block of a SUB-BLOCK: m[M][r0:r1, c0:c1][ir0:ir1, ic0:ic1] *= a
#   ("tvscale", a, M, r0, r1, c0, c1)
#         a sub-block of the TRANSPOSED view: m[M].T[r0:r1, c0:c1] *= a
#
# Both express an alias relation the graph can only describe by COMPOSING a
# chain of ``View`` descriptors (and, for tvscale, by mapping the box through a
# permutation). They exist for the alias-derivation equivalence shard, which
# needs a corpus where those two compositions actually occur.
#
# Opt-in (``rank_views=True``, likewise drawn by no existing shard):
#
#   ("ivscale", a, M, row, c0, c1)          m[M][row, c0:c1] *= a
#   ("ivaxpy",  a, src, M, row, c0, c1)     m[M][row, c0:c1] += a*v[src]
#   ("i3scale", a, T, k, r0, r1, c0, c1)    t[T][r0:r1, c0:c1, k] *= a
#   ("i3axpy",  a, src, T, k, r0, r1, c0, c1)
#                                           t[T][r0:r1, c0:c1, k] += a*m[src]
#
# These are RANK-REDUCING views: an axis is dropped rather than sliced, which is
# what ``cg.view_indexed`` exists for and what ``cg.view`` cannot express. They
# are here for the deferred shard, because the alias bug that shipped lived
# exactly where a rank-reducing view of a DEFERRED parent registered an address:
# two such parents of one shape, dropped at the same index, presented views on
# identical byte spans and the pointer derivation merged the parents. The
# matrix form drops the leading axis, so the view is strided; the rank-three
# form drops the trailing one, so it is contiguous.
# ──────────────────────────────────────────────────────────────────────────


def _scalar(rng):
    return float(np.round(rng.uniform(-1.0, 1.0), 4))


def _d(rng, dims=DIMS):
    return int(dims[int(rng.integers(0, len(dims)))])


def _pick(rng, by_shape, shape, exclude=()):
    cands = [i for i in by_shape.get(shape, ()) if i not in exclude]
    return int(rng.choice(cands)) if cands else None


def _pick_mat(rng, shape, exclude=()):
    return _pick(rng, MAT_BY_SHAPE, shape, exclude)


def _pick_vec(rng, length, exclude=()):
    return _pick(rng, VEC_BY_LEN, length, exclude)


def _pick_r3(rng, shape, exclude=()):
    return _pick(rng, R3_BY_SHAPE, shape, exclude)


def _fallback(rng):
    return ("scale", _scalar(rng), int(rng.integers(0, len(MAT_SHAPES))))


def _gen_block(rng, depth, max_stmts, rich_views=False, rank_views=False):
    stmts = []
    n = int(rng.integers(1, max_stmts + 1))
    for _ in range(n):
        roll = rng.random()
        if depth > 0 and roll < 0.18:
            cnt = int(rng.integers(1, 4))
            stmts.append(("loop", cnt, _gen_block(rng, depth - 1, max_stmts, rich_views, rank_views)))
        elif depth > 0 and roll < 0.30:
            flag = bool(rng.integers(0, 2))
            then = _gen_block(rng, depth - 1, max_stmts, rich_views, rank_views)
            els = _gen_block(rng, depth - 1, max_stmts, rich_views, rank_views)
            stmts.append(("cond", flag, then, els))
        else:
            stmts.append(_gen_primitive(rng, rich_views, rank_views))
    return stmts


def _gen_chained_view(rng):
    """A sub-block of a sub-block, or a sub-block of a transpose.

    Drawn only when a caller asks for ``rich_views``. Every existing shard's
    corpus is defined by its seed, so adding these to the common roll would
    renumber every program the whole suite has ever generated.
    """
    M = int(rng.integers(0, len(MAT_SHAPES)))
    R, C = MAT_SHAPES[M]
    a = _scalar(rng)
    if rng.random() < 0.5:
        # Outer block, then a block of it. The inner extents are relative to the
        # outer window, which is exactly the composition the derivation has to
        # get right.
        sr = int(rng.integers(1, R + 1))
        sc = int(rng.integers(1, C + 1))
        r0 = int(rng.integers(0, R - sr + 1))
        c0 = int(rng.integers(0, C - sc + 1))
        isr = int(rng.integers(1, sr + 1))
        isc = int(rng.integers(1, sc + 1))
        ir0 = int(rng.integers(0, sr - isr + 1))
        ic0 = int(rng.integers(0, sc - isc + 1))
        return ("vvscale", a, M, r0, r0 + sr, c0, c0 + sc, ir0, ir0 + isr, ic0, ic0 + isc)
    # The transpose has shape (C, R), so the block bounds swap roles.
    sr = int(rng.integers(1, C + 1))
    sc = int(rng.integers(1, R + 1))
    r0 = int(rng.integers(0, C - sr + 1))
    c0 = int(rng.integers(0, R - sc + 1))
    return ("tvscale", a, M, r0, r0 + sr, c0, c0 + sc)


def _gen_rank_view(rng):
    """A view with an axis DROPPED rather than sliced, at a non-zero offset.

    Drawn only when a caller asks for ``rank_views``, for the reason
    ``_gen_chained_view`` gives: an existing shard's corpus is defined by its
    seed, and adding to the common roll would renumber every program the suite
    has ever generated.
    """
    if rng.random() < 0.5:
        # A ROW of a matrix: axis 0 dropped, axis 1 sliced. Strided, since the
        # tensors are column major, so nothing here can be mistaken for a
        # contiguous block by a derivation that only looks at extents.
        M = int(rng.integers(0, len(MAT_SHAPES)))
        R, C = MAT_SHAPES[M]
        row = int(rng.integers(0, R))
        sc = int(rng.integers(1, C + 1))
        c0 = int(rng.integers(0, C - sc + 1))
        src = _pick_vec(rng, sc)
        if src is None or rng.random() < 0.5:
            return ("ivscale", _scalar(rng), M, row, c0, c0 + sc)
        return ("ivaxpy", _scalar(rng), src, M, row, c0, c0 + sc)
    # A SLAB of a rank-three tensor: the trailing axis dropped, the leading two
    # sliced. Contiguous, which is the other half of the pair.
    T = int(rng.integers(0, len(R3_SHAPES)))
    A, B, Cc = R3_SHAPES[T]
    k = int(rng.integers(0, Cc))
    sr = int(rng.integers(1, A + 1))
    r0 = int(rng.integers(0, A - sr + 1))
    sc = int(rng.integers(1, B + 1))
    c0 = int(rng.integers(0, B - sc + 1))
    src = _pick_mat(rng, (sr, sc))
    if src is None or rng.random() < 0.5:
        return ("i3scale", _scalar(rng), T, k, r0, r0 + sr, c0, c0 + sc)
    return ("i3axpy", _scalar(rng), src, T, k, r0, r0 + sr, c0, c0 + sc)


def _gen_primitive(rng, rich_views=False, rank_views=False):
    if rich_views and rng.random() < 0.25:
        return _gen_chained_view(rng)
    if rank_views and rng.random() < 0.30:
        return _gen_rank_view(rng)
    op = int(rng.integers(0, 14))
    a = _scalar(rng)
    if op == 13:  # gemm whose A operand is a *view* block of a larger matrix
        m, k, n = _d(rng), _d(rng), _d(rng)
        cand = [sh for sh in MAT_BY_SHAPE if sh[0] >= m and sh[1] >= k]
        if not cand:
            return _fallback(rng)
        R, Cc = tuple(cand[int(rng.integers(0, len(cand)))])
        M = _pick_mat(rng, (R, Cc))
        B = _pick_mat(rng, (k, n))
        if M is None or B is None:
            return _fallback(rng)
        C = _pick_mat(rng, (m, n), (M, B))
        if C is None:
            return _fallback(rng)
        ar0 = int(rng.integers(0, R - m + 1))
        ac0 = int(rng.integers(0, Cc - k + 1))
        return ("vgemm", a, M, ar0, ar0 + m, ac0, ac0 + k, B, float(rng.integers(0, 2)), C)
    if op == 12:  # element-wise unary transform (in-place, self-modifying)
        return ("etransform", int(rng.integers(0, len(ETRANSFORM_FNS))), int(rng.integers(0, len(MAT_SHAPES))))
    if op == 0:
        return ("scale", a, int(rng.integers(0, len(MAT_SHAPES))))
    if op in (1, 2):  # axpy / axpby, same shape, distinct
        sh = (_d(rng), _d(rng))
        x = _pick_mat(rng, sh)
        y = _pick_mat(rng, sh, (x,))
        if x is None or y is None:
            return _fallback(rng)
        return ("axpy", a, x, y) if op == 1 else ("axpby", a, x, _scalar(rng), y)
    if op == 3:  # gemm: A(m,k) B(k,n) C(m,n)
        m, k, n = _d(rng), _d(rng), _d(rng)
        A, B = _pick_mat(rng, (m, k)), _pick_mat(rng, (k, n))
        if A is None or B is None:
            return _fallback(rng)
        C = _pick_mat(rng, (m, n), (A, B))
        return ("gemm", a, A, B, float(rng.integers(0, 2)), C) if C is not None else _fallback(rng)
    if op == 4:  # einsum (mixed shapes)
        spec = _EINSUM_SPECS[int(rng.integers(0, len(_EINSUM_SPECS)))]
        _, shape_rule = EINSUM_PATTERNS[spec]
        sa, sb, sc = shape_rule(_d(rng), _d(rng), _d(rng))
        A, B = _pick_mat(rng, sa), _pick_mat(rng, sb)
        if A is None or B is None:
            return _fallback(rng)
        C = _pick_mat(rng, sc, (A, B))
        if C is None:
            return _fallback(rng)
        # conj_a/conj_b: meaningful for the complex dtypes, a no-op for the real
        # ones; exercises the native conj dispatch and the passes' conj guards.
        ca, cb = bool(rng.integers(0, 2)), bool(rng.integers(0, 2))
        return ("einsum", spec, a, A, B, float(rng.integers(0, 2)), C, ca, cb)
    if op == 5:  # perm (transpose): A(i,j) -> C(j,i)
        i, j = _d(rng), _d(rng)
        A = _pick_mat(rng, (i, j))
        C = _pick_mat(rng, (j, i), (A,))
        return ("perm", a, float(rng.integers(0, 2)), A, C) if A is not None and C is not None else _fallback(rng)
    if op == 6:  # symm: A(n,n) B(n,p) C(p,p)
        n, p = _d(rng), _d(rng)
        A, B = _pick_mat(rng, (n, n)), _pick_mat(rng, (n, p))
        if A is None or B is None:
            return _fallback(rng)
        C = _pick_mat(rng, (p, p), (A, B))
        return ("symm", A, B, C) if C is not None else _fallback(rng)
    if op == 7:  # gemv: A(m,n) x(n) y(m), x != y
        m, n = _d(rng), _d(rng)
        A = _pick_mat(rng, (m, n))
        x = _pick_vec(rng, n)
        y = _pick_vec(rng, m, (x,) if m == n else ())
        if A is None or x is None or y is None:
            return _fallback(rng)
        return ("gemv", a, A, x, float(rng.integers(0, 2)), y)
    if op == 8:  # ger: A(m,n) x(m) y(n)
        m, n = _d(rng), _d(rng)
        A, x, y = _pick_mat(rng, (m, n)), _pick_vec(rng, m), _pick_vec(rng, n)
        return ("ger", a, x, y, A) if A is not None and x is not None and y is not None else _fallback(rng)
    if op == 9:  # batched einsum on the rank-3 pool
        spec = _BEINSUM_SPECS[int(rng.integers(0, len(_BEINSUM_SPECS)))]
        _, shape_rule = BEINSUM_PATTERNS[spec]
        sa, sb, sc = shape_rule(_d(rng, R3_DIMS), _d(rng, R3_DIMS), _d(rng, R3_DIMS), _d(rng, R3_DIMS))
        A, B = _pick_r3(rng, sa), _pick_r3(rng, sb)
        if A is None or B is None:
            return _fallback(rng)
        C = _pick_r3(rng, sc, (A, B))
        if C is None:
            return _fallback(rng)
        ca, cb = bool(rng.integers(0, 2)), bool(rng.integers(0, 2))
        return ("beinsum", spec, a, A, B, float(rng.integers(0, 2)), C, ca, cb)
    if op == 10:  # vscale: scale a sub-block view of a matrix
        M = int(rng.integers(0, len(MAT_SHAPES)))
        R, C = MAT_SHAPES[M]
        sr = int(rng.integers(1, R + 1))
        sc = int(rng.integers(1, C + 1))
        r0 = int(rng.integers(0, R - sr + 1))
        c0 = int(rng.integers(0, C - sc + 1))
        return ("vscale", a, M, r0, r0 + sr, c0, c0 + sc)
    # op == 11: vaxpy: view(M)[block] += a*src, src a full matrix of the block shape
    M = int(rng.integers(0, len(MAT_SHAPES)))
    R, C = MAT_SHAPES[M]
    sr_choices = [d for d in DIMS if d <= R]
    sc_choices = [d for d in DIMS if d <= C]
    if not sr_choices or not sc_choices:
        return _fallback(rng)
    sr = int(rng.choice(sr_choices))
    sc = int(rng.choice(sc_choices))
    src = _pick_mat(rng, (sr, sc), (M,))
    if src is None:
        return _fallback(rng)
    r0 = int(rng.integers(0, R - sr + 1))
    c0 = int(rng.integers(0, C - sc + 1))
    return ("vaxpy", a, src, M, r0, r0 + sr, c0, c0 + sc)


# ──────────────────────────────────────────────────────────────────────────
# numpy oracle interpreter
# ──────────────────────────────────────────────────────────────────────────


def interp_np(stmts, m, v, t, dt=None):
    # When dt is given the oracle is kept in that precision: a Python-float
    # scalar times a float32 array would otherwise promote to float64, making
    # the oracle more accurate than the float32 graph and creating spurious
    # mismatches. Slice assignments (vscale/vaxpy) cast automatically into the
    # already-typed destination array, so only the rebinding ops need a cast.
    cast = (lambda x: np.asarray(x).astype(dt, copy=False)) if dt is not None else (lambda x: x)
    for s in stmts:
        k = s[0]
        if k == "scale":
            _, a, x = s
            m[x] = cast(m[x] * a)
        elif k == "axpy":
            _, a, x, y = s
            m[y] = cast(m[y] + a * m[x])
        elif k == "axpby":
            _, a, x, b, y = s
            m[y] = cast(a * m[x] + b * m[y])
        elif k == "gemm":
            _, a, A, B, b, C = s
            m[C] = cast(a * (m[A] @ m[B]) + b * m[C])
        elif k == "einsum":
            spec, ab, A, B, cpf, C = s[1:7]
            ca, cb = (s[7], s[8]) if len(s) > 7 else (False, False)
            opA = np.conj(m[A]) if ca else m[A]
            opB = np.conj(m[B]) if cb else m[B]
            m[C] = cast(ab * EINSUM_PATTERNS[spec][0](opA, opB) + cpf * m[C])
        elif k == "beinsum":
            spec, ab, A, B, cpf, C = s[1:7]
            ca, cb = (s[7], s[8]) if len(s) > 7 else (False, False)
            opA = np.conj(t[A]) if ca else t[A]
            opB = np.conj(t[B]) if cb else t[B]
            t[C] = cast(ab * BEINSUM_PATTERNS[spec][0](opA, opB) + cpf * t[C])
        elif k == "leinsum":
            spec, ab, Av, Bt, cpf, Cm = s[1:7]
            m[Cm] = cast(ab * LEINSUM_PATTERNS[spec][0](v[Av], t[Bt]) + cpf * m[Cm])
        elif k == "perm":
            _, a, cpf, A, C = s
            m[C] = cast(a * m[A].T + cpf * m[C])
        elif k == "symm":
            _, A, B, C = s
            m[C] = cast(m[B].T @ m[A] @ m[B])
        elif k == "gemv":
            _, a, A, x, b, y = s
            v[y] = cast(a * (m[A] @ v[x]) + b * v[y])
        elif k == "ger":
            _, a, x, y, A = s
            m[A] = cast(m[A] + a * np.outer(v[x], v[y]))
        elif k == "etransform":
            _, fn, M = s
            m[M] = cast(ETRANSFORM_FNS[fn](m[M]))
        elif k == "vgemm":
            _, a, M, r0, r1, c0, c1, B, b, C = s
            m[C] = cast(a * (m[M][r0:r1, c0:c1] @ m[B]) + b * m[C])
        elif k == "vscale":
            _, a, M, r0, r1, c0, c1 = s
            m[M][r0:r1, c0:c1] = m[M][r0:r1, c0:c1] * a
        elif k == "vaxpy":
            _, a, src, M, r0, r1, c0, c1 = s
            m[M][r0:r1, c0:c1] = m[M][r0:r1, c0:c1] + a * m[src]
        elif k == "vvscale":
            _, a, M, r0, r1, c0, c1, ir0, ir1, ic0, ic1 = s
            block = m[M][r0:r1, c0:c1]
            block[ir0:ir1, ic0:ic1] = block[ir0:ir1, ic0:ic1] * a
            m[M][r0:r1, c0:c1] = block
        elif k == "tvscale":
            _, a, M, r0, r1, c0, c1 = s
            m[M][c0:c1, r0:r1] = m[M][c0:c1, r0:r1] * a
        elif k == "ivscale":
            _, a, M, row, c0, c1 = s
            m[M][row, c0:c1] = m[M][row, c0:c1] * a
        elif k == "ivaxpy":
            _, a, src, M, row, c0, c1 = s
            m[M][row, c0:c1] = m[M][row, c0:c1] + a * v[src]
        elif k == "i3scale":
            _, a, T, kk, r0, r1, c0, c1 = s
            t[T][r0:r1, c0:c1, kk] = t[T][r0:r1, c0:c1, kk] * a
        elif k == "i3axpy":
            _, a, src, T, kk, r0, r1, c0, c1 = s
            t[T][r0:r1, c0:c1, kk] = t[T][r0:r1, c0:c1, kk] + a * m[src]
        elif k == "loop":
            _, n, body = s
            for _ in range(n):
                interp_np(body, m, v, t, dt)
        elif k == "cond":
            _, flag, then, els = s
            interp_np(then if flag else els, m, v, t, dt)
        else:  # pragma: no cover
            raise AssertionError(f"unknown opcode {k!r}")


# ──────────────────────────────────────────────────────────────────────────
# ComputeGraph builder
# ──────────────────────────────────────────────────────────────────────────


def _emit_primitive(s, m, v, t):
    k = s[0]
    if k == "scale":
        _, a, x = s
        einsums.linalg.scale(a, m[x])
    elif k == "axpy":
        _, a, x, y = s
        einsums.linalg.axpy(a, m[x], m[y])
    elif k == "axpby":
        _, a, x, b, y = s
        einsums.linalg.axpby(a, m[x], b, m[y])
    elif k == "gemm":
        _, a, A, B, b, C = s
        einsums.linalg.gemm(a, m[A], m[B], b, m[C])
    elif k == "einsum":
        spec, ab, A, B, cpf, C = s[1:7]
        ca, cb = (s[7], s[8]) if len(s) > 7 else (False, False)
        einsums.einsum(spec, m[C], m[A], m[B], c_pf=cpf, ab_pf=ab, conj_a=ca, conj_b=cb)
    elif k == "beinsum":
        spec, ab, A, B, cpf, C = s[1:7]
        ca, cb = (s[7], s[8]) if len(s) > 7 else (False, False)
        einsums.einsum(spec, t[C], t[A], t[B], c_pf=cpf, ab_pf=ab, conj_a=ca, conj_b=cb)
    elif k == "leinsum":
        spec, ab, Av, Bt, cpf, Cm = s[1:7]
        einsums.einsum(LEINSUM_PATTERNS[spec][1], m[Cm], v[Av], t[Bt], c_pf=cpf, ab_pf=ab)
    elif k == "perm":
        _, a, cpf, A, C = s
        einsums.permute("ij <- ji", m[C], m[A], c_pf=cpf, a_pf=a)
    elif k == "symm":
        _, A, B, C = s
        einsums.linalg.symm_gemm(m[A], m[B], m[C])
    elif k == "gemv":
        _, a, A, x, b, y = s
        einsums.linalg.gemv(a, m[A], v[x], b, v[y])
    elif k == "ger":
        _, a, x, y, A = s
        einsums.linalg.ger(a, v[x], v[y], m[A])
    elif k == "etransform":
        _, fn, M = s
        einsums.linalg.element_transform(m[M], ETRANSFORM_FNS[fn])
    elif k == "vgemm":
        _, a, M, r0, r1, c0, c1, B, b, C = s
        einsums.linalg.gemm(a, cg.view(m[M], [(r0, r1), (c0, c1)]), m[B], b, m[C])
    elif k == "vscale":
        _, a, M, r0, r1, c0, c1 = s
        einsums.linalg.scale(a, cg.view(m[M], [(r0, r1), (c0, c1)]))
    elif k == "vaxpy":
        _, a, src, M, r0, r1, c0, c1 = s
        einsums.linalg.axpy(a, m[src], cg.view(m[M], [(r0, r1), (c0, c1)]))
    elif k == "vvscale":
        _, a, M, r0, r1, c0, c1, ir0, ir1, ic0, ic1 = s
        outer = cg.view(m[M], [(r0, r1), (c0, c1)])
        einsums.linalg.scale(a, cg.view(outer, [(ir0, ir1), (ic0, ic1)]))
    elif k == "tvscale":
        _, a, M, r0, r1, c0, c1 = s
        einsums.linalg.scale(a, cg.view(cg.permute_view(m[M], [1, 0]), [(r0, r1), (c0, c1)]))
    elif k == "ivscale":
        _, a, M, row, c0, c1 = s
        einsums.linalg.scale(a, cg.view_indexed(m[M], [(2, row, 0), (1, c0, c1)]))
    elif k == "ivaxpy":
        _, a, src, M, row, c0, c1 = s
        einsums.linalg.axpy(a, v[src], cg.view_indexed(m[M], [(2, row, 0), (1, c0, c1)]))
    elif k == "i3scale":
        _, a, T, kk, r0, r1, c0, c1 = s
        einsums.linalg.scale(a, cg.view_indexed(t[T], [(1, r0, r1), (1, c0, c1), (2, kk, 0)]))
    elif k == "i3axpy":
        _, a, src, T, kk, r0, r1, c0, c1 = s
        einsums.linalg.axpy(a, m[src], cg.view_indexed(t[T], [(1, r0, r1), (1, c0, c1), (2, kk, 0)]))
    else:  # pragma: no cover
        raise AssertionError(f"not a primitive: {k!r}")


def build_cg(stmts, graph, m, v, t, tag):
    i = 0
    n = len(stmts)
    while i < n:
        run = []
        while i < n and stmts[i][0] not in ("loop", "cond"):
            run.append(stmts[i])
            i += 1
        if run:
            with cg.capture(graph):
                for s in run:
                    _emit_primitive(s, m, v, t)
        if i < n:
            s = stmts[i]
            i += 1
            if s[0] == "loop":
                _, cnt, body = s
                bg = graph.add_loop(f"{tag}_loop{i}", cnt, lambda it, c=cnt: it < c - 1)
                build_cg(body, bg, m, v, t, f"{tag}_l{i}")
            else:  # cond
                _, flag, then, els = s
                then_g, else_g = graph.add_conditional(f"{tag}_cond{i}", lambda f=flag: f)
                build_cg(then, then_g, m, v, t, f"{tag}_t{i}")
                build_cg(els, else_g, m, v, t, f"{tag}_e{i}")


def _make_pool(m_arrays, v_arrays, t_arrays, name):
    # The tensor dtype is inferred per seed array, so the same builder serves
    # both the real and complex suites (complex arrays → complex128 tensors).
    def mk(prefix, arrays):
        out = []
        for idx, arr in enumerate(arrays):
            tn = einsums.create_zero_tensor(f"{name}_{prefix}{idx}", list(arr.shape), dtype=str(arr.dtype))
            np.asarray(tn)[...] = arr
            out.append(tn)
        return out

    return mk("m", m_arrays), mk("v", v_arrays), mk("t", t_arrays)


# ──────────────────────────────────────────────────────────────────────────
# Trial driver
# ──────────────────────────────────────────────────────────────────────────


def _run_program(prog, m_arrays, v_arrays, t_arrays, name, optimize):
    mats, vecs, r3s = _make_pool(m_arrays, v_arrays, t_arrays, name)
    g = cg.Graph(name)
    build_cg(prog, g, mats, vecs, r3s, name)
    if optimize:
        g.apply(cg.default_pass_manager())
    g.execute()
    return ([np.asarray(x).copy() for x in mats],
            [np.asarray(x).copy() for x in vecs],
            [np.asarray(x).copy() for x in r3s])


def _usable(*pools, cap=1e8):
    """A trial is only meaningful if the oracle stayed numerically sane. Bigger
    programs with repeated accumulation can overflow to inf/NaN (or grow so large
    the dtype loses enough absolute resolution that the tolerance is dominated by
    fp noise); such cases test floating-point overflow, not pass soundness, so we
    skip them. The cap is tighter for single precision (see _DTYPE_CAP)."""
    for pool in pools:
        for arr in pool:
            if arr.size and (not np.all(np.isfinite(arr)) or np.max(np.abs(arr)) > cap):
                return False
    return True


def check_program(prog, m_arrays, v_arrays, t_arrays, label, dtype="float64"):
    rtol, atol = _DTYPE_TOL[dtype]
    cap = _DTYPE_CAP[dtype]
    dt = np.dtype(dtype)
    om = [a.copy() for a in m_arrays]
    ov = [a.copy() for a in v_arrays]
    ot = [a.copy() for a in t_arrays]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        interp_np(prog, om, ov, ot, dt)
    if not _usable(om, ov, ot, cap=cap):
        pytest.skip("oracle overflowed — numerically degenerate program")

    rm, rv, rt = _run_program(prog, m_arrays, v_arrays, t_arrays, f"{label}_raw", optimize=False)
    pm, pv, pt = _run_program(prog, m_arrays, v_arrays, t_arrays, f"{label}_opt", optimize=True)

    def _cmp(stage, got, oracle, kind):
        for idx in range(len(oracle)):
            if not np.allclose(got[idx], oracle[idx], rtol=rtol, atol=atol):
                raise AssertionError(
                    f"{stage} disagrees with oracle on {kind}{idx} (dtype={dtype})"
                    f"{' (a pass miscompiled)' if stage == 'OPTIMIZED' else ''}\n"
                    f"program={prog!r}\ngot=\n{got[idx]}\noracle=\n{oracle[idx]}"
                )

    for stage, (gm, gv, gt) in (("RAW", (rm, rv, rt)), ("OPTIMIZED", (pm, pv, pt))):
        _cmp(stage, gm, om, "m")
        _cmp(stage, gv, ov, "v")
        _cmp(stage, gt, ot, "t")


# ──────────────────────────────────────────────────────────────────────────
# Cross-executor differential
#
# check_program uses the default (Sequential) executor, so it validates the
# passes. The parallel executors, OpenMP (task-based) and Dataflow (TaskPool
# continuations), instead schedule independent nodes concurrently from the
# graph's dependency edges (RAW/WAR/WAW, plus effective_io for control-flow
# subtrees). A missing or wrong edge there does not show up under Sequential at
# all; it surfaces as a divergent result here (and, run under a sanitizer, as a
# data race). So we replay each random program through all three executors, raw
# and optimized, and demand every run agrees with the numpy oracle.
# ──────────────────────────────────────────────────────────────────────────

_CROSS_EXECUTORS = [
    ("Sequential", cg.SequentialExecutor),
    ("OpenMP", cg.OpenMPExecutor),
    ("Dataflow", cg.DataflowExecutor),
]


def _run_program_exec(prog, m_arrays, v_arrays, t_arrays, name, optimize, exec_cls):
    mats, vecs, r3s = _make_pool(m_arrays, v_arrays, t_arrays, name)
    g = cg.Graph(name)
    build_cg(prog, g, mats, vecs, r3s, name)
    if optimize:
        g.apply(cg.default_pass_manager())
    g.execute(exec_cls())
    return ([np.asarray(x).copy() for x in mats],
            [np.asarray(x).copy() for x in vecs],
            [np.asarray(x).copy() for x in r3s])


def check_program_cross_executor(prog, m_arrays, v_arrays, t_arrays, label, dtype="float64"):
    rtol, atol = _DTYPE_TOL[dtype]
    cap = _DTYPE_CAP[dtype]
    dt = np.dtype(dtype)
    om = [a.copy() for a in m_arrays]
    ov = [a.copy() for a in v_arrays]
    ot = [a.copy() for a in t_arrays]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        interp_np(prog, om, ov, ot, dt)
    if not _usable(om, ov, ot, cap=cap):
        pytest.skip("oracle overflowed — numerically degenerate program")

    def _cmp(stage, got, oracle, kind):
        for idx in range(len(oracle)):
            if not np.allclose(got[idx], oracle[idx], rtol=rtol, atol=atol):
                raise AssertionError(
                    f"{stage} disagrees with oracle on {kind}{idx}\n"
                    f"program={prog!r}\ngot=\n{got[idx]}\noracle=\n{oracle[idx]}"
                )

    for ex_name, exec_cls in _CROSS_EXECUTORS:
        for optimize in (False, True):
            stage = f"{ex_name}/{'opt' if optimize else 'raw'}"
            tag = f"{label}_{ex_name}_{'opt' if optimize else 'raw'}"
            gm, gv, gt = _run_program_exec(prog, m_arrays, v_arrays, t_arrays, tag, optimize, exec_cls)
            _cmp(stage, gm, om, "m")
            _cmp(stage, gv, ov, "v")
            _cmp(stage, gt, ot, "t")


def _seed_arrays(rng, dtype="float64"):
    is_complex = dtype in ("complex64", "complex128")

    def gen(sh):
        a = rng.standard_normal(sh)
        if is_complex:
            a = a + 1j * rng.standard_normal(sh)
        return a.astype(np.dtype(dtype))

    m = [gen(sh) for sh in MAT_SHAPES]
    v = [gen((L,)) for L in VEC_LENS]
    t = [gen(sh) for sh in R3_SHAPES]
    return m, v, t


def _square_seed_arrays(rng, n_mats=4, n_vecs=3, n=3, n_r3=2):
    """Square N×N matrices, length-N vectors, and a couple N×N×2 rank-3
    tensors, for the hand-written regressions that index a small fixed pool."""
    m = [rng.standard_normal((n, n)) for _ in range(n_mats)]
    v = [rng.standard_normal((n,)) for _ in range(n_vecs)]
    t = [rng.standard_normal((n, n, 2)) for _ in range(n_r3)]
    return m, v, t


_SQ = "ij <- ik ; kj"  # square-friendly einsum (A@B); any pattern works on n×n


def _sq_pool(rng, count, n=3):
    return [rng.standard_normal((n, n)) for _ in range(count)]

# Passes exposed to Python that are individually sound and (should be)
# order-independent for correctness on eager tensors. A random permutation that
# miscompiles is either a real bug or an undocumented ordering constraint.
_SAFE_PASSES = [
    "ScaleAbsorption", "ElementWiseFusion",
    "ConstantFolding", "CSE", "DeadNodeElimination", "LoopInvariantHoisting",
    "SymmetryPropagation", "MemoryPlanning", "InplaceOptimization", "Reorder",
]

# ──────────────────────────────────────────────────────────────────────────
# Region identity round trip
#
# The differential shards ask "does the graph agree with numpy". This one asks
# a question no other shard covers and that no tolerance applies to: does a
# program survive being RAISED into the algebraic IR and LOWERED back?
#
# `lower_region` rebuilds every node from the expression rather than reusing
# what it raised, so anything the IR fails to carry - a conjugation flag, a
# destination prefactor, an index list a pass rewrote through the live block
# rather than the snapshot - comes back as a different number rather than as a
# missing field nobody notices.
#
# BITWISE, not allclose. A lowered node runs the same kernel over the same
# values in the same order as the node it replaced, so anything short of
# identical is a defect and not a floating-point question. That also makes this
# shard immune to the overflow skip the tolerance-based ones need: inf == inf
# and a NaN pool is compared with equal_nan, so a numerically degenerate
# program is still a perfectly good round-trip test.
#
# `_REGION_STATS` counts what actually formed a region, so a shard can assert
# the corpus has not quietly degraded into programs with nothing raisable in
# them.
# ──────────────────────────────────────────────────────────────────────────

_REGION_STATS = {"attempted": 0, "with_regions": 0, "rewritten": 0}


def _run_program_region_identity(prog, m_arrays, v_arrays, t_arrays, name):
    """Build, raise every region and lower it unchanged, execute."""
    _REGION_STATS["attempted"] += 1

    mats, vecs, r3s = _make_pool(m_arrays, v_arrays, t_arrays, name)
    g = cg.Graph(name)
    build_cg(prog, g, mats, vecs, r3s, name)

    identity = cg.RegionIdentity()
    pm = cg.PassManager()
    pm.add(identity)
    pm.run(g)

    if identity.regions_formed:
        _REGION_STATS["with_regions"] += 1
    if identity.regions_rewritten:
        _REGION_STATS["rewritten"] += 1

    g.execute()
    return ([np.asarray(x).copy() for x in mats],
            [np.asarray(x).copy() for x in vecs],
            [np.asarray(x).copy() for x in r3s]), identity


def check_program_region_identity(prog, m_arrays, v_arrays, t_arrays, label, dtype="float64"):
    """Raise every region, lower it unchanged, demand the same bits."""
    raw = _run_program(prog, m_arrays, v_arrays, t_arrays, f"{label}_raw", optimize=False)
    (identity_pools, identity) = _run_program_region_identity(
        prog, m_arrays, v_arrays, t_arrays, f"{label}_id")

    for kind, got, expected in zip("mvt", identity_pools, raw):
        for idx in range(len(expected)):
            # equal_nan, because a degenerate program is still a valid
            # round-trip case and skipping it would throw away exactly the
            # inputs most likely to expose a dropped prefactor.
            if not np.array_equal(got[idx], expected[idx], equal_nan=True):
                raise AssertionError(
                    f"a raise/lower round trip changed {kind}{idx} (dtype={dtype})\n"
                    f"program={prog!r}\n"
                    f"regions formed={identity.regions_formed} "
                    f"rewritten={identity.regions_rewritten}\n"
                    f"captured=\n{expected[idx]}\nafter round trip=\n{got[idx]}"
                )


# ──────────────────────────────────────────────────────────────────────────
# Save / load round trip
#
# The differential above asks "does the graph agree with numpy". This one asks
# a narrower question that no other shard covers: does a program SURVIVE being
# written to a file and read back, with its answer unchanged?
#
# The skip rule mirrors _usable's. A random program routinely holds something
# the IR cannot yet carry - a View node, a gemv, an anonymous element
# transform - and `Graph.serializability_report` is what names those; a save
# refuses exactly that set, so the refusal IS the skip signal and there is no
# second list to keep in step. `_ROUNDTRIP_STATS` counts what actually round
# tripped so a shard can assert the corpus has not quietly degraded to all
# skips.
# ──────────────────────────────────────────────────────────────────────────

_ROUNDTRIP_STATS = {"attempted": 0, "round_tripped": 0}


def _run_program_roundtrip(prog, m_arrays, v_arrays, t_arrays, name):
    """Build, save, load, bind by manifest name, execute.

    Returns the loaded run's pools, or None when the program holds something a
    file cannot carry (the save refuses it)."""
    _ROUNDTRIP_STATS["attempted"] += 1

    mats, vecs, r3s = _make_pool(m_arrays, v_arrays, t_arrays, name)
    g = cg.Graph(name)
    build_cg(prog, g, mats, vecs, r3s, name)

    with tempfile.TemporaryDirectory() as scratch:
        path = os.path.join(scratch, f"{name}.eig.json")
        try:
            cg.save_graph(g, path)
        except Exception:
            return None
        loaded = cg.load_graph(path)

    # A SECOND pool under the same names, seeded identically. Same names because
    # the manifest binds by name and `_make_pool` derives a tensor's name from
    # the pool's; a different prefix would leave every slot unbound.
    b_mats, b_vecs, b_r3s = _make_pool(m_arrays, v_arrays, t_arrays, name)
    by_name = {}
    for prefix, pool in (("m", b_mats), ("v", b_vecs), ("t", b_r3s)):
        for idx, tensor in enumerate(pool):
            by_name[f"{name}_{prefix}{idx}"] = tensor

    for key in loaded.manifest_names():
        if key not in by_name:
            # A slot the pool cannot supply would silently execute against the
            # loader's placeholder storage, which is a green test that proved
            # nothing. Refuse the trial instead.
            return None
        loaded.bind(key, by_name[key])

    loaded.execute()
    _ROUNDTRIP_STATS["round_tripped"] += 1
    return ([np.asarray(x).copy() for x in b_mats],
            [np.asarray(x).copy() for x in b_vecs],
            [np.asarray(x).copy() for x in b_r3s])


def check_program_roundtrip(prog, m_arrays, v_arrays, t_arrays, label, dtype="float64"):
    """A saved-and-reloaded program computes what the original computed."""
    cap = _DTYPE_CAP[dtype]
    dt = np.dtype(dtype)
    om = [a.copy() for a in m_arrays]
    ov = [a.copy() for a in v_arrays]
    ot = [a.copy() for a in t_arrays]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        interp_np(prog, om, ov, ot, dt)
    if not _usable(om, ov, ot, cap=cap):
        pytest.skip("oracle overflowed - numerically degenerate program")

    got = _run_program_roundtrip(prog, m_arrays, v_arrays, t_arrays, f"{label}_rt")
    if got is None:
        pytest.skip("program holds a node the IR cannot carry yet")

    _assert_pools(got, (om, ov, ot), prog, "ROUND-TRIPPED")


def _oracle(prog, m, v, t, runs=1):
    om = [a.copy() for a in m]
    ov = [a.copy() for a in v]
    ot = [a.copy() for a in t]
    # Overflow/NaN in a degenerate program is expected and handled by _usable;
    # don't spam warnings for it.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for _ in range(runs):
            interp_np(prog, om, ov, ot)
    return om, ov, ot


def _build(prog, m, v, t, name):
    mats, vecs, r3 = _make_pool(m, v, t, name)
    g = cg.Graph(name)
    build_cg(prog, g, mats, vecs, r3, name)
    return g, mats, vecs, r3


def _assert_pools(got, oracle, prog, label, extra=""):
    for kind, gs, os in zip("mvt", got, oracle):
        for idx in range(len(os)):
            if not np.allclose(gs[idx], os[idx], rtol=RTOL, atol=ATOL):
                raise AssertionError(
                    f"{label} disagrees with oracle on {kind}{idx}{extra}\n"
                    f"program={prog!r}\ngot=\n{gs[idx]}\noracle=\n{os[idx]}"
                )


# ──────────────────────────────────────────────────────────────────────────
# Single-pass tier measurement
#
# The shards above ask "does the optimized graph agree with numpy". Sorting a
# pass into one of the Part 5.1 tiers asks something narrower and quantitative:
# run ONE pass alone, and report how far its answer sits from the same graph
# with no passes at all.
#
# Two properties of what follows are the point rather than conveniences.
#
# It reports whether the pass FIRED. A single-pass differential that finds no
# difference has shown the pass faithful, or has shown that it never ran, and
# those are opposite conclusions drawn from identical output. Every candidate
# below carries a counter for exactly this reason.
#
# And it MEASURES rather than asserts. A pass that is not bitwise over this
# corpus is not thereby broken, it is re-associating, and the number it deviates
# by is the bound its tier test should pin. An assertion of bit equality here
# would answer the classification question by assuming it, which is the mistake
# the GCC leg already caught once.
#
# Non-finite positions are held apart from the arithmetic. A bitwise-exact pass
# may legitimately differ there, since removing a multiplication also removes
# its ability to produce a NaN, so folding that into a max-deviation number
# would report inf for a pass that is exact everywhere it computes anything.
# ──────────────────────────────────────────────────────────────────────────

#: Every structural-algebraic pass a tier classification has to cover, mapped to
#: the attribute naming how many times it fired. Adding a pass here is what puts
#: it in the measurement, so the table is the list of what has been classified.
TIER_CANDIDATES = {
    "DeltaElimination": "num_eliminated",
    "ConstantFolding": "num_folded",
    "ScaleAbsorption": "num_absorbed",
    "PermuteFusion": "num_rewrites",
    "CSE": "num_eliminated",
    "DeadNodeElimination": "num_eliminated",
    "SymmetrizedAccumulation": "num_rewritten",
    "ElementWiseFusion": "num_fused",
    "LinearCombinationContractionFolding": "num_eliminated",
    "DistributiveFactoring": "num_eliminated",
    "LoopInvariantHoisting": "num_hoisted",
    "ContractionPlanning": "chains_restructured",
    "LayoutAssignment": "num_relaid_out",
    "MultiTermFactorization": "num_shared",
}

#: Per-pass tallies across a shard, so a run can say how much evidence it
#: actually gathered instead of only that nothing failed.
_TIER_STATS = {}


def _tier_stat(pass_name):
    return _TIER_STATS.setdefault(
        pass_name, {"attempted": 0, "usable": 0, "fired": 0, "bitwise": 0})


def _written_pool_indices(g, counts):
    """Which pool buffers @p g still WRITES, per kind.

    Needed because a pass may legitimately ORPHAN a buffer: PermuteFusion folds
    a transpose into its consumer and the transposed temporary is then never
    written, exactly as CSE leaves an eliminated duplicate's storage alone. Such
    a buffer still holds its seed value, and comparing it would report the seed
    against the computed answer as if the pass had corrupted something. It is
    not part of what the graph computes, so it is not part of the measurement.

    Read off the graph rather than declared per generator, because a hand-kept
    list of dead buffers is one more table to fall out of step with the passes.
    """
    doc = json.loads(g.to_json())
    names = {t["id"]: t.get("name", "") for t in doc.get("tensors", [])}

    written_names = set()

    def walk(nodes):
        for n in nodes or []:
            for oid in (n.get("outputs") or []):
                nm = names.get(oid)
                if nm:
                    written_names.add(nm)
            # Loop bodies and conditional branches carry their own node lists.
            for key in ("body", "then_body", "else_body", "nodes"):
                sub = n.get(key)
                if isinstance(sub, dict):
                    walk(sub.get("nodes"))
                elif isinstance(sub, list):
                    walk(sub)

    walk(doc.get("nodes"))
    return {kind: {i for i in range(count)
                   if any(nm.endswith(f"_{kind}{i}") for nm in written_names)}
            for kind, count in counts.items()}


def _build_for_measurement(prog, g, mats, vecs, r3s, name):
    """Lay a trial's work into @p g.

    ``prog`` is either a statement list, which goes through @ref build_cg as
    every other shard's does, or a CALLABLE ``(graph, m, v, t, name)``. The
    callable form exists because several passes cannot be provoked by a
    statement list at all: their guards want graph-owned scratch, or a delta, or
    a loop body holding an invariant, and the pool the statement format draws
    from is user tensors and nothing else. An opportunity generator for such a
    pass builds its own graph and is handed the same seeded pool.
    """
    if callable(prog):
        prog(g, mats, vecs, r3s, name)
    else:
        build_cg(prog, g, mats, vecs, r3s, name)


def _run_program_single_pass(prog, m_arrays, v_arrays, t_arrays, name, pass_name):
    """Build, apply exactly one pass, execute.

    Returns ``(pools, fired_count, written)``, where ``written`` names the pool
    buffers the rewritten graph still produces.
    """
    mats, vecs, r3s = _make_pool(m_arrays, v_arrays, t_arrays, name)
    g = cg.Graph(name)
    _build_for_measurement(prog, g, mats, vecs, r3s, name)

    pass_obj = getattr(cg, pass_name)()
    # A pass that is off by default still has to be measured, and the classification is about what
    # the pass DOES to the numbers rather than about whether a default pipeline reaches it. The
    # programmatic switch exists precisely so a driver need not mutate process-global config.
    if hasattr(pass_obj, "set_search_enabled"):
        pass_obj.set_search_enabled(True)
    pm = cg.PassManager()
    pm.add(pass_obj)
    # Materialization is correctness-enabling rather than an optimization: it
    # allocates the deferred tensors a generator declared and does nothing else,
    # and a graph holding one cannot execute without it. It runs on BOTH sides of
    # the comparison, so it contributes nothing to the gap being measured, and it
    # is inert for every generator that allocates its own scratch eagerly.
    pm.add(cg.Materialization())
    pm.run(g)

    counts = {"m": len(mats), "v": len(vecs), "t": len(r3s)}
    written = _written_pool_indices(g, counts)

    g.execute()

    fired = int(getattr(pass_obj, TIER_CANDIDATES[pass_name]))
    return ([np.asarray(x).copy() for x in mats],
            [np.asarray(x).copy() for x in vecs],
            [np.asarray(x).copy() for x in r3s]), fired, written


def _run_program_raw_written(prog, m_arrays, v_arrays, t_arrays, name):
    """The unoptimized run, plus which pool buffers it writes."""
    mats, vecs, r3s = _make_pool(m_arrays, v_arrays, t_arrays, name)
    g = cg.Graph(name)
    _build_for_measurement(prog, g, mats, vecs, r3s, name)
    counts = {"m": len(mats), "v": len(vecs), "t": len(r3s)}
    written = _written_pool_indices(g, counts)
    # The other half of the pair; see the note in _run_program_single_pass.
    pm = cg.PassManager()
    pm.add(cg.Materialization())
    pm.run(g)
    g.execute()
    return ([np.asarray(x).copy() for x in mats],
            [np.asarray(x).copy() for x in vecs],
            [np.asarray(x).copy() for x in r3s]), written


def _gap(got, expected):
    """Absolute, relative, ULP and norm-relative gap over FINITE positions.

    Returns ``(max_abs, max_rel, max_ulp, sq_err, sq_ref, nonfinite_diff)``.
    The two squared sums are returned rather than a ratio so a caller can
    accumulate a norm-relative figure across a whole pool.
    """
    finite = np.isfinite(got) & np.isfinite(expected)
    # Disagreeing about WHERE the non-finite values sit is a real difference and
    # a categorical one, so it is reported as a flag rather than folded into a
    # magnitude that would come out inf.
    nonfinite_diff = bool(np.any(np.isfinite(got) != np.isfinite(expected)))
    if not np.any(finite):
        return 0.0, 0.0, 0.0, 0.0, 0.0, nonfinite_diff

    g = got[finite].astype(np.float64)
    e = expected[finite].astype(np.float64)
    diff = np.abs(g - e)
    max_abs = float(diff.max())

    nz = np.abs(e) > 0
    max_rel = float((diff[nz] / np.abs(e[nz])).max()) if np.any(nz) else 0.0
    # np.spacing of a zero reference is a denormal, which would turn any gap at
    # all into ~1e300 ulps and say nothing; those positions are the abs figure's.
    max_ulp = float((diff[nz] / np.spacing(np.abs(e[nz]))).max()) if np.any(nz) else 0.0

    return max_abs, max_rel, max_ulp, float((diff ** 2).sum()), float((e ** 2).sum()), nonfinite_diff


def measure_program_single_pass(prog, m_arrays, v_arrays, t_arrays, label,
                                pass_name, dtype="float64"):
    """Run one pass alone against the unoptimized graph and report the gap.

    Returns ``None`` when the trial is not usable (the oracle overflowed, which
    tests floating point rather than the pass). Otherwise a dict:

    ``fired``          how many rewrites the pass reported making
    ``bitwise``        every finite element identical to the last bit
    ``max_abs``        largest absolute gap over finite positions
    ``max_rel``        largest element-wise relative gap
    ``max_ulp``        largest gap in units in the last place
    ``norm_rel``       Frobenius-norm relative gap over the whole pool
    ``nonfinite_diff`` the two runs disagree about where non-finite values are
    ``orphaned``       buffers excluded because the pass folded their producer
                       away, so the rewritten graph no longer writes them

    Deliberately not an assertion: which tier a pass belongs to is what this is
    evidence for, so the caller decides what the number means.
    """
    stat = _tier_stat(pass_name)
    stat["attempted"] += 1

    cap = _DTYPE_CAP[dtype]
    raw, raw_written = _run_program_raw_written(
        prog, m_arrays, v_arrays, t_arrays, f"{label}_raw")
    if not _usable(*raw, cap=cap):
        return None
    stat["usable"] += 1

    pools, fired, opt_written = _run_program_single_pass(
        prog, m_arrays, v_arrays, t_arrays, f"{label}_{pass_name}", pass_name)

    max_abs = max_rel = max_ulp = 0.0
    sq_err = sq_ref = 0.0
    nonfinite_diff = False
    bitwise = True
    orphaned = 0

    for kind, got, expected in zip("mvt", pools, raw):
        for idx in range(len(expected)):
            # A buffer the raw run produced and the rewritten one no longer
            # does is orphaned, not wrong: it holds its seed value because the
            # pass folded its producer away. See _written_pool_indices.
            if idx in raw_written[kind] and idx not in opt_written[kind]:
                orphaned += 1
                continue
            if not np.array_equal(got[idx], expected[idx], equal_nan=True):
                bitwise = False
            a, r, u, se, sr, nf = _gap(got[idx], expected[idx])
            max_abs = max(max_abs, a)
            max_rel = max(max_rel, r)
            max_ulp = max(max_ulp, u)
            sq_err += se
            sq_ref += sr
            nonfinite_diff = nonfinite_diff or nf

    if fired:
        stat["fired"] += 1
    if bitwise:
        stat["bitwise"] += 1

    return {
        "pass_name": pass_name,
        "fired": fired,
        "bitwise": bitwise,
        "max_abs": max_abs,
        "max_rel": max_rel,
        "max_ulp": max_ulp,
        "norm_rel": (sq_err ** 0.5 / sq_ref ** 0.5) if sq_ref > 0 else 0.0,
        "nonfinite_diff": nonfinite_diff,
        "orphaned": orphaned,
        "program": prog,
    }


__all__ = [
    'fuzz_seeds',
    'DIMS',
    'R3_DIMS',
    'COPIES',
    '_DTYPE_TOL',
    '_DTYPE_CAP',
    'RTOL',
    'ATOL',
    'MAT_SHAPES',
    'VEC_LENS',
    'R3_SHAPES',
    'MAT_BY_SHAPE',
    'VEC_BY_LEN',
    'R3_BY_SHAPE',
    'EINSUM_PATTERNS',
    '_EINSUM_SPECS',
    'BEINSUM_PATTERNS',
    '_BEINSUM_SPECS',
    'LEINSUM_PATTERNS',
    'ETRANSFORM_FNS',
    '_scalar',
    '_d',
    '_pick',
    '_pick_mat',
    '_pick_vec',
    '_pick_r3',
    '_fallback',
    '_gen_block',
    '_gen_primitive',
    '_gen_chained_view',
    '_gen_rank_view',
    'interp_np',
    '_emit_primitive',
    'build_cg',
    '_make_pool',
    '_run_program',
    '_usable',
    'check_program',
    '_CROSS_EXECUTORS',
    '_run_program_exec',
    'check_program_cross_executor',
    '_REGION_STATS',
    '_run_program_region_identity',
    'check_program_region_identity',
    '_ROUNDTRIP_STATS',
    '_run_program_roundtrip',
    'check_program_roundtrip',
    '_seed_arrays',
    '_square_seed_arrays',
    '_SQ',
    '_sq_pool',
    '_oracle',
    '_build',
    '_assert_pools',
    '_SAFE_PASSES',
    '_G',
    'TIER_CANDIDATES',
    '_TIER_STATS',
    '_tier_stat',
    '_build_for_measurement',
    '_written_pool_indices',
    '_run_program_single_pass',
    '_run_program_raw_written',
    '_gap',
    'measure_program_single_pass',
]
