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
from _permutation_operators import P_SHAPES, apply_operator, operator_prefix, shaped_operator
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

# Graph-owned scratch matrices, drawn only by the ``rich_ops`` generator. They
# sit in the matrix pool AFTER every user matrix, at SCRATCH_BASE + j, so a
# statement names one exactly as it names a user matrix; the rich arms create
# them with ``Graph.create_zero_tensor`` (intermediate) and never compare them,
# since a pass may legitimately free or elide a graph-owned buffer. They exist
# because PermuteFusion, CSE and DeadNodeElimination decline anything the caller
# holds, and a pool of nothing but user tensors never gave them a candidate.
SCRATCH_SHAPES = [(r, c) for r in DIMS for c in DIMS for _ in range(2)]
SCRATCH_BASE = len(MAT_SHAPES)
SCRATCH_BY_SHAPE: dict[tuple[int, int], list[int]] = {}
for _idx, _sh in enumerate(SCRATCH_SHAPES):
    SCRATCH_BY_SHAPE.setdefault(_sh, []).append(SCRATCH_BASE + _idx)

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
#
# Opt-in (``rich_ops=True``, drawn only by the optimization-level and rich
# random-pipeline shards):
#
#   ("xeinsum", spec, op, ab, Aref, Bref, cpf, Cref, ca, cb)
#         C = ab * op(A (x) B) + cpf * C, spec a key of EINSUM_PATTERNS or
#         BEINSUM_PATTERNS and op a permutation operator from
#         _permutation_operators (or None). Each operand is a reference:
#           ("m", i)                       the whole matrix m[i]
#           ("mv", i, r0, r1, c0, c1)      the block m[i][r0:r1, c0:c1]
#           ("mt", i)                      the transpose view of m[i]
#           ("t", i)                       the rank-3 tensor t[i]
#         so one opcode carries an operator, a view read, and a view WRITE.
#
# Opt-in (``all_passes=True``, drawn only by the all-passes shuffled shard):
#
#   ("aperm", kind, op, a, src, cpf, C)
#         C = a * op(src) + cpf * C, a permute under a permutation operator on
#         the matrix pool (kind "m", letters i,j) or the rank-3 pool (kind "t",
#         letters i,j,k)
#   ("dot", kind, out, A, B)                v[out] = [sum(A * B)]
#   ("ddiv", a, A, D, cpf, C)               m[C] = a * m[A] / m[D] + cpf * m[C]
#
# plus the statements above, over the extra pool slots ``ALLPASS_*`` describes.
#
# A ``rich_ops`` program may also name graph-owned scratch: matrix slots at
# SCRATCH_BASE and above, which only ``_build_with_scratch`` creates. Such a
# program goes through the rich arms, never through ``check_program``.
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


def _gen_block(rng, depth, max_stmts, rich_views=False, rank_views=False, rich_ops=False, all_passes=False):
    stmts = []
    n = int(rng.integers(1, max_stmts + 1))
    for _ in range(n):
        roll = rng.random()
        if depth > 0 and roll < 0.18:
            cnt = int(rng.integers(1, 4))
            stmts.append(("loop", cnt, _gen_block(rng, depth - 1, max_stmts, rich_views, rank_views, rich_ops,
                                                  all_passes)))
        elif depth > 0 and roll < 0.30:
            flag = bool(rng.integers(0, 2))
            then = _gen_block(rng, depth - 1, max_stmts, rich_views, rank_views, rich_ops, all_passes)
            els = _gen_block(rng, depth - 1, max_stmts, rich_views, rank_views, rich_ops, all_passes)
            stmts.append(("cond", flag, then, els))
        else:
            prim = _gen_primitive(rng, rich_views, rank_views, rich_ops, all_passes)
            # Only the rich_ops and all_passes draws return a run of statements
            # (a batchable cluster, a motif), so the other corpora never reach
            # the extend.
            if isinstance(prim, list):
                stmts.extend(prim)
            else:
                stmts.append(prim)
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


def _view_ref(rng, shape, exclude=()):
    """A block of some matrix at least @p shape, at a random offset.

    The parent is drawn from every matrix big enough rather than from the
    matrices of exactly that shape, so a full-extent view (the block IS the
    parent) and a strict sub-block are both reachable.
    """
    r, c = shape
    cands = [i for i, (R, C) in enumerate(MAT_SHAPES) if R >= r and C >= c and i not in exclude]
    if not cands:
        return None
    M = int(rng.choice(cands))
    R, C = MAT_SHAPES[M]
    r0 = int(rng.integers(0, R - r + 1))
    c0 = int(rng.integers(0, C - c + 1))
    return ("mv", M, r0, r0 + r, c0, c0 + c)


def _transposed_ref(rng, shape, exclude=()):
    """The transpose view of a whole matrix whose transpose has @p shape.

    A different alias relation from a block: the view covers every element of
    its parent but maps its axes crosswise, so a pass that compares a view with
    its parent by extents alone sees two tensors of different shapes and no
    overlap at all.
    """
    M = _pick_mat(rng, (shape[1], shape[0]), exclude)
    return ("mt", M) if M is not None else None


def _gen_rank2_operator(rng):
    # P(x0/x1) is the only table row that fits a two-letter output. The letter
    # order is drawn so both spellings, P(i/j) and P(j/i), reach the parser.
    chosen = ["i", "j"] if rng.random() < 0.5 else ["j", "i"]
    return ([[chosen[0]], [chosen[1]]], 0, chosen)


def _gen_rich_op(rng):
    """An einsum carrying a permutation operator, or a view read or written.

    Drawn only when a caller asks for ``rich_ops``, for the reason
    ``_gen_chained_view`` gives. These are the shapes the curated default
    pipeline sees in real code and the shuffled pipelines never did: an
    operator is what AntisymmetrizerExpansion expands and what CSE, PermuteFusion
    and GEMMBatching must each refuse to treat as the plain product, and a view
    operand is where alias-aware scheduling and every pass that moves a node
    past another have to agree on what overlaps.
    """
    kind = int(rng.choice(8, p=[0.15, 0.12, 0.14, 0.14, 0.08, 0.08, 0.12, 0.17]))
    if kind == 6:
        return _gen_batch_cluster(rng)
    if kind == 7:
        return _gen_scratch_motif(rng)
    a = _scalar(rng)
    cpf = float(rng.integers(0, 2))
    ca, cb = bool(rng.integers(0, 2)), bool(rng.integers(0, 2))

    if kind == 1:  # rank-3 batched einsum under an operator over C's letters
        spec = _BEINSUM_SPECS[int(rng.integers(0, len(_BEINSUM_SPECS)))]
        _, shape_rule = BEINSUM_PATTERNS[spec]
        c_idx = ["i", "j", "b"]
        shape_index = int(rng.integers(0, 4))  # every row with at most three letters
        groups_shape, _ = P_SHAPES[shape_index]
        n = sum(len(g) for g in groups_shape)
        chosen = [str(x) for x in rng.permutation(c_idx)[:n]]
        # Forced equal rather than filtered for, as the einsum fuzzer does: an
        # operator permutes axes into each other, so they must share an extent,
        # and waiting for independent draws to agree reaches the three-letter
        # rows almost never.
        ext = {x: _d(rng, R3_DIMS) for x in ("i", "k", "j", "b")}
        common = _d(rng, R3_DIMS)
        for x in chosen:
            ext[x] = common
        op = ([[chosen[i] for i in g] for g in groups_shape], shape_index, chosen)
        sa, sb, sc = shape_rule(ext["i"], ext["k"], ext["j"], ext["b"])
        A, B = _pick_r3(rng, sa), _pick_r3(rng, sb)
        if A is None or B is None:
            return _fallback(rng)
        C = _pick_r3(rng, sc, (A, B))
        if C is None:
            return _fallback(rng)
        return ("xeinsum", spec, op, a, ("t", A), ("t", B), cpf, ("t", C), ca, cb)

    spec = _EINSUM_SPECS[int(rng.integers(0, len(_EINSUM_SPECS)))]
    _, shape_rule = EINSUM_PATTERNS[spec]
    if kind == 0:
        want_op = True
    else:
        want_op = rng.random() < 0.5
    ni, nk, nj = _d(rng), _d(rng), _d(rng)
    if want_op:
        nj = ni
    op = _gen_rank2_operator(rng) if want_op else None
    sa, sb, sc = shape_rule(ni, nk, nj)

    if kind == 0:  # operator, plain operands
        A, B = _pick_mat(rng, sa), _pick_mat(rng, sb)
        if A is None or B is None:
            return _fallback(rng)
        C = _pick_mat(rng, sc, (A, B))
        if C is None:
            return _fallback(rng)
        return ("xeinsum", spec, op, a, ("m", A), ("m", B), cpf, ("m", C), ca, cb)

    if kind == 2:  # one operand READ through a view
        if rng.random() < 0.5:
            Aref = _view_ref(rng, sa)
            B = _pick_mat(rng, sb)
            if Aref is None or B is None:
                return _fallback(rng)
            Bref = ("m", B)
        else:
            Bref = _view_ref(rng, sb)
            A = _pick_mat(rng, sa)
            if Bref is None or A is None:
                return _fallback(rng)
            Aref = ("m", A)
        C = _pick_mat(rng, sc, (Aref[1], Bref[1]))
        if C is None:
            return _fallback(rng)
        return ("xeinsum", spec, op, a, Aref, Bref, cpf, ("m", C), ca, cb)

    if kind == 4:  # one operand READ through a transposed view of a whole matrix
        if rng.random() < 0.5:
            Aref = _transposed_ref(rng, sa)
            B = _pick_mat(rng, sb)
            if Aref is None or B is None:
                return _fallback(rng)
            Bref = ("m", B)
        else:
            Bref = _transposed_ref(rng, sb)
            A = _pick_mat(rng, sa)
            if Bref is None or A is None:
                return _fallback(rng)
            Aref = ("m", A)
        C = _pick_mat(rng, sc, (Aref[1], Bref[1]))
        if C is None:
            return _fallback(rng)
        return ("xeinsum", spec, op, a, Aref, Bref, cpf, ("m", C), ca, cb)

    # kinds 3 and 5: the output WRITTEN through a view, a block for 3 and a
    # transpose for 5. Neither input may live in the view's parent: an output
    # overlapping an input is rejected unless the index lists match, and that
    # rejection is a different test.
    A, B = _pick_mat(rng, sa), _pick_mat(rng, sb)
    if A is None or B is None:
        return _fallback(rng)
    Cref = _view_ref(rng, sc, (A, B)) if kind == 3 else _transposed_ref(rng, sc, (A, B))
    if Cref is None:
        return _fallback(rng)
    return ("xeinsum", spec, op, a, ("m", A), ("m", B), cpf, Cref, ca, cb)


def _gen_batch_cluster(rng):
    """Two to four same-shaped contractions into distinct outputs, back to back.

    GEMMBatching only fires on a run of independent contractions that share a
    spec and a shape, which the one-statement draws above almost never line up.
    Some members carry an operator and some do not, because a batched GEMM
    computes the unpermuted product and a pass that batched an operator member
    alongside a plain one returned the plain product for both.
    """
    spec = _EINSUM_SPECS[int(rng.integers(0, len(_EINSUM_SPECS)))]
    _, shape_rule = EINSUM_PATTERNS[spec]
    n = _d(rng)
    sa, sb, sc = shape_rule(n, _d(rng), n)
    # One prefactor pair and one conjugation choice for the whole cluster: the
    # pass batches only bit-equal alpha and beta and skips a conjugated member,
    # so independent draws would decline nearly every cluster.
    ab, cpf = _scalar(rng), float(rng.integers(0, 2))
    ca = cb = bool(rng.random() < 0.2)
    members = []
    written = set()
    read = set()
    for _ in range(int(rng.integers(2, 5))):
        A = _pick_mat(rng, sa, tuple(written))
        B = _pick_mat(rng, sb, tuple(written))
        if A is None or B is None:
            break
        C = _pick_mat(rng, sc, tuple(written | read | {A, B}))
        if C is None:
            break
        read |= {A, B}
        written.add(C)
        op = _gen_rank2_operator(rng) if rng.random() < 0.4 else None
        members.append(("xeinsum", spec, op, ab, ("m", A), ("m", B), cpf, ("m", C), ca, cb))
    return members if members else _fallback(rng)


def _gen_scratch_motif(rng):
    """A short run over graph-owned scratch that an O1 pass is built to rewrite.

    Three shapes, one per pass family: a pure transpose into scratch read by an
    einsum (PermuteFusion), the same contraction computed twice into two
    scratch buffers and both read (CSE, then DeadNodeElimination), and a
    contraction into scratch that is scaled or transformed before it is read
    (ScaleAbsorption, ElementWiseFusion). The consumer einsum may carry an
    operator, because a fused or deduplicated operand under an antisymmetrizer
    is the combination the default order never produces on its own.
    """
    spec = _EINSUM_SPECS[int(rng.integers(0, len(_EINSUM_SPECS)))]
    _, shape_rule = EINSUM_PATTERNS[spec]
    want_op = rng.random() < 0.4
    ni, nk, nj = _d(rng), _d(rng), _d(rng)
    if want_op:
        nj = ni
    op = _gen_rank2_operator(rng) if want_op else None
    sa, sb, sc = shape_rule(ni, nk, nj)
    ab, cpf = _scalar(rng), float(rng.integers(0, 2))
    ca, cb = bool(rng.integers(0, 2)), bool(rng.integers(0, 2))
    roll = int(rng.integers(0, 3))

    if roll == 0:
        # The transpose must be pure (alpha 1, beta 0) and its output read once,
        # or the pass declines; both are what this draws.
        if rng.random() < 0.5:
            S = _pick(rng, SCRATCH_BY_SHAPE, sa)
            src = _pick_mat(rng, (sa[1], sa[0]))
            B = _pick_mat(rng, sb)
            if src is None or B is None:
                return _fallback(rng)
            C = _pick_mat(rng, sc, (src, B))
            Aref, Bref = ("m", S), ("m", B)
        else:
            S = _pick(rng, SCRATCH_BY_SHAPE, sb)
            src = _pick_mat(rng, (sb[1], sb[0]))
            A = _pick_mat(rng, sa)
            if src is None or A is None:
                return _fallback(rng)
            C = _pick_mat(rng, sc, (src, A))
            Aref, Bref = ("m", A), ("m", S)
        if C is None:
            return _fallback(rng)
        return [("perm", 1.0, 0.0, src, S),
                ("xeinsum", spec, op, ab, Aref, Bref, cpf, ("m", C), ca, cb)]

    A, B = _pick_mat(rng, sa), _pick_mat(rng, sb)
    if A is None or B is None:
        return _fallback(rng)
    S1 = _pick(rng, SCRATCH_BY_SHAPE, sc)
    C = _pick_mat(rng, sc, (A, B))
    if C is None:
        return _fallback(rng)
    first = ("xeinsum", spec, op, ab, ("m", A), ("m", B), 0.0, ("m", S1), ca, cb)

    if roll == 1:
        S2 = _pick(rng, SCRATCH_BY_SHAPE, sc, (S1,))
        C2 = _pick_mat(rng, sc, (A, B, C))
        out = [first, first[:7] + (("m", S2),) + first[8:], ("axpy", _scalar(rng), S2, C)]
        if C2 is not None and rng.random() < 0.5:
            out.append(("axpy", _scalar(rng), S1, C2))
        return out

    middle = (("scale", _scalar(rng), S1) if rng.random() < 0.5
              else ("etransform", int(rng.integers(0, len(ETRANSFORM_FNS))), S1))
    return [first, middle, ("axpby", _scalar(rng), S1, _scalar(rng), C)]


def _scratch_indices(stmts):
    """Every scratch slot @p stmts names, at any nesting depth.

    Relies on pool indices being the only integers in a statement that can
    reach SCRATCH_BASE: extents, offsets, loop counts, table rows and flags are
    all bounded by the four-wide dims well below it.
    """
    found = set()

    def walk(x):
        if isinstance(x, (list, tuple)):
            for y in x:
                walk(y)
        elif isinstance(x, int) and not isinstance(x, bool) and x >= SCRATCH_BASE:
            found.add(x)

    walk(stmts)
    return found


def rich_op_census(stmts, out=None):
    """How many operator einsums, view reads and view writes @p stmts holds.

    Recurses into loop bodies and both branches of a conditional, because the
    guards that read this want to know what the pipeline was SHOWN, and a
    statement in an untaken branch is still a node every pass walks.
    """
    if out is None:
        out = {"statements": 0, "operator": 0, "view_read": 0, "view_write": 0, "scratch": 0}
    for s in stmts:
        if s[0] == "loop":
            rich_op_census(s[2], out)
            continue
        if s[0] == "cond":
            rich_op_census(s[2], out)
            rich_op_census(s[3], out)
            continue
        out["statements"] += 1
        out["scratch"] += bool(_scratch_indices([s]))
        if s[0] == "xeinsum":
            _, _, op, _, Aref, Bref, _, Cref, _, _ = s
            out["operator"] += op is not None
            out["view_read"] += Aref[0] in ("mv", "mt") or Bref[0] in ("mv", "mt")
            out["view_write"] += Cref[0] in ("mv", "mt")
        elif s[0] in ("vgemm",):
            out["view_read"] += 1
        elif s[0] in ("vscale", "vaxpy"):
            out["view_write"] += 1
    return out


def _gen_primitive(rng, rich_views=False, rank_views=False, rich_ops=False, all_passes=False):
    # Checked first and drawn only when asked for, so no other corpus consumes
    # an extra random number and every existing seed keeps its program.
    if all_passes and rng.random() < 0.45:
        return _gen_all_passes_motif(rng)
    if rich_views and rng.random() < 0.25:
        return _gen_chained_view(rng)
    if rank_views and rng.random() < 0.30:
        return _gen_rank_view(rng)
    if rich_ops and rng.random() < 0.40:
        return _gen_rich_op(rng)
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


def _ref_read(ref, m, t):
    if ref[0] == "t":
        return t[ref[1]]
    if ref[0] == "m":
        return m[ref[1]]
    if ref[0] == "mt":
        return m[ref[1]].T
    _, M, r0, r1, c0, c1 = ref
    return m[M][r0:r1, c0:c1]


def _ref_write(ref, m, t, value, cast):
    # A whole-tensor write REBINDS the slot, so it needs the cast the rebinding
    # ops above apply; a block write assigns into the typed parent, which casts.
    if ref[0] == "t":
        t[ref[1]] = cast(value)
    elif ref[0] == "m":
        m[ref[1]] = cast(value)
    elif ref[0] == "mt":
        m[ref[1]] = cast(np.asarray(value).T)
    else:
        _, M, r0, r1, c0, c1 = ref
        m[M][r0:r1, c0:c1] = value


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
            fn = EINSUM_PATTERNS[spec][0] if spec in EINSUM_PATTERNS else _MATMUL_SPELLINGS[spec]
            m[C] = cast(ab * fn(opA, opB) + cpf * m[C])
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
        elif k == "xeinsum":
            _, spec, op, ab, Aref, Bref, cpf, Cref, ca, cb = s
            patterns = BEINSUM_PATTERNS if Cref[0] == "t" else EINSUM_PATTERNS
            opA = _ref_read(Aref, m, t)
            opB = _ref_read(Bref, m, t)
            opA = np.conj(opA) if ca else opA
            opB = np.conj(opB) if cb else opB
            c_idx = list(spec.split("<-")[0].strip())
            base = apply_operator(op, c_idx, patterns[spec][0](opA, opB))
            _ref_write(Cref, m, t, ab * base + cpf * _ref_read(Cref, m, t), cast)
        elif k == "aperm":
            _, kind, op, a, src, cpf, C = s
            pool = m if kind == "m" else t
            letters = _APERM_LETTERS[kind]
            pool[C] = cast(a * apply_operator(op, letters, pool[src]) + cpf * pool[C])
        elif k == "dot":
            _, kind, out, A, B = s
            pool = m if kind == "m" else t
            v[out] = cast(np.array([np.sum(pool[A] * pool[B])]))
        elif k == "ddiv":
            _, a, A, D, cpf, C = s
            m[C] = cast(a * (m[A] / m[D]) + cpf * m[C])
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
    elif k == "xeinsum":
        _, spec, op, ab, Aref, Bref, cpf, Cref, ca, cb = s
        lhs, rhs = spec.split("<-")
        full = f"{lhs.strip()} <- {operator_prefix(op)}{rhs.strip()}"
        einsums.einsum(full, _ref_tensor(Cref, m, t), _ref_tensor(Aref, m, t), _ref_tensor(Bref, m, t),
                       c_pf=cpf, ab_pf=ab, conj_a=ca, conj_b=cb)
    elif k == "aperm":
        _, kind, op, a, src, cpf, C = s
        pool = m if kind == "m" else t
        letters = ",".join(_APERM_LETTERS[kind])
        einsums.permute(f"{letters} <- {operator_prefix(op)}{letters}", pool[C], pool[src], c_pf=cpf, a_pf=a)
    elif k == "dot":
        _, kind, out, A, B = s
        pool = m if kind == "m" else t
        einsums.linalg.dot(v[out], pool[A], pool[B])
    elif k == "ddiv":
        _, a, A, D, cpf, C = s
        einsums.linalg.direct_division(a, m[A], m[D], cpf, m[C])
    else:  # pragma: no cover
        raise AssertionError(f"not a primitive: {k!r}")


def _ref_tensor(ref, m, t):
    if ref[0] == "t":
        return t[ref[1]]
    if ref[0] == "m":
        return m[ref[1]]
    if ref[0] == "mt":
        return cg.permute_view(m[ref[1]], [1, 0])
    _, M, r0, r1, c0, c1 = ref
    return cg.view(m[M], [(r0, r1), (c0, c1)])


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


def region_pass_manager():
    """The structural-algebraic phase, with the search on.

    ``default_pass_manager`` does not run it: the search is off by default
    because its runtime is a function of how many candidates a graph offers.
    These passes are the ones that descend into loop bodies and conditional
    branches, so a control-flow corpus is the only place their descent is
    exercised against a numeric oracle rather than against a counter.
    """
    mtf = cg.MultiTermFactorization()
    mtf.set_search_enabled(True)
    pm = cg.PassManager()
    # No wall-clock allowance, for the reason the region shard's own pipeline
    # gives: a search that runs out of one emits a valid but different graph, so
    # what a slower machine would be comparing is not what a faster one compared.
    pm.set_optimizer_budget(0)
    for p in (cg.DeltaElimination(), cg.LinearCombinationContractionFolding(),
              cg.DistributiveFactoring(), mtf, cg.LayoutAssignment(),
              cg.ContractionPlanning(), cg.Materialization()):
        pm.add(p)
    return pm


def check_program_region_pipeline(prog, m_arrays, v_arrays, t_arrays, label, dtype="float64"):
    """The region pipeline over a program full of loops and conditionals.

    The same oracle comparison ``check_program`` makes, with the structural
    phase in place of the default pipeline. What it is for is the descent: a
    region rewrite now raises a loop body, and a rewrite that is correct on a
    flat run and wrong inside a body shows up here as a wrong number rather than
    as a counter nobody reads.
    """
    rtol, atol = _DTYPE_TOL[dtype]
    cap = _DTYPE_CAP[dtype]
    dt = np.dtype(dtype)
    om = [a.copy() for a in m_arrays]
    ov = [a.copy() for a in v_arrays]
    ot = [a.copy() for a in t_arrays]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        interp_np(prog, om, ov, ot, dt)
    if not _usable(om, ov, ot, cap=cap):
        pytest.skip("oracle overflowed - numerically degenerate program")

    mats, vecs, r3s = _make_pool(m_arrays, v_arrays, t_arrays, f"{label}_reg")
    g = cg.Graph(f"{label}_reg")
    build_cg(prog, g, mats, vecs, r3s, f"{label}_reg")
    g.apply(region_pass_manager())
    g.execute()

    for kind, got, oracle in (("m", mats, om), ("v", vecs, ov), ("t", r3s, ot)):
        for idx in range(len(oracle)):
            value = np.asarray(got[idx])
            if not np.allclose(value, oracle[idx], rtol=rtol, atol=atol):
                raise AssertionError(
                    f"REGION PIPELINE disagrees with oracle on {kind}{idx} (dtype={dtype})\n"
                    f"program={prog!r}\ngot=\n{value}\noracle=\n{oracle[idx]}")


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
# Optimization levels
#
# ``check_program`` runs ``default_pass_manager``, which is the O2 list built
# by a different entry point. What a user who writes ``graph.optimize(O1)``
# gets is ``PassManager::create_for``: a separately written list that only
# claims to match the head of the default one. Nothing else runs it against a
# numeric oracle, so a pass that is sound only because a later pass cleans up
# after it would be wrong at O1 and green everywhere else.
#
# ``_OPT_LEVEL_STATS`` counts, per level, the trials that ran and the ones the
# pipeline actually changed, so a shard can assert the levels did something:
# an O1 that silently stopped modifying anything passes every comparison.
# ──────────────────────────────────────────────────────────────────────────

OPT_LEVELS = {"O1": einsums._core.OptLevel.O1, "O2": einsums._core.OptLevel.O2}

_OPT_LEVEL_STATS = {name: {"attempted": 0, "modified": 0} for name in OPT_LEVELS}


def _oracle_typed(prog, m_arrays, v_arrays, t_arrays, dtype, runs=1):
    """The numpy oracle in @p dtype, or a pytest skip if it overflowed.

    The scratch slots start at zero, as ``create_zero_tensor`` does, and are
    dropped from what is returned: only the caller's tensors are observable.
    """
    dt = np.dtype(dtype)
    om = [a.copy() for a in m_arrays] + [np.zeros(sh, dtype=dt) for sh in SCRATCH_SHAPES]
    ov = [a.copy() for a in v_arrays]
    ot = [a.copy() for a in t_arrays]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for _ in range(runs):
            interp_np(prog, om, ov, ot, dt)
    om = om[:len(m_arrays)]
    if not _usable(om, ov, ot, cap=_DTYPE_CAP[dtype]):
        pytest.skip("oracle overflowed - numerically degenerate program")
    return om, ov, ot


def _build_with_scratch(prog, m_arrays, v_arrays, t_arrays, name):
    """``_build``, plus a graph-owned intermediate for every scratch slot @p prog names.

    Returns the matrix list truncated to the caller's tensors as well, which is
    what gets compared.
    """
    mats, vecs, r3s = _make_pool(m_arrays, v_arrays, t_arrays, name)
    g = cg.Graph(name)
    used = _scratch_indices(prog)
    dtype = str(m_arrays[0].dtype)
    full = list(mats)
    for j, sh in enumerate(SCRATCH_SHAPES):
        slot = SCRATCH_BASE + j
        if slot not in used:
            full.append(None)
        elif j % 2 == 0:
            # The first copy of each shape is DEFERRED: Materialization gives
            # it storage and FreeInsertion, InplaceOptimization and
            # MemoryPlanning only ever act on a buffer whose lifetime the graph
            # owns from allocation to free, which an eager tensor is not.
            full.append(g.declare_zero_tensor(f"{name}_s{j}", list(sh), intermediate=True, dtype=dtype))
        else:
            full.append(g.create_zero_tensor(f"{name}_s{j}", list(sh), intermediate=True, dtype=dtype))
    build_cg(prog, g, full, vecs, r3s, name)
    return g, mats, vecs, r3s


def _assert_pools_typed(got, oracle, prog, stage, dtype, extra=""):
    rtol, atol = _DTYPE_TOL[dtype]
    for kind, gs, os_ in zip("mvt", got, oracle):
        for idx in range(len(os_)):
            if not np.allclose(gs[idx], os_[idx], rtol=rtol, atol=atol):
                raise AssertionError(
                    f"{stage} disagrees with oracle on {kind}{idx} (dtype={dtype}){extra}\n"
                    f"program={prog!r}\ngot=\n{gs[idx]}\noracle=\n{os_[idx]}")


def check_program_opt_level(prog, m_arrays, v_arrays, t_arrays, label, level, dtype="float64", runs=1):
    """``graph.optimize(level)``, executed @p runs times, against the oracle.

    ``runs`` greater than one replays the optimized graph, which is where a
    pass that inserted a Free or reused a buffer on the assumption of a single
    execution shows up.
    """
    oracle = _oracle_typed(prog, m_arrays, v_arrays, t_arrays, dtype, runs)
    g, mats, vecs, r3s = _build_with_scratch(prog, m_arrays, v_arrays, t_arrays, f"{label}_{level}")
    stat = _OPT_LEVEL_STATS[level]
    stat["attempted"] += 1
    if g.optimize(OPT_LEVELS[level]):
        stat["modified"] += 1
    for _ in range(runs):
        g.execute()
    got = ([np.asarray(x).copy() for x in mats],
           [np.asarray(x).copy() for x in vecs],
           [np.asarray(x).copy() for x in r3s])
    _assert_pools_typed(got, oracle, prog, f"optimize({level})", dtype, extra=f" runs={runs}")


# ──────────────────────────────────────────────────────────────────────────
# Shuffled pipelines over operators and views
#
# The random-pipeline shards shuffle ``_SAFE_PASSES`` over the base corpus,
# which never carries a permutation operator and never writes an einsum or a
# transpose through a view. The passes that most need an order-independence
# check on exactly those shapes are the ones the list left out:
# AntisymmetrizerExpansion (rewrites an operator into its terms, so every pass
# on either side of it sees a different graph), PermuteFusion (folds a
# transpose into its consumer, which is wrong through an aliasing view) and
# GEMMBatching (must not batch an operator member as if it were the plain
# product). They go into this arm's list, and the programs come from
# ``rich_ops``.
#
# GEMMBatching has no Python class, so it runs as the default pipeline with
# every other pass switched off. ``_DEFAULT_PASS_NAMES`` is the list that makes
# that possible, and the run checks it: a default pass missing from it would
# run alongside GEMMBatching, and the check names that as stale rather than
# letting it widen the pipeline under test.
# ──────────────────────────────────────────────────────────────────────────

# Materialization is here because the rich corpus declares deferred scratch,
# which cannot execute without it; shuffling it tests that no pass assumes
# storage exists yet. FreeInsertion is left out: its 1 MiB floor is not
# settable from Python and no tensor this pool holds reaches it, so it would be
# shuffled and never fire.
_RICH_SAFE_PASSES = _SAFE_PASSES + ["PermuteFusion", "AntisymmetrizerExpansion", "GEMMBatching",
                                     "Materialization"]

#: Every pass ``populate_default`` can build, in any build configuration.
#: Names that this build does not have are harmless: they match nothing.
_DEFAULT_PASS_NAMES = (
    "ProvenancePropagation", "TiledExpansion", "DeltaElimination",
    "AntisymmetryDetection", "AntisymmetrizerLinearity", "AntisymmetryInference",
    "AntisymmetrizerFolding", "AntisymmetrizerExpansion", "ConstantFolding", "ScaleAbsorption",
    "PermuteFusion", "CSE", "DeadNodeElimination", "SymmetrizedAccumulation", "ElementWiseFusion",
    "LinearCombinationContractionFolding", "DistributiveFactoring", "LoopInvariantHoisting",
    "ScratchPrivatization", "MultiTermFactorization", "LayoutAssignment", "ContractionPlanning",
    "GEMMBatching", "Reorder", "IOPrefetch", "DistributionPlanning", "Materialization",
    "SymmetryPropagation", "SpacePropagation", "CrossSpaceValidation", "ScalingAnalysis",
    "StreamContractionFusion", "GPUPlacement", "TransferInsertion", "TransferElimination",
    "GPUDiagnostics", "StreamAssignment", "InputSlicing", "SUMMAExpansion",
    "CommunicationInsertion", "CommunicationElimination", "CommunicationScheduling",
    "InplaceOptimization", "FreeInsertion", "MemoryPlanning",
)

#: Per pass: how many shuffled trials ran it and how many times it changed the
#: graph. A pass in the list that never fires has been shuffled, not tested.
_RICH_PIPELINE_STATS = {name: {"ran": 0, "modified": 0} for name in _RICH_SAFE_PASSES}


def _isolated_default_pass(pass_name):
    """The default pipeline with every pass but @p pass_name switched off."""
    pm = cg.PassManager()
    pm.populate_default()
    for other in _DEFAULT_PASS_NAMES:
        if other != pass_name:
            pm.disable(other)
    return pm


def _apply_one_pass(g, pass_name):
    """Run one pass of the shuffled list on @p g; True if it changed the graph."""
    if hasattr(_G, pass_name):
        pm = cg.PassManager()
        pm.add(getattr(_G, pass_name)())
        return g.apply(pm)
    pm = _isolated_default_pass(pass_name)
    modified = g.apply(pm)
    skipped = pm.disabled_passes()
    if pass_name in skipped or len(skipped) != pm.size - 1:
        raise AssertionError(
            f"isolating {pass_name} left {pm.size - len(skipped)} of {pm.size} default passes on; "
            f"_DEFAULT_PASS_NAMES is stale (skipped: {skipped})")
    return modified


def check_program_rich_pipeline(prog, m_arrays, v_arrays, t_arrays, label, rng, dtype="float64", runs=1):
    """A random order of ``_RICH_SAFE_PASSES``, applied one pass at a time.

    One pass per PassManager so each pass's own verdict lands in
    ``_RICH_PIPELINE_STATS``; the sequence of runs is the same pipeline a single
    manager holding the whole order would be.
    """
    oracle = _oracle_typed(prog, m_arrays, v_arrays, t_arrays, dtype, runs)
    order = list(_RICH_SAFE_PASSES)
    rng.shuffle(order)
    g, mats, vecs, r3s = _build_with_scratch(prog, m_arrays, v_arrays, t_arrays, label)
    for name in order:
        _RICH_PIPELINE_STATS[name]["ran"] += 1
        if _apply_one_pass(g, name):
            _RICH_PIPELINE_STATS[name]["modified"] += 1
    # Every pipeline above O0 ends in Materialization, which is correctness-enabling rather than
    # an optimization: a structural pass that runs after the shuffled one (AntisymmetrizerExpansion
    # does) leaves deferred scratch only a later Materialization gives storage. Shuffled position
    # is still exercised above; this closing one is not counted.
    _apply_one_pass(g, "Materialization")
    for _ in range(runs):
        g.execute()
    got = ([np.asarray(x).copy() for x in mats],
           [np.asarray(x).copy() for x in vecs],
           [np.asarray(x).copy() for x in r3s])
    _assert_pools_typed(got, oracle, prog, "RICH-RANDOM-PIPELINE", dtype, extra=f"  order={order} runs={runs}")

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
    # A tier measurement compares what one pass did to the numbers, so it has to
    # be the pass deciding and not the clock: a search cut off by a wall-clock
    # allowance emits a different graph and the tier would be classifying that.
    pm.set_optimizer_budget(0)
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


# ──────────────────────────────────────────────────────────────────────────
# Every default pass in a shuffled order
#
# ``check_program_rich_pipeline`` shuffles the passes that act on operators,
# views and scratch. The default pipeline holds eleven more that no shuffled
# arm ever ran: the delta and provenance pair, the four antisymmetrizer passes,
# SymmetrizedAccumulation, the two sum-factoring folds, and the two passes that
# re-bracket products. Each is a recognizer for a shape the random corpus never
# draws (a tensor declared an identity, a bound antisymmetric input, a
# transposed accumulation, a chain large enough for a cost model to have an
# opinion), so shuffling them over that corpus would shuffle them without ever
# running them.
#
# So the arm carries its own motifs, one per recognizer, and a pool extended by
# the tensors those motifs need. The extension lives AFTER every slot the other
# corpora know about, so no existing program or seed changes:
#
#   matrix slots ALLPASS_XM_BASE + j, described by ALLPASS_XM_SLOTS[j]
#   rank-3 slots ALLPASS_XT_BASE + j, described by ALLPASS_XT_SLOTS[j]
#   vector slots ALLPASS_XV_BASE + j, described by ALLPASS_XV_SLOTS[j]
#
# Each slot has a role. Roles whose data the declarations and the
# data-reading passes rely on, which no generator writes except to rescale an
# ``anti`` slot, which keeps it antisymmetric:
#
#   delta        an identity, declared one with ``cg.annotate(tag="identity")``
#   anti, anti3  data antisymmetric in the axes an operator permutes (axes 1
#                and 2 for rank three, the group of P(i/jk))
#   sym          a symmetric, strictly positive denominator
#   occ, zero    space annotations that make their shared letter range over
#                two disjoint spaces; ``zero`` holds zeros, so the data honours
#                what the annotation claims
#
# Source roles, random data the motifs read. The linearity, fold, multi-term
# and distributive motifs now and then write one of theirs between the
# statements a pass rewrites, which is a write that pass must see:
#
#   linsrc       the sources of the linearity and fold motifs
#   mtfsrc       the factors of the multi-term motif
#   cpsrc        the factors of the chain motif
#   dfsrc        the operands of the distributive motif
#   lccft, lccfv the rank-3 and vector operands of the 2J-K motif (lccft also
#                serves as a rank-3 fold source)
#
# Roles only one motif writes: ``mtfout``, ``cpout``, ``dfout``, ``lccfout``
# and ``symout``, each that motif's result. Graph-owned roles, created only
# when a program names them: ``scratch``, ``symacc``, ``foldscr``, ``cpscr``
# and ``deltascr``.
#
# The dedicated roles keep each motif's shape intact wherever it lands: a
# recognizer finds the single writer or the untouched operand it looks for,
# unless the motif itself drew the write that must make it decline (see the
# motif docstrings for which writes each one draws).
# ──────────────────────────────────────────────────────────────────────────

#: The two extents of the re-bracketing motifs: a product through the long one
#: is what ContractionPlanning and MultiTermFactorization re-bracket.
ALLPASS_BS, ALLPASS_BL = 3, 12

ALLPASS_XM_BASE = SCRATCH_BASE + len(SCRATCH_SHAPES)
ALLPASS_XM_SLOTS: list[tuple[tuple[int, ...], str]] = (
    [((n, n), "delta") for n in DIMS for _ in range(2)]
    + [((n, n), "anti") for n in DIMS[1:] for _ in range(2)]
    + [((n, n), "sym") for n in DIMS for _ in range(2)]
    + [((n, n), "occ") for n in DIMS[1:]]
    + [((n, n), "zero") for n in DIMS[1:]]
    + [((n, n), "linsrc") for n in DIMS[1:] for _ in range(2)]
    + [((n, n), "symout") for n in DIMS for _ in range(2)]
    + [((a, b), "dfsrc") for a in (2, 3) for b in (2, 3) for _ in range(4)]
    + [((a, b), "dfout") for a in (2, 3) for b in (2, 3) for _ in range(2)]
    + [((n, n), "lccfout") for n in R3_DIMS for _ in range(2)]
    + [((ALLPASS_BS, ALLPASS_BL), "mtfsrc") for _ in range(3)]
    + [((ALLPASS_BL, ALLPASS_BS), "mtfsrc")]
    + [((ALLPASS_BS, ALLPASS_BL), "mtfout") for _ in range(2)]
    + [((ALLPASS_BL, ALLPASS_BS), "cpsrc") for _ in range(2)]
    + [((ALLPASS_BS, ALLPASS_BL), "cpsrc")]
    + [((ALLPASS_BL, ALLPASS_BS), "cpout")]
    + [((n, n), "scratch") for n in DIMS for _ in range(4)]
    + [((n, n), "symacc") for n in DIMS for _ in range(2)]
    + [((n, n), "deltascr") for n in DIMS for _ in range(2)]
    + [((n, n), "foldscr") for n in DIMS[1:] for _ in range(12)]
    + [((ALLPASS_BL, ALLPASS_BL), "scratch") for _ in range(2)]
    + [((ALLPASS_BL, ALLPASS_BL), "cpscr") for _ in range(4)]
)

ALLPASS_XT_BASE = len(R3_SHAPES)
ALLPASS_XT_SLOTS: list[tuple[tuple[int, ...], str]] = (
    [((k, n, n), "lccft") for k in R3_DIMS for n in R3_DIMS]
    + [((n, n, n), "anti3") for n in R3_DIMS[1:] for _ in range(2)]
    + [((n, n, n), "scratch") for n in R3_DIMS[1:] for _ in range(4)]
)


def _slots_by_role(slots, base):
    out: dict[str, dict[tuple[int, ...], list[int]]] = {}
    for j, (shape, role) in enumerate(slots):
        out.setdefault(role, {}).setdefault(shape, []).append(base + j)
    return out


ALLPASS_XV_BASE = len(VEC_LENS)
ALLPASS_XV_SLOTS: list[tuple[tuple[int, ...], str]] = [((k,), "lccfv") for k in R3_DIMS]

ALLPASS_XM_BY = _slots_by_role(ALLPASS_XM_SLOTS, ALLPASS_XM_BASE)
ALLPASS_XV_BY = _slots_by_role(ALLPASS_XV_SLOTS, ALLPASS_XV_BASE)

#: Roles whose slots are graph-owned and start at zero. ``symacc`` is scratch
#: that only the SymmetrizedAccumulation motif touches (see _motif_symacc) and
#: ``foldscr`` scratch only the rank-2 fold motif writes, so its operands keep
#: the single writer the antisymmetrizer passes look for; ``cpscr`` is the
#: chain motif's interior, for the same reason; ``deltascr`` holds a transposed
#: delta and nothing else (see _motif_delta).
_ALLPASS_SCRATCH_ROLES = ("scratch", "symacc", "foldscr", "cpscr", "deltascr")
ALLPASS_XT_BY = _slots_by_role(ALLPASS_XT_SLOTS, ALLPASS_XT_BASE)

#: The letters an ``aperm`` spells its operand and result with, per pool.
_APERM_LETTERS = {"m": ["i", "j"], "t": ["i", "j", "k"]}

_MATMUL = "ij <- ik ; kj"

#: Other spellings of the plain matrix product, so a chain can name each
#: statement's letters apart the way hand-written CC code does. Kept out of
#: EINSUM_PATTERNS, whose key list the other corpora draw from.
_MATMUL_SPELLINGS = {spec: (lambda A, B: A @ B) for spec in (
    "pr <- pq ; qr", "ps <- pr ; rs", "qs <- qr ; rs", "ps <- pq ; qs",
    "tv <- tu ; uv", "tw <- tv ; vw")}


def _xm(rng, role, shape, exclude=()):
    return _pick(rng, ALLPASS_XM_BY.get(role, {}), shape, exclude)


def _xt(rng, role, shape, exclude=()):
    return _pick(rng, ALLPASS_XT_BY.get(role, {}), shape, exclude)


def _pick_distinct(picker, count, *args):
    """@p count distinct slots from @p picker, or None when the pool runs short."""
    got = []
    for _ in range(count):
        s = picker(*args, exclude=tuple(got))
        if s is None:
            return None
        got.append(s)
    return got


def _motif_delta(rng):
    """A contraction against a declared identity, feeding a second contraction.

    DeltaElimination's shape. A third of the time the identity is first
    transposed into scratch, which is an identity only because
    ProvenancePropagation carries the tag across the permute, so whether the
    delta is seen depends on which of the two ran first.
    """
    spec = _EINSUM_SPECS[int(rng.integers(0, len(_EINSUM_SPECS)))]
    _, shape_rule = EINSUM_PATTERNS[spec]
    ni, nk, nj = _d(rng), _d(rng), _d(rng)
    on_left = rng.random() < 0.5
    # The delta operand must be square, so the two letters it carries share an extent.
    if on_left:
        ni = nk
    else:
        nj = nk
    sa, sb, sc = shape_rule(ni, nk, nj)
    n = sa[0] if on_left else sb[0]
    delta = _xm(rng, "delta", (n, n))
    out = []
    if rng.random() < 0.33:
        # Half the time a plain transpose into a ``deltascr`` slot, which nothing
        # else writes, so the copy is an identity the tag may follow. Otherwise the
        # copy scales or accumulates, lands in scratch other statements share,
        # or is overwritten before the contraction reads it, and the tag must
        # not follow (``test_provenance_propagation_tags_only_a_plain_copy``).
        if rng.random() < 0.5:
            moved = _xm(rng, "deltascr", (n, n))
            out.append(("perm", 1.0, 0.0, delta, moved))
        else:
            moved = _xm(rng, "scratch", (n, n)) if rng.random() < 0.5 else _pick(rng, SCRATCH_BY_SHAPE, (n, n))
            a = 1.0 if rng.random() < 0.5 else _scalar(rng)
            out.append(("perm", a, 0.0 if rng.random() < 0.7 else 1.0, delta, moved))
            if rng.random() < 0.25:
                X, Y = _pick_mat(rng, (n, n)), _pick_mat(rng, (n, n))
                out.append(("einsum", _MATMUL, _scalar(rng), X, Y, 0.0, moved, False, False))
        delta = moved
    other = _pick_mat(rng, sb if on_left else sa)
    if other is None:
        return _fallback(rng)
    A, B = (delta, other) if on_left else (other, delta)
    # Usually graph scratch, which is what the pass dissolves; sometimes a user
    # tensor, whose write the rewrite must keep.
    into_scratch = rng.random() < 0.75
    if into_scratch:
        tmp = _pick(rng, SCRATCH_BY_SHAPE, sc, (A, B))
    else:
        tmp = _pick_mat(rng, sc, (A, B))
    if tmp is None:
        return _fallback(rng)
    # A unit prefactor into scratch read by the next contraction is a chain
    # MultiTermFactorization dissolves, spelled with letters (i, j, k at
    # whatever extents the draw gave) that other statements in the region
    # reuse at other extents; see ``_respelled``.
    ab = 1.0 if rng.random() < 0.6 else _scalar(rng)
    cpf = 0.0 if rng.random() < 0.8 else 1.0
    ca, cb = bool(rng.random() < 0.2), bool(rng.random() < 0.2)
    out.append(("einsum", spec, ab, A, B, cpf, tmp, ca, cb))
    q = _d(rng)
    E = _pick_mat(rng, (sc[1], q))
    C2 = _pick_mat(rng, (sc[0], q), (A, B, tmp, E))
    if E is None or C2 is None:
        return _fallback(rng)
    out.append(("einsum", _MATMUL, _scalar(rng), tmp, E, float(rng.integers(0, 2)), C2,
                bool(rng.random() < 0.2), False))
    return out


def _motif_zero_block(rng):
    """A contraction whose summed letter ranges over two disjoint spaces."""
    n = int(rng.choice(DIMS[1:]))
    A, Z = _xm(rng, "occ", (n, n)), _xm(rng, "zero", (n, n))
    C = _pick_mat(rng, (n, n))
    cpf = [0.0, 1.0, 2.0, _scalar(rng)][int(rng.integers(0, 4))]
    return [("einsum", _MATMUL, _scalar(rng), A, Z, cpf, C, False, False)]


def _unit_or_scalar(rng):
    """Exactly one half of the time, the prefactor a hand-written operator most often carries."""
    return 1.0 if rng.random() < 0.5 else _scalar(rng)


def _antisym_producer(rng, op, n, src, out):
    """``out = a * op(src)``, as a permute or as an einsum under the operator.

    Both forms draw ``a`` half of the time and use one otherwise. A fold that
    repoints the contraction at the permute's source must carry the permute's
    prefactor with it
    (``test_antisymmetrizer_folding_keeps_the_operator_prefactor``).
    """
    if rng.random() < 0.7:
        return ("aperm", "m", op, _unit_or_scalar(rng), src, 0.0, out)
    k = _d(rng)
    A, B = _pick_mat(rng, (n, k)), _pick_mat(rng, (k, n))
    return ("xeinsum", _MATMUL, op, _unit_or_scalar(rng), ("m", A), ("m", B), 0.0, ("m", out), False, False)


def _motif_antisym_fold(rng):
    """An antisymmetrized matrix contracted to a scalar against an antisymmetric one.

    The AntisymmetrizerFolding shape, with the premise on the other operand
    established two ways: structurally (a second antisymmetrizer's output,
    which AntisymmetryInference tags) or through an invariant denominator
    (inference carries the tag across a division by a ``sym`` slot, which
    AntisymmetryDetection finds invariant). A bound antisymmetric matrix as the
    other operand is not a premise the passes establish for P(i/j), whose
    groups are singletons, so it is not drawn; _motif_antisym_fold3 is where a
    premise read from the data is exercised.
    """
    n = int(rng.choice(DIMS[1:]))
    op = _gen_rank2_operator(rng)
    # A source is sometimes a matrix other statements write, and a quarter of
    # the time one is rescaled between the operators and the dot. The fold
    # repoints the dot at the source, so it must see that write
    # (``test_antisymmetrizer_folding_sees_its_source_overwritten``). A
    # rescale keeps an ``anti`` slot antisymmetric, which is what
    # AntisymmetryDetection reads from it.
    roll = rng.random()
    src_v = (_xm(rng, "anti", (n, n)) if roll < 0.3 else _xm(rng, "linsrc", (n, n)) if roll < 0.8
             else _pick_mat(rng, (n, n)))
    scratch = _pick_distinct(lambda exclude=(): _xm(rng, "foldscr", (n, n), exclude), 3)
    out = _pick_vec(rng, 1)
    if scratch is None or out is None:
        return _fallback(rng)
    W, V, Wd = scratch
    # The folded operand is always a permute: the pass folds only that form.
    # Now and then a destination prefactor: the output is no longer the
    # operator's value, so the fold must decline.
    stmts = [("aperm", "m", op, _unit_or_scalar(rng), src_v, 1.0 if rng.random() < 0.1 else 0.0, V)]
    src_w = _xm(rng, "anti", (n, n)) if rng.random() < 0.5 else _xm(rng, "linsrc", (n, n))
    stmts.insert(0, _antisym_producer(rng, op, n, src_w, W))
    if rng.random() < 0.25:
        stmts.append(("scale", _scalar(rng), src_v if rng.random() < 0.7 else src_w))
    other = W
    if rng.random() < 0.5:
        stmts.append(("ddiv", 1.0 if rng.random() < 0.5 else _scalar(rng), W, _xm(rng, "sym", (n, n)), 0.0, Wd))
        other = Wd
    pair = (other, V) if rng.random() < 0.5 else (V, other)
    stmts.append(("dot", "m", out) + pair)
    return stmts


def _motif_antisym_fold3(rng):
    """The coset form on rank three: P(i/jk) of a source antisymmetric in (j,k).

    Only AntisymmetryDetection establishes the within-group premise, so the fold
    here rests on data rather than on structure.
    """
    n = int(rng.choice(R3_DIMS[1:]))
    shape_index = int(rng.choice([1, 1, 3, 2]))
    op = shaped_operator(shape_index, ("i", "j", "k"))
    # Read-only sources. The rank-3 pool has no whole-tensor write, and a slab
    # write would break the antisymmetry AntisymmetryDetection reads from an
    # ``anti3`` slot; _motif_antisym_fold draws a source written before the dot.
    src_w = _xt(rng, "anti3", (n, n, n)) if rng.random() < 0.7 else _xt(rng, "lccft", (n, n, n))
    src_v = _xt(rng, "lccft", (n, n, n)) if rng.random() < 0.7 else _xt(rng, "anti3", (n, n, n))
    scratch = _pick_distinct(lambda exclude=(): _xt(rng, "scratch", (n, n, n), exclude), 2)
    out = _pick_vec(rng, 1)
    if scratch is None or out is None or src_w is None or src_v is None:
        return _fallback(rng)
    W, V = scratch
    # Drawn prefactors, for the reason _antisym_producer gives.
    return [("aperm", "t", op, _unit_or_scalar(rng), src_w, 0.0, W),
            ("aperm", "t", op, _unit_or_scalar(rng), src_v, 0.0, V),
            ("dot", "t", out) + ((W, V) if rng.random() < 0.5 else (V, W))]


def _motif_linearity(rng):
    """``X = a P(A)``, ``Y = c P(B)``, ``X += s Y``, then X read: AntisymmetrizerLinearity."""
    n = int(rng.choice(DIMS[1:]))
    op = _gen_rank2_operator(rng)
    # B is sometimes a matrix other statements write, and a quarter of the time
    # the second operator's source is rescaled between the two operators. The
    # merged sum is built at the FIRST operator, so it must see that write
    # (``test_antisymmetrizer_linearity_sees_a_write_between_the_operators``).
    A = _xm(rng, "anti", (n, n)) if rng.random() < 0.3 else _xm(rng, "linsrc", (n, n))
    B = _xm(rng, "linsrc", (n, n), (A,)) if rng.random() < 0.8 else _pick_mat(rng, (n, n))
    scratch = _pick_distinct(lambda exclude=(): _xm(rng, "scratch", (n, n), exclude), 2)
    if scratch is None:
        return _fallback(rng)
    X, Y = scratch
    # Now and then the second operator is spelled with its letters swapped, which
    # names the same antisymmetrizer through a different group order.
    op_y = op if rng.random() < 0.8 else _gen_rank2_operator(rng)
    prod = [("aperm", "m", op, _scalar(rng), A, 0.0, X), ("aperm", "m", op_y, _scalar(rng), B, 0.0, Y)]
    if rng.random() < 0.5:
        prod.reverse()
    if rng.random() < 0.25:
        prod.insert(1, ("scale", _scalar(rng), prod[1][4]))
    stmts = prod + [("axpby", _scalar(rng), Y, 1.0, X)]
    if rng.random() < 0.5:
        C = _pick_mat(rng, (n, n))
        stmts.append(("axpy", _scalar(rng), X, C))
    else:
        stmts.append(("dot", "m", _pick_vec(rng, 1), X, _pick_mat(rng, (n, n))))
    return stmts


def _motif_symacc(rng):
    """``r2 += s tmp; tmpP = tmp^T; r2 += s tmpP``: SymmetrizedAccumulation.

    ``tmp`` may be a user tensor, which is a program a caller can write and
    whose final value the caller can read. ``tmpP`` is usually graph scratch of
    the ``symacc`` role, which nothing but the transpose writes, and otherwise
    a user tensor the caller reads back
    (``test_symmetrized_accumulation_keeps_a_caller_held_transpose``). A
    fifth of the time the site is followed by an accumulate into ``tmpP``
    and a read of it
    (``test_symmetrized_accumulation_sees_a_later_accumulate_into_the_transpose``).
    Either way the rewrite must keep ``tmpP``'s write. Now and then a loop
    that reads ``r2`` sits right after the transpose, ahead of a half, and the
    rewrite must not move that half's contribution up past it
    (``test_symmetrized_accumulation_sees_a_loop_between_the_halves``). A
    shuffled Reorder or LoopInvariantHoisting also moves loops between the
    halves when the motif lands both in a block and inside one of its loops.
    """
    n, k = _d(rng), _d(rng)
    A, B = _pick_mat(rng, (n, k)), _pick_mat(rng, (k, n))
    tmp = _xm(rng, "scratch", (n, n)) if rng.random() < 0.7 else _pick_mat(rng, (n, n), (A, B))
    tmpP = _xm(rng, "symacc", (n, n)) if rng.random() < 0.8 else _pick_mat(rng, (n, n), (A, B, tmp))
    r2 = _xm(rng, "symout", (n, n))
    if None in (A, B, tmp, tmpP, r2):
        return _fallback(rng)
    s1 = _scalar(rng)
    s2 = s1 if rng.random() < 0.8 else _scalar(rng)
    e = ("einsum", _MATMUL, _scalar(rng), A, B, 0.0, tmp, False, False)
    a1 = ("axpby", s1, tmp, 1.0, r2)
    p = ("perm", 1.0, 0.0, tmp, tmpP)
    a2 = ("axpby", s2, tmpP, 1.0, r2)
    orders = ([e, a1, p, a2], [e, p, a1, a2], [e, p, a2, a1])
    stmts = list(orders[int(rng.integers(0, len(orders)))])
    if rng.random() < 0.15:
        Z = _pick_mat(rng, (n, n), (tmp, tmpP))
        stmts.insert(stmts.index(p) + 1, ("loop", int(rng.integers(1, 3)), [("axpy", _scalar(rng), r2, Z)]))
    if rng.random() < 0.2:
        X, Z = _pick_mat(rng, (n, n), (tmpP,)), _pick_mat(rng, (n, n), (tmpP,))
        stmts += [("axpy", _scalar(rng), X, tmpP), ("axpy", _scalar(rng), tmpP, Z)]
    return stmts


def _motif_lccf(rng):
    """One rank-3 tensor read twice under transposed axes into one output: the 2J-K shape.

    Every operand is a slot only this motif touches (``lccft``, ``lccfv``,
    ``lccfout``). Now and then a loop that rescales the output sits between
    the first two members, and the fold must see the write inside it
    (``test_linear_combination_folding_sees_a_loop_between_the_members``). A
    shuffled Reorder or LoopInvariantHoisting also moves loops between the
    members when the motif lands both in a block and inside one of its loops.
    """
    k, n = _d(rng, R3_DIMS), _d(rng, R3_DIMS)
    T = _pick(rng, ALLPASS_XT_BY["lccft"], (k, n, n))
    vec = _pick(rng, ALLPASS_XV_BY["lccfv"], (k,))
    C = _xm(rng, "lccfout", (n, n))
    if T is None or vec is None or C is None:
        return _fallback(rng)
    pats = ["kij", "kji"]
    if rng.random() < 0.5:
        pats.reverse()
    if rng.random() < 0.25:
        pats.append(pats[int(rng.integers(0, 2))])
    first_cpf = [0.0, 1.0, _scalar(rng)][int(rng.integers(0, 3))]
    stmts = [("leinsum", p, _scalar(rng), vec, T, first_cpf if i == 0 else 1.0, C) for i, p in enumerate(pats)]
    if rng.random() < 0.15:
        stmts.insert(1, ("loop", int(rng.integers(1, 3)), [("scale", _scalar(rng), C)]))
    return stmts


def _motif_distributive(rng):
    """Two or three contractions accumulating into one output, sharing one operand.

    Operands are ``dfsrc`` slots and the output a ``dfout`` slot only this
    motif touches. A quarter of the time a write through a VIEW of an operand
    or of the output lands between two members, which the pass must see
    through the alias
    (``test_distributive_factoring_sees_a_view_write_between_the_members``).
    A fifth of the time a second group follows into the same output, each
    group headed by its own destination prefactor, and the two replacements
    must keep program order
    (``test_distributive_factoring_keeps_two_groups_in_program_order``).
    """
    ni, nj = _d(rng, (2, 3)), _d(rng, (2, 3))
    C = _xm(rng, "dfout", (ni, nj))
    stmts = _distributive_group(rng, ni, nj, C)
    if stmts is None:
        return _fallback(rng)
    if rng.random() < 0.2:
        stmts += _distributive_group(rng, ni, nj, C) or []
    return stmts


def _distributive_group(rng, ni, nj, C):
    """One shared-operand group into the (@p ni, @p nj) output @p C, or None when the pool runs short."""
    spec = _EINSUM_SPECS[int(rng.integers(0, len(_EINSUM_SPECS)))]
    _, shape_rule = EINSUM_PATTERNS[spec]
    sa, sb, sc = shape_rule(ni, _d(rng, (2, 3)), nj)
    count = int(rng.integers(2, 4))
    shared_left = rng.random() < 0.5
    shared = _xm(rng, "dfsrc", sa if shared_left else sb)
    others = _pick_distinct(lambda exclude=(): _xm(rng, "dfsrc", sb if shared_left else sa, exclude + (shared,)), count)
    if shared is None or others is None:
        return None
    first_cpf = [1.0, 0.5, 2.0][int(rng.integers(0, 3))]
    stmts = []
    for i, other in enumerate(others):
        A, B = (shared, other) if shared_left else (other, shared)
        stmts.append(("einsum", spec, _scalar(rng), A, B, first_cpf if i == 0 else 1.0, C,
                      bool(rng.random() < 0.05), False))
    if rng.random() < 0.25:
        roll = rng.random()
        M, shape = ((shared, sa if shared_left else sb) if roll < 0.4
                    else (others[1], sb if shared_left else sa) if roll < 0.8 else (C, sc))
        write = _view_write(rng, M, shape)
        if write is not None:
            stmts.insert(int(rng.integers(1, len(stmts))), write)
    return stmts


def _view_write(rng, M, shape):
    """``M[block] += a * src`` through a view of a random block of @p M, or None."""
    sr, sc = int(rng.integers(1, shape[0] + 1)), int(rng.integers(1, shape[1] + 1))
    src = _pick_mat(rng, (sr, sc))
    if src is None:
        return None
    r0, c0 = int(rng.integers(0, shape[0] - sr + 1)), int(rng.integers(0, shape[1] - sc + 1))
    return ("vaxpy", _scalar(rng), src, M, r0, r0 + sr, c0, c0 + sc)


def _motif_multi_term(rng):
    """Two three-factor products sharing ``A B``, maybe bracketed apart: MultiTermFactorization."""
    s, L = ALLPASS_BS, ALLPASS_BL
    # The factors are ``mtfsrc`` slots. A quarter of the time a factor of the
    # shared pair is rescaled just ahead of the chains, and the shared product
    # must be computed after that write
    # (``test_multi_term_factorization_shares_after_a_factor_is_written``).
    factors = _pick_distinct(lambda exclude=(): _xm(rng, "mtfsrc", (s, L), exclude), 3)
    outs = _pick_distinct(lambda exclude=(): _xm(rng, "mtfout", (s, L), exclude), 2)
    wide = None if factors is None or outs is None else factors + outs
    B = _xm(rng, "mtfsrc", (L, s))
    T1, T1b = _pick_distinct(lambda exclude=(): _xm(rng, "scratch", (s, s), exclude), 2) or (None, None)
    T2 = _xm(rng, "scratch", (L, L))
    if wide is None or T1 is None:
        return _fallback(rng)
    A, C, D, R1, R2 = wide
    # Letters p, q, r, s name each statement's axes apart at fixed extents
    # (p = r = 3, q = s = 12); see ``_respelled`` for the other spelling.
    stmts = [("einsum", "pr <- pq ; qr", 1.0, A, B, 0.0, T1, False, False),
             ("einsum", "ps <- pr ; rs", _scalar(rng), T1, C, float(rng.integers(0, 2)), R1, False, False)]
    if rng.random() < 0.5:
        stmts += [("einsum", "qs <- qr ; rs", 1.0, B, D, 0.0, T2, False, False),
                  ("einsum", "ps <- pq ; qs", _scalar(rng), A, T2, float(rng.integers(0, 2)), R2, False, False)]
    else:
        stmts += [("einsum", "pr <- pq ; qr", 1.0, A, B, 0.0, T1b, False, False),
                  ("einsum", "ps <- pr ; rs", _scalar(rng), T1b, D, float(rng.integers(0, 2)), R2, False, False)]
    if rng.random() < 0.25:
        stmts.insert(0, ("scale", _scalar(rng), B if rng.random() < 0.5 else A))
    return _respelled(rng, stmts)


def _respelled(rng, stmts):
    """@p stmts, a third of the time with every product spelled ``ij <- ik ; kj``.

    Letters are scoped to a statement, so in ``(A B) C`` that spelling gives
    k an extent of 12 in the first product and 3 in the second, j the reverse,
    and the rest of the region gives i, j and k whatever extents it draws. A
    pass that keys extents by letter across a region sizes the intermediate it
    introduces wrong
    (``test_multi_term_factorization_sizes_a_reused_letter_by_its_own_statement``).
    """
    if rng.random() >= 1 / 3:
        return stmts
    return [st[:1] + (_MATMUL,) + st[2:] if st[0] == "einsum" else st for st in stmts]


def _motif_chain(rng):
    """``R = (A B) C`` through a (12, 12) intermediate, where ``A (B C)`` is far cheaper."""
    s, L = ALLPASS_BS, ALLPASS_BL
    srcs = _pick_distinct(lambda exclude=(): _xm(rng, "cpsrc", (L, s), exclude), 2)
    tall = None if srcs is None else srcs + [_xm(rng, "cpout", (L, s))]
    B = _xm(rng, "cpsrc", (s, L))
    T = _xm(rng, "cpscr", (L, L))
    if tall is None:
        return _fallback(rng)
    A, C, R = tall
    cpf = 0.0 if rng.random() < 0.8 else 1.0
    # Letters t, u, v, w, or the reused spelling ``_respelled`` draws.
    return _respelled(rng, [
        ("einsum", "tv <- tu ; uv", 1.0 if rng.random() < 0.7 else _scalar(rng), A, B, 0.0, T, False, False),
        ("einsum", "tw <- tv ; vw", _scalar(rng), T, C, cpf, R, False, False)])


#: Each motif with its draw weight. The folds and the re-bracketing motifs
#: are drawn more often because their passes fire on a smaller share of the
#: orders a shuffle produces: a fold needs AntisymmetryInference ahead of it and
#: AntisymmetrizerExpansion behind it, one order in six, and a chain is
#: re-bracketed by whichever of MultiTermFactorization and ContractionPlanning
#: reaches it first. The identity contraction, the transposed accumulation
#: and the shared-operand sum are drawn more often too, since a share of their
#: instances carry a shape the pass must decline on (half the transposed
#: identities, about a third of the sites and a quarter of the sums).
_ALLPASS_MOTIFS = ((_motif_delta, 1.5), (_motif_zero_block, 1.0), (_motif_antisym_fold, 3.0),
                   (_motif_antisym_fold3, 2.0), (_motif_linearity, 1.5), (_motif_symacc, 1.5),
                   (_motif_lccf, 1.0), (_motif_distributive, 1.5), (_motif_multi_term, 2.0),
                   (_motif_chain, 2.0))
_ALLPASS_MOTIF_P = np.array([w for _, w in _ALLPASS_MOTIFS]) / sum(w for _, w in _ALLPASS_MOTIFS)


def _gen_all_passes_motif(rng):
    """One motif, drawn by weight. See the section comment for what each provokes."""
    return _ALLPASS_MOTIFS[int(rng.choice(len(_ALLPASS_MOTIFS), p=_ALLPASS_MOTIF_P))][0](rng)


def allpass_named_slots(prog):
    """Every pool integer @p prog names at or above SCRATCH_BASE, nested bodies included."""
    return _scratch_indices(prog)


def allpass_program(rng, depth, max_stmts):
    """A program for the all-passes arm: ``_gen_block`` with its motifs, run as drawn."""
    return _gen_block(rng, depth=depth, max_stmts=max_stmts, all_passes=True)


def _deferred_scratch(prog):
    """The scratch slots the builder DECLARES: the even ones."""
    slots = {SCRATCH_BASE + j for j in range(0, len(SCRATCH_SHAPES), 2)}
    slots |= {ALLPASS_XM_BASE + j for j, (_, role) in enumerate(ALLPASS_XM_SLOTS)
              if role in _ALLPASS_SCRATCH_ROLES and j % 2 == 0}
    return slots


def _deferred_scratch_t(prog):
    return {ALLPASS_XT_BASE + j for j, (_, role) in enumerate(ALLPASS_XT_SLOTS)
            if role in _ALLPASS_SCRATCH_ROLES and j % 2 == 0}


def allpass_seed_arrays(rng, dtype="float64"):
    """``_seed_arrays`` plus the extension's user slots, as ``(m, v, t, xm, xt, xv)``.

    ``xm[j]`` and ``xt[j]`` are None for scratch slots, which start at zero.
    """
    m, v, t = _seed_arrays(rng, dtype)
    dt = np.dtype(dtype)
    is_complex = dt.kind == "c"

    def gen(sh):
        a = rng.standard_normal(sh)
        if is_complex:
            a = a + 1j * rng.standard_normal(sh)
        return a

    def fill(shape, role):
        if role in _ALLPASS_SCRATCH_ROLES:
            return None
        if role == "delta":
            return np.eye(shape[0]).astype(dt)
        if role == "zero":
            return np.zeros(shape, dtype=dt)
        if role == "anti":
            x = gen(shape)
            return (x - x.T).astype(dt)
        if role == "anti3":
            x = gen(shape)
            return (x - x.transpose(0, 2, 1)).astype(dt)
        if role == "sym":
            x = rng.standard_normal(shape)
            return (2.0 + np.abs(x + x.T)).astype(dt)
        return gen(shape).astype(dt)

    xm = [fill(sh, role) for sh, role in ALLPASS_XM_SLOTS]
    xt = [fill(sh, role) for sh, role in ALLPASS_XT_SLOTS]
    xv = [fill(sh, role) for sh, role in ALLPASS_XV_SLOTS]
    return m, v, t, xm, xt, xv


def _oracle_all_passes(prog, arrays, dtype, runs=1):
    """The numpy oracle over the extended pool, returning only what a caller can observe."""
    m, v, t, xm, xt, xv = arrays
    dt = np.dtype(dtype)
    om = ([a.copy() for a in m] + [np.zeros(sh, dtype=dt) for sh in SCRATCH_SHAPES]
          + [a.copy() if a is not None else np.zeros(sh, dtype=dt) for a, (sh, _) in zip(xm, ALLPASS_XM_SLOTS)])
    ot = [a.copy() for a in t] + [a.copy() if a is not None else np.zeros(sh, dtype=dt)
                                  for a, (sh, _) in zip(xt, ALLPASS_XT_SLOTS)]
    ov = [a.copy() for a in v] + [a.copy() for a in xv]
    # A DEFERRED zero tensor is materialized and zeroed at the start of every
    # execute, where an eager one keeps its value across replays; a replay
    # re-zeroes exactly the slots the builder declares.
    deferred_m = sorted(_deferred_scratch(prog))
    deferred_t = sorted(_deferred_scratch_t(prog))
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for _ in range(runs):
            for slot in deferred_m:
                om[slot] = np.zeros_like(om[slot])
            for slot in deferred_t:
                ot[slot] = np.zeros_like(ot[slot])
            interp_np(prog, om, ov, ot, dt)
    out_m = om[:len(m)] + [om[ALLPASS_XM_BASE + j] for j, a in enumerate(xm) if a is not None]
    out_t = ot[:len(t)] + [ot[ALLPASS_XT_BASE + j] for j, a in enumerate(xt) if a is not None]
    if not _usable(out_m, ov, out_t, cap=_DTYPE_CAP[dtype]):
        pytest.skip("oracle overflowed: a numerically degenerate program")
    return out_m, ov, out_t


def _allpass_registry(graph):
    """occ and virt share no element; aux is related to nothing. Private to @p graph."""
    registry = cg.private_space_registry(graph)
    occ = registry.register_space(cg.index_space("occ", "o", 4.0))
    virt = registry.register_space(cg.index_space("virt", "v", 4.0))
    registry.register_space(cg.index_space("aux", "x", 4.0))
    registry.declare_disjoint(occ, virt)
    return registry


def _build_all_passes(prog, arrays, name):
    """Build @p prog over the extended pool.

    Returns the graph and the tensors a caller can observe, in the order
    ``_oracle_all_passes`` returns them. The declarations each role needs are
    made on the root graph for every slot the program names: a delta is tagged
    an identity before capture, and the space-annotated slots get their spaces
    after it.
    """
    m, v, t, xm, xt, xv = arrays
    mats, vecs, r3s = _make_pool(m, v, t, name)
    g = cg.Graph(name)
    _allpass_registry(g)
    used = _scratch_indices(prog)
    dtype = str(m[0].dtype)
    # Kept apart: matrix and rank-3 slot numbers overlap.
    deferred_m, deferred_t = _deferred_scratch(prog), _deferred_scratch_t(prog)

    def scratch(label, slot, shape, deferred):
        # Alternate deferred and eager, as _build_with_scratch does, so both
        # storage modes meet every pass.
        if slot in deferred:
            return g.declare_zero_tensor(label, list(shape), intermediate=True, dtype=dtype)
        return g.create_zero_tensor(label, list(shape), intermediate=True, dtype=dtype)

    def user(label, arr):
        tn = einsums.create_zero_tensor(label, list(arr.shape), dtype=dtype)
        np.asarray(tn)[...] = arr
        return tn

    full_m = list(mats)
    for j, sh in enumerate(SCRATCH_SHAPES):
        full_m.append(scratch(f"{name}_s{j}", SCRATCH_BASE + j, sh, deferred_m) if SCRATCH_BASE + j in used else None)
    x_users = []
    late_spaces = []
    for j, (sh, role) in enumerate(ALLPASS_XM_SLOTS):
        slot = ALLPASS_XM_BASE + j
        if role in _ALLPASS_SCRATCH_ROLES:
            full_m.append(scratch(f"{name}_xs{j}", slot, sh, deferred_m) if slot in used else None)
            continue
        tn = user(f"{name}_x{j}", xm[j])
        if slot in used and role == "delta":
            cg.annotate(tn, tag="identity", graph=g)
        elif slot in used and role in ("occ", "zero"):
            late_spaces.append((tn, ("aux", "occ") if role == "occ" else ("virt", "aux")))
        full_m.append(tn)
        x_users.append(tn)
    full_t = list(r3s)
    xt_users = []
    for j, (sh, role) in enumerate(ALLPASS_XT_SLOTS):
        slot = ALLPASS_XT_BASE + j
        if role in _ALLPASS_SCRATCH_ROLES:
            full_t.append(scratch(f"{name}_xts{j}", slot, sh, deferred_t) if slot in used else None)
            continue
        tn = user(f"{name}_xt{j}", xt[j])
        full_t.append(tn)
        xt_users.append(tn)
    x_vecs = [user(f"{name}_xv{j}", xv[j]) for j in range(len(xv))]
    build_cg(prog, g, full_m, vecs + x_vecs, full_t, name)
    # Spaces AFTER capture, as a Python caller annotating a finished program
    # does: capture binds each letter to one space and rejects a contraction
    # whose letter would bind two, which is the very contraction the zero-block
    # rewrite exists for. The passes read the handles as they stand when they run.
    for tn, spaces in late_spaces:
        cg.annotate(tn, spaces, graph=g)
    return g, (mats + x_users, vecs + x_vecs, r3s + xt_users)


#: The eleven default passes no other shuffled arm runs.
_ALL_PASSES_ADDED = [
    "ProvenancePropagation", "DeltaElimination", "AntisymmetryDetection", "AntisymmetrizerLinearity",
    "AntisymmetryInference", "AntisymmetrizerFolding", "SymmetrizedAccumulation",
    "LinearCombinationContractionFolding", "DistributiveFactoring", "MultiTermFactorization",
    "ContractionPlanning",
]

#: What the all-passes arm shuffles: the rich list plus the eleven above.
_ALL_PASSES = _RICH_SAFE_PASSES + _ALL_PASSES_ADDED

#: The counters that say a pass acted, where its return value cannot. The
#: analysis passes (detection, inference, provenance) never report a change to
#: the node set, so "modified" would call them vacuous when they are not.
_ALL_PASSES_FIRE_ATTRS = {
    "ProvenancePropagation": ("num_propagated",),
    "DeltaElimination": ("num_eliminated", "num_zero_blocks"),
    "AntisymmetryDetection": ("num_found",),
    "AntisymmetrizerLinearity": ("num_merged",),
    "AntisymmetryInference": ("num_tagged",),
    "AntisymmetrizerFolding": ("num_folded",),
    "SymmetrizedAccumulation": ("num_rewritten",),
    "LinearCombinationContractionFolding": ("num_eliminated",),
    "DistributiveFactoring": ("num_eliminated",),
    "MultiTermFactorization": ("num_shared", "num_rebracketed"),
    "ContractionPlanning": ("chains_restructured",),
}

#: Per pass: trials that ran it and trials on which it fired.
_ALL_PASSES_STATS = {name: {"ran": 0, "fired": 0} for name in _ALL_PASSES}


def _make_all_passes_pass(name):
    """The pass object the arm runs for @p name, or None to isolate it from the default list.

    MultiTermFactorization has its search switched on, since off is its default
    and a shuffle would otherwise run it without it doing anything.
    DistributiveFactoring skips its cost model, which declines everything this
    pool's extents can express: the rewrite is what is under test, not the price.
    """
    if name == "MultiTermFactorization":
        p = _G.MultiTermFactorization()
        p.set_search_enabled(True)
        return p
    if name == "DistributiveFactoring":
        return _G.DistributiveFactoring(_G.Factor.Always)
    if hasattr(_G, name):
        return getattr(_G, name)()
    return None


def _apply_all_passes_one(g, name):
    """Run one pass on @p g; returns how many rewrites it reports (1 or 0 without a counter)."""
    p = _make_all_passes_pass(name)
    if p is None:
        return int(bool(_apply_one_pass(g, name)))
    pm = cg.PassManager()
    # No wall-clock allowance, so a search is never cut off and the graph a
    # trial runs does not depend on how fast the machine is.
    pm.set_optimizer_budget(0)
    pm.add(p)
    modified = g.apply(pm)
    attrs = _ALL_PASSES_FIRE_ATTRS.get(name)
    return sum(int(getattr(p, a)) for a in attrs) if attrs else int(bool(modified))


def check_program_all_passes(prog, arrays, label, rng, dtype="float64", runs=1):
    """A random order of ``_ALL_PASSES``, one pass per manager, then Materialization.

    Returns the set of passes that fired on this trial. A pass that throws, a
    graph the verifier rejects, and a wrong number are all reported with the
    order and the program, which is what a reduction starts from. The drawn
    order runs exactly as drawn.
    """
    oracle = _oracle_all_passes(prog, arrays, dtype, runs)
    order = list(_ALL_PASSES)
    rng.shuffle(order)
    g, observed = _build_all_passes(prog, arrays, label)
    fired = set()
    stage = None
    try:
        for stage in order:
            _ALL_PASSES_STATS[stage]["ran"] += 1
            if _apply_all_passes_one(g, stage):
                _ALL_PASSES_STATS[stage]["fired"] += 1
                fired.add(stage)
        # Correctness-enabling rather than shuffled: see check_program_rich_pipeline.
        stage = "closing Materialization"
        _apply_all_passes_one(g, "Materialization")
        stage = "execute"
        for _ in range(runs):
            g.execute()
    except Exception as exc:
        raise AssertionError(f"ALL-PASSES pipeline failed at {stage}: {exc}\n"
                             f"order={order}\nprogram={prog!r}") from exc
    got = tuple([np.asarray(x).copy() for x in pool] for pool in observed)
    _assert_pools_typed(got, oracle, prog, "ALL-PASSES-PIPELINE", dtype,
                        extra=f"  order={order} runs={runs}")
    return fired


def run_program_all_passes_default(prog, arrays, label, dtype="float64", level=None, runs=1):
    """The same program under ``default_pass_manager()``, or ``optimize(level)`` when given.

    For triage: whether a shuffled-order failure is also reachable in an order
    a user gets.
    """
    oracle = _oracle_all_passes(prog, arrays, dtype, runs)
    g, observed = _build_all_passes(prog, arrays, label)
    if level is None:
        g.apply(cg.default_pass_manager())
    else:
        g.optimize(OPT_LEVELS[level])
    for _ in range(runs):
        g.execute()
    got = tuple([np.asarray(x).copy() for x in pool] for pool in observed)
    _assert_pools_typed(got, oracle, prog, f"ALL-PASSES {level or 'default'}", dtype, extra=f" runs={runs}")


__all__ = [
    'fuzz_seeds',
    'check_program_region_pipeline',
    'region_pass_manager',
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
    '_gen_rich_op',
    '_gen_batch_cluster',
    '_view_ref',
    '_transposed_ref',
    'rich_op_census',
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
    'OPT_LEVELS',
    '_OPT_LEVEL_STATS',
    '_oracle_typed',
    '_build_with_scratch',
    '_scratch_indices',
    '_gen_scratch_motif',
    'SCRATCH_SHAPES',
    'SCRATCH_BASE',
    'SCRATCH_BY_SHAPE',
    '_assert_pools_typed',
    'check_program_opt_level',
    '_RICH_SAFE_PASSES',
    '_DEFAULT_PASS_NAMES',
    '_RICH_PIPELINE_STATS',
    '_isolated_default_pass',
    '_apply_one_pass',
    'check_program_rich_pipeline',
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
    'ALLPASS_BS',
    'ALLPASS_BL',
    'ALLPASS_XM_BASE',
    'ALLPASS_XM_SLOTS',
    'ALLPASS_XT_BASE',
    'ALLPASS_XT_SLOTS',
    'ALLPASS_XM_BY',
    'ALLPASS_XT_BY',
    'ALLPASS_XV_BASE',
    'ALLPASS_XV_SLOTS',
    'ALLPASS_XV_BY',
    '_gen_all_passes_motif',
    'allpass_program',
    'allpass_named_slots',
    'allpass_seed_arrays',
    '_oracle_all_passes',
    '_build_all_passes',
    '_ALL_PASSES_ADDED',
    '_ALL_PASSES',
    '_ALL_PASSES_FIRE_ATTRS',
    '_ALL_PASSES_STATS',
    '_make_all_passes_pass',
    '_apply_all_passes_one',
    'check_program_all_passes',
    'run_program_all_passes_default',
]
