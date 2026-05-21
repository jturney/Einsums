# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""Differential / fuzz harness for ComputeGraph + the optimization passes.

The idea is simple and brutal: generate a random *program* over a pool of
tensors, then run it three ways and demand they all agree:

  1. **numpy oracle** — a pure-numpy interpreter of the program.
  2. **raw graph**     — the program replayed into a ``cg.Graph`` and executed
                         with *no* optimization passes.
  3. **optimized graph** — the same program, then ``default_pass_manager()``
                         applied before execution.

If (raw == oracle) but (optimized != oracle), a pass miscompiled the graph.
If (raw != oracle), the executor itself disagrees with numpy. Either way the
seed and the offending program are printed so the failure reproduces.

The tensor pool is deliberately diverse:
  * **matrices** over every (r, c) with r, c ∈ {2, 3, 4} (several copies each),
  * **vectors** of each length,
  * **rank-3 tensors** over {2, 3}^3 for batched contractions.

The generator picks shape-compatible operands for each op, so contractions
exercise non-square M/N/K and batched (rank-3) gemms.

The op set stresses the read-modify-write hazards that have broken passes
before, across the BLAS levels, the einsum/permute path, batched gemm, and
**views** (sub-block aliases — these stress the scheduler's alias resolution,
since a write through a view must be seen as a write to the parent):

  * ``scale`` / ``axpy`` / ``axpby`` — level-1 in-place / accumulate.
  * ``gemm``   — ``C = a*A@B + b*C`` (mixed M/N/K, overwrite or accumulate).
  * ``einsum`` — three contraction patterns over mixed shapes.
  * ``beinsum``— rank-3 batched einsum ``ijb<-ikb;kjb`` (BatchedGemm path).
  * ``perm``   — transpose ``C = a*A^T + c*C``.
  * ``symm``   — symmetric double multiply ``C = B^T A B``.
  * ``gemv`` / ``ger`` — matrix×vector and rank-1 update.
  * ``vscale`` / ``vaxpy`` — scale / axpy applied through a *view* of a
                 matrix sub-block; mixed with full-matrix writes to the same
                 tensor to exercise alias-aware scheduling.

plus control flow:

  * ``loop``   — run a body sub-program a fixed number of times.
  * ``cond``   — generation-time coin flip selecting then/else branch.

Conditionals use a coin flip rather than a data-dependent predicate on purpose:
a predicate near its threshold would flip differently under fp noise between the
oracle and the executor, making the *whole branch* diverge and the test flaky.
Data-dependent branching is covered by the dedicated SCF/MP2 tests.
"""

from __future__ import annotations

import numpy as np
import pytest

import einsums
import einsums.graph as cg

# ──────────────────────────────────────────────────────────────────────────
# Pool layout — fixed so the generator and the per-trial seed arrays agree.
# ──────────────────────────────────────────────────────────────────────────

DIMS = (2, 3, 4)
R3_DIMS = (2, 3)
COPIES = 3  # copies of each matrix shape / vector length / rank-3 shape
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


def _gen_block(rng, depth, max_stmts):
    stmts = []
    n = int(rng.integers(1, max_stmts + 1))
    for _ in range(n):
        roll = rng.random()
        if depth > 0 and roll < 0.18:
            cnt = int(rng.integers(1, 4))
            stmts.append(("loop", cnt, _gen_block(rng, depth - 1, max_stmts)))
        elif depth > 0 and roll < 0.30:
            flag = bool(rng.integers(0, 2))
            then = _gen_block(rng, depth - 1, max_stmts)
            els = _gen_block(rng, depth - 1, max_stmts)
            stmts.append(("cond", flag, then, els))
        else:
            stmts.append(_gen_primitive(rng))
    return stmts


def _gen_primitive(rng):
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
    if op in (1, 2):  # axpy / axpby — same shape, distinct
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
        return ("einsum", spec, a, A, B, float(rng.integers(0, 2)), C) if C is not None else _fallback(rng)
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
        return ("beinsum", spec, a, A, B, float(rng.integers(0, 2)), C) if C is not None else _fallback(rng)
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


def interp_np(stmts, m, v, t):
    for s in stmts:
        k = s[0]
        if k == "scale":
            _, a, x = s
            m[x] = m[x] * a
        elif k == "axpy":
            _, a, x, y = s
            m[y] = m[y] + a * m[x]
        elif k == "axpby":
            _, a, x, b, y = s
            m[y] = a * m[x] + b * m[y]
        elif k == "gemm":
            _, a, A, B, b, C = s
            m[C] = a * (m[A] @ m[B]) + b * m[C]
        elif k == "einsum":
            _, spec, ab, A, B, cpf, C = s
            m[C] = ab * EINSUM_PATTERNS[spec][0](m[A], m[B]) + cpf * m[C]
        elif k == "beinsum":
            _, spec, ab, A, B, cpf, C = s
            t[C] = ab * BEINSUM_PATTERNS[spec][0](t[A], t[B]) + cpf * t[C]
        elif k == "perm":
            _, a, cpf, A, C = s
            m[C] = a * m[A].T + cpf * m[C]
        elif k == "symm":
            _, A, B, C = s
            m[C] = m[B].T @ m[A] @ m[B]
        elif k == "gemv":
            _, a, A, x, b, y = s
            v[y] = a * (m[A] @ v[x]) + b * v[y]
        elif k == "ger":
            _, a, x, y, A = s
            m[A] = m[A] + a * np.outer(v[x], v[y])
        elif k == "etransform":
            _, fn, M = s
            m[M] = ETRANSFORM_FNS[fn](m[M])
        elif k == "vgemm":
            _, a, M, r0, r1, c0, c1, B, b, C = s
            m[C] = a * (m[M][r0:r1, c0:c1] @ m[B]) + b * m[C]
        elif k == "vscale":
            _, a, M, r0, r1, c0, c1 = s
            m[M][r0:r1, c0:c1] = m[M][r0:r1, c0:c1] * a
        elif k == "vaxpy":
            _, a, src, M, r0, r1, c0, c1 = s
            m[M][r0:r1, c0:c1] = m[M][r0:r1, c0:c1] + a * m[src]
        elif k == "loop":
            _, n, body = s
            for _ in range(n):
                interp_np(body, m, v, t)
        elif k == "cond":
            _, flag, then, els = s
            interp_np(then if flag else els, m, v, t)
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
        _, spec, ab, A, B, cpf, C = s
        einsums.einsum(spec, m[C], m[A], m[B], c_pf=cpf, ab_pf=ab)
    elif k == "beinsum":
        _, spec, ab, A, B, cpf, C = s
        einsums.einsum(spec, t[C], t[A], t[B], c_pf=cpf, ab_pf=ab)
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
    def mk(prefix, arrays):
        out = []
        for idx, arr in enumerate(arrays):
            tn = einsums.create_zero_tensor(f"{name}_{prefix}{idx}", list(arr.shape))
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


def _usable(*pools):
    """A trial is only meaningful if the oracle stayed numerically sane. Bigger
    programs with repeated accumulation can overflow to inf/NaN (or to values so
    large that a 1e-5 relative tolerance is dominated by fp noise); such cases
    test floating-point overflow, not pass soundness, so we skip them."""
    for pool in pools:
        for arr in pool:
            if arr.size and (not np.all(np.isfinite(arr)) or np.max(np.abs(arr)) > 1e8):
                return False
    return True


def check_program(prog, m_arrays, v_arrays, t_arrays, label):
    om = [a.copy() for a in m_arrays]
    ov = [a.copy() for a in v_arrays]
    ot = [a.copy() for a in t_arrays]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        interp_np(prog, om, ov, ot)
    if not _usable(om, ov, ot):
        pytest.skip("oracle overflowed — numerically degenerate program")

    rm, rv, rt = _run_program(prog, m_arrays, v_arrays, t_arrays, f"{label}_raw", optimize=False)
    pm, pv, pt = _run_program(prog, m_arrays, v_arrays, t_arrays, f"{label}_opt", optimize=True)

    def _cmp(stage, got, oracle, kind):
        for idx in range(len(oracle)):
            if not np.allclose(got[idx], oracle[idx], rtol=RTOL, atol=ATOL):
                raise AssertionError(
                    f"{stage} disagrees with oracle on {kind}{idx}"
                    f"{' (a pass miscompiled)' if stage == 'OPTIMIZED' else ''}\n"
                    f"program={prog!r}\ngot=\n{got[idx]}\noracle=\n{oracle[idx]}"
                )

    for stage, (gm, gv, gt) in (("RAW", (rm, rv, rt)), ("OPTIMIZED", (pm, pv, pt))):
        _cmp(stage, gm, om, "m")
        _cmp(stage, gv, ov, "v")
        _cmp(stage, gt, ot, "t")


def _seed_arrays(rng):
    m = [rng.standard_normal(sh) for sh in MAT_SHAPES]
    v = [rng.standard_normal((L,)) for L in VEC_LENS]
    t = [rng.standard_normal(sh) for sh in R3_SHAPES]
    return m, v, t


def _square_seed_arrays(rng, n_mats=4, n_vecs=3, n=3, n_r3=2):
    """Square N×N matrices, length-N vectors, and a couple N×N×2 rank-3
    tensors — for the hand-written regressions that index a small fixed pool."""
    m = [rng.standard_normal((n, n)) for _ in range(n_mats)]
    v = [rng.standard_normal((n,)) for _ in range(n_vecs)]
    t = [rng.standard_normal((n, n, 2)) for _ in range(n_r3)]
    return m, v, t


# ──────────────────────────────────────────────────────────────────────────
# Fuzz: many seeds, all four structural shapes
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("seed", range(400))
def test_fuzz_flat(seed):
    rng = np.random.default_rng(seed)
    prog = _gen_block(rng, depth=0, max_stmts=10)
    check_program(prog, *_seed_arrays(rng), f"flat{seed}")


@pytest.mark.parametrize("seed", range(400))
def test_fuzz_with_control_flow(seed):
    rng = np.random.default_rng(10_000 + seed)
    prog = _gen_block(rng, depth=3, max_stmts=6)
    check_program(prog, *_seed_arrays(rng), f"cf{seed}")


@pytest.mark.parametrize("seed", range(300))
def test_fuzz_deep_nesting(seed):
    rng = np.random.default_rng(50_000 + seed)
    prog = _gen_block(rng, depth=4, max_stmts=4)
    check_program(prog, *_seed_arrays(rng), f"deep{seed}")


# ──────────────────────────────────────────────────────────────────────────
# Fixed regression programs (index a small square pool).
# ──────────────────────────────────────────────────────────────────────────


def test_regression_mutable_reuse_chain():
    prog = [
        ("axpby", 1.0, 0, 0.0, 1),
        ("axpby", 1.0, 0, 0.0, 2),
        ("scale", 0.5, 1),
        ("axpy", 2.0, 1, 2),
        ("gemm", 1.0, 1, 2, 1.0, 3),
    ]
    check_program(prog, *_square_seed_arrays(np.random.default_rng(123)), "reuse_chain")


def test_regression_nested_loop():
    prog = [("loop", 3, [("scale", 0.9, 0), ("loop", 2, [("axpy", 0.5, 0, 1), ("gemm", 1.0, 0, 1, 1.0, 2)])])]
    check_program(prog, *_square_seed_arrays(np.random.default_rng(456)), "nested_loop")


def test_regression_sequential_loops():
    prog = [
        ("loop", 3, [("scale", 0.8, 0), ("axpy", 1.0, 0, 1)]),
        ("loop", 2, [("scale", 1.1, 1), ("axpby", 0.5, 1, 0.5, 2)]),
    ]
    check_program(prog, *_square_seed_arrays(np.random.default_rng(789)), "seq_loops")


def test_regression_loop_with_conditional():
    prog_then = [("loop", 2, [("cond", True, [("axpy", 0.5, 0, 1)], [("scale", 0.0, 1)]), ("gemm", 1.0, 0, 1, 0.0, 2)])]
    prog_else = [("loop", 2, [("cond", False, [("axpy", 0.5, 0, 1)], [("scale", 2.0, 1)]), ("gemm", 1.0, 0, 1, 0.0, 2)])]
    m, v, t = _square_seed_arrays(np.random.default_rng(1011))
    check_program(prog_then, m, v, t, "loop_cond_then")
    check_program(prog_else, m, v, t, "loop_cond_else")


def test_regression_accumulating_gemm_in_loop():
    prog = [("loop", 2, [("gemm", -0.8137, 3, 3, 1.0, 2), ("gemm", -0.1093, 1, 0, 1.0, 3)])]
    check_program(prog, *_square_seed_arrays(np.random.default_rng(5026)), "accum_gemm_loop")


def test_regression_einsum_accumulate_in_loop():
    prog = [("loop", 3, [("einsum", "ij <- ik ; kj", 1.0, 0, 1, 1.0, 2)])]
    check_program(prog, *_square_seed_arrays(np.random.default_rng(2024)), "einsum_accum_loop")


def test_regression_einsum_transpose_patterns():
    prog = [
        ("einsum", "ij <- ik ; kj", 1.0, 0, 1, 0.0, 2),
        ("einsum", "ij <- ki ; kj", 1.0, 0, 1, 0.0, 3),
        ("einsum", "ij <- ik ; jk", 0.5, 2, 3, 0.0, 0),
    ]
    check_program(prog, *_square_seed_arrays(np.random.default_rng(333)), "einsum_patterns")


def test_regression_permute_accumulate_in_loop():
    prog = [("perm", 1.0, 0.0, 0, 1), ("loop", 3, [("perm", 0.5, 1.0, 0, 1)])]
    check_program(prog, *_square_seed_arrays(np.random.default_rng(666)), "perm_accum")


def test_regression_symm_gemm():
    prog = [("loop", 2, [("symm", 0, 1, 2), ("scale", 0.5, 3)])]
    check_program(prog, *_square_seed_arrays(np.random.default_rng(444)), "symm")


def test_regression_gemv_ger_in_loop():
    prog = [("loop", 3, [("gemv", 0.7, 0, 0, 1.0, 1), ("ger", 0.3, 1, 0, 2)])]
    check_program(prog, *_square_seed_arrays(np.random.default_rng(555)), "gemv_ger")


def test_regression_batched_einsum_in_loop():
    """Accumulating batched einsum (rank-3) inside a loop reads its destination
    each iteration — exercises the BatchedGemm path under LIH/scheduling."""
    prog = [
        ("beinsum", "ijb <- ikb ; kjb", 1.0, 0, 1, 0.0, 2),   # t2 = t0@t1 (batched, overwrite)
        ("loop", 3, [("beinsum", "ijb <- ikb ; kjb", 0.5, 0, 1, 1.0, 2)]),  # t2 += 0.5*(t0@t1), must stay in loop
    ]
    rng = np.random.default_rng(1717)
    m: list[np.ndarray] = []
    v: list[np.ndarray] = []
    t = [rng.standard_normal((3, 3, 2)) for _ in range(3)]  # t0,t1 inputs; t2 output
    check_program(prog, m, v, t, "batched_einsum")


def test_regression_view_alias_ordering():
    """Mix full-matrix writes and view (sub-block) writes on the same matrix,
    then read the whole matrix — stresses the scheduler's alias resolution
    (a write through a view must be seen as a write to the parent)."""
    prog = [
        ("scale", 0.5, 0),                 # whole m0 *= 0.5
        ("vscale", 3.0, 0, 1, 3, 1, 3),    # m0[1:3,1:3] *= 3
        ("vaxpy", 1.0, 1, 0, 0, 2, 0, 2),  # m0[0:2,0:2] += m1   (m1 is 3x3; block 2x2 -> use a 2x2 src)
        ("gemm", 1.0, 0, 2, 0.0, 3),       # m3 = m0 @ m2  (reads the aliased m0)
    ]
    # m1 must be 2x2 to match the vaxpy block; give a custom square+small pool.
    rng = np.random.default_rng(2929)
    m = [rng.standard_normal((3, 3)), rng.standard_normal((2, 2)),
         rng.standard_normal((3, 3)), rng.standard_normal((3, 3))]
    v: list[np.ndarray] = []
    t: list[np.ndarray] = []
    check_program(prog, m, v, t, "view_alias")


def test_regression_overlapping_views():
    """Two views of the same matrix with *overlapping* regions, written by
    different ops, then the whole matrix read — the partial writes must keep
    their relative order (both resolve to the same owner)."""
    prog = [
        ("vscale", 2.0, 0, 0, 2, 0, 2),     # m0[0:2,0:2] *= 2
        ("vaxpy", 1.0, 1, 0, 1, 3, 1, 3),   # m0[1:3,1:3] += m1 (overlaps [1:2,1:2])
        ("vscale", 0.5, 0, 0, 3, 0, 3),     # m0[0:3,0:3] *= 0.5 (covers both)
        ("gemm", 1.0, 0, 2, 0.0, 3),        # m3 = m0 @ m2
    ]
    rng = np.random.default_rng(8642)
    m = [rng.standard_normal((3, 3)), rng.standard_normal((2, 2)),
         rng.standard_normal((3, 3)), rng.standard_normal((3, 3))]
    check_program(prog, m, [], [], "overlap_views")


def test_regression_element_transform_in_loop():
    """In-place element-wise transform inside a loop, mixed with a full write."""
    prog = [("loop", 3, [("etransform", 2, 0), ("axpy", 0.5, 1, 0)])]
    check_program(prog, *_square_seed_arrays(np.random.default_rng(9753)), "etransform_loop")


def test_regression_view_in_loop():
    """A view write inside a loop, interleaved with a full-tensor write."""
    prog = [
        ("loop", 3, [
            ("vscale", 0.9, 0, 0, 2, 0, 2),   # m0[0:2,0:2] *= 0.9
            ("scale", 1.0, 0),                # whole-tensor touch (identity scale)
        ]),
    ]
    check_program(prog, *_square_seed_arrays(np.random.default_rng(3131)), "view_loop")


def test_regression_mixed_shape_einsum_chain():
    rng = np.random.default_rng(2468)
    m = [rng.standard_normal((2, 4)), rng.standard_normal((4, 3)),
         rng.standard_normal((2, 3)), rng.standard_normal((3, 2))]
    v: list[np.ndarray] = []
    t: list[np.ndarray] = []
    prog = [
        ("einsum", "ij <- ik ; kj", 1.0, 0, 1, 0.0, 2),
        ("perm", 1.0, 0.0, 2, 3),
        ("scale", 0.5, 2),
    ]
    check_program(prog, m, v, t, "mixed_einsum")


# ══════════════════════════════════════════════════════════════════════════
# Harder attack modes — same program generator, meaner ways of running it.
# ══════════════════════════════════════════════════════════════════════════

import einsums._core.graph as _G  # noqa: E402  (pass classes for random pipelines)

# Passes exposed to Python that are individually sound and (should be)
# order-independent for correctness on eager tensors. A random permutation that
# miscompiles is either a real bug or an undocumented ordering constraint.
_SAFE_PASSES = [
    "ScaleAbsorption", "ElementWiseFusion", "ChainParenthesization",
    "ConstantFolding", "CSE", "DeadNodeElimination", "LoopInvariantHoisting",
    "SymmetryPropagation", "MemoryPlanning", "InplaceOptimization", "Reorder",
]


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


@pytest.mark.parametrize("seed", range(300))
def test_fuzz_reexecution(seed):
    """Execute the optimized graph twice without resetting inputs and compare to
    the oracle applied twice. Iterative solvers replay a graph repeatedly, so a
    pass that frees a still-needed buffer or corrupts state on replay must fail
    here even though a single execution passes."""
    rng = np.random.default_rng(20_000 + seed)
    prog = _gen_block(rng, depth=3, max_stmts=6)
    m, v, t = _seed_arrays(rng)
    oracle = _oracle(prog, m, v, t, runs=2)
    if not _usable(*oracle):
        pytest.skip("oracle overflowed — numerically degenerate program")

    g, mats, vecs, r3 = _build(prog, m, v, t, f"replay{seed}")
    g.apply(cg.default_pass_manager())
    g.execute()
    g.execute()
    got = ([np.asarray(x).copy() for x in mats],
           [np.asarray(x).copy() for x in vecs],
           [np.asarray(x).copy() for x in r3])
    _assert_pools(got, oracle, prog, "REEXECUTED")


@pytest.mark.parametrize("seed", range(400))
def test_fuzz_random_pipeline(seed):
    """Apply a random *permutation* of the individually-sound passes (rather than
    the curated default order) and demand the result still matches the oracle.
    This is the strongest interaction test: any ordering that miscompiles is a
    real soundness bug or an undocumented ordering dependency."""
    rng = np.random.default_rng(30_000 + seed)
    prog = _gen_block(rng, depth=3, max_stmts=6)
    m, v, t = _seed_arrays(rng)
    oracle = _oracle(prog, m, v, t)
    if not _usable(*oracle):
        pytest.skip("oracle overflowed — numerically degenerate program")

    order = list(_SAFE_PASSES)
    rng.shuffle(order)

    g, mats, vecs, r3 = _build(prog, m, v, t, f"rndpipe{seed}")
    pm = cg.PassManager()
    for name in order:
        pm.add(getattr(_G, name)())
    g.apply(pm)
    g.execute()
    got = ([np.asarray(x).copy() for x in mats],
           [np.asarray(x).copy() for x in vecs],
           [np.asarray(x).copy() for x in r3])
    _assert_pools(got, oracle, prog, "RANDOM-PIPELINE", extra=f"  order={order}")


@pytest.mark.parametrize("seed", range(200))
def test_fuzz_double_optimize(seed):
    """Apply the default pass manager twice before executing. A non-idempotent
    pass (one that doesn't reach a fixpoint, or that mis-handles its own prior
    output) would diverge on the second application."""
    rng = np.random.default_rng(40_000 + seed)
    prog = _gen_block(rng, depth=3, max_stmts=4)
    m, v, t = _seed_arrays(rng)
    oracle = _oracle(prog, m, v, t)
    if not _usable(*oracle):
        pytest.skip("oracle overflowed — numerically degenerate program")

    g, mats, vecs, r3 = _build(prog, m, v, t, f"dblopt{seed}")
    g.apply(cg.default_pass_manager())
    g.apply(cg.default_pass_manager())
    g.execute()
    got = ([np.asarray(x).copy() for x in mats],
           [np.asarray(x).copy() for x in vecs],
           [np.asarray(x).copy() for x in r3])
    _assert_pools(got, oracle, prog, "DOUBLE-OPTIMIZE")


@pytest.mark.parametrize("seed", range(300))
def test_fuzz_random_pipeline_replay(seed):
    """The meanest combination: optimize with a *random* pass order, then
    execute *twice*. A randomly-optimized graph must still replay correctly —
    catches interaction bugs that only manifest on re-execution (e.g. a Free
    inserted by one pass ordering that a second run then needs)."""
    rng = np.random.default_rng(60_000 + seed)
    prog = _gen_block(rng, depth=3, max_stmts=6)
    m, v, t = _seed_arrays(rng)
    oracle = _oracle(prog, m, v, t, runs=2)
    if not _usable(*oracle):
        pytest.skip("oracle overflowed — numerically degenerate program")

    order = list(_SAFE_PASSES)
    rng.shuffle(order)

    g, mats, vecs, r3 = _build(prog, m, v, t, f"rndreplay{seed}")
    pm = cg.PassManager()
    for name in order:
        pm.add(getattr(_G, name)())
    g.apply(pm)
    g.execute()
    g.execute()
    got = ([np.asarray(x).copy() for x in mats],
           [np.asarray(x).copy() for x in vecs],
           [np.asarray(x).copy() for x in r3])
    _assert_pools(got, oracle, prog, "RANDOM-PIPELINE+REPLAY", extra=f"  order={order}")
