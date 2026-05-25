# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""Differential fuzzer for the numpy-style ergonomics layer.

Generates a random program of numpy-style operations — operators (``@`` /
``+`` / ``-`` / ``*`` / ``/`` / unary), in-place RMW (``+=`` / ``-=`` / ``*=``
/ ``/=``), ``.T`` / ``.transpose`` / ``.swapaxes``, slicing and rank-reducing
indexing, and the ``.sum`` / ``.mean`` / ``.max`` reductions — over a pool of
tensors, then runs the *same* program four ways and demands they all agree:

  1. **numpy oracle** — the program applied to numpy arrays.
  2. **einsums eager** — the same Python expressions on einsums tensors.
  3. **einsums capture** — the same expressions recorded into a ``cg.Graph``,
     then ``execute()``-d with no optimization passes.
  4. **einsums optimized** — the captured graph after ``default_pass_manager()``,
     then executed. If #3 matches but #4 doesn't, a pass miscompiled the graph.

In-place ops only target owning storage (not views) so the oracle stays simple;
they stress the read-modify-write hazards under capture + the alias-aware
scheduler that the optimization passes rely on.

Seed range note: the committed range (200 seeds) is green on all four arms.
It used to be capped at 80 to dodge bug-optimizer-scale-view-alias — a
view-of-a-view (e.g. ``(A.T)[1:]``) created inside capture took an eager
raw-slice path with ``aliases == 0``, so the scheduler (Reorder) couldn't see
that a reduction over it depended on an in-place ``Scale`` of the owning
tensor and floated the read ahead of the write. The fix wired slicing a view
through ``cg.view``/``view_indexed`` (now instantiated for ``RuntimeTensorView``
parents, with the capture-aware ``__getitem__`` installed on the view classes),
so the slice aliases its parent chain. The cap is lifted; raise it further to
widen coverage.

Because numpy arrays and einsums tensors share the operator/method syntax, a
single interpreter (:func:`_run_chain`) drives all three; only the operand
pool (and, for #3, the surrounding ``cg.capture``) differ.

The point is the **eager-vs-capture** axis: it systematically catches the
capture-only bugs that were previously found by hand. This harness already
turned up two real ones:

  * range-slice views reported full-parent dims at capture time, so operators
    that size their output / divisor from a captured view's dims (``@``,
    ``.mean``, the elementwise allocators) mis-sized them;
  * a slice-of-a-slice collided in ``get_slot`` (the inner view's placeholder
    shared the parent's data pointer), so the outer slice resolved the wrong
    parent.

Both are fixed in ``cg::view_runtime``'s placeholder (constant Range bounds now
contribute their real extent and offset). The generator only emits valid
shape-compatible programs (it tracks numpy shapes as it builds), and reductions
are kept terminal so numpy's scalar broadcast can't diverge from einsums'
no-broadcast semantics. Failures print the seed + program so they reproduce.
"""

from __future__ import annotations

import numpy as np
import pytest

import einsums

# Don't hang on the attach-debugger prompt if a regression ever hard-faults
# (no-op once the runtime is already up).
einsums.rc.debug_no_attach_debugger = True

import einsums.graph as cg
from einsums.testing import ALL_DTYPES

_DIMS = (2, 3, 4)
_TOL = {"float32": (1e-3, 1e-3), "complex64": (1e-3, 1e-3),
        "float64": (1e-5, 1e-5), "complex128": (1e-5, 1e-5)}
_CAP = {"float32": 1e3, "complex64": 1e3, "float64": 1e8, "complex128": 1e8}
_N_STEPS = 6


def _seed_shapes(rng):
    """Three matrices + two vectors over dims {2,3,4}."""
    return ([(int(rng.choice(_DIMS)), int(rng.choice(_DIMS))) for _ in range(3)]
            + [(int(rng.choice(_DIMS)),) for _ in range(2)])


# Ops that mutate pool[args[0]] in place (RMW) instead of appending a result.
_INPLACE = frozenset({"iadd", "isub", "iscale", "idiv"})


def _gen_program(rng, shapes, n_steps):
    """A list of ``(op, arg_indices, scalar)`` steps. Functional ops append a
    new pool tensor; in-place ops (``_INPLACE``) mutate an existing slot.
    Shapes are tracked so only valid operands are chosen; ``is_view`` tracks
    which slots are zero-copy views (transpose/slice/index) so in-place ops
    only target owning storage (mutating through a view would alias its
    parent — sound but excluded here to keep the oracle simple)."""
    shapes = list(shapes)
    is_view = [False] * len(shapes)
    alias_root = list(range(len(shapes)))  # ultimate owning tensor each slot aliases
    steps = []
    _VIEW_OPS = ("transpose", "slice", "index")
    for _ in range(n_steps):
        # candidate := (op, args, out_shape | None, result_is_view); None => in-place
        cands = []
        for i, si in enumerate(shapes):
            owning_i = not is_view[i]
            for op in ("smul", "sadd", "ssub", "sdiv", "neg"):
                cands.append((op, (i,), si, False))
            if len(si) == 2:
                cands.append(("transpose", (i,), (si[1], si[0]), True))
            if si[0] >= 2:                                  # slice: drop row 0
                cands.append(("slice", (i,), (si[0] - 1, *si[1:]), True))
            if len(si) >= 2:                                # index: rank-reduce
                cands.append(("index", (i,), si[1:], True))
            if owning_i:                                    # in-place scalar
                cands.append(("iscale", (i,), None, False))
                cands.append(("idiv", (i,), None, False))
            for j, sj in enumerate(shapes):
                if si == sj:
                    for op in ("add", "sub", "had"):
                        cands.append((op, (i, j), si, False))
                    # In-place accumulate, but never when the source aliases the
                    # target (e.g. A -= A.T) — that's an order-dependent
                    # read/write overlap that numpy resolves by copying and BLAS
                    # axpy does not, so the two legitimately disagree.
                    if owning_i and alias_root[j] != i:
                        cands.append(("iadd", (i, j), None, False))
                        cands.append(("isub", (i, j), None, False))
                if len(si) == 2 and len(sj) == 2 and si[1] == sj[0]:
                    cands.append(("matmul", (i, j), (si[0], sj[1]), False))
                if len(si) == 2 and len(sj) == 1 and si[1] == sj[0]:
                    cands.append(("matmul", (i, j), (si[0],), False))
                if len(si) == 1 and len(sj) == 2 and si[0] == sj[0]:
                    cands.append(("matmul", (i, j), (sj[1],), False))
        op, args, out_shape, res_view = cands[int(rng.integers(len(cands)))]
        sc = float(rng.uniform(0.5, 2.0)) * (1.0 if rng.random() < 0.5 else -1.0)
        steps.append((op, args, sc))
        if out_shape is not None:        # functional op grows the pool
            shapes.append(out_shape)
            is_view.append(res_view)
            # A view aliases its source's root; an owning result is its own root.
            alias_root.append(alias_root[args[0]] if op in _VIEW_OPS else len(alias_root))
    return steps


def _apply(step, pool):
    """One step's expression — identical syntax for numpy arrays and einsums
    tensors, which is why it serves all three backends."""
    op, args, sc = step
    a = pool[args[0]]
    if op == "add": return a + pool[args[1]]
    if op == "sub": return a - pool[args[1]]
    if op == "had": return a * pool[args[1]]
    if op == "matmul": return a @ pool[args[1]]
    if op == "smul": return sc * a
    if op == "sadd": return a + sc
    if op == "ssub": return a - sc
    if op == "sdiv": return a / sc
    if op == "neg": return -a
    if op == "transpose": return a.T
    if op == "slice": return a[1:]
    if op == "index": return a[0]
    raise AssertionError(f"unknown op {op}")


def _run_chain(steps, pool):
    pool = list(pool)
    for op, args, sc in steps:
        if op in _INPLACE:
            # Augmented assignment exercises __iadd__/__isub__/__imul__/__itruediv__
            # (in place for numpy and eager; an RMW node under capture). Reassigning
            # the slot is fine — those dunders return the same (mutated) object.
            if op == "iadd": pool[args[0]] += pool[args[1]]
            elif op == "isub": pool[args[0]] -= pool[args[1]]
            elif op == "iscale": pool[args[0]] *= sc
            elif op == "idiv": pool[args[0]] /= sc
        else:
            pool.append(_apply((op, args, sc), pool))
    return pool


def _reductions(pool, real):
    out = []
    for t in pool:
        out.append(t.sum())
        out.append(t.mean())
        if real:
            out.append(t.max())
    return out


def _as_flat(x):
    return np.asarray(x).reshape(-1)


@pytest.mark.parametrize("dtype", ALL_DTYPES)
@pytest.mark.parametrize("seed", range(200))
def test_fuzz_ergonomics(seed, dtype):
    rng = np.random.default_rng(seed)
    real = not dtype.startswith("complex")
    shapes = _seed_shapes(rng)

    def mk(sh):
        a = rng.standard_normal(sh)
        if not real:
            a = a + 1j * rng.standard_normal(sh)
        return a.astype(dtype)

    inputs = [mk(s) for s in shapes]
    steps = _gen_program(rng, shapes, _N_STEPS)

    # numpy oracle
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        npool = _run_chain(steps, [a.copy() for a in inputs])
        nred = _reductions(npool, real)
    cap = _CAP[dtype]
    if any(p.size and (not np.all(np.isfinite(p)) or np.max(np.abs(p)) > cap) for p in npool):
        pytest.skip("oracle overflowed — numerically degenerate program")

    # Each einsums arm gets its OWN fresh inputs — in-place ops mutate them.
    def fresh(prefix):
        return [einsums.asarray(a, name=f"{prefix}{i}") for i, a in enumerate(inputs)]

    # einsums eager
    epool = _run_chain(steps, fresh("e"))
    ered = _reductions(epool, real)

    # einsums capture -> execute  (no optimization passes)
    g = cg.Graph(f"fuzz_ergo_{seed}")
    with cg.capture(g):
        cpool = _run_chain(steps, fresh("c"))
        cred = _reductions(cpool, real)
    g.execute()

    # einsums capture -> default passes -> execute. If the raw arm matches but
    # this one doesn't, a pass miscompiled the graph.
    g_opt = cg.Graph(f"fuzz_ergo_opt_{seed}")
    with cg.capture(g_opt):
        opool = _run_chain(steps, fresh("o"))
        ored = _reductions(opool, real)
    g_opt.apply(cg.default_pass_manager())
    g_opt.execute()

    rtol, atol = _TOL[dtype]

    def cmp(label, got, oracle):
        for k in range(len(oracle)):
            a, b = _as_flat(got[k]), _as_flat(oracle[k])
            if a.shape != b.shape or not np.allclose(a, b, rtol=rtol, atol=atol):
                raise AssertionError(
                    f"{label}[{k}] disagrees with numpy (seed={seed}, dtype={dtype})\n"
                    f"  steps={steps}\n  got={a}\n  oracle={b}"
                )

    cmp("eager-pool", epool, npool)
    cmp("eager-reduction", ered, nred)
    cmp("capture-pool", cpool, npool)
    cmp("capture-reduction", cred, nred)
    cmp("optimized-pool", opool, npool)
    cmp("optimized-reduction", ored, nred)
