# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""Differential fuzzer for the numpy-style ergonomics layer.

Generates a random program of numpy-style operations — operators (``@`` /
``+`` / ``-`` / ``*`` / ``/`` / unary), ``.T`` / ``.transpose`` /
``.swapaxes``, slicing and rank-reducing indexing, and the ``.sum`` /
``.mean`` / ``.max`` reductions — over a pool of tensors, then runs the *same*
program three ways and demands they all agree:

  1. **numpy oracle** — the program applied to numpy arrays.
  2. **einsums eager** — the same Python expressions on einsums tensors.
  3. **einsums capture** — the same expressions recorded into a ``cg.Graph``,
     then ``execute()``-d.

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


def _gen_program(rng, shapes, n_steps):
    """A list of ``(op, arg_indices, scalar)`` steps, each producing a new
    pool tensor. Shapes are tracked so only valid operands are chosen."""
    shapes = list(shapes)
    steps = []
    for _ in range(n_steps):
        cands = []
        for i, si in enumerate(shapes):
            for op in ("smul", "sadd", "ssub", "sdiv", "neg"):
                cands.append((op, (i,), si))
            if len(si) == 2:
                cands.append(("transpose", (i,), (si[1], si[0])))
            if si[0] >= 2:                                  # slice: drop row 0
                cands.append(("slice", (i,), (si[0] - 1, *si[1:])))
            if len(si) >= 2:                                # index: rank-reduce
                cands.append(("index", (i,), si[1:]))
            for j, sj in enumerate(shapes):
                if si == sj:
                    for op in ("add", "sub", "had"):
                        cands.append((op, (i, j), si))
                if len(si) == 2 and len(sj) == 2 and si[1] == sj[0]:
                    cands.append(("matmul", (i, j), (si[0], sj[1])))
                if len(si) == 2 and len(sj) == 1 and si[1] == sj[0]:
                    cands.append(("matmul", (i, j), (si[0],)))
                if len(si) == 1 and len(sj) == 2 and si[0] == sj[0]:
                    cands.append(("matmul", (i, j), (sj[1],)))
        op, args, out_shape = cands[int(rng.integers(len(cands)))]
        sc = float(rng.uniform(0.5, 2.0)) * (1.0 if rng.random() < 0.5 else -1.0)
        steps.append((op, args, sc))
        shapes.append(out_shape)
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
    for step in steps:
        pool.append(_apply(step, pool))
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
@pytest.mark.parametrize("seed", range(80))
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

    # einsums eager
    epool = _run_chain(steps, [einsums.asarray(a, name=f"e{i}") for i, a in enumerate(inputs)])
    ered = _reductions(epool, real)

    # einsums capture -> execute
    g = cg.Graph(f"fuzz_ergo_{seed}")
    with cg.capture(g):
        cpool = _run_chain(steps, [einsums.asarray(a, name=f"c{i}") for i, a in enumerate(inputs)])
        cred = _reductions(cpool, real)
    g.execute()

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
