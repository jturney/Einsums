# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""Hypothesis differential: graph-captured element-wise family vs numpy.

Covers scale / axpy / axpby / direct_product / direct_division, the
named elementary ops (exp, log, log1p, expm1, sign), and the real-only
maximum / minimum / max_axes / min_axes / compare / where with
non-contiguous view operands, degenerate (size-1) dims, all four dtypes,
prefactors (incl. accumulation), and the optional default pass manager.

This is the TensorImpl vectorization surface (get_incx / is_totally_vectorable /
query_vectorable_params + the per-op layout guards). The ``@example`` entries
pin the reducers that were once miscompiled:
  * axpy into a (1,2,1) view (strides 2,1,2): a size-1 boundary dim's inflated
    stride was taken as the increment, so the op under-processed.
  * direct_product into a transposed (3,3,1) C view: the coarse is_column_major
    flag matched while the stride order didn't, pairing mismatched elements.
"""
from __future__ import annotations

import itertools

import numpy as np
from hypothesis import HealthCheck, example, given, settings
from _sanitizer_scaling import sanitizer_examples
from hypothesis import strategies as st

import einsums
import einsums.graph as cg
from _dtype_draws import DTYPES, assert_rounding_close, is_complex, random_array, rounded

_ctr = itertools.count()


def _nm() -> str:
    return f"hew{next(_ctr)}"


def _mkv(arr, use_view, dt, rng):
    if not use_view or arr.ndim < 2:
        return einsums.asarray(arr, dtype=dt)
    perm = list(rng.permutation(arr.ndim))
    if perm == list(range(arr.ndim)):
        perm = perm[::-1]
    t = einsums.asarray(np.ascontiguousarray(np.transpose(arr, perm)), dtype=dt)
    return t.permute_view(list(np.argsort(perm)))


def _rnd(shape, dt, rng):
    return random_array(shape, dt, rng)


def _shift_from_zero(B0, dt):
    """Move each element of ``B0`` two units away from zero.

    Rounded to ``dt`` again, so the tensor holds the divisor the oracle divides by.
    """
    shifted = B0 + 2 * np.sign(B0.real if is_complex(dt) else B0)
    return shifted.astype(dt).astype(B0.dtype)


@st.composite
def _shape(draw):
    r = draw(st.integers(1, 3))
    return tuple(draw(st.integers(1, 4)) for _ in range(r))


_ELEMENTARY = {"exp": np.exp, "log": np.log, "log1p": np.log1p, "expm1": np.expm1, "sign": np.sign}
_REAL_ONLY_ELEMENTARY = ("log1p", "expm1", "sign")


_MAX_MIN_OPS = ("maximum", "minimum", "max_axes", "min_axes", "compare", "compare_scalar", "where")
_COMPARISONS = {"gt": np.greater, "ge": np.greater_equal, "lt": np.less, "le": np.less_equal, "eq": np.equal, "ne": np.not_equal}


@given(op=st.sampled_from(["scale", "axpy", "axpby", "direct_product", "direct_division", "element_op", *_MAX_MIN_OPS]),
       element_op=st.sampled_from(sorted(_ELEMENTARY)),
       shape=_shape(), alpha=st.sampled_from([1.0, -2.0, 0.5]), beta=st.sampled_from([0.0, 1.0, -1.5]),
       dt=DTYPES,
       va=st.booleans(), vb=st.booleans(), vc=st.booleans(), passes=st.booleans(),
       axes_mask=st.integers(0, 7), special=st.booleans(), cmp=st.sampled_from(sorted(_COMPARISONS)))
@settings(max_examples=sanitizer_examples(300), deadline=None,
          suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large, HealthCheck.filter_too_much])
@example(op="axpy", element_op="exp", shape=(1, 2, 1), alpha=1.0, beta=0.0, dt="float64", va=False, vb=True, vc=False,
         passes=False, axes_mask=0, special=False, cmp="gt")
@example(op="direct_product", element_op="exp", shape=(3, 3, 1), alpha=1.0, beta=0.0, dt="complex128", va=False, vb=False,
         vc=True, passes=False, axes_mask=0, special=False, cmp="gt")
def test_hyp_elementwise_diff(op, element_op, shape, alpha, beta, dt, va, vb, vc, passes, axes_mask, special, cmp):
    rng = np.random.default_rng(0)
    g = cg.Graph(_nm())
    if op in _MAX_MIN_OPS:
        if is_complex(dt):
            dt = "float64" if dt == "complex128" else "float32"
        A0 = _rnd(shape, dt, rng)
        B0 = _rnd(shape, dt, rng)
        if special:
            # NaN, -inf and both zeros, where a naive comparison or a seeded fold goes wrong.
            flat_a, flat_b = A0.reshape(-1), B0.reshape(-1)
            flat_a[0] = np.nan
            flat_b[-1] = -np.inf
            flat_a[len(flat_a) // 2] = -0.0
            flat_b[len(flat_b) // 2] = 0.0
        At, Bt = _mkv(A0, va, dt, rng), _mkv(B0, vb, dt, rng)
        if special:
            # Ties too, so ge/le/eq see equality and not only random distinct values.
            B0.reshape(-1)[0] = A0.reshape(-1)[-1] if A0.size > 1 else B0.reshape(-1)[0]
            Bt = _mkv(B0, vb, dt, rng)
        if op in ("compare", "compare_scalar"):
            rhs = B0 if op == "compare" else 0.0
            with np.errstate(invalid="ignore"):
                oracle = _COMPARISONS[cmp](A0, rhs).astype(A0.dtype)
            out = einsums.create_zero_tensor(_nm(), list(shape), dtype=dt)
            with cg.capture(g):
                einsums.linalg.compare(out, At, Bt if op == "compare" else 0.0, cmp)
        elif op == "where":
            mask0 = (rng.integers(0, 3, size=shape) - 1).astype(A0.dtype)  # -1, 0, 1: any nonzero is true
            if special:
                mask0.reshape(-1)[-1] = np.nan  # NaN is truthy
                A0.reshape(-1)[-1] = np.inf
                B0.reshape(-1)[-1] = np.nan  # the branch not taken must not leak
                At, Bt = _mkv(A0, va, dt, rng), _mkv(B0, vb, dt, rng)
            oracle = np.where(mask0.astype(bool), A0, B0)
            Mt = _mkv(mask0, vc, dt, rng)
            out = einsums.create_zero_tensor(_nm(), list(shape), dtype=dt)
            with cg.capture(g):
                einsums.linalg.where(out, Mt, At, Bt)
        elif op in ("maximum", "minimum"):
            oracle = getattr(np, op)(A0, B0)
            out = einsums.create_zero_tensor(_nm(), list(shape), dtype=dt)
            with cg.capture(g):
                getattr(einsums.linalg, op)(out, At, Bt)
        else:
            axes = [a for a in range(len(shape)) if axes_mask & (1 << a)]
            oracle = getattr(A0, op[:3])(axis=tuple(axes)) if axes else A0.copy()
            out = einsums.create_zero_tensor(_nm(), list(np.shape(oracle)), dtype=dt)
            with cg.capture(g):
                getattr(einsums.linalg, op)(out, At, axes)
        if passes:
            g.apply(cg.default_pass_manager())
        g.execute()
        g.execute()  # an overwrite replays to the same answer
        # Selections, not arithmetic: the answer is one of the inputs, exactly, sign of zero included.
        got = np.asarray(out)
        np.testing.assert_array_equal(got, oracle, err_msg=f"op={op} cmp={cmp} shape={shape} dt={dt} va={va} vb={vb} axes_mask={axes_mask}")
        np.testing.assert_array_equal(np.signbit(got), np.signbit(oracle), err_msg=f"op={op} signed zeros")
        return
    if op == "element_op":
        if is_complex(dt) and element_op in _REAL_ONLY_ELEMENTARY:
            element_op = "log"
        A0 = _rnd(shape, dt, rng)
        if element_op in ("log", "log1p"):
            # Stay inside the domain: |x| + 0.25 has a positive real part and keeps log1p finite.
            A0 = rounded(np.abs(A0.real) + 0.25 + (1j * A0.imag if is_complex(dt) else 0), dt)
        oracle = _ELEMENTARY[element_op](A0)
        # Each map is applied to one rounded element, so its error is relative to the result,
        # plus |x| for a log landing near zero, where the input's own rounding dominates.
        scale = np.abs(oracle) + np.abs(A0)
        At = _mkv(A0, va, dt, rng)
        with cg.capture(g):
            einsums.linalg.element_transform(At, element_op)
        out = At
    elif op == "scale":
        A0 = _rnd(shape, dt, rng)
        oracle = alpha * A0
        scale = np.abs(oracle)
        At = _mkv(A0, va, dt, rng)
        with cg.capture(g):
            einsums.linalg.scale(alpha, At)
        out = At
    elif op == "axpy":
        X0 = _rnd(shape, dt, rng)
        Y0 = _rnd(shape, dt, rng)
        oracle = Y0 + alpha * X0
        scale = np.abs(Y0) + abs(alpha) * np.abs(X0)
        Xt, Yt = _mkv(X0, va, dt, rng), _mkv(Y0, vb, dt, rng)
        with cg.capture(g):
            einsums.linalg.axpy(alpha, Xt, Yt)
        out = Yt
    elif op == "axpby":
        X0 = _rnd(shape, dt, rng)
        Y0 = _rnd(shape, dt, rng)
        oracle = alpha * X0 + beta * Y0
        scale = abs(alpha) * np.abs(X0) + abs(beta) * np.abs(Y0)
        Xt, Yt = _mkv(X0, va, dt, rng), _mkv(Y0, vb, dt, rng)
        with cg.capture(g):
            einsums.linalg.axpby(alpha, Xt, beta, Yt)
        out = Yt
    elif op == "direct_product":
        A0 = _rnd(shape, dt, rng)
        B0 = _rnd(shape, dt, rng)
        C0 = _rnd(shape, dt, rng)
        oracle = alpha * A0 * B0 + beta * C0
        scale = abs(alpha) * np.abs(A0 * B0) + abs(beta) * np.abs(C0)
        At, Bt, Ct = _mkv(A0, va, dt, rng), _mkv(B0, vb, dt, rng), _mkv(C0, vc, dt, rng)
        with cg.capture(g):
            einsums.linalg.direct_product(alpha, At, Bt, beta, Ct)
        out = Ct
    else:  # direct_division -- keep |B| >= ~1 to avoid blow-up
        A0 = _rnd(shape, dt, rng)
        B0 = _rnd(shape, dt, rng)
        B0 = _shift_from_zero(B0, dt)
        C0 = _rnd(shape, dt, rng)
        oracle = alpha * A0 / B0 + beta * C0
        scale = abs(alpha) * np.abs(A0 / B0) + abs(beta) * np.abs(C0)
        At, Bt, Ct = _mkv(A0, va, dt, rng), _mkv(B0, vb, dt, rng), _mkv(C0, vc, dt, rng)
        with cg.capture(g):
            einsums.linalg.direct_division(alpha, At, Bt, beta, Ct)
        out = Ct
    if passes:
        g.apply(cg.default_pass_manager())
    g.execute()
    assert_rounding_close(
        out, oracle, dt, scale,
        err_msg=f"op={op} element_op={element_op} shape={shape} alpha={alpha} beta={beta} dt={dt} "
                f"va={va} vb={vb} vc={vc} passes={passes}")
