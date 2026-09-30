# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.

"""The string einsum's route cascade, seen from Python, for every dtype.

A contraction that falls off its fast path onto the generic loop still
computes the right values, so a value test cannot see the regression; only
``einsums.graph.last_dispatch_route()`` shows it. The C++ tripwire in
EagerParityGaps.cpp pins the same routes; these cases pin them through the
bindings, where each dtype reaches its own instantiation of the engine, and
check the values against numpy as well so a route that fires and computes
nothing cannot pass.

Where the route depends on the dtype it is the conjugation flags that decide:
conjugating a complex operand leaves the BLAS routes for PackedGemm (or, for a
scalar output, the conjugating dot), while on a real operand the flag is the
identity and the plain BLAS route answers.
"""

from __future__ import annotations

import itertools
import zlib

import numpy as np
import pytest

import einsums
import einsums.graph as cg
from einsums.testing import COMPLEX_DTYPES, assert_close

_ctr = itertools.count()


def _is_complex(dtype: str) -> bool:
    return dtype in COMPLEX_DTYPES


def _data(shape, dtype, rng):
    """Order-one data in ``dtype``, genuinely complex for a complex dtype."""
    data = rng.standard_normal(shape)
    if _is_complex(dtype):
        data = data + 1j * rng.standard_normal(shape)
    return data.astype(dtype)


def _tensor(values, dtype):
    tensor = einsums.create_zero_tensor(f"route{next(_ctr)}", list(values.shape), dtype=dtype)
    if values.size:
        np.asarray(tensor)[...] = values
    return tensor


def _prefactors(dtype):
    """``(c_pf, ab_pf)``: complex for a complex dtype, so a kernel that drops an imaginary part fails."""
    if _is_complex(dtype):
        return 0.3 - 0.2j, 0.7 + 0.4j
    return 0.3, 0.7


# (spec, numpy spec, shape per letter, route) for the specs whose route does not depend on the dtype.
_CASCADE = [
    ("ij <- ik ; kj", "ik,kj->ij", {"i": 4, "j": 5, "k": 3}, "gemm_direct_runtime"),
    ("i <- ij ; j", "ij,j->i", {"i": 4, "j": 3}, "gemv_mat_vec_runtime"),
    ("j <- i ; ij", "i,ij->j", {"i": 4, "j": 3}, "gemv_vec_mat_runtime"),
    ("ij <- i ; j", "i,j->ij", {"i": 6, "j": 6}, "ger_runtime"),
    (" <- i ; i", "i,i->", {"i": 6}, "dot_runtime"),
    (" <- ij ; ij", "ij,ij->", {"i": 4, "j": 5}, "dot_runtime"),
    ("ij <- ij ; ij", "ij,ij->ij", {"i": 4, "j": 5}, "direct_product_runtime"),
    # The same product with operands in another letter order: permuted into C's order first.
    ("ij <- ij ; ji", "ij,ji->ij", {"i": 4, "j": 5}, "direct_product_permuted_runtime"),
    ("ijk <- kij ; jki", "kij,jki->ijk", {"i": 3, "j": 4, "k": 5}, "direct_product_permuted_runtime"),
    # Two M and two N indices, so PackedGemm forms the contraction rather than deferring to a GEMM.
    ("ijkl <- ijm ; mkl", "ijm,mkl->ijkl", {"i": 3, "j": 4, "k": 5, "l": 6, "m": 2}, "packed_gemm"),
    # A small outer product: no link to contract, and too few elements for PackedGemm's setup to pay.
    ("ijk <- ij ; k", "ij,k->ijk", {"i": 2, "j": 3, "k": 4}, "generic_loop"),
    # A repeated letter folds into one strided axis, and the folded contraction takes the fast path it
    # fits: an outer product, and a GEMM whose diagonal operand has no unit stride, so PackedGemm packs it.
    ("ij <- ii ; jj", "ii,jj->ij", {"i": 3, "j": 4}, "diagonal:ger_runtime"),
    ("ij <- iik ; kj", "iik,kj->ij", {"i": 4, "j": 5, "k": 3}, "diagonal:packed_gemm"),
    # A letter summed over one operand alone is summed there first, and the rest takes its route.
    ("ij <- ijk ; ij", "ijk,ij->ij", {"i": 3, "j": 4, "k": 2}, "lone_reduced:direct_product_runtime"),
    ("ij <- ikm ; j", "ikm,j->ij", {"i": 3, "j": 4, "k": 2, "m": 5}, "lone_reduced:ger_runtime"),
    # Every letter of the first operand is lone, so it would sum to a scalar; the loop takes it.
    ("ij <- k ; ij", "k,ij->ij", {"i": 3, "j": 4, "k": 2}, "generic_loop_lone_summed"),
]


def _run(spec, np_spec, extents, dtype, conj_a=False, conj_b=False):
    """Run ``spec`` eagerly with prefactors on a nonzero C; return (route, actual, expected)."""
    rng = np.random.default_rng(zlib.crc32(f"{spec}|{dtype}|{conj_a}|{conj_b}".encode()))
    c_letters, rest = spec.split("<-")
    a_letters, b_letters = (part.strip() for part in rest.split(";"))
    c_letters = c_letters.strip()

    a = _data(tuple(extents[ch] for ch in a_letters), dtype, rng)
    b = _data(tuple(extents[ch] for ch in b_letters), dtype, rng)
    c_shape = tuple(extents[ch] for ch in c_letters) if c_letters else (1,)
    c0 = _data(c_shape, dtype, rng)
    c_pf, ab_pf = _prefactors(dtype)

    A, B, C = _tensor(a, dtype), _tensor(b, dtype), _tensor(c0, dtype)
    einsums.einsum(spec, C, A, B, c_pf, ab_pf, conj_a=conj_a, conj_b=conj_b)
    route = cg.last_dispatch_route()

    product = np.einsum(np_spec, np.conj(a) if conj_a else a, np.conj(b) if conj_b else b)
    expected = c_pf * c0 + ab_pf * np.reshape(product, c_shape)
    return route, np.asarray(C), expected.astype(dtype)


@pytest.mark.parametrize("spec, np_spec, extents, route", _CASCADE, ids=[case[3] + ":" + case[0] for case in _CASCADE])
def test_cascade_route_and_value(spec, np_spec, extents, route, dtype):
    got_route, actual, expected = _run(spec, np_spec, extents, dtype)
    assert got_route == route, f"{spec} on {dtype} took {got_route!r}, expected {route!r}"
    assert_close(actual, expected, dtype=dtype)


# (spec, numpy spec, extents, route on a real dtype, route on a complex dtype), each with conj on A and B.
_CONJUGATED = [
    ("ij <- ik ; kj", "ik,kj->ij", {"i": 4, "j": 5, "k": 3}, "gemm_direct_runtime", "packed_gemm"),
    ("i <- ij ; j", "ij,j->i", {"i": 4, "j": 3}, "gemv_mat_vec_runtime", "packed_gemm"),
    (" <- i ; i", "i,i->", {"i": 6}, "dot_runtime", "true_dot_runtime"),
]


@pytest.mark.parametrize("conj_a, conj_b", [(True, False), (False, True), (True, True)])
@pytest.mark.parametrize(
    "spec, np_spec, extents, real_route, complex_route", _CONJUGATED, ids=[case[0] for case in _CONJUGATED]
)
def test_conjugated_route_and_value(spec, np_spec, extents, real_route, complex_route, conj_a, conj_b, dtype):
    got_route, actual, expected = _run(spec, np_spec, extents, dtype, conj_a=conj_a, conj_b=conj_b)
    want = complex_route if _is_complex(dtype) else real_route
    assert got_route == want, f"{spec} conj_a={conj_a} conj_b={conj_b} on {dtype} took {got_route!r}, expected {want!r}"
    assert_close(actual, expected, dtype=dtype)


def test_strided_batched_route_on_replay(dtype):
    """A batch index trailing every operand replays as one strided batched GEMM."""
    rng = np.random.default_rng(7)
    a = _data((3, 5, 4), dtype, rng)
    b = _data((5, 2, 4), dtype, rng)
    A, B = _tensor(a, dtype), _tensor(b, dtype)
    C = einsums.create_zero_tensor(f"route{next(_ctr)}", [3, 2, 4], dtype=dtype)

    graph = cg.Graph(f"route{next(_ctr)}")
    with cg.capture(graph):
        einsums.einsum("ijb;jkb->ikb", C, A, B)
    graph.execute()

    assert cg.last_dispatch_route() == "strided_batched_gemm"
    assert_close(C, np.einsum("ijb,jkb->ikb", a, b).astype(dtype), dtype=dtype)


def test_real_conjugation_keeps_strided_batched_route(dtype):
    """Conjugating a real operand is dropped at capture, so the node keeps the strided route."""
    rng = np.random.default_rng(11)
    a = _data((3, 5, 4), dtype, rng)
    b = _data((5, 2, 4), dtype, rng)
    A, B = _tensor(a, dtype), _tensor(b, dtype)
    C = einsums.create_zero_tensor(f"route{next(_ctr)}", [3, 2, 4], dtype=dtype)

    graph = cg.Graph(f"route{next(_ctr)}")
    with cg.capture(graph):
        einsums.einsum("ikb <- conj(ijb) ; jkb", C, A, B)
    graph.execute()

    route = cg.last_dispatch_route()
    if _is_complex(dtype):
        assert route != "strided_batched_gemm"
    else:
        assert route == "strided_batched_gemm"
    assert_close(C, np.einsum("ijb,jkb->ikb", np.conj(a), b).astype(dtype), dtype=dtype)
