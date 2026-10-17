# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""Hypothesis differential: einsum over three to five operands vs numpy.einsum.

Draws contractions the two-operand fuzzer cannot express: a letter carried by
three or more operands and summed, a letter carried by one operand alone and
summed (a trace folded into the first contraction it takes part in), a letter
repeated within an operand (a diagonal), an empty output, zero and unit
extents, the same tensor passed twice, conjugated operands, non-contiguous
permuted views, both prefactors, and every dtype. Each draw runs eagerly, or
captured with or without the default passes. numpy.einsum is the oracle; the
einsum under test runs as the binary steps einsums.graph.plan_nary_einsum
chooses, so a wrong plan, a dropped letter, or a misplaced conjugate shows up
as a mismatch.
"""

from __future__ import annotations

import itertools

import numpy as np
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st
from _sanitizer_scaling import sanitizer_examples

import einsums
import einsums.graph as cg
from einsums.testing import ALL_DTYPES, assert_close

_LETTERS = "ijklmn"
_ctr = itertools.count()


def _rnd(shape, dtype, rng):
    data = rng.standard_normal(shape)
    if dtype.startswith("complex"):
        data = data + 1j * rng.standard_normal(shape)
    return data.astype(dtype)


def _tensor(array, dtype, view, rng):
    """``array`` as an einsums tensor, or as a permuted, non-contiguous view of one."""
    if not view or array.ndim < 2:
        return einsums.asarray(array, dtype=dtype)
    perm = list(rng.permutation(array.ndim))
    if perm == list(range(array.ndim)):
        perm = perm[::-1]
    stored = einsums.asarray(np.ascontiguousarray(np.transpose(array, perm)), dtype=dtype)
    return stored.permute_view(list(np.argsort(perm)))


@st.composite
def _problem(draw):
    count = draw(st.integers(3, 5))
    pool = _LETTERS[: draw(st.integers(2, len(_LETTERS)))]
    operands = []
    for _ in range(count):
        rank = draw(st.integers(1, 3))
        operands.append([draw(st.sampled_from(pool)) for _ in range(rank)])
    used = sorted({letter for operand in operands for letter in operand})
    # numpy's einsum cannot write one output axis twice, so the output names each letter once.
    output = draw(st.permutations(used))
    output = output[: draw(st.integers(0, min(len(output), 4)))]
    extent = {letter: draw(st.sampled_from([0, 1, 2, 3, 4] if draw(st.booleans()) else [1, 2, 3])) for letter in used}
    dtype = draw(st.sampled_from(ALL_DTYPES))
    complex_ = dtype.startswith("complex")
    conj = [draw(st.booleans()) and complex_ for _ in range(count)]
    views = [draw(st.booleans()) for _ in range(count)]
    # The first two operands are the same tensor when their index lists have the same extents.
    alias = draw(st.booleans()) and [extent[x] for x in operands[0]] == [extent[x] for x in operands[1]]
    c_pf = draw(st.sampled_from([0.0, 1.0, -0.5] + ([0.5 - 0.25j] if complex_ else [])))
    ab_pf = draw(st.sampled_from([1.0, 2.0, -1.5] + ([1.0 + 0.5j] if complex_ else [])))
    mode = draw(st.sampled_from(["eager", "graph", "graph+passes"]))
    seed = draw(st.integers(0, 2**31 - 1))
    return operands, output, extent, dtype, conj, views, alias, c_pf, ab_pf, mode, seed


def _spec(operands, output, conj):
    def spell(letters, conjugated):
        joined = ",".join(letters)
        return f"conj({joined})" if conjugated else joined

    return ",".join(output) + " <- " + " ; ".join(spell(op, c) for op, c in zip(operands, conj))


@given(problem=_problem())
@settings(max_examples=sanitizer_examples(200), deadline=None, suppress_health_check=[HealthCheck.too_slow])
# The first two operands sum to a scalar, which a two-operand spec cannot read; the partial product
# keeps a letter instead, summed by the next contraction.
@example(problem=([["i"], ["i"], ["j"]], [], {"i": 2, "j": 3}, "float64", [False] * 3, [False] * 3, False, 0.0, 1.0, "eager", 0))
@example(problem=([["i"], ["j"], ["k"]], [], {"i": 2, "j": 3, "k": 1}, "complex128", [True, False, True], [False] * 3, False, 0.5,
                  2.0, "graph+passes", 1))
def test_hyp_nary_einsum_diff(problem):
    operands, output, extent, dtype, conj, views, alias, c_pf, ab_pf, mode, seed = problem
    rng = np.random.default_rng(seed)

    arrays = [_rnd(tuple(extent[x] for x in operand), dtype, rng) for operand in operands]
    if alias:
        arrays[1] = arrays[0]
    tensors = [_tensor(array, dtype, view, rng) for array, view in zip(arrays, views)]
    if alias:
        tensors[1] = tensors[0]

    start = _rnd(tuple(extent[x] for x in output), dtype, rng)
    C = einsums.asarray(start.copy(), dtype=dtype)

    spec = _spec(operands, output, conj)
    if mode == "eager":
        einsums.einsum(spec, C, *tensors, c_pf=c_pf, ab_pf=ab_pf)
    else:
        graph = cg.Graph(f"nary{next(_ctr)}")
        with cg.capture(graph):
            einsums.einsum(spec, C, *tensors, c_pf=c_pf, ab_pf=ab_pf)
        if mode == "graph+passes":
            graph.apply(cg.default_pass_manager())
        graph.execute()

    oracle_inputs = [np.conj(a) if c else a for a, c in zip(arrays, conj)]
    numpy_spec = ",".join("".join(op) for op in operands) + "->" + "".join(output)
    want = c_pf * start + ab_pf * np.einsum(numpy_spec, *oracle_inputs)
    assert_close(np.asarray(C), np.asarray(want, dtype=dtype))
