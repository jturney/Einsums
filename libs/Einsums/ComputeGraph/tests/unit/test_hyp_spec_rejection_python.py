# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""Specs drawn at random: a malformed one must be REJECTED, a valid permute must match numpy.

The differential fuzzers only ever draw well-formed, rank-matched specs, so the paths that decide
whether a spec is acceptable at all were covered by a handful of fixed cases. Three bugs lived
there: the rank check ran only for C++ string literals, a rank-1 output letter carried by neither
operand ran as a dot that wrote C[0], and the permute parser accepted any character as an index.
Each was a spec that computed something instead of being refused. These properties draw that
space instead of listing it, on the eager path and the captured one.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import einsums
import einsums.graph as cg
from _sanitizer_scaling import sanitizer_examples

#: Every rejection the bindings translate: std::invalid_argument, std::out_of_range and raw throws.
_REJECTED = (ValueError, RuntimeError, IndexError)

#: Index names, single-letter and multi-character, including the '_' a permute spec admits.
_NAMES = ["i", "j", "k", "a", "b", "mu", "nu_1", "k_x", "p2"]

_SETTINGS = dict(
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large, HealthCheck.filter_too_much],
)


def _tensor(name, shape, rng):
    t = einsums.create_zero_tensor(name, list(shape), dtype="float64")
    if int(np.prod(shape)) > 0:
        np.asarray(t)[...] = rng.standard_normal(shape)
    return t


def _run(mode, call):
    """Run @p call eagerly or inside a captured graph that is then executed."""
    if mode == "eager":
        call()
        return
    graph = cg.Graph("spec_fuzz")
    with cg.capture(graph):
        call()
    graph.execute()


def _spell(names):
    """An index list, comma separated so multi-character names stay one index."""
    return ",".join(names)


@st.composite
def _malformed_einsum(draw):
    """A contraction made invalid in one of two ways the parser or the rank check must refuse.

    ``output_only``: C names a letter neither operand carries, with a C tensor of matching rank,
    so only the spec's roles are wrong. ``rank``: one operand's tensor has an axis more or fewer
    than the spec names for it, so only the shapes are wrong.
    """
    letters = draw(st.permutations(["i", "j", "k", "l"]))
    a_idx = list(letters[: draw(st.integers(1, 3))])
    b_idx = list(letters[draw(st.integers(0, 2)) : draw(st.integers(3, 4))]) or ["i"]
    shared = [x for x in a_idx if x in b_idx]
    c_idx = [x for x in dict.fromkeys(a_idx + b_idx) if x not in shared]
    extent = {x: draw(st.integers(1, 3)) for x in "ijklz"}

    how = draw(st.sampled_from(["output_only", "rank"]))
    a_shape = [extent[x] for x in a_idx]
    b_shape = [extent[x] for x in b_idx]
    if how == "output_only":
        c_idx = c_idx + ["z"]
    else:
        target = draw(st.sampled_from(["a", "b"]))
        grow = draw(st.booleans())
        shape = a_shape if target == "a" else b_shape
        if grow or len(shape) == 1:
            shape.append(2)
        else:
            shape.pop()
    c_shape = [extent[x] for x in c_idx]
    spec = f"{_spell(c_idx)} <- {_spell(a_idx)} ; {_spell(b_idx)}"
    return spec, a_shape, b_shape, c_shape, how


@settings(max_examples=sanitizer_examples(150), **_SETTINGS)
@given(problem=_malformed_einsum(), mode=st.sampled_from(["eager", "graph"]), seed=st.integers(0, 2**16))
def test_malformed_einsum_is_rejected(problem, mode, seed):
    spec, a_shape, b_shape, c_shape, how = problem
    rng = np.random.default_rng(seed)
    A = _tensor("A", a_shape, rng)
    B = _tensor("B", b_shape, rng)
    C = _tensor("C", c_shape, rng)
    with pytest.raises(_REJECTED):
        _run(mode, lambda: einsums.einsum(spec, C, A, B))


@st.composite
def _permutation(draw):
    rank = draw(st.integers(1, 4))
    # One multi-character name cannot be spelled alone: with no comma to mark the multi-character
    # mode, "mu" is the two indices m and u. So a rank-one draw takes a single-letter name.
    pool = [x for x in _NAMES if len(x) == 1] if rank == 1 else _NAMES
    names = draw(st.permutations(pool))[:rank]
    order = draw(st.permutations(list(range(rank))))
    extent = [draw(st.integers(0, 3)) for _ in range(rank)]
    return list(names), list(order), extent


@settings(max_examples=sanitizer_examples(150), **_SETTINGS)
@given(problem=_permutation(), mode=st.sampled_from(["eager", "graph"]), seed=st.integers(0, 2**16))
def test_random_permute_matches_numpy(problem, mode, seed):
    names, order, extent = problem
    rng = np.random.default_rng(seed)
    A = _tensor("A", extent, rng)
    c_names = [names[p] for p in order]
    C = _tensor("C", [extent[p] for p in order], rng)
    spec = f"{_spell(c_names)} <- {_spell(names)}"
    _run(mode, lambda: einsums.permute(spec, C, A))
    np.testing.assert_allclose(np.asarray(C), np.transpose(np.asarray(A), order), rtol=0.0, atol=0.0)


@st.composite
def _malformed_permute(draw):
    """A permutation made invalid: a character no index name may hold, an output name the input
    does not carry, the two sides naming different counts of axes, or tensors of a rank the spec
    does not name."""
    rank = draw(st.integers(2, 3))
    names = list(draw(st.permutations(["i", "j", "k"]))[:rank])
    c_names = list(draw(st.permutations(names)))
    how = draw(st.sampled_from(["character", "stray", "count", "rank"]))
    if how == "character":
        bad = draw(st.sampled_from(["@", "$", ".", "!", "#"]))
        c_names[0] = c_names[0] + bad
    elif how == "stray":
        c_names[0] = "z"
    elif how == "count":
        c_names = c_names[:-1]
    return names, c_names, how


@settings(max_examples=sanitizer_examples(100), **_SETTINGS)
@given(problem=_malformed_permute(), mode=st.sampled_from(["eager", "graph"]), seed=st.integers(0, 2**16))
def test_malformed_permute_is_rejected(problem, mode, seed):
    names, c_names, how = problem
    rng = np.random.default_rng(seed)
    # "rank": a well-formed spec over tensors with one axis more than it names. That ran over the
    # named axes only and left the rest of C untouched, so it has to be refused like the others.
    extra = 1 if how == "rank" else 0
    A = _tensor("A", [2] * (len(names) + extra), rng)
    C = _tensor("C", [2] * (len(c_names) + extra), rng)
    spec = f"{_spell(c_names)} <- {_spell(names)}"
    with pytest.raises(_REJECTED):
        _run(mode, lambda: einsums.permute(spec, C, A))
