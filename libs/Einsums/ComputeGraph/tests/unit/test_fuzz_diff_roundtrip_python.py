# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""Differential fuzz: a random program survives a save/load round trip.

The other shards ask whether the graph agrees with numpy. This one asks a
narrower question nothing else covers: written to a file, read back in the same
process, rebound by manifest NAME to storage the loaded graph has never seen,
does the program still compute what it computed?

Every step of that sentence is a place the IR can lose something quietly - a
prefactor's dtype widened, an index list reordered, a body left pointing at the
loader's placeholder buffer instead of the bound one - and none of those shows
up as an error. They show up here as a wrong number.

Programs the IR cannot carry yet (a ``View`` node, a ``gemv``, an anonymous
element transform) are skipped, and the refusal from ``save_graph`` is what
decides that: there is exactly one list of what is reconstructible and it lives
in the library. ``test_roundtrip_corpus_is_not_all_skips`` is the guard that
keeps the skip rule from turning the shard into a no-op.

The shared harness lives in _fuzz_diff_common.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from einsums.testing import ALL_DTYPES

from _fuzz_diff_common import *  # shared fuzz/differential harness

# Statement kinds whose captured nodes are reconstructible today. Drawing only
# these is a DENSITY choice, not a correctness one: an unlisted kind still gets
# refused at save and skipped, and the pinned case plus the not-all-skips guard
# below are what catch the list going stale. Without it roughly nine trials in
# ten skip, because the generator draws views and BLAS-2 ops freely and neither
# survives a save yet.
_SAVEABLE_KINDS = frozenset({"scale", "axpy", "axpby", "gemm", "einsum", "beinsum", "perm", "loop", "cond"})


def _all_saveable(stmts):
    """Whether every statement, at every nesting level, is a saveable kind."""
    for s in stmts:
        if s[0] not in _SAVEABLE_KINDS:
            return False
        if s[0] == "loop" and not _all_saveable(s[2]):
            return False
        if s[0] == "cond" and not (_all_saveable(s[2]) and _all_saveable(s[3])):
            return False
    return True


def _draw_saveable(rng, depth, max_stmts, tries=32):
    """Draw until the program holds only saveable kinds, or give up."""
    for _ in range(tries):
        prog = _gen_block(rng, depth=depth, max_stmts=max_stmts)
        if prog and _all_saveable(prog):
            return prog
    return None


@pytest.mark.parametrize("dtype", ALL_DTYPES)
@pytest.mark.parametrize("seed", fuzz_seeds(120))
def test_fuzz_roundtrip_flat(seed, dtype):
    rng = np.random.default_rng(seed)
    prog = _draw_saveable(rng, depth=0, max_stmts=8)
    if prog is None:
        pytest.skip("no saveable program drawn for this seed")
    check_program_roundtrip(prog, *_seed_arrays(rng, dtype), f"rtflat{seed}", dtype=dtype)


@pytest.mark.parametrize("seed", fuzz_seeds(60))
def test_fuzz_roundtrip_control_flow(seed):
    # depth 1 lets the generator draw loops and conditionals, so the fragment
    # encoding and its boundary references are exercised rather than only the
    # flat node list.
    rng = np.random.default_rng(10_000 + seed)
    prog = _draw_saveable(rng, depth=1, max_stmts=6)
    if prog is None:
        pytest.skip("no saveable program drawn for this seed")
    check_program_roundtrip(prog, *_seed_arrays(rng), f"rtcf{seed}")


def test_roundtrip_of_a_pinned_reconstructible_program():
    """One program that is reconstructible BY CONSTRUCTION.

    The random corpus skips whatever the IR cannot carry, which is the right
    rule and a poor guarantee: a regression that made everything unsaveable
    would turn this shard green and silent. This case cannot skip.
    """
    # A SQUARE pool, so every statement below is shape-valid whichever slots it
    # names; the mixed-shape pool the random corpus draws from would make the
    # pinned program a statement about shapes rather than about the round trip.
    rng = np.random.default_rng(4242)
    m, v, t = _square_seed_arrays(rng)
    prog = [
        ("scale", 0.5, 0),
        ("axpby", 0.25, 0, 0.75, 1),
        ("gemm", 1.5, 0, 1, 0.0, 2),
        ("perm", 2.0, 0.0, 2, 3),
    ]
    got = _run_program_roundtrip(prog, m, v, t, "rtpinned")
    assert got is not None, "a program of scale/axpby/gemm/permute must be saveable"
    _assert_pools(got, _oracle(prog, m, v, t), prog, "ROUND-TRIPPED")


def test_roundtrip_corpus_is_not_all_skips():
    """The skip rule must not swallow the whole corpus.

    Ordered last in the file so the parametrized cases above have run and the
    counter means something. If this fails, either the reconstructible set
    shrank or the generator stopped drawing anything the IR can carry - both are
    regressions, and both would otherwise show up as a shard that passes without
    testing anything.
    """
    attempted = _ROUNDTRIP_STATS["attempted"]
    tripped = _ROUNDTRIP_STATS["round_tripped"]
    assert attempted > 0, "no round-trip trial ran at all"
    assert tripped > 0, f"every one of {attempted} trials skipped; nothing was actually round-tripped"
