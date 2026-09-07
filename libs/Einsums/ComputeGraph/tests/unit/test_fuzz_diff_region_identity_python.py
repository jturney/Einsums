# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""Differential fuzz: raise every region into the algebraic IR and lower it back.

The raise/lower gate, driven over the differential corpus rather than over a
hand-written list. The hand-written cases live in ``RegionIdentity.cpp`` and
state shapes precisely; this shard supplies the breadth, which is what actually
finds a field the IR forgot to carry.

Bitwise, not allclose. ``lower_region`` rebuilds every node from the expression
rather than reusing what it raised, so it runs the same kernel over the same
values in the same order; anything short of identical is a defect and not a
tolerance question. That also makes this shard immune to the overflow skip the
tolerance-based shards need, so the degenerate programs - the ones most likely
to expose a dropped prefactor - stay in the corpus instead of being thrown out.
"""

from __future__ import annotations

import numpy as np
import pytest

import einsums
import einsums.graph as cg
from einsums import linalg
from einsums.testing import ALL_DTYPES

from _fuzz_diff_common import *  # shared fuzz/differential harness


@pytest.mark.parametrize("dtype", ALL_DTYPES)
@pytest.mark.parametrize("seed", fuzz_seeds(120))
def test_fuzz_region_identity_flat(seed, dtype):
    rng = np.random.default_rng(seed)
    prog = _gen_block(rng, depth=0, max_stmts=10)
    check_program_region_identity(prog, *_seed_arrays(rng, dtype), f"rid{seed}", dtype=dtype)


@pytest.mark.parametrize("seed", fuzz_seeds(60))
def test_fuzz_region_identity_control_flow(seed):
    # Loops and conditionals are BARRIERS: the region rule stops at them rather
    # than descending, so what this exercises is that a graph full of barriers
    # still round-trips the runs BETWEEN them. A framework that silently
    # swallowed a loop body would show up here as a wrong number, and one that
    # refused to form any region at all would show up in the corpus assertion
    # below.
    rng = np.random.default_rng(90_000 + seed)
    prog = _gen_block(rng, depth=3, max_stmts=6)
    check_program_region_identity(prog, *_seed_arrays(rng, "float64"), f"ridcf{seed}")


def test_the_corpus_actually_formed_regions():
    """A green shard that raised nothing would prove nothing.

    The shards above skip nothing, so every trial reached the pass; what this
    asserts is that the pass had something to do on most of them. A generator
    change that stopped emitting raisable ops would otherwise leave this file
    passing while testing an empty set, which is the failure mode a coverage
    counter exists to catch.
    """
    attempted = _REGION_STATS["attempted"]
    if attempted == 0:
        # Selected on its own, so the counter it reads was never fed. Skipping
        # rather than failing, because the counter is a property of the shards
        # above and this only has something to say after they have run; a full
        # run of this file always does.
        pytest.skip("no trials ran in this selection; run the whole shard")

    with_regions = _REGION_STATS["with_regions"]
    rewritten = _REGION_STATS["rewritten"]
    assert with_regions > attempted // 2, (
        f"only {with_regions} of {attempted} trials formed any region; "
        "the generator may have stopped emitting raisable operations"
    )
    # Every region that formed was rewritten, because RegionIdentity always
    # returns true. A gap between the two means a raise or a lower declined,
    # which is worth seeing rather than tolerating.
    assert rewritten == with_regions, (
        f"{with_regions} trials formed regions but only {rewritten} were lowered; "
        "a raise or a lower declined - check the pass's skip reasons"
    )


def test_the_dump_renders_the_algebra_not_the_node_list():
    """The region dump, asserted rather than assumed.

    A dump nobody reads is a dump that rots, and this is the property that makes
    it worth having at all: it reads as algebra. A node-list dump would name
    kinds and ids; this names tensors and index letters, which is what a person
    comparing two versions of a contraction actually needs.
    """
    import einsums

    rng = np.random.default_rng(7)
    ta = einsums.create_zero_tensor("A", [3, 3], dtype="float64")
    tb = einsums.create_zero_tensor("B", [3, 3], dtype="float64")
    tc = einsums.create_zero_tensor("C", [3, 3], dtype="float64")
    np.asarray(ta)[...] = rng.standard_normal((3, 3))
    np.asarray(tb)[...] = rng.standard_normal((3, 3))

    g = cg.Graph("dump")
    with cg.capture(g):
        einsums.einsum("ij <- ik ; kj", tc, ta, tb)

    identity = cg.RegionIdentity()
    identity.set_dump(True)
    pm = cg.PassManager()
    pm.add(identity)
    pm.run(g)

    text = identity.dump_text
    assert text, "the dump was empty with dumping switched on"
    # Reads as algebra: the tensors by name, the letters they are contracted
    # over, and an assignment rather than a node kind.
    assert "C[i,j]" in text
    assert "A[i,k]" in text
    assert "B[k,j]" in text
    assert "before:" in text and "after:" in text
    # An identity rewrite renders identically on both sides.
    before = text.split("before:")[1].split("after:")[0]
    after = text.split("after:")[1]
    assert before.strip() == after.strip()


# ──────────────────────────────────────────────────────────────────────────
# Grouped families
#
# A grouped node is one operation over a family of members whose extents
# differ, and it raises to a term carrying one more free letter. The corpus
# here is drawn separately from the one above because the shape is: members are
# per-entity SLICES of one padded store, which is how a local-correlation
# method reaches these calls and what makes the leading dimensions differ from
# the extents.
#
# Bitwise, for the reason the shards above are: the lowering rebuilds every
# grouped node from the algebra and runs the same kernel over the same values.
# ──────────────────────────────────────────────────────────────────────────

_GROUPED_STATS = {"attempted": 0, "with_regions": 0, "rewritten": 0}


def _ragged_slices(rng, store, extents, cols):
    """One per-member view of ``store``, each its own shape, none overlapping."""
    out = []
    row = 0
    for n, c in zip(extents, cols):
        out.append(store[row:row + n, 0:c])
        row += n
    return out


def _grouped_pools(rng, count, dtype):
    """Per-member operand slices with member-varying extents, out of padded stores."""
    rows = [int(rng.integers(1, 5)) for _ in range(count)]
    links = [int(rng.integers(1, 5)) for _ in range(count)]
    cols = [int(rng.integers(1, 5)) for _ in range(count)]

    def store(name, r, c):
        data = rng.standard_normal((sum(r), max(c))).astype(dtype)
        if np.issubdtype(np.dtype(dtype), np.complexfloating):
            data = data + 1j * rng.standard_normal(data.shape).astype(
                np.float64 if np.dtype(dtype) == np.complex128 else np.float32)
        return einsums.array(np.asarray(data, dtype=dtype), name=name)

    a_store = store("gA", rows, links)
    b_store = store("gB", links, cols)
    c_store = store("gC", rows, cols)
    d_store = store("gD", rows, cols)
    r_store = store("gR", [1] * count, [1] * count)
    return {
        "rows": rows, "links": links, "cols": cols,
        "stores": [a_store, b_store, c_store, d_store, r_store],
        "a": _ragged_slices(rng, a_store, rows, links),
        "b": _ragged_slices(rng, b_store, links, cols),
        "c": _ragged_slices(rng, c_store, rows, cols),
        "d": _ragged_slices(rng, d_store, rows, cols),
        "r": _ragged_slices(rng, r_store, [1] * count, [1] * count),
    }


def _emit_grouped(rng, pools, count):
    """A short program of grouped statements, drawn over the whole family."""
    picks = []
    for _ in range(int(rng.integers(2, 6))):
        picks.append(int(rng.integers(0, 5)))
    alphas = lambda: [float(rng.integers(-3, 4)) for _ in range(count)]
    betas = lambda: [float(rng.integers(0, 3)) for _ in range(count)]
    for pick in picks:
        if pick == 0:
            cg.grouped_batched_gemm(float(rng.integers(1, 3)), pools["a"], pools["b"],
                                    float(rng.integers(0, 2)), pools["c"])
        elif pick == 1:
            linalg.grouped_axpby(alphas(), pools["c"], betas(), pools["d"])
        elif pick == 2:
            linalg.grouped_dot(pools["r"], pools["c"], pools["d"])
        elif pick == 3:
            linalg.grouped_direct_product(alphas(), pools["c"], pools["d"], betas(), pools["d"])
        else:
            linalg.grouped_permute("ab <- ab", pools["d"], pools["c"], betas(), alphas())
    return picks


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("seed", fuzz_seeds(40))
def test_fuzz_region_identity_grouped(seed, dtype):
    count = int(np.random.default_rng(seed).integers(2, 6))

    def run(identity):
        rng = np.random.default_rng(seed)
        pools = _grouped_pools(rng, count, dtype)
        g = cg.Graph(f"grouped-rid{seed}")
        with cg.capture(g):
            _emit_grouped(rng, pools, count)
        stats = None
        if identity:
            pass_ = cg.RegionIdentity()
            pm = cg.PassManager()
            pm.add(pass_)
            pm.set_optimizer_budget(0)
            # Budget zero, so no decision here is the clock's; RegionIdentity
            # has no search to cut off and reports none.
            g.apply(pm)
            stats = (pass_.regions_formed, pass_.regions_rewritten)
        g.execute()
        return [np.array(t, copy=True) for t in pools["stores"]], stats

    expected, _ = run(False)
    got, stats = run(True)

    _GROUPED_STATS["attempted"] += 1
    if stats[0]:
        _GROUPED_STATS["with_regions"] += 1
        _GROUPED_STATS["rewritten"] += 1 if stats[1] == stats[0] else 0

    for idx, (a, b) in enumerate(zip(expected, got)):
        if not np.array_equal(a, b, equal_nan=True):
            raise AssertionError(
                f"a raise/lower round trip changed grouped store {idx} "
                f"(dtype={dtype}, seed={seed}, members={count})\n"
                f"captured=\n{a}\nafter round trip=\n{b}")


def test_the_grouped_corpus_actually_raised_families():
    """A green grouped shard that raised nothing would prove nothing."""
    attempted = _GROUPED_STATS["attempted"]
    if attempted == 0:
        pytest.skip("no grouped trials ran in this selection; run the whole shard")
    assert _GROUPED_STATS["with_regions"] == attempted, (
        f"only {_GROUPED_STATS['with_regions']} of {attempted} grouped trials formed a region; "
        "the grouped kinds may have stopped being raisable")
    assert _GROUPED_STATS["rewritten"] == attempted, (
        f"only {_GROUPED_STATS['rewritten']} of {attempted} grouped trials lowered every region; "
        "a raise or a lower declined - check the pass's skip reasons")
