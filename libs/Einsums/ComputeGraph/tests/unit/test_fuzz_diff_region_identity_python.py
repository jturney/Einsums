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

import einsums.graph as cg
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
