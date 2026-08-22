# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""``GateFlags`` and ``Graph.add_conditional_flag`` through the bindings.

The C++ side is covered in ``GateFlags.cpp``; what this adds is the binding, and
the binding is the whole point of the type. A conditional whose predicate is a
Python callable takes the GIL every time it is evaluated, so a graph with
hundreds of them replayed across a thread team serializes on it. A flag array is
written from Python once and read from C++ without ever coming back.
"""

from __future__ import annotations

import numpy as np
import pytest

import einsums
import einsums.graph as cg
from einsums import linalg as la


def _tensor(name, value, dim=4):
    t = einsums.create_zero_tensor(name, [dim, dim], dtype="float64")
    np.asarray(t)[...] = value
    return t


def test_flags_read_back_what_was_written():
    flags = cg.GateFlags(4)
    assert flags.size == 4
    assert [flags.get(i) for i in range(4)] == [False] * 4

    flags.set(2, True)
    assert [flags.get(i) for i in range(4)] == [False, False, True, False]

    flags.fill(True)
    assert all(flags.get(i) for i in range(4))

    # One bulk write, which is how a solver that knows every answer sets them.
    flags.assign([1, 0, 0, 1])
    assert [flags.get(i) for i in range(4)] == [True, False, False, True]

    flags.resize(6, True)
    assert flags.size == 6
    assert flags.get(5)

    with pytest.raises(Exception):
        flags.get(6)
    with pytest.raises(Exception):
        flags.set(6, True)


def test_a_flag_gate_selects_a_branch_and_flips_between_replays():
    out = _tensor("out", 0.0)
    flags = cg.GateFlags(1, True)

    g = cg.Graph("flip")
    with cg.capture(g):
        la.scale(0.0, out)
    then_g, else_g = g.add_conditional_flag("branch", flags, 0)
    with cg.capture(then_g):
        la.axpby(2.0, _tensor("ones", 1.0), 1.0, out)
    with cg.capture(else_g):
        la.axpby(5.0, _tensor("ones else", 1.0), 1.0, out)

    g.execute()
    assert np.allclose(np.asarray(out), 2.0)

    # No re-capture and no re-plan: the node reads the array it was handed.
    flags.set(0, False)
    g.execute()
    assert np.allclose(np.asarray(out), 5.0)

    flags.set(0, True)
    g.execute()
    assert np.allclose(np.asarray(out), 2.0)


def test_an_index_past_the_end_reads_false():
    out = _tensor("oob out", 1.0)
    flags = cg.GateFlags(1, True)

    g = cg.Graph("oob")
    then_g, _ = g.add_conditional_flag("branch", flags, 5)
    with cg.capture(then_g):
        la.scale(2.0, out)

    g.execute()
    assert np.allclose(np.asarray(out), 1.0)


def test_a_flag_gate_and_a_lambda_gate_compute_the_same_bits():
    """The two spellings differ in how the answer is fetched, not in the answer.

    Four blocks, two of them open, each block accumulating a different multiple
    of the same operand, so a block taken in one graph and not the other shows
    up immediately. Bit-identical rather than close: the flag gate exists to be
    a drop-in for the callable, and a tolerance would hide a real difference.
    """
    live = [True, False, True, False]
    source = _tensor("source", 0.25)

    def build(gate_for):
        out = einsums.create_zero_tensor("acc", [4, 4], dtype="float64")
        g = cg.Graph("blocks")
        for block in range(len(live)):
            then_g, _ = gate_for(g, block)
            with cg.capture(then_g):
                la.axpby(1.0 + block, source, 1.0, out)
        g.execute()
        return np.array(np.asarray(out), copy=True)

    flags = cg.GateFlags(len(live), False)
    for block, on in enumerate(live):
        flags.set(block, on)
    by_flags = build(lambda g, b: g.add_conditional_flag(f"block [{b}]", flags, b))
    by_lambda = build(
        lambda g, b: g.add_conditional(f"block [{b}]", lambda _b=b: live[_b]))

    assert np.array_equal(by_flags, by_lambda)
    # And they did something: blocks 0 and 2 ran, blocks 1 and 3 did not.
    assert np.allclose(by_flags, (1.0 + 3.0) * 0.25)


def test_a_copy_of_the_handle_shares_the_array():
    """Copies alias, which is what lets a solver hand the array around."""
    flags = cg.GateFlags(2, False)
    alias = flags
    flags.set(0, True)
    assert alias.get(0)


def test_the_array_outlives_the_handle_the_node_was_built_from():
    """The node bakes in the buffer, not the handle."""
    out = _tensor("outlives", 1.0)

    def build():
        g = cg.Graph("outlives")
        temporary = cg.GateFlags(1, True)
        then_g, _ = g.add_conditional_flag("branch", temporary, 0)
        with cg.capture(then_g):
            la.scale(3.0, out)
        return g

    g = build()
    g.execute()
    assert np.allclose(np.asarray(out), 3.0)
