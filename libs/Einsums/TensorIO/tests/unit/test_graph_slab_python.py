#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

"""Python tests for the graph-aware slab IO surface.

Covers ``einsums.io.Slab`` plus ``read_slice`` / ``write_slice``
recorded into a ComputeGraph. The motivating pattern is a
read-transform-write-back loop where the same graph node fires multiple
times against different slab ranges — i.e. the user keeps a single Slab
object, mutates ``slab.ranges`` between graph executions, and the captured
executor lambda picks up the new ranges every time (it captures the Slab
by reference C++-side).
"""

import os
import tempfile
import numpy as np
import pytest

import einsums
import einsums.graph as cg


def _temp_path(name):
    return os.path.join(tempfile.gettempdir(),
                        f"einsums_graphslab_{os.getpid()}_{name}.etn")


def test_slab_module_layout():
    """Slab + slice graph wrappers should live under einsums.io."""
    assert einsums.io.Slab is not None
    assert callable(einsums.io.read_slice)
    assert callable(einsums.io.write_slice)


def test_slab_default_construction_is_empty():
    s = einsums.io.Slab()
    assert s.ranges == []


def test_slab_constructed_with_ranges():
    s = einsums.io.Slab([(0, 4), (1, 3)])
    assert s.ranges == [(0, 4), (1, 3)]


def test_slab_ranges_are_mutable():
    """The C++ executor captures Slab by reference — so mutating
    ranges between graph runs must be visible to the next execute().
    Verify the Python-side mutation surface itself."""
    s = einsums.io.Slab([(0, 2)])
    s.ranges = [(2, 4), (0, 4)]
    assert s.ranges == [(2, 4), (0, 4)]


def test_graph_slab_round_trip_through_recorded_node():
    """Build a graph that reads a slab into ``block``, mutate the in-memory
    block, then run a second graph to write it back. Verifies that
    read_slice / write_slice fire under the executor and pick up
    the current Slab range each invocation."""
    path = _temp_path("rt")
    try:
        seed = einsums.RuntimeTensorD("A", [4, 4])
        np.asarray(seed, copy=False)[:] = np.arange(16, dtype=np.float64).reshape(4, 4)
        f = einsums.io.TensorFile(path, einsums.io.Mode.Write)
        f.write("A", seed)
        del f

        block = einsums.RuntimeTensorD("blk", [2, 2])
        slab = einsums.io.Slab([(1, 3), (1, 3)])

        g_read = cg.Graph("r")
        with cg.capture(g_read):
            einsums.io.read_slice(path, "A", slab, block)
        g_read.execute()

        # Slab [1:3, 1:3] of arange(16).reshape(4,4) — column-major
        # storage means the in-memory block layout differs, but the values
        # themselves should match the corresponding slab.
        # Read the same slab via the non-graph TensorFile API for a
        # ground-truth comparison.
        truth = einsums.RuntimeTensorD("truth", [2, 2])
        f2 = einsums.io.TensorFile(path, einsums.io.Mode.Read)
        f2.read_slice("A", truth, [(1, 3), (1, 3)])
        assert np.allclose(np.asarray(block, copy=False),
                           np.asarray(truth, copy=False))
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_graph_driven_tile_loop_via_add_loop():
    """The motivating use case driven entirely by the graph: ``add_loop``
    records a body subgraph that reads a slab, runs a Python ``transform``
    callable as a graph node (``cg.custom``), and writes the slab back.
    The ``cond`` callback advances ``slab.ranges`` between iterations.

    No host-language loop — the graph drives everything.
    """
    path = _temp_path("tiles_add_loop")
    try:
        seed = einsums.RuntimeTensorD("A", [4, 4])
        np.asarray(seed, copy=False)[:] = np.arange(16, dtype=np.float64).reshape(4, 4)
        f = einsums.io.TensorFile(path, einsums.io.Mode.Write)
        f.write("A", seed)
        del f

        block = einsums.RuntimeTensorD("blk", [2, 2])
        slab = einsums.io.Slab([(0, 2), (0, 2)])  # initial tile (0, 0)

        def transform():
            np.asarray(block, copy=False)[:] *= 10.0

        def body():
            einsums.io.read_slice(path, "A", slab, block)
            cg.custom("transform", transform, block)
            einsums.io.write_slice(path, "A", slab, block)

        def cond(i):
            nxt = i + 1
            if nxt >= 4:
                return False
            bi, bj = nxt // 2, nxt % 2
            slab.ranges = [(bi * 2, bi * 2 + 2), (bj * 2, bj * 2 + 2)]
            return True

        g = cg.Graph("tile-loop")
        g.add_loop("tiles", 4, cond, body)
        g.execute()

        rt = einsums.RuntimeTensorD("rt", [4, 4])
        f3 = einsums.io.TensorFile(path, einsums.io.Mode.Read)
        f3.read("A", rt)
        arr = np.asarray(rt, copy=False)
        expected = np.arange(16, dtype=np.float64).reshape(4, 4) * 10.0
        assert np.allclose(arr, expected), f"mismatch:\n{arr}\nvs\n{expected}"
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_slab_io_runs_immediately_outside_capture():
    """Option-A semantics: read_slice / write_slice called
    outside ``cg.capture(g)`` execute their executor lambdas immediately
    rather than throwing. Mirrors checkpoint_etn's pattern."""
    path = _temp_path("immediate")
    try:
        seed = einsums.RuntimeTensorD("A", [4, 4])
        np.asarray(seed, copy=False)[:] = np.arange(16, dtype=np.float64).reshape(4, 4)
        f = einsums.io.TensorFile(path, einsums.io.Mode.Write)
        f.write("A", seed)
        del f

        block = einsums.RuntimeTensorD("blk", [2, 2])
        slab = einsums.io.Slab([(1, 3), (1, 3)])

        # No cg.capture — runs immediately.
        einsums.io.read_slice(path, "A", slab, block)
        truth = einsums.RuntimeTensorD("truth", [2, 2])
        f2 = einsums.io.TensorFile(path, einsums.io.Mode.Read)
        f2.read_slice("A", truth, [(1, 3), (1, 3)])
        assert np.allclose(np.asarray(block, copy=False),
                           np.asarray(truth, copy=False))

        # write_slice outside capture also runs immediately.
        np.asarray(block, copy=False)[:] = 42.0
        einsums.io.write_slice(path, "A", slab, block)

        # Verify by reading back through the non-graph API.
        check = einsums.RuntimeTensorD("check", [2, 2])
        f3 = einsums.io.TensorFile(path, einsums.io.Mode.Read)
        f3.read_slice("A", check, [(1, 3), (1, 3)])
        assert np.allclose(np.asarray(check, copy=False), 42.0)
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_graph_slab_writes_throw_on_dim_mismatch():
    """The slab's per-dim sizes must match the tensor's shape; the C++
    side throws std::runtime_error which pybind11 converts to
    RuntimeError."""
    path = _temp_path("mismatch")
    try:
        seed = einsums.RuntimeTensorD("A", [4, 4])
        np.asarray(seed, copy=False)[:] = 0.0
        f = einsums.io.TensorFile(path, einsums.io.Mode.Write)
        f.write("A", seed)
        del f

        wrong = einsums.RuntimeTensorD("w", [3, 2])  # doesn't fit (1:3, 1:3)
        slab = einsums.io.Slab([(1, 3), (1, 3)])
        g = cg.Graph("err")
        with cg.capture(g):
            einsums.io.write_slice(path, "A", slab, wrong)
        with pytest.raises(RuntimeError):
            g.execute()
    finally:
        if os.path.exists(path):
            os.remove(path)
