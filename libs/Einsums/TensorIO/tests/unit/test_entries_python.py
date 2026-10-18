#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

"""Copying, renaming and swapping the entries of ``.etn`` files.

``einsums.io.copy``, ``einsums.io.rename`` and ``einsums.io.swap`` run at once
outside a capture and record a node that runs at every replay inside one. The
expected values are the numpy arrays that were written.
"""

import os
import tempfile

import numpy as np
import pytest

import einsums
import einsums.graph as cg
from einsums.testing import ALL_DTYPES, assert_close


@pytest.fixture
def etn_path():
    paths = []

    def make(stem="entries"):
        fd, path = tempfile.mkstemp(suffix=".etn", prefix=f"einsums_{stem}_")
        os.close(fd)
        os.remove(path)
        paths.append(path)
        return path

    yield make
    for path in paths:
        if os.path.exists(path):
            os.remove(path)


def _filled(name, shape, dtype, seed):
    values = (seed + np.arange(int(np.prod(shape)), dtype=np.float64)).reshape(shape)
    if np.dtype(dtype).kind == "c":
        values = values - 1j * values
    tensor = einsums.create_zero_tensor(name, list(shape), dtype=dtype)
    np.asarray(tensor, copy=False)[:] = values.astype(dtype)
    return tensor


def _read_back(path, entry, dtype):
    shape = einsums.io.TensorFile(path).dims(entry)
    out = einsums.create_zero_tensor(entry, list(shape), dtype=dtype)
    einsums.io.read(path, entry, out)
    return np.asarray(out, copy=False).copy()


def _names(path):
    return list(einsums.io.TensorFile(path).tensor_names())


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_copy_rename_and_swap_against_numpy(etn_path, dtype):
    path, other = etn_path(), etn_path("other")
    a = _filled("a", (5, 4, 3), dtype, 1.0)
    b = _filled("b", (5, 4, 3), dtype, 100.0)
    einsums.io.write(path, "a", a)
    einsums.io.write(path, "b", b)

    einsums.io.copy(path, "a", path, "c")
    einsums.io.copy(path, "b", other, "b")
    assert_close(_read_back(path, "c", dtype), np.asarray(a), dtype=dtype)
    assert_close(_read_back(other, "b", dtype), np.asarray(b), dtype=dtype)

    einsums.io.swap(path, "a", "b")
    assert_close(_read_back(path, "a", dtype), np.asarray(b), dtype=dtype)
    assert_close(_read_back(path, "b", dtype), np.asarray(a), dtype=dtype)

    einsums.io.rename(path, "c", "d")
    assert sorted(_names(path)) == ["a", "b", "d"]
    assert_close(_read_back(path, "d", dtype), np.asarray(a), dtype=dtype)


def test_refusals_name_the_problem(etn_path):
    path = etn_path()
    einsums.io.write(path, "a", _filled("a", (3, 2), "float64", 1.0))
    einsums.io.write(path, "b", _filled("b", (2, 3), "float64", 1.0))
    with pytest.raises(ValueError, match="already holds an entry named 'b'"):
        einsums.io.rename(path, "a", "b")
    with pytest.raises(ValueError, match="never changes an entry's shape or type"):
        einsums.io.copy(path, "a", path, "b")
    with pytest.raises(RuntimeError, match="not found"):
        einsums.io.swap(path, "a", "gone")
    assert _names(path) == ["a", "b"]


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_one_graph_steps_by_swapping_its_entries(etn_path, dtype):
    """Each replay reads "cur", stores twice it in "next", and swaps the two."""
    path = etn_path()
    start = _filled("start", (4, 3), dtype, 1.0)
    einsums.io.write(path, "cur", start)

    g = cg.Graph("step")
    cur = g.declare_tensor("cur", [4, 3], dtype=dtype)
    nxt = g.declare_tensor("next", [4, 3], dtype=dtype)
    cg.annotate(cur, backing=einsums.io.etn(path, "cur"), graph=g)
    cg.annotate(nxt, backing=einsums.io.etn(path, "next", write=True), graph=g)
    with cg.capture(g):
        einsums.permute("i,j <- i,j", nxt, cur, c_pf=0.0, a_pf=2.0)
        einsums.io.swap(path, "cur", "next")
    g.apply(cg.default_pass_manager())

    for _ in range(3):
        g.execute()
    assert_close(_read_back(path, "cur", dtype), 8.0 * np.asarray(start), dtype=dtype)
    assert_close(_read_back(path, "next", dtype), 4.0 * np.asarray(start), dtype=dtype)
