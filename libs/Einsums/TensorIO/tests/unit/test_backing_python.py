#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

"""A ``.etn`` entry as the storage behind a tensor a graph reads but never holds.

``cg.annotate(tensor, backing=einsums.io.etn(path, entry))`` on a tensor
declared on the graph. Every replay fills it from the file; the expected
values come from numpy over what was written. With ``write=True`` the graph
produces the tensor and the entry receives it, checked by reading the file
back.
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
    fd, path = tempfile.mkstemp(suffix=".etn", prefix="einsums_backing_")
    os.close(fd)
    os.remove(path)
    yield path
    if os.path.exists(path):
        os.remove(path)


def _filled(name, shape, dtype, scale):
    values = scale * (np.arange(int(np.prod(shape)), dtype=np.float64) % 17 - 8.0).reshape(shape)
    if np.dtype(dtype).kind == "c":
        values = values - 0.25j * values
    tensor = einsums.create_zero_tensor(name, list(shape), dtype=dtype)
    np.asarray(tensor, copy=False)[:] = values.astype(dtype)
    return tensor


def _product_graph(path, dtype, A, C):
    """C[i,j] = A[i,k] B[k,j], with B declared on the graph and backed by the file."""
    g = cg.Graph("backed_product")
    B = g.declare_tensor("B", [4, 3], dtype=dtype)
    cg.annotate(B, tag="eri", backing=einsums.io.etn(path, "B"), graph=g)
    with cg.capture(g):
        einsums.einsum("ij <- ik ; kj", C, A, B)
    return g, B


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_backed_tensor_replays_against_numpy(etn_path, dtype):
    b = _filled("B", (4, 3), dtype, 0.5)
    einsums.io.write(etn_path, "B", b)
    A = _filled("A", (5, 4), dtype, 0.25)
    C = einsums.create_zero_tensor("C", [5, 3], dtype=dtype)

    g, B = _product_graph(etn_path, dtype, A, C)
    assert g.tensor_backing(B).kind == "tensor_io.etn"
    assert g.tensor_tag(B).name == "eri"

    g.apply(cg.default_pass_manager())
    g.execute()
    expected = np.asarray(A, copy=False) @ np.asarray(b, copy=False)
    assert_close(np.asarray(C, copy=False), expected, dtype=dtype)


def test_saved_graph_keeps_the_backing(etn_path, tmp_path):
    b = _filled("B", (4, 3), "float64", 0.5)
    einsums.io.write(etn_path, "B", b)
    A = _filled("A", (5, 4), "float64", 0.25)
    C = einsums.create_zero_tensor("C", [5, 3], dtype="float64")
    g, _ = _product_graph(etn_path, "float64", A, C)

    saved = str(tmp_path / "backed.eig.json")
    cg.save_graph(g, saved)
    loaded = cg.load_graph(saved)

    A2 = _filled("A2", (5, 4), "float64", 0.25)
    C2 = einsums.create_zero_tensor("C2", [5, 3], dtype="float64")
    cg.bind(loaded, {"A": A2, "C": C2})
    loaded.apply(cg.default_pass_manager())
    loaded.execute()
    expected = np.asarray(A2, copy=False) @ np.asarray(b, copy=False)
    assert_close(np.asarray(C2, copy=False), expected, dtype="float64")


def test_a_process_that_only_loads_finds_the_kind(etn_path, tmp_path):
    """The .etn kind registers itself at start-up, not on first use of einsums.io.etn.

    A process that loads a saved graph and never builds a backing itself is the
    case a lazily registered kind would fail, so it runs in a fresh interpreter.
    """
    import subprocess
    import sys

    einsums.io.write(etn_path, "B", _filled("B", (4, 3), "float64", 0.5))
    A = _filled("A", (5, 4), "float64", 0.25)
    C = einsums.create_zero_tensor("C", [5, 3], dtype="float64")
    g, _ = _product_graph(etn_path, "float64", A, C)
    saved = str(tmp_path / "backed.eig.json")
    cg.save_graph(g, saved)

    script = (
        "import einsums.graph as cg\n"
        f"g = cg.load_graph({saved!r})\n"
        "print('loaded')\n"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=os.environ.copy())
    assert result.returncode == 0, result.stderr
    assert "loaded" in result.stdout


def test_a_tensor_the_caller_holds_cannot_be_backed(etn_path):
    einsums.io.write(etn_path, "B", _filled("B", (4, 3), "float64", 0.5))
    g = cg.Graph("held")
    held = einsums.create_zero_tensor("B", [4, 3], dtype="float64")
    with pytest.raises(ValueError, match="already allocated"):
        cg.annotate(held, backing=einsums.io.etn(etn_path, "B"), graph=g)


def test_annotate_needs_something_to_say():
    with pytest.raises(TypeError, match="spaces=, tag=, backing="):
        cg.annotate(einsums.create_zero_tensor("X", [2], dtype="float64"), graph=cg.Graph("empty"))


def _read_back(path, entry, shape, dtype):
    out = einsums.create_zero_tensor(entry, list(shape), dtype=dtype)
    einsums.io.read(path, entry, out)
    return np.asarray(out, copy=False).copy()


def _producer_graph(path, dtype, A, B, name="produce"):
    """T[i,j] = A[i,k] B[k,j] into a tensor declared on the graph and backed for writing."""
    g = cg.Graph(name)
    T = g.declare_tensor("T", [A.shape[0], B.shape[1]], dtype=dtype)
    cg.annotate(T, backing=einsums.io.etn(path, "T", write=True), graph=g)
    with cg.capture(g):
        einsums.einsum("ij <- ik ; kj", T, A, B)
    return g, T


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_a_write_backed_tensor_is_stored_to_its_entry(etn_path, dtype):
    A = _filled("A", (5, 4), dtype, 0.25)
    B = _filled("B", (4, 3), dtype, 0.5)
    assert not os.path.exists(etn_path)
    g, T = _producer_graph(etn_path, dtype, A, B)
    assert os.path.exists(etn_path)
    assert g.tensor_backing(T).write
    assert g.verify() == []

    g.apply(cg.default_pass_manager())
    g.execute()
    expected = np.asarray(A, copy=False) @ np.asarray(B, copy=False)
    assert_close(_read_back(etn_path, "T", (5, 3), dtype), expected, dtype=dtype)


def test_a_write_backing_reuses_a_matching_entry_and_refuses_another(etn_path):
    einsums.io.write(etn_path, "T", _filled("T", (5, 3), "float64", -9.0))
    einsums.io.write(etn_path, "X", _filled("X", (2, 2), "float64", 3.0))
    A = _filled("A", (5, 4), "float64", 0.25)
    B = _filled("B", (4, 3), "float64", 0.5)
    g, _ = _producer_graph(etn_path, "float64", A, B)
    g.apply(cg.default_pass_manager())
    g.execute()
    expected = np.asarray(A, copy=False) @ np.asarray(B, copy=False)
    assert_close(_read_back(etn_path, "T", (5, 3), "float64"), expected, dtype="float64")
    assert_close(_read_back(etn_path, "X", (2, 2), "float64"), np.asarray(_filled("X", (2, 2), "float64", 3.0), copy=False),
                 dtype="float64")

    wide = _filled("B", (4, 6), "float64", 0.5)
    with pytest.raises(ValueError, match="of another shape or type"):
        _producer_graph(etn_path, "float64", A, wide, name="reshaped")


def test_a_write_backing_nothing_writes_fails_verify(etn_path):
    g = cg.Graph("unwritten")
    T = g.declare_tensor("T", [2, 3], dtype="float64")
    cg.annotate(T, backing=einsums.io.etn(etn_path, "T", write=True), graph=g)
    problems = g.verify()
    assert len(problems) == 1
    assert "no node writes it" in problems[0]


def test_slab_streaming_produces_the_entry_a_block_at_a_time(etn_path):
    A = _filled("A", (5, 4), "float64", 0.25)
    B = _filled("B", (4, 7), "float64", 0.5)
    E = _filled("E", (7, 2), "float64", 0.75)
    D = einsums.create_zero_tensor("D", [5, 2], dtype="float64")
    g, T = _producer_graph(etn_path, "float64", A, B)
    with cg.capture(g):
        einsums.einsum("il <- ij ; jl", D, T, E)

    stream = cg.SlabStreaming()
    stream.set_slab_budget(2 * 5 * 8)
    pm = cg.PassManager()
    pm.add(stream)
    pm.add(cg.Materialization())
    g.apply(pm)
    assert stream.num_written == 1
    assert stream.num_streamed == 2
    g.execute()

    a, b, e = (np.asarray(x, copy=False) for x in (A, B, E))
    assert_close(_read_back(etn_path, "T", (5, 7), "float64"), a @ b, dtype="float64")
    assert_close(np.asarray(D, copy=False), (a @ b) @ e, dtype="float64")
