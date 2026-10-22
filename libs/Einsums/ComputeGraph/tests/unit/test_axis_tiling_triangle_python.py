# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""AxisTiling's triangle, reached from Python through a declared symmetry.

The density-fitted particle ladder of a closed-shell coupled-cluster residual::

    H[Q,a,f,i,j] = sum_e   B[Q,a,e] tau[i,j,e,f]
    r2[i,j,a,b] += sum_Qf  H[Q,a,f,i,j] B[Q,b,f]

is unchanged when (i,a) trades places with (j,b), because tau is. Declared on tau, that lets
AxisTiling run only the occupied tiles on and above the diagonal and write the rest as mirrors.
Every case compares with numpy, and the declaration itself is checked on every replay.
"""

from __future__ import annotations

import numpy as np
import pytest

import einsums
import einsums._core.graph as _G
import einsums.graph as cg
from einsums.testing import ALL_DTYPES, assert_close

_O, _V, _Q = 6, 4, 5


def _pair_exchange():
    """tau[i,j,a,b] = tau[j,i,b,a]: the pair at positions (0, 2) trades places with (1, 3)."""
    return einsums.SymmetryDescriptor([einsums.SymmetryOp.group_swap([0, 2], [1, 3])])


def _antisymmetric_pairs():
    """Spin-orbital: negated by exchanging i with j, and by exchanging a with b."""
    return einsums.SymmetryDescriptor([einsums.SymmetryOp.swap(0, 1, -1), einsums.SymmetryOp.swap(2, 3, -1)])


def _random(shape, dtype, rng):
    data = rng.standard_normal(shape)
    if np.dtype(dtype).kind == "c":
        data = data + 1j * rng.standard_normal(shape)
    return data.astype(dtype)


def _tensor(name, array):
    tensor = einsums.create_zero_tensor(name, list(array.shape), dtype=str(array.dtype))
    np.asarray(tensor)[...] = array
    return tensor


class _Ladder:
    """The ladder's inputs as numpy arrays and as the einsums tensors the graph reads."""

    def __init__(self, dtype, kind, seed=0):
        rng = np.random.default_rng(seed)
        raw = _random((_O, _O, _V, _V), dtype, rng)
        if kind == "symmetric":
            tau = raw + raw.transpose(1, 0, 3, 2)
        elif kind == "antisymmetric":
            tau = raw - raw.transpose(1, 0, 2, 3) - raw.transpose(0, 1, 3, 2) + raw.transpose(1, 0, 3, 2)
        else:
            tau = raw
        self.b = _random((_Q, _V, _V), dtype, rng)
        self.tau = tau
        self.r2 = _random((_O, _O, _V, _V), dtype, rng)
        self.dtype = dtype
        self.B = _tensor("Bvv", self.b)
        self.Tau = _tensor("Tau", self.tau)
        self.R2 = _tensor("r2", self.r2)
        if kind == "symmetric":
            self.Tau.set_symmetry(_pair_exchange())
        elif kind == "antisymmetric":
            self.Tau.set_symmetry(_antisymmetric_pairs())

    def expected(self):
        return self.r2 + np.einsum("qae,ijef,qbf->ijab", self.b, self.tau, self.b)

    def graph(self):
        graph = cg.Graph("ladder")
        H = graph.scratch("H", [_Q, _V, _V, _O, _O], self.dtype)
        with cg.capture(graph):
            einsums.einsum("Q,a,f,i,j <- Q,a,e ; i,j,e,f", H, self.B, self.Tau)
            einsums.einsum("i,j,a,b <- Q,a,f,i,j ; Q,b,f", self.R2, H, self.B, c_pf=1.0)
        return graph


def _tile(graph, block, dtype="float64"):
    """AxisTiling alone, at a cap that holds a tile of H and not the whole of it, block pinned."""
    tiling = _G.AxisTiling()
    # Priced with the built-in rates: the runner's own would move a close choice between machines.
    tiling.use_reference_cost_model()
    tiling.set_memory_cap(_Q * _V * _V * _O * _O * np.dtype(dtype).itemsize // 2)
    tiling.set_block_extent(block)
    manager = cg.PassManager()
    manager.add(tiling)
    graph.apply(manager)
    return tiling


def _run(graph):
    graph.apply(cg.default_pass_manager())
    graph.execute()


def test_the_descriptor_binds_and_a_tensor_carries_it():
    tensor = einsums.create_zero_tensor("t", [3, 3, 2, 2])
    assert not tensor.has_symmetry()
    tensor.set_symmetry(_pair_exchange())
    assert tensor.has_symmetry()
    tensor.clear_symmetry()
    assert not tensor.has_symmetry()
    assert einsums.SymmetryDescriptor.eri_8fold().size() == 3
    assert einsums.SymmetryDescriptor().empty()


@pytest.mark.parametrize("dtype", ALL_DTYPES)
@pytest.mark.parametrize("block", [2, 4])
def test_a_declared_symmetric_tau_runs_the_ladder_as_a_triangle(dtype, block):
    # Blocks of 2 divide the six occupied orbitals; blocks of 4 leave a remainder of 2.
    ladder = _Ladder(dtype, "symmetric")
    graph = ladder.graph()
    tiling = _tile(graph, block, dtype)
    assert tiling.emission == "triangle"
    assert tiling.triangle_sign == 1
    _run(graph)
    assert_close(ladder.R2, ladder.expected(), dtype=dtype)


def test_an_antisymmetric_tau_mirrors_with_the_opposite_sign():
    ladder = _Ladder("float64", "antisymmetric")
    graph = ladder.graph()
    tiling = _tile(graph, 2)
    assert tiling.emission == "triangle"
    assert tiling.triangle_sign == -1
    _run(graph)
    assert_close(ladder.R2, ladder.expected(), dtype="float64")


def test_without_a_declaration_every_tile_runs():
    ladder = _Ladder("float64", "general")
    graph = ladder.graph()
    tiling = _tile(graph, 2)
    assert tiling.emission == "range"
    assert tiling.triangle_sign == 0
    _run(graph)
    assert_close(ladder.R2, ladder.expected(), dtype="float64")


def test_a_broken_declaration_stops_the_replay():
    ladder = _Ladder("float64", "symmetric")
    graph = ladder.graph()
    assert _tile(graph, 2).emission == "triangle"
    _run(graph)
    np.asarray(ladder.Tau)[0, 1, 2, 3] += 1.0
    with pytest.raises(RuntimeError, match="declared symmetric"):
        graph.execute()
