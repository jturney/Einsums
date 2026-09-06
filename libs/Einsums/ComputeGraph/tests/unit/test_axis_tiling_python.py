# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""AxisTiling on a real molecule's full-axis DF-MP2, offline.

The program is the one ``test_laplace_mp2_python.py`` runs, the correlation energy written over
all four orbital indices rather than pair by pair::

    K[i,a,j,b] = sum_Q B[Q,i,a] B[Q,j,b]
    T          = K * D
    E          = sum_iajb (2 K - K[i,b,j,a]) T

At water/cc-pVDZ that declares five tensors of 5 x 19 x 5 x 19 doubles, 72200 bytes each, which
is exactly the object density fitting exists to avoid. The hand-written pair-driven form in
``examples/psi4-bridge/df_mp2_graph.py`` holds one 19 x 19 block instead. This shard is what says
the pass derives the second schedule from the first.

Every case asserts the mechanism rather than that something happened: which axes, at which
extents, into how many slices, in chunks of what, and what the largest intermediate measures on
each side of the decision.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

import einsums
import einsums._core.graph as _G
import einsums.graph as cg
from einsums import linalg as la

_HERE = os.path.dirname(os.path.abspath(__file__))
_FIXTURE = os.path.normpath(
    os.path.join(_HERE, "..", "..", "..", "..", "..", "examples", "dlpno", "fixtures", "water-ccpvdz.npz"))

#: A cap between one occupied pair of the four-index tensor and one occupied row of it. The pair
#: is 19 x 19 doubles, the row is 5 x 19 x 19, so this is the cap that forces per-pair streaming
#: and nothing coarser.
_PAIR_CAP = 4096


def _tensor(name, array):
    tensor = einsums.create_zero_tensor(name, list(array.shape))
    np.asarray(tensor)[...] = np.ascontiguousarray(array)
    return tensor


@pytest.fixture(scope="module")
def water():
    """Canonical orbitals, the fitted three-index tensor, and the orbital energies."""
    if not os.path.exists(_FIXTURE):
        pytest.skip(f"fixture not present: {_FIXTURE}")
    z = np.load(_FIXTURE, allow_pickle=True)

    overlap = _tensor("S", z["S"])
    fock = _tensor("F", z["F"])
    nbf = overlap.dim(0)
    orthogonalizer = la.pow(overlap, -0.5, 1e-10)
    half = einsums.create_zero_tensor("X'F", [nbf, nbf])
    ortho = einsums.create_zero_tensor("X'FX", [nbf, nbf])
    la.gemm(1.0, orthogonalizer, fock, 0.0, half, trans_a=True)
    la.gemm(1.0, half, orthogonalizer, 0.0, ortho)
    energies = einsums.create_zero_tensor("eps", [nbf])
    la.syev(ortho, energies, compute_eigenvectors=True)
    coefficients = einsums.create_zero_tensor("C", [nbf, nbf])
    la.gemm(1.0, orthogonalizer, ortho, 0.0, coefficients)

    nocc = int(z["C_occ"].shape[1])
    nvir = nbf - nocc
    naux = int(z["metric"].shape[0])
    eps = np.array(np.asarray(energies))
    coeff = np.array(np.asarray(coefficients))

    ao = _tensor("(Q|mn)", np.asarray(z["eri_3index"], dtype=np.float64))
    metric = _tensor("(P|Q)", np.asarray(z["metric"], dtype=np.float64))
    occupied = _tensor("C_occ", coeff[:, :nocc])
    virtual = _tensor("C_vir", coeff[:, nocc:])
    partial = einsums.create_zero_tensor("(Q|in)", [naux, nocc, nbf])
    three = einsums.create_zero_tensor("(Q|ia)", [naux, nocc, nvir])
    einsums.einsum("Q,m,n ; m,i -> Q,i,n", partial, ao, occupied)
    einsums.einsum("Q,i,n ; n,a -> Q,i,a", three, partial, virtual)
    fitted = einsums.create_zero_tensor("B", [naux, nocc, nvir])
    einsums.einsum("P,Q ; Q,i,a -> P,i,a", fitted, la.pow(metric, -0.5, 1e-10), three)

    return {
        "nocc": nocc, "nvir": nvir, "naux": naux,
        "occupied_energies": _tensor("eps_occ", eps[:nocc]),
        "virtual_energies": _tensor("eps_vir", eps[nocc:]),
        "three": three, "metric": metric, "fitted": fitted,
        "reference": float(z["energy_psi4_df_mp2"]),
    }


def _shape(water):
    return [water["nocc"], water["nvir"], water["nocc"], water["nvir"]]


def _denominator(water, name="D"):
    denominator = einsums.create_zero_tensor(name, _shape(water))
    la.outer_sum(denominator,
                 [water["occupied_energies"], water["virtual_energies"],
                  water["occupied_energies"], water["virtual_energies"]],
                 [1.0, -1.0, 1.0, -1.0])
    la.element_transform(denominator, lambda x: 1.0 / x)
    return denominator


def _capture(graph, water, denominator, energy):
    """The full-axis energy. ``K`` is formed twice because the exchange term needs its own copy."""
    shape = _shape(water)
    B = water["fitted"]
    K = graph.scratch("K", shape, "float64")
    T = graph.scratch("T", shape, "float64")
    again = graph.scratch("K_again", shape, "float64")
    exchange = graph.scratch("K_exchange", shape, "float64")
    combination = graph.scratch("Kbar", shape, "float64")
    with cg.capture(graph):
        einsums.einsum("Q,i,a ; Q,j,b -> i,a,j,b", K, B, B)
        la.direct_product(1.0, K, denominator, 0.0, T)
        einsums.einsum("Q,i,a ; Q,j,b -> i,a,j,b", again, B, B)
        einsums.permute("iajb <- ibja", exchange, again)
        la.axpby(2.0, again, 0.0, combination)
        la.axpby(-1.0, exchange, 1.0, combination)
        la.dot(energy, combination, T)


def _tiled(graph, cap):
    """Apply the pass alone at @p cap and hand back what it decided."""
    tiling = _G.AxisTiling()
    tiling.set_memory_cap(cap)
    manager = cg.PassManager()
    manager.add(tiling)
    graph.apply(manager)
    return tiling


def _full_axis_graph(water, label="mp2 full axis"):
    energy = einsums.create_zero_tensor("E", [1])
    graph = cg.Graph(label)
    _capture(graph, water, _denominator(water), energy)
    return graph, energy


def _reason_fragments(tiling):
    return " | ".join(reason for reason, _ in tiling.skip_reasons)


def test_the_chosen_axes_are_the_two_occupied_ones(water):
    """The decision, on the program the design was written against.

    The virtual axes are not rejected on their size, which at this molecule is smaller than the
    occupied ones. They are rejected because the exchange permutation exchanges them, so a slice
    of the permuted tensor would need a slice of its source at a pair the body is not at.
    """
    graph, _ = _full_axis_graph(water)
    tiling = _tiled(graph, _PAIR_CAP)

    assert tiling.axis_letters == ["i", "j"], _reason_fragments(tiling)
    assert tiling.axis_names == ["K[0]", "K[2]"]
    assert tiling.axis_extents == [water["nocc"], water["nocc"]]
    assert tiling.slice_count == water["nocc"] * water["nocc"]
    assert tiling.accumulator == "E"


def test_the_largest_intermediate_falls_from_the_four_index_tensor_to_one_pair(water):
    """o^2 v^2 becomes v^2, which is the whole of what the pass is for."""
    nocc, nvir = water["nocc"], water["nvir"]
    graph, _ = _full_axis_graph(water)
    tiling = _tiled(graph, _PAIR_CAP)

    assert tiling.largest_before == nocc * nvir * nocc * nvir * 8
    assert tiling.largest_after == nvir * nvir * 8
    assert tiling.depth == 1
    assert tiling.iterations == nocc * nocc

    # Every four-index intermediate streams, and so does the three-index integral, because one
    # of its axes is an occupied one. The energy is the only thing left whole, and it is the
    # accumulation.
    assert set(tiling.streamed) == {"K", "T", "K_again", "K_exchange", "Kbar", "B", "D"}
    assert tiling.whole == ["E"]


def test_the_cap_decides_and_a_program_that_fits_is_left_alone(water):
    """The decline that makes the pass a no-op on a form that never needed it."""
    nocc, nvir = water["nocc"], water["nvir"]
    graph, _ = _full_axis_graph(water)
    tiling = _tiled(graph, nocc * nvir * nocc * nvir * 8)

    assert tiling.slice_count == 0
    assert "already fits the cap" in _reason_fragments(tiling)


def test_raising_the_cap_moves_the_decision_from_the_pair_to_the_row(water):
    """What the traffic criterion buys, measured rather than asserted.

    A cap of five pairs admits the occupied ROW as well, and the row wins: it reads each
    ``B[Q,i,:]`` slab once where the pair schedule re-reads it once per ``j``. The chunk of
    pairs is never reached at this molecule, because the number of occupied orbitals and the
    number of pairs stand in exactly the ratio that makes the two thresholds coincide.
    """
    nocc, nvir = water["nocc"], water["nvir"]
    graph, _ = _full_axis_graph(water)
    tiling = _tiled(graph, nocc * nvir * nvir * 8)

    assert tiling.axis_letters == ["i"]
    assert tiling.slice_count == nocc
    assert tiling.largest_after == nocc * nvir * nvir * 8


def test_after_the_laplace_transform_the_pass_declines(water):
    """The order between the two passes, read off the program rather than assumed.

    The transform dissolves the four-index denominator, and what it leaves behind at this
    molecule is BIGGER than what it replaced: one exponential-dressed integral per quadrature
    point, ``Q o v t``, against ``o^2 v^2``. So the pass declines, and the reason the tally gives
    is that no candidate axis set brings the largest intermediate under the cap rather than that
    there is nothing left to tile.
    """
    energy = einsums.create_zero_tensor("E_laplace", [1])
    graph = cg.Graph("mp2 laplace")
    denominator = _denominator(water, "D_tagged")
    _capture(graph, water, denominator, energy)
    graph.annotate_tag(denominator, _G.LaplaceTransform.denominator_tag(
        ["eps_occ", "eps_vir", "eps_occ", "eps_vir"], "+-+-"))

    transform = cg.LaplaceTransform()
    transform.set_epsilon(1e-5)
    transform.add_energy("eps_occ", water["occupied_energies"])
    transform.add_energy("eps_vir", water["virtual_energies"])
    manager = cg.PassManager()
    manager.add(transform)
    assert graph.apply(manager), f"the transform declined: {transform.skip_reasons}"

    plain, _ = _full_axis_graph(water, "mp2 plain")
    before = _tiled(plain, _PAIR_CAP).largest_before

    tiling = _tiled(graph, _PAIR_CAP)
    assert tiling.slice_count == 0
    assert "under the cap" in _reason_fragments(tiling)
    assert tiling.largest_before > before, (
        "the transform is expected to widen the intermediates at this molecule, and the decline "
        "below is about that rather than about the four-index tensor being gone")
