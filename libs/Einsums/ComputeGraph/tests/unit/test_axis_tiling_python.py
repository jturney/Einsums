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

At water/cc-pVDZ that declares four tensors of 5 x 19 x 5 x 19 doubles, 72200 bytes each, which
is exactly the object density fitting exists to avoid. The hand-written pair-driven form in
``examples/psi4-bridge/df_mp2_graph.py`` holds one 19 x 19 block instead. This shard is what says
the pass derives the second schedule from the first.

Every case asserts the mechanism rather than that something happened: which axes, at which
extents, into how many slices, in chunks of what, and what the largest intermediate measures on
each side of the decision.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

import einsums
import einsums._core.graph as _G
import einsums.graph as cg
from einsums import linalg as la

from _region_invariants import assert_materialization_invariants

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
    """The full-axis energy, over one integral that four statements read."""
    shape = _shape(water)
    B = water["fitted"]
    K = graph.scratch("K", shape, "float64")
    T = graph.scratch("T", shape, "float64")
    exchange = graph.scratch("K_exchange", shape, "float64")
    combination = graph.scratch("Kbar", shape, "float64")
    with cg.capture(graph):
        einsums.einsum("Q,i,a ; Q,j,b -> i,a,j,b", K, B, B)
        la.direct_product(1.0, K, denominator, 0.0, T)
        einsums.permute("iajb <- ibja", exchange, K)
        la.axpby(2.0, K, 0.0, combination)
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
    """A graph, its energy, and the denominator it reads.

    The denominator comes back with the rest because the graph holds it by ADDRESS: a caller
    who lets it fall out of scope leaves the graph pointing at freed storage, and a pass that
    reads a handle at optimize time finds it there rather than at execute.
    """
    energy = einsums.create_zero_tensor("E", [1])
    graph = cg.Graph(label)
    denominator = _denominator(water)
    _capture(graph, water, denominator, energy)
    return graph, energy, denominator


def _reason_fragments(tiling):
    return " | ".join(reason for reason, _ in tiling.skip_reasons)


def test_the_chosen_axes_are_the_two_occupied_ones(water):
    """The decision, on the program the design was written against.

    The virtual axes are not rejected on their size, which at this molecule is smaller than the
    occupied ones. They are rejected because the exchange permutation exchanges them, so a slice
    of the permuted tensor would need a slice of its source at a pair the body is not at.
    """
    graph, _, _held = _full_axis_graph(water)
    tiling = _tiled(graph, _PAIR_CAP)

    assert tiling.axis_letters == ["i", "j"], _reason_fragments(tiling)
    assert tiling.axis_names == ["K[0]", "K[2]"]
    assert tiling.axis_extents == [water["nocc"], water["nocc"]]
    assert tiling.slice_count == water["nocc"] * water["nocc"]
    assert tiling.accumulator == "E"


def test_the_largest_intermediate_falls_from_the_four_index_tensor_to_one_pair(water):
    """o^2 v^2 becomes v^2, which is the whole of what the pass is for."""
    nocc, nvir = water["nocc"], water["nvir"]
    graph, _, _held = _full_axis_graph(water)
    tiling = _tiled(graph, _PAIR_CAP)

    assert tiling.largest_before == nocc * nvir * nocc * nvir * 8
    assert tiling.largest_after == nvir * nvir * 8
    assert tiling.depth == 1
    assert tiling.iterations == nocc * nocc

    # Every four-index intermediate streams, and so does the three-index integral, because one
    # of its axes is an occupied one. The energy is the only thing left whole, and it is the
    # accumulation.
    assert set(tiling.streamed) == {"K", "T", "K_exchange", "Kbar", "B", "D"}
    assert tiling.whole == ["E"]


def test_the_cap_decides_and_a_program_that_fits_is_left_alone(water):
    """The decline that makes the pass a no-op on a form that never needed it."""
    nocc, nvir = water["nocc"], water["nvir"]
    graph, _, _held = _full_axis_graph(water)
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
    graph, _, _held = _full_axis_graph(water)
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

    plain, _, _held = _full_axis_graph(water, "mp2 plain")
    before = _tiled(plain, _PAIR_CAP).largest_before

    tiling = _tiled(graph, _PAIR_CAP)
    assert tiling.slice_count == 0
    assert "under the cap" in _reason_fragments(tiling)
    assert tiling.largest_before > before, (
        "the transform is expected to widen the intermediates at this molecule, and the decline "
        "below is about that rather than about the four-index tensor being gone")


# ── The rewrite ─────────────────────────────────────────────────────────────

#: The re-associating tier's bound at double precision, which is what the accumulation across
#: iterations owes: the sum is taken pair by pair rather than over the whole four-index tensor.
_TIER = 1024.0 * 2.2e-16


def _pair_driven_energy(water):
    """The hand-written pair loop, in numpy, which never forms a four-index tensor.

    This is ``examples/psi4-bridge/df_mp2_graph.py``'s algorithm stated as an oracle: for each
    occupied pair, one ``v`` by ``v`` integral block, its transpose, the denominator over the
    virtual pair, and one reduction. Written outside einsums on purpose, so what the tiled
    graph is compared against is the ALGORITHM rather than another einsums program.
    """
    B = np.asarray(water["fitted"])
    eo = np.asarray(water["occupied_energies"])
    ev = np.asarray(water["virtual_energies"])
    total = 0.0
    for i in range(water["nocc"]):
        for j in range(water["nocc"]):
            block = B[:, i, :].T @ B[:, j, :]
            denominator = 1.0 / (eo[i] + eo[j] - ev[:, None] - ev[None, :])
            total += float(np.sum((2.0 * block - block.T) * (block * denominator)))
    return total


def _replayed(water, cap):
    """Capture, tile at @p cap, optimize, execute. Returns the energy and the graph."""
    energy = einsums.create_zero_tensor("E", [1])
    graph = cg.Graph("mp2 tiled")
    denominator = _denominator(water)
    _capture(graph, water, denominator, energy)
    tiling = _tiled(graph, cap)
    graph.apply(cg.default_pass_manager())
    graph.execute()
    return float(np.asarray(energy)[0]), graph, tiling, denominator


def _untiled(water):
    energy = einsums.create_zero_tensor("E_untiled", [1])
    graph = cg.Graph("mp2 untiled")
    denominator = _denominator(water)
    _capture(graph, water, denominator, energy)
    graph.apply(cg.default_pass_manager())
    graph.execute()
    return float(np.asarray(energy)[0])


def test_the_tiled_loop_replays_the_untiled_energy(water):
    """The number, against the program with no tiling on it and against the pair loop."""
    reference = _untiled(water)
    energy, _, tiling, _held = _replayed(water, _PAIR_CAP)

    assert tiling.num_tiled == 1
    assert energy == pytest.approx(reference, rel=_TIER, abs=1e-13)
    assert energy == pytest.approx(_pair_driven_energy(water), rel=1e-12, abs=1e-13)
    assert energy == pytest.approx(water["reference"], abs=1e-9)


def test_the_rewritten_program_is_one_loop_and_the_zeroing_of_its_accumulation(water):
    """The node set, so the assertion is about what was emitted rather than that it ran.

    Six captured nodes, one contraction each for the integral and the reduction and four
    statements between them, and two emitted: the zeroing of the accumulation and the loop that
    is the rest of the program. The captured count was seven while this shard formed the
    integral a second time for the exchange combination to read; nothing about the schedule
    depended on that, which is why the copy is gone.
    """
    energy = einsums.create_zero_tensor("E_shape", [1])
    graph = cg.Graph("mp2 tiled")
    denominator = _denominator(water)
    _capture(graph, water, denominator, energy)
    captured = graph.num_nodes()
    tiling = _tiled(graph, _PAIR_CAP)

    assert captured == 6
    assert tiling.num_tiled == 1
    assert graph.num_nodes() == 2
    assert [n["kind"] for n in json.loads(graph.to_json())["nodes"]] == ["Scale", "Loop"]


def test_the_tiled_graph_keeps_the_storage_invariants(water):
    """No buffer allocated for a tensor the rewrite dissolved, and none allocated twice."""
    _, graph, _, _held = _replayed(water, _PAIR_CAP)
    assert_materialization_invariants(graph, "axis tiling")


def test_a_replay_restarts_the_sweep_rather_than_continuing_it(water):
    """The slice index is a cursor, so the second replay has to begin at the first pair again."""
    energy = einsums.create_zero_tensor("E_replay", [1])
    graph = cg.Graph("mp2 tiled")
    denominator = _denominator(water)
    _capture(graph, water, denominator, energy)
    _tiled(graph, _PAIR_CAP)
    graph.apply(cg.default_pass_manager())

    graph.execute()
    first = float(np.asarray(energy)[0])
    graph.execute()
    assert float(np.asarray(energy)[0]) == pytest.approx(first, rel=1e-14)


# ── The round trip ──────────────────────────────────────────────────────────

def test_the_tiled_loop_is_re_derived_on_the_other_side_of_a_file(water, tmp_path):
    """A saved graph carries the ALGEBRA and a load re-derives the schedule from it.

    That is the phase rule rather than a limitation met halfway: tiling is a resource decision,
    it is taken against the machine the replay runs on, and a cap stated where the graph was
    built is not the cap in force where it is loaded. So the file holds the full-axis program,
    the load rebinds it to fresh buffers, and the loop is emitted there.
    """
    graph, _, denominator = _full_axis_graph(water, "mp2 to save")
    path = str(tmp_path / "mp2_full_axis.eig")
    cg.save_graph(graph, path)

    loaded = cg.load_graph(path)
    names = set(loaded.manifest_names())
    replayed = einsums.create_zero_tensor("E", [1])
    mapping = {"B": water["fitted"], "D": denominator, "E": replayed}
    cg.bind(loaded, {name: tensor for name, tensor in mapping.items() if name in names})

    tiling = _tiled(loaded, _PAIR_CAP)
    assert tiling.num_tiled == 1
    assert tiling.axis_letters == ["i", "j"]
    assert tiling.largest_after == water["nvir"] * water["nvir"] * 8

    loaded.apply(cg.default_pass_manager())
    loaded.execute()
    assert float(np.asarray(replayed)[0]) == pytest.approx(_untiled(water), rel=_TIER, abs=1e-13)


def test_a_tiled_graph_says_which_of_its_nodes_a_file_cannot_hold(water):
    """The narrowing, pinned rather than left to be rediscovered.

    A tiled graph does not save, and the report names the two reasons. The slice index is a
    ``write_param`` whose source is a callback, because a parameter that advances by a chunk
    every iteration is arithmetic no ``BoundExpr`` arm expresses. And a ``View`` node is not
    reconstructible at all: it has no builder entry and no IR encoding, so a parametric slice
    cannot cross a file whatever its bounds are spelled as.

    Neither costs anything, because the schedule is never what a file is supposed to hold. What
    would have to change for a tiled graph to save is the View node kind, and the slice index
    after it.
    """
    graph, _, _held = _full_axis_graph(water, "mp2 blockers")
    assert _tiled(graph, _PAIR_CAP).num_tiled == 1

    blockers = graph.serializability_report()
    assert blockers, "a tiled graph is expected not to save"
    kinds = {b.kind_name for b in blockers}
    assert kinds == {"WriteParam", "View"}, kinds

    by_kind = {b.kind_name: b.reason for b in blockers}
    assert "callback arm" in by_kind["WriteParam"]
    assert "not yet reconstructible" in by_kind["View"]
    assert all(b.subgraph_path.startswith("loop(") for b in blockers), (
        "every blocker is expected to be inside the emitted body")


# ── The composition ─────────────────────────────────────────────────────────

#: The family, declared with the extents ``test_naf_python.py`` declares them with. The registry
#: is process-global and a derived space cannot be declared twice from two different families,
#: so the two shards have to agree; only the RATIOS matter to the comparison.
_FAMILY = (("occ", "o", 300.0, "nocc"), ("vir", "v", 2700.0, "nvir"), ("aux", "x", 8100.0, "naux"))


@pytest.fixture(scope="module")
def naf_family():
    registry = cg.global_space_registry()
    for name, symbol, extent, dim in _FAMILY:
        registry.register_space(cg.index_space(name, symbol, extent, cg.GrowthClass.linear(), dim))
    _G.NaturalAuxiliaryFactorization.register_naf_space(cg.Graph("naf_family"))
    return registry


def test_an_auxiliary_truncation_composes_with_the_tiling_on_one_program(water, naf_family):
    """Two passes, two phases, one program: the algebra is truncated, then the schedule sliced.

    A truncation shortens an index and changes nothing about which axes are free, so the tiling
    decision has to come out the same on the truncated program as on the full one. That is the
    claim, and it is what makes the two composable at all: one is a statement about the algebra
    and the other about the machine, and the phase order is what keeps them from arguing.
    """
    energy = einsums.create_zero_tensor("E_naf", [1])
    graph = cg.Graph("mp2 naf then tiled")
    denominator = _denominator(water)
    _capture(graph, water, denominator, energy)

    B = water["fitted"]
    cg.annotate(B, ("aux", "occ", "vir"), graph=graph)
    graph.annotate_tag(B, _G.ProvenanceTag.make("eri"))

    registry = _G.FactorizationRegistry()
    registry.add(_G.NaturalAuxiliaryFactorization("eri", B, 1e-1))
    factorization = _G.FactorizationPass(registry)
    manager = cg.PassManager()
    manager.add(cg.ProvenancePropagation())
    manager.add(factorization)
    assert graph.apply(manager), f"the truncation declined: {factorization.skip_reasons}"

    records = graph.approximations()
    # One record per substituted occurrence of the integral, and the program forms it once.
    assert {r.pass_name for r in records} == {"NaturalAuxiliary"}
    assert len(records) == 1

    # The schedule, decided on the truncated program. The occupied pair is still the answer:
    # what the truncation moved is the length of a summed index, and a summed index was never a
    # candidate. The CAP has to move, and that is the composition's one visible cost: the
    # truncation leaves a Y-by-Y coupling that carries no orbital index at all, so it is
    # loop-invariant, it is formed whole outside the loop, and it is now the largest live
    # intermediate. A cap tight enough to force per-pair streaming on the untruncated program
    # does not clear it, and the pass says so rather than tiling around it.
    assert _tiled(graph, _PAIR_CAP).num_tiled == 0

    coupling = 39 * 39 * 8
    tiling = _tiled(graph, 13000)
    assert tiling.num_tiled == 1, _reason_fragments(tiling)
    assert tiling.axis_letters == ["i", "j"]
    assert tiling.largest_after == coupling

    graph.apply(cg.default_pass_manager())
    graph.execute()
    value = float(np.asarray(energy)[0])

    exact = _untiled(water)
    bound = sum(r.bound for r in records) * abs(exact)
    assert abs(value - exact) <= bound + 1e-13, (
        f"the composed energy {value} is outside the truncation's own bound {bound:.3e} against {exact}")
    assert value != exact, "the truncation is expected to move the number it approximates"
