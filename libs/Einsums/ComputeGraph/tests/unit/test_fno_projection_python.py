# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""The other half of ``BasisTruncation``: projecting the caller's tensors into the space it built.

``test_fno_python`` pins the SETUP, which decides what the truncated space is.
This file pins what the pass does with it: every tensor the algebra reads whose
slot carries the space being replaced is projected into a graph-owned entry
``<name>@fno`` the setup writes, the algebra is pointed at the projected entries,
the intermediates it writes on the way are re-declared over the truncated space,
and the orbital energies over the replaced space are REPLACED by the
semicanonical ones rather than projected.

The case that matters is a tensor with TWO axes over the space being replaced,
which is every four-external integral and every amplitude. Its projection is a
chain of two contractions with an intermediate between them, and the intermediate
is declared on the setup body before the capture guard opens so the resource phase
places its lifecycle inside the body. Nothing about the chain is optional: with
that lifecycle taken away, execute stops at
``BasisTruncation(vir)/setup_fno_project_0`` and says it was never given storage.

The water numbers, cc-pVDZ with 5 occupied and 19 virtual orbitals: a cutoff of
1e-2 keeps 4, 1e-3 keeps 9 and 1e-4 keeps 17, and the correction the record
carries at 1e-3 is -2.0920e-2 hartree.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

import einsums
import einsums._core.graph as _G
import einsums.graph as cg

_HERE = os.path.dirname(os.path.abspath(__file__))
_FIXTURE = os.path.normpath(
    os.path.join(_HERE, "..", "..", "..", "..", "..", "examples", "dlpno", "fixtures", "water-ccpvdz.npz"))

#: The family the caller declares, the same ratios ``test_naf_python`` uses.
_FAMILY = (("occ", "o", 300.0, "nocc"), ("vir", "v", 2700.0, "nvir"), ("aux", "x", 8100.0, "naux"))


def _tensor(name, array):
    tensor = einsums.create_zero_tensor(name, list(array.shape))
    np.asarray(tensor)[...] = np.ascontiguousarray(array)
    return tensor


@pytest.fixture(scope="module", autouse=True)
def family():
    """The family and the truncated space declared once, on the process-global registry."""
    registry = cg.global_space_registry()
    for name, symbol, extent, dim in _FAMILY:
        registry.register_space(cg.index_space(name, symbol, extent, cg.GrowthClass.linear(), dim))
    _G.BasisTruncation.register_fno_space(cg.Graph("fno_projection_family"))
    return registry


@pytest.fixture(scope="module")
def water():
    """Canonical orbitals, the density-fitted integrals, and the MP2 amplitudes."""
    if not os.path.exists(_FIXTURE):
        pytest.skip(f"fixture not present: {_FIXTURE}")
    z = np.load(_FIXTURE, allow_pickle=False)

    overlap, fock = z["S"], z["F"]
    nbf = overlap.shape[0]
    values, vectors = np.linalg.eigh(overlap)
    orthogonalizer = vectors @ np.diag(values ** -0.5) @ vectors.T
    energies, rotation = np.linalg.eigh(orthogonalizer.T @ fock @ orthogonalizer)
    coefficients = orthogonalizer @ rotation
    nocc = int(z["C_occ"].shape[1])

    metric = z["metric"]
    mvals, mvecs = np.linalg.eigh(metric)
    inv_sqrt = mvecs @ np.diag(np.where(mvals > 1e-10, mvals ** -0.5, 0.0)) @ mvecs.T
    three_ao = np.einsum("PQ,Qmn->Pmn", inv_sqrt, z["eri_3index"])
    three = np.einsum("Pmn,mp,nq->Ppq", three_ao, coefficients, coefficients)

    o, v = slice(0, nocc), slice(nocc, nbf)
    B_ov = np.ascontiguousarray(three[:, o, v])
    ovov = np.einsum("Qia,Qjb->iajb", B_ov, B_ov)
    gaps = (energies[o, None, None, None] - energies[None, v, None, None]
            + energies[None, None, o, None] - energies[None, None, None, v])
    return {
        "nocc": nocc, "nvir": nbf - nocc, "naux": three.shape[0],
        "B_ov": B_ov,
        "ovov": ovov,
        "amplitudes": np.ascontiguousarray(ovov / gaps),
        "fock_vv": np.ascontiguousarray(np.diag(energies[v])),
        "eps_occ": np.ascontiguousarray(energies[o]),
        "eps_vir": np.ascontiguousarray(energies[v]),
        "reference": float(z["energy_psi4_df_mp2"]),
    }


def _mp2(ovov, eps_occ, eps_vir):
    """The MP2 correlation energy from a set of integrals and two energy vectors."""
    gaps = (eps_occ[:, None, None, None] - eps_vir[None, :, None, None]
            + eps_occ[None, None, :, None] - eps_vir[None, None, None, :])
    amplitudes = ovov / gaps
    return float(np.einsum("iajb,iajb->", 2.0 * ovov - ovov.transpose(0, 3, 2, 1), amplitudes))


def _ladder(problem, name):
    """``K[i,a,j,b] = sum_Q B[Q,i,a] B[Q,j,b]``, the shape a density-fitted program writes.

    ``K`` is a graph-owned intermediate rather than the caller's tensor, which is
    what lets its extents follow the space: an interface tensor the algebra WRITES
    belongs to the caller and the pass declines to shrink it. It is read by a pair
    quantity the caller owns, because an intermediate nothing reads is one dead-node
    elimination is right to take away and the sweep needs its values.
    """
    nocc, nvir = problem["nocc"], problem["nvir"]
    three = _tensor("B_ov", problem["B_ov"])
    pairs = einsums.create_zero_tensor("pairs", [nocc, nocc])

    graph = cg.Graph(name)
    integrals = graph.declare_tensor("K", [nocc, nvir, nocc, nvir], True)
    with cg.capture(graph):
        einsums.einsum("Q,i,a ; Q,j,b -> i,a,j,b", integrals, three, three)
        einsums.einsum("i,a,j,b ; i,a,j,b -> i,j", pairs, integrals, integrals)
    cg.annotate(three, ("aux", "occ", "vir"), graph=graph)
    cg.annotate(integrals, ("occ", "vir", "occ", "vir"), graph=graph)
    return {"graph": graph, "three": three, "integrals": integrals, "pairs": pairs}


def _two_axis(problem, name):
    """The caller's own four-index integrals, read by the algebra over TWO virtual axes.

    The case the projection is a chain for: one contraction per axis, with an
    intermediate between them that the setup body has to declare and the resource
    phase has to place inside it.
    """
    nocc = problem["nocc"]
    integrals = _tensor("K", problem["ovov"])
    pairs = einsums.create_zero_tensor("pairs", [nocc, nocc])

    graph = cg.Graph(name)
    with cg.capture(graph):
        einsums.einsum("i,a,j,b ; i,a,j,b -> i,j", pairs, integrals, integrals)
    cg.annotate(integrals, ("occ", "vir", "occ", "vir"), graph=graph)
    return {"graph": graph, "integrals": integrals, "pairs": pairs}


def _truncate(program, problem, occupation, verbosity=1):
    """Run the pass over @p program at @p occupation."""
    held = {
        "amplitudes": _tensor("t2", problem["amplitudes"]),
        "fock": _tensor("F_vv", problem["fock_vv"]),
        "eps_occ": _tensor("eps_occ", problem["eps_occ"]),
    }
    truncation = _G.BasisTruncation()
    truncation.set_amplitudes(held["amplitudes"])
    truncation.set_fock(held["fock"])
    truncation.set_occupied_energies(held["eps_occ"])
    truncation.set_occupation(occupation)

    manager = cg.PassManager()
    manager.set_verbosity(verbosity)
    manager.add(truncation)
    program.update(held)
    program["pass"] = truncation
    program["fired"] = program["graph"].apply(manager)
    return program


def _replay(program, verbosity=1):
    """Optimize and execute, and hand back the truncated integrals and energies."""
    manager = cg.default_pass_manager()
    manager.set_verbosity(verbosity)
    program["graph"].apply(manager)
    program["graph"].execute()
    return (np.array(np.asarray(program["integrals"]), copy=True),
            np.array(np.asarray(program["pass"].energies()), copy=True))


# ──────────────────────────────────────────────────────────────────────────
# The chain, which is the case a four-index tensor poses
# ──────────────────────────────────────────────────────────────────────────


def test_a_tensor_with_two_virtual_axes_is_projected_through_a_two_step_chain(water, capfd):
    """The mechanism, named part by part rather than inferred from a right answer.

    A new graph-owned entry at the truncated extents, spaces that say the axes
    moved, an algebra that reads the projected entry and not the caller's, and a
    chain intermediate whose lifecycle is inside the setup body. The last is the
    one the previous attempt at this died on: an intermediate nothing allocates
    is a replay that stops in the second contraction.
    """
    program = _truncate(_two_axis(water, "fno_chain"), water, 1e-3)
    assert program["fired"], dict(program["pass"].skip_reasons)
    kept = program["pass"].kept
    assert 0 < kept < water["nvir"]

    # The projected entry, by name, and NOT as something a bind supplies.
    assert list(program["pass"].projected) == ["K@fno"]
    names = set(program["graph"].manifest_names())
    assert "K@fno_in" in names, sorted(names)
    assert "K@fno" not in names, "a projected tensor is graph-owned, not something a bind supplies"

    manager = cg.default_pass_manager()
    manager.set_verbosity(2)
    program["graph"].apply(manager)
    captured = capfd.readouterr()
    assert "'BasisTruncation(vir)/setup_fno_project_0' inside setup body 'BasisTruncation(vir)/setup'" in captured.err, captured.err
    assert not cg.duplicate_materializations(program["graph"])
    assert not cg.stranded_materializations(program["graph"])

    program["graph"].execute()
    U = np.asarray(program["pass"].transformation())
    projected = np.einsum("iajb,ay,bz->iyjz", water["ovov"], U, U)
    assert np.allclose(np.asarray(program["pairs"]), np.einsum("iyjz,iyjz->ij", projected, projected), rtol=1e-10, atol=1e-14)


def test_the_algebra_reads_the_projected_tensor_and_not_the_one_it_replaced(water):
    """Both halves of the redirect, checked where each of them shows.

    The operand LISTS are what every later pass reads and the SLOTS are what an
    executor baked at capture resolves through. A rewrite of the ids alone leaves
    the replay reading the untruncated buffer and reporting the untruncated
    answer, which nothing else here would notice: the numbers below are what say
    the slot moved, and the extents are what say the list did.
    """
    program = _truncate(_ladder(water, "fno_redirect"), water, 1e-3)
    assert program["fired"]
    got, _energies = _replay(program)

    untruncated = np.einsum("Qia,Qjb->iajb", water["B_ov"], water["B_ov"])
    assert got.shape != untruncated.shape
    # And the value is the truncated contraction rather than a truncated view of
    # the untruncated one, which a half-done redirect would also produce.
    U = np.asarray(program["pass"].transformation())
    assert not np.allclose(got, untruncated[:, :got.shape[1], :, :got.shape[3]], atol=1e-8)
    assert np.allclose(got, np.einsum("iajb,ay,bz->iyjz", untruncated, U, U), rtol=1e-10, atol=1e-12)


def test_the_energy_vector_is_replaced_rather_than_projected(water):
    """A rank-1 tensor over the space being replaced is a different question.

    Projecting the orbital energies gives the diagonal of the Fock block in the
    natural-orbital basis, where a denominator wants the SEMICANONICAL
    eigenvalues; they are different numbers and only one of them makes a
    denominator right. So the pass points the algebra at the vector its own setup
    produces, and checks the vector it is replacing against the Fock block it
    holds before believing it is the energies at all.
    """
    nocc, nvir = water["nocc"], water["nvir"]
    weights = _tensor("X_ov", water["B_ov"][0])
    energies = _tensor("eps_vir", water["eps_vir"])
    out = einsums.create_zero_tensor("weighted", [nocc])

    graph = cg.Graph("fno_energies")
    with cg.capture(graph):
        einsums.einsum("i,a ; a -> i", out, weights, energies)
    cg.annotate(weights, ("occ", "vir"), graph=graph)
    cg.annotate(energies, ("vir",), graph=graph)

    program = _truncate({"graph": graph, "integrals": out}, water, 1e-3)
    assert program["fired"], dict(program["pass"].skip_reasons)
    assert list(program["pass"].projected) == ["X_ov@fno"]

    graph.apply(cg.default_pass_manager())
    graph.execute()
    U = np.asarray(program["pass"].transformation())
    semicanonical = np.asarray(program["pass"].energies())
    assert np.allclose(np.asarray(out), (water["B_ov"][0] @ U) @ semicanonical, rtol=1e-10, atol=1e-12)
    # Not the leading entries of the canonical energies, which is what projecting
    # rather than replacing would have produced.
    assert not np.allclose(semicanonical, water["eps_vir"][:len(semicanonical)], atol=1e-6)


# ──────────────────────────────────────────────────────────────────────────
# The proving ground
# ──────────────────────────────────────────────────────────────────────────


def test_the_occupation_sweep_reports_the_count_the_error_and_the_record(water):
    """The water table: the virtual count, the energy error, and what the record says.

    The uncorrected error is the truncation's whole cost and falls as the cutoff
    tightens. The corrected one does not fall with it, and that is a property of
    the proving ground rather than a defect: the record carries
    ``E_MP2(full) - E_MP2(fno)`` measured exactly, so adding it back to an MP2
    energy in the truncated space returns the untruncated MP2 energy to rounding
    at every cutoff. What the sweep says is that the record is exact for the
    method it was measured on, which is what makes it worth adding to a
    correlated energy the truncation was not measured on.
    """
    full = _mp2(water["ovov"], water["eps_occ"], water["eps_vir"])
    assert full == pytest.approx(water["reference"], abs=1e-9)

    rows = []
    for cutoff in (1e-2, 1e-3, 1e-4):
        program = _truncate(_ladder(water, f"fno_sweep_{cutoff}"), water, cutoff)
        assert program["fired"], dict(program["pass"].skip_reasons)
        integrals, energies = _replay(program)

        truncated = _mp2(integrals, water["eps_occ"], energies)
        correction = program["pass"].correction
        rows.append((cutoff, program["pass"].kept, abs(truncated - full), abs(truncated + correction - full)))

        records = program["graph"].approximations()
        assert [r.pass_name for r in records] == ["BasisTruncation"]
        assert records[0].origin == _G.ApproximationOrigin.Measured
        assert records[0].effect == _G.ApproximationEffect.EnergyLike
        assert records[0].tolerance == pytest.approx(cutoff)
        assert sorted(records[0].spaces) == ["fno", "vir"]
        assert records[0].setup == "BasisTruncation(vir)/setup"
        assert records[0].bound == pytest.approx(abs(correction), rel=1e-12)

    counts = [kept for _c, kept, _u, _r in rows]
    assert counts == sorted(counts) and counts[0] < counts[-1] < water["nvir"], rows
    assert counts == [4, 9, 17], rows

    uncorrected = [error for _c, _k, error, _r in rows]
    assert uncorrected == sorted(uncorrected, reverse=True), rows
    for cutoff, _kept, before, after in rows:
        assert after < before, f"cutoff {cutoff:g}: the correction did not help"
        assert after < 1e-12, f"cutoff {cutoff:g}: the correction left {after:.3e} behind"

    # The number the design records, on this molecule at this cutoff.
    assert rows[1][3] < 1e-12
    assert [row for row in rows if row[0] == 1e-3][0][2] == pytest.approx(2.0920e-2, rel=1e-3)


def test_the_projected_graph_saves_loads_rebinds_and_replays(water, tmp_path):
    """A file, a fresh set of buffers, and the same numbers on the other side.

    Saved BEFORE the default manager runs, because a ``Materialize`` node holds an
    allocating closure and allocation is a resource decision this design re-derives
    on load. Three things the loaded graph asks for are worth naming: the
    untruncated tensor arrives under the projection's own name, the rectangular
    selection matrix arrives as an interface tensor because taking leading columns
    is spelled as a contraction against an identity, and the amplitudes and Fock
    block arrive because the setup rebuilds the space from them on every bind.
    """
    program = _truncate(_ladder(water, "fno_roundtrip"), water, 1e-3)
    assert program["fired"]
    assert program["graph"].serializability_report() == []
    kept = program["pass"].kept

    path = str(tmp_path / "fno.eig")
    cg.save_graph(program["graph"], path)
    in_process, energies = _replay(program)

    loaded = cg.load_graph(path)
    records = loaded.approximations()
    assert [r.pass_name for r in records] == ["BasisTruncation"]
    assert records[0].origin == _G.ApproximationOrigin.Measured

    names = set(loaded.manifest_names())
    assert {"B_ov@fno_in", "fno_keep", "fno_transformation", "fno_energies", "t2", "F_vv"} <= names, sorted(names)

    nvir = water["nvir"]
    identity = np.zeros((nvir, kept))
    identity[np.arange(kept), np.arange(kept)] = 1.0
    fresh = {
        "B_ov": _tensor("B_ov", water["B_ov"]),
        "B_ov@fno_in": _tensor("B_ov", water["B_ov"]),
        "t2": _tensor("t2", water["amplitudes"]),
        "F_vv": _tensor("F_vv", water["fock_vv"]),
        "fno_keep": _tensor("fno_keep", identity),
        "fno_transformation": einsums.create_zero_tensor("fno_transformation", [nvir, kept]),
        "fno_energies": einsums.create_zero_tensor("fno_energies", [kept]),
    }
    cg.bind(loaded, {name: tensor for name, tensor in fresh.items() if name in names})
    loaded.apply(cg.default_pass_manager())
    loaded.execute()

    # REBUILT rather than replayed from stored factors: the density was
    # diagonalized again on the bound amplitudes and the transformation written.
    replayed_U = np.asarray(fresh["fno_transformation"])
    assert np.allclose(replayed_U.T @ replayed_U, np.eye(kept), atol=1e-12)
    assert np.allclose(np.asarray(fresh["fno_energies"]), energies, rtol=1e-10, atol=1e-13)

    projected = np.einsum("Qia,ay->Qiy", water["B_ov"], replayed_U)
    gap = np.linalg.norm(np.einsum("Qiy,Qjz->iyjz", projected, projected) - in_process)
    assert gap / np.linalg.norm(in_process) <= 1e-10, f"norm-relative gap {gap:.3e}"


# ──────────────────────────────────────────────────────────────────────────
# What it declines
# ──────────────────────────────────────────────────────────────────────────


def test_composition_with_a_grid_fit_on_the_same_space_is_declined(water):
    """A collocation matrix is over the UNTRUNCATED basis, and this pass says so.

    Whether a grid fitted to the full virtual space says anything about the
    truncated one is a separate question, and answering it by projecting the
    collocation matrix would be a guess. The decline is reported through the tally
    rather than left as a wrong number.
    """
    _G.ThcFactorization.register_grid_space(cg.Graph("fno_grid_family"))
    nocc, nvir = water["nocc"], water["nvir"]
    npoints = 11
    collocation = _tensor("X_vP", np.zeros((nvir, npoints)))
    weights = _tensor("W_ov", water["B_ov"][0])
    out = einsums.create_zero_tensor("grid_out", [nocc, npoints])

    graph = cg.Graph("fno_grid")
    with cg.capture(graph):
        einsums.einsum("i,a ; a,P -> i,P", out, weights, collocation)
    cg.annotate(weights, ("occ", "vir"), graph=graph)
    cg.annotate(collocation, ("vir", _G.ThcFactorization.grid_space_name()), graph=graph)

    program = _truncate({"graph": graph, "integrals": out}, water, 1e-3)
    assert not program["fired"]
    reasons = dict(program["pass"].skip_reasons)
    assert any("projecting a collocation matrix" in reason for reason in reasons), reasons


def test_a_tensor_the_graph_writes_over_that_space_is_declined(water):
    """An output's extents belong to the caller, and shrinking one is not this pass's call."""
    nocc, nvir = water["nocc"], water["nvir"]
    three = _tensor("B_ov", water["B_ov"])
    integrals = einsums.create_zero_tensor("K_caller", [nocc, nvir, nocc, nvir])

    graph = cg.Graph("fno_written")
    with cg.capture(graph):
        einsums.einsum("Q,i,a ; Q,j,b -> i,a,j,b", integrals, three, three)
    cg.annotate(three, ("aux", "occ", "vir"), graph=graph)
    cg.annotate(integrals, ("occ", "vir", "occ", "vir"), graph=graph)

    program = _truncate({"graph": graph, "integrals": integrals}, water, 1e-3)
    assert not program["fired"]
    reasons = dict(program["pass"].skip_reasons)
    assert any("belong to the caller" in reason for reason in reasons), reasons


def test_a_rank_one_tensor_that_is_not_the_orbital_energies_is_declined(water):
    """Recognized by its VALUES, because what else a vector over that space means is a guess."""
    nocc, nvir = water["nocc"], water["nvir"]
    weights = _tensor("X_ov", water["B_ov"][0])
    occupations = _tensor("something_else", np.ones(nvir))
    out = einsums.create_zero_tensor("weighted", [nocc])

    graph = cg.Graph("fno_wrong_vector")
    with cg.capture(graph):
        einsums.einsum("i,a ; a -> i", out, weights, occupations)
    cg.annotate(weights, ("occ", "vir"), graph=graph)
    cg.annotate(occupations, ("vir",), graph=graph)

    program = _truncate({"graph": graph, "integrals": out}, water, 1e-3)
    assert not program["fired"]
    reasons = dict(program["pass"].skip_reasons)
    assert any("not the orbital energies the Fock block carries" in reason for reason in reasons), reasons


def test_a_projection_that_cannot_measure_its_own_effect_is_declined(water):
    """An unmeasured bound is not the record this rewrite owes.

    Building the space asks for neither the occupied energies nor a measurement,
    which is why a graph with nothing to project still truncates; projecting is
    the lossy act, and a lossy act with no measured effect is declined.
    """
    program = _ladder(water, "fno_unmeasured")
    truncation = _G.BasisTruncation()
    truncation.set_amplitudes(_tensor("t2", water["amplitudes"]))
    truncation.set_fock(_tensor("F_vv", water["fock_vv"]))
    truncation.set_occupation(1e-3)
    manager = cg.PassManager()
    manager.add(truncation)

    assert not program["graph"].apply(manager)
    reasons = dict(truncation.skip_reasons)
    assert any("cannot be measured without the occupied orbital energies" in reason for reason in reasons), reasons


def test_a_graph_with_nothing_over_that_space_still_builds_the_space(water):
    """The setup ships on its own, which is what makes it checkable on its own."""
    graph = cg.Graph("fno_setup_only")
    program = _truncate({"graph": graph, "integrals": None}, water, 1e-3)
    assert program["fired"], dict(program["pass"].skip_reasons)
    assert list(program["pass"].projected) == []
    assert program["pass"].correction == 0.0
    assert graph.approximations() == [], "a truncation that projected nothing has changed no arithmetic"
