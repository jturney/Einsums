# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""``BasisTruncation``, the first pass that replaces a SPACE rather than an expression.

Every other lossy pass rewrites arithmetic. This one truncates the virtual index
to the natural orbitals a first-order density says carry occupation, and the
method then runs unchanged over shorter axes.

This file pins the SETUP, which is the half that decides what the truncated
space is: how many orbitals survive a cutoff, that the transformation it builds
is an isometry, that the orbital energies it hands back are the eigenvalues of
the truncated Fock block in ascending order, and that a correlated energy
computed in the space it produces is the one numpy computes in the same space.
The projection of the caller's tensors through it is pinned beside it, in
``test_fno_projection_python``.

The water numbers, cc-pVDZ with 5 occupied and 19 virtual orbitals: the natural
occupations run from 2.3e-2 down to 4.7e-5, so a cutoff of 1e-3 keeps 9 and one
of 1e-4 keeps 17. There is no cutoff that halves this molecule's virtual space
without taking a large part of its correlation energy with it, which is a
property of a small basis rather than of the method.
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
    """The family declared once, on the process-global registry.

    A load resolves space names against the process-global registry, so a graph
    annotated against a private one cannot be read back; and the truncated
    space's typical extent is derived from the space it sits inside, which is a
    declaration that must be made once rather than per program.
    """
    registry = cg.global_space_registry()
    for name, symbol, extent, dim in _FAMILY:
        registry.register_space(cg.index_space(name, symbol, extent, cg.GrowthClass.linear(), dim))
    graph = cg.Graph("fno_family")
    _G.BasisTruncation.register_fno_space(graph)
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
    nvir = nbf - nocc

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
        "nocc": nocc, "nvir": nvir, "naux": three.shape[0],
        "B_ov": B_ov,
        "B_vv": np.ascontiguousarray(three[:, v, v]),
        "oooo": np.ascontiguousarray(np.einsum("Ppq,Prs->pqrs", three, three)[o, o, o, o]),
        "ovov": ovov,
        "amplitudes": np.ascontiguousarray(ovov / gaps),
        "fock_vv": np.ascontiguousarray(np.diag(energies[v])),
        "eps_occ": np.ascontiguousarray(energies[o]),
        "eps_vir": np.ascontiguousarray(energies[v]),
        "reference": float(z["energy_psi4_df_mp2"]),
    }


def numpy_natural_orbitals(problem):
    """The same construction in numpy, largest occupation first.

    Written out here because every assertion below is against it: what the pass
    computes has to be the quantity a reader can check by hand, not merely a
    number that stays the same between two runs of the pass.
    """
    t = problem["amplitudes"]
    combined = 2.0 * t - t.transpose(0, 3, 2, 1)
    raw = np.einsum("iajc,ibjc->ab", t, combined)
    density = raw + raw.T
    occupations, vectors = np.linalg.eigh(density)
    return occupations[::-1], vectors[:, ::-1]


def numpy_truncation(problem, kept):
    """The transformation and the energies the pass should produce for @p kept."""
    _occupations, vectors = numpy_natural_orbitals(problem)
    natural = vectors[:, :kept]
    block = natural.T @ problem["fock_vv"] @ natural
    energies, rotation = np.linalg.eigh(block)
    return natural @ rotation, energies


def _mp2(ovov, eps_occ, eps_vir):
    gaps = (eps_occ[:, None, None, None] - eps_vir[None, :, None, None]
            + eps_occ[None, None, :, None] - eps_vir[None, None, None, :])
    amplitudes = ovov / gaps
    return float(np.einsum("iajb,iajb->", 2.0 * ovov - ovov.transpose(0, 3, 2, 1), amplitudes))


def truncate(problem, occupation, name="fno_setup"):
    """Run the pass over an otherwise empty graph and execute its setup."""
    amplitudes = _tensor("t2", problem["amplitudes"])
    fock = _tensor("F_vv", problem["fock_vv"])
    graph = cg.Graph(name)

    truncation = _G.BasisTruncation()
    truncation.set_amplitudes(amplitudes)
    truncation.set_fock(fock)
    truncation.set_occupation(occupation)
    manager = cg.PassManager()
    manager.add(truncation)
    fired = graph.apply(manager)
    return {"graph": graph, "pass": truncation, "fired": fired,
            "amplitudes": amplitudes, "fock": fock}


def run_setup(state):
    """Execute the graph so the setup body computes the transformation."""
    state["graph"].apply(cg.default_pass_manager())
    state["graph"].execute()
    return (np.array(np.asarray(state["pass"].transformation()), copy=True),
            np.array(np.asarray(state["pass"].energies()), copy=True))


# ──────────────────────────────────────────────────────────────────────────
# What the setup builds
# ──────────────────────────────────────────────────────────────────────────


def test_the_occupations_are_the_eigenvalues_of_the_density_numpy_computes(water):
    """The pass and the reader see the same density, largest occupation first."""
    state = truncate(water, 1e-3)
    assert state["fired"], dict(state["pass"].skip_reasons)
    occupations, _vectors = numpy_natural_orbitals(water)
    assert np.allclose(np.array(state["pass"].occupations), occupations, rtol=1e-10, atol=1e-14)


def test_the_kept_count_is_the_orbitals_above_the_cutoff(water):
    """The count against the cutoff, over a sweep, and it only ever grows."""
    occupations, _vectors = numpy_natural_orbitals(water)
    counts = []
    for cutoff in (1e-2, 5e-3, 1e-3, 5e-4, 1e-4):
        state = truncate(water, cutoff, name=f"fno_count_{cutoff}")
        assert state["fired"], dict(state["pass"].skip_reasons)
        assert state["pass"].kept == int((occupations > cutoff).sum())
        counts.append(state["pass"].kept)
    assert counts == sorted(counts), counts
    assert counts[0] < counts[-1] < water["nvir"]


def test_the_transformation_is_an_isometry_into_the_truncated_space(water):
    """``U^T U`` is the identity of the truncated space, and ``U`` is not square.

    The two rotations the setup composes are each orthogonal, so their product
    over the kept columns has orthonormal columns and nothing else: it is an
    isometry rather than a rotation, which is what a truncation is.
    """
    state = truncate(water, 1e-3)
    assert state["fired"]
    U, _energies = run_setup(state)

    kept = state["pass"].kept
    assert U.shape == (water["nvir"], kept)
    assert kept < water["nvir"]
    assert np.allclose(U.T @ U, np.eye(kept), atol=1e-12)
    # And NOT a rotation of the full space: the projector it induces is not the
    # identity, which is the whole of what "truncated" means.
    assert not np.allclose(U @ U.T, np.eye(water["nvir"]), atol=1e-8)


def test_the_energies_are_the_truncated_fock_block_in_ascending_order(water):
    """A semicanonical basis, which is what makes a denominator diagonal again.

    The kept natural orbitals are not canonical, so their Fock block is not
    diagonal and a denominator built from the old orbital energies would be
    wrong. The second diagonalization is what fixes that, and what it hands back
    is the eigenvalues of the truncated block, ascending.
    """
    state = truncate(water, 1e-3)
    assert state["fired"]
    U, energies = run_setup(state)

    assert list(energies) == sorted(energies)
    block = U.T @ water["fock_vv"] @ U
    assert np.allclose(block, np.diag(energies), atol=1e-11), "the transformed Fock block is not diagonal"

    _expected_U, expected = numpy_truncation(water, state["pass"].kept)
    assert np.allclose(energies, expected, rtol=1e-10, atol=1e-13)
    # The truncated space is not a slice of the canonical one: its energies are
    # not a subset of the original ones, which is what semicanonicalization does.
    assert not np.allclose(energies, water["eps_vir"][:len(energies)], atol=1e-6)


def test_the_correlation_energy_in_the_truncated_space_is_the_one_numpy_computes(water):
    """The setup's whole purpose, measured on the quantity a method would use.

    The integrals are projected through the transformation the setup built and
    the MP2 energy is taken in the truncated space, against numpy doing the same
    from its own eigendecomposition. What is compared is the SPACE rather than
    the energy: the two must agree to rounding, and the gap to the untruncated
    energy is what the truncation costs.
    """
    state = truncate(water, 1e-3)
    assert state["fired"]
    U, energies = run_setup(state)

    projected = np.einsum("Qia,ab->Qib", water["B_ov"], U)
    ovov = np.einsum("Qib,Qjc->ibjc", projected, projected)
    truncated = _mp2(ovov, water["eps_occ"], energies)

    expected_U, expected_energies = numpy_truncation(water, state["pass"].kept)
    expected_projected = np.einsum("Qia,ab->Qib", water["B_ov"], expected_U)
    expected_ovov = np.einsum("Qib,Qjc->ibjc", expected_projected, expected_projected)
    oracle = _mp2(expected_ovov, water["eps_occ"], expected_energies)

    assert truncated == pytest.approx(oracle, rel=1e-9)
    # And it is a real truncation of a real quantity: the full-space energy is
    # psi4's own DF-MP2 and the truncated one is meaningfully below it.
    full = _mp2(water["ovov"], water["eps_occ"], water["eps_vir"])
    assert full == pytest.approx(water["reference"], abs=1e-9)
    assert abs(truncated - full) > 1e-3


# ──────────────────────────────────────────────────────────────────────────
# What it declines
# ──────────────────────────────────────────────────────────────────────────


def test_a_cutoff_that_drops_nothing_is_declined(water):
    """A truncation that truncates nothing is a rotation, and says so."""
    state = truncate(water, 1e-9, name="fno_no_drop")
    assert not state["fired"]
    reasons = dict(state["pass"].skip_reasons)
    assert any("rotation rather than a truncation" in reason for reason in reasons), reasons


def test_a_cutoff_that_keeps_nothing_is_declined(water):
    """And so is one that takes the whole space away."""
    state = truncate(water, 1.0, name="fno_no_keep")
    assert not state["fired"]
    reasons = dict(state["pass"].skip_reasons)
    assert any("keeps no natural orbital" in reason for reason in reasons), reasons


def test_a_pass_handed_nothing_declines_rather_than_guessing(water):
    """Amplitudes and a Fock block are what the pass has instead of a graph to read."""
    graph = cg.Graph("fno_empty")
    truncation = _G.BasisTruncation()
    manager = cg.PassManager()
    manager.add(truncation)
    assert not graph.apply(manager)
    reasons = dict(truncation.skip_reasons)
    assert any("no amplitudes or no Fock block" in reason for reason in reasons), reasons

    # A Fock block whose extent disagrees with the amplitudes is a decline too,
    # rather than a truncation of whichever one happened to be read first.
    truncation.set_amplitudes(_tensor("t2", water["amplitudes"]))
    truncation.set_fock(_tensor("F_bad", np.eye(water["nvir"] + 1)))
    graph2 = cg.Graph("fno_mismatch")
    manager2 = cg.PassManager()
    manager2.add(truncation)
    assert not graph2.apply(manager2)
    assert any("the density could not be formed" in reason for reason in dict(truncation.skip_reasons))


def test_amplitudes_of_the_wrong_shape_are_refused_at_the_call(water):
    """A rank-4 tensor whose paired axes disagree is not ``t[i,a,j,b]``."""
    truncation = _G.BasisTruncation()
    with pytest.raises(ValueError):
        truncation.set_amplitudes(_tensor("bad", np.zeros((2, 3, 2, 4))))
    with pytest.raises(ValueError):
        truncation.set_fock(_tensor("bad", np.zeros((3, 4))))
    with pytest.raises(ValueError):
        truncation.set_occupation(-1e-12)


def test_the_truncated_space_is_declared_inside_the_virtual_one(water):
    """Containment and scale order, the second client of the relation."""
    graph = cg.Graph("fno_containment")
    registry = graph.space_registry
    truncated, full = registry.find("fno"), registry.find("vir")
    assert registry.is_contained(truncated, full) == cg.Tristate.Yes
    assert registry.is_less(truncated, full) == cg.Tristate.Yes
    assert registry.space(truncated).dim_symbol == "nfno"
    assert registry.space(truncated).typical_extent == pytest.approx(0.5 * registry.space(full).typical_extent)
