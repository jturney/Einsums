# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""ThcFactorization on a real molecule's integrals, offline.

``examples/psi4-bridge/df_mp2_laplace.py --thc`` is the same validation driven
from psi4. This runs it from the DLPNO fixture's own buffers, which already
carry a psi4 Becke grid: ``grid_phi_flat``, ``grid_w_flat`` and the per-block
basis maps beside it are the collocation matrix this needs, so no new fixture
had to be written.

Three things are checked, and the first is what makes the other two mean
anything.

The SHIPPED provider is run on this molecule, through ``FactorizationPass``, on
the contraction ``C[m,n] = sum_pq (mn|pq) T[p,q]``. What comes out is compared
against the same contraction over the exact density-fitted integral, so the
error being measured is the grid fit's own on a real basis rather than a
synthetic one. The fit is also written out here in numpy, and the two agree,
which is what lets the energy arms below use the written-out one: they need the
factors as buffers and the pass keeps them inside a graph.

Then the four-index residual, ``(ia|jb)`` through the grid against the same
integral through the density fit, and the MP2 correlation energy through three
arms: density fitting alone, the grid fit on top of it, and the grid fit with
the Laplace quadrature on the denominator. The exact arm reproduces psi4's own
DF-MP2, so what the others are compared against is the right quantity.

The grid is PRUNED, and that is a setup the caller performs rather than
something the pass decides: the pass takes the collocation matrix it is handed.
The selection here is the pivoted one interpolative separable density fitting
uses, which is the cheapest thing that answers the only question that matters,
which points carry independent basis-pair products.
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


def _tensor(name, array):
    tensor = einsums.create_zero_tensor(name, list(array.shape))
    np.asarray(tensor)[...] = np.ascontiguousarray(array)
    return tensor


def _select_points(pair, count):
    """Pivoted selection of the points whose basis-pair products are independent.

    Modified Gram-Schmidt with column pivoting over ``pair[mn, P]``: take the
    column of largest remaining norm, project it out, repeat. It stops early
    when the remaining norms collapse, which is the grid saying it has no more
    independent directions to give and is the honest answer to "how many points
    are enough".
    """
    remaining = pair.copy()
    pivots = []
    for _ in range(count):
        norms = np.einsum("ij,ij->j", remaining, remaining)
        best = int(np.argmax(norms))
        if norms[best] <= 1e-20:
            break
        pivots.append(best)
        direction = remaining[:, best] / np.sqrt(norms[best])
        remaining -= np.outer(direction, direction @ remaining)
    return np.array(pivots, dtype=int)


@pytest.fixture(scope="module")
def water():
    """Canonical orbitals, the fitted three-index tensor, and a pruned grid."""
    if not os.path.exists(_FIXTURE):
        pytest.skip(f"fixture not present: {_FIXTURE}")
    z = np.load(_FIXTURE, allow_pickle=False)
    if not bool(z["has_grid"]):
        pytest.skip("the fixture carries no grid")

    overlap, fock = z["S"], z["F"]
    nbf = overlap.shape[0]
    values, vectors = np.linalg.eigh(overlap)
    orthogonalizer = vectors @ np.diag(values ** -0.5) @ vectors.T
    energies, rotation = np.linalg.eigh(orthogonalizer.T @ fock @ orthogonalizer)
    coefficients = orthogonalizer @ rotation

    nocc = int(z["C_occ"].shape[1])

    # The density fit, in the molecular-orbital basis. This is what the grid fit
    # is fitted FROM and what it is measured against.
    metric = z["metric"]
    mvals, mvecs = np.linalg.eigh(metric)
    inv_sqrt = mvecs @ np.diag(np.where(mvals > 1e-10, mvals ** -0.5, 0.0)) @ mvecs.T
    three_ao = np.einsum("PQ,Qmn->Pmn", inv_sqrt, z["eri_3index"])
    three = np.einsum("Pmn,mp,nq->Ppq", three_ao, coefficients, coefficients)

    # The Becke grid, unpacked from the fixture's per-block buffers into one
    # dense collocation matrix and rotated into the same orbital basis.
    phi_flat, weights = z["grid_phi_flat"], z["grid_w_flat"]
    bfmap, npoints, nbfs = z["grid_bfmap_flat"], z["grid_npoints"], z["grid_nbf"]
    blocks, offset, mapped = [], 0, 0
    for count, width in zip(npoints, nbfs):
        count, width = int(count), int(width)
        block = phi_flat[offset:offset + count * width].reshape(count, width)
        columns = bfmap[mapped:mapped + width].astype(int)
        full = np.zeros((count, nbf))
        full[:, columns] = block
        blocks.append(full)
        offset += count * width
        mapped += width
    collocation_ao = np.concatenate(blocks, axis=0).T
    # The quadrature weight, split four ways, because four collocation factors
    # meet in the integral this fits.
    collocation = (coefficients.T @ collocation_ao) * (np.abs(weights) ** 0.25)

    pairs = np.triu_indices(nbf)
    pivots = _select_points(collocation[pairs[0], :] * collocation[pairs[1], :], nbf * (nbf + 1) // 2)
    grid = np.ascontiguousarray(collocation[:, pivots])

    return {
        "nbf": nbf, "nocc": nocc, "nvir": nbf - nocc, "ngrid": grid.shape[1],
        "raw_grid_points": int(collocation_ao.shape[1]),
        "orbital_energies": energies, "three": three, "grid": grid,
        "reference": float(z["energy_psi4_df_mp2"]),
    }


#: The drop threshold this grid needs, and NOT the provider's default.
#:
#: ``S`` here runs from ``4.9e-14`` to ``2.8`` with no gap: an absolute cutoff
#: at the default ``1e-10`` keeps 268 of 280 directions, and the ones between
#: ``1e-10`` and ``1e-8`` are near-null directions whose inverse amplifies
#: rounding into the fit. At ``1e-8`` the fit is at its most accurate on this
#: grid and two implementations of the same formula agree to five digits; at
#: ``1e-10`` they do not agree at all, because the answer is then decided by
#: which side of the cutoff an eigenvalue happens to land on. A Coulomb metric
#: does not behave this way, which is why the default is what it is.
_DROP = 1e-8


def _numpy_fit(water_problem, threshold=_DROP):
    """The provider's own algebra, written out.

    Used by the energy arms because they need the factors as buffers and the
    pass keeps them inside a graph; validated against the pass in
    ``test_the_shipped_provider_fits_this_molecule``, which is what makes it a
    reference rather than a second implementation nobody checks.
    """
    grid, three = water_problem["grid"], water_problem["three"]
    gram = grid.T @ grid
    metric = gram * gram
    values, vectors = np.linalg.eigh(metric)
    inv_sqrt = vectors @ np.diag(np.where(values > threshold, values ** -0.5, 0.0)) @ vectors.T
    inverse = inv_sqrt @ inv_sqrt
    projected = np.einsum("Amn,mP,nP->AP", three, grid, grid)
    return inverse @ (projected.T @ projected) @ inverse, inverse, projected


def _dense(water_problem):
    three = water_problem["three"]
    return np.einsum("Amn,Apq->mnpq", three, three)


def _thc_dense(water_problem):
    coupling, _inverse, _projected = _numpy_fit(water_problem)
    grid = water_problem["grid"]
    left = np.einsum("mP,nP,PQ->Qmn", grid, grid, coupling)
    right = np.einsum("pQ,qQ->Qpq", grid, grid)
    return np.einsum("Qmn,Qpq->mnpq", left, right), left, right


def test_the_shipped_provider_fits_this_molecule(water):
    """FactorizationPass with ThcFactorization, on this molecule's own integrals.

    The contraction the fit is measured through is an ordinary one,
    ``C[m,n] = sum_pq (mn|pq) T[p,q]``, so what is compared is the grid fit's
    error on a real basis rather than the machinery reproducing itself.
    """
    dense = _dense(water)
    operand = np.random.default_rng(20260904).standard_normal((water["nbf"], water["nbf"]))
    exact = np.einsum("mnpq,pq->mn", dense, operand)

    M = _tensor("(mn|pq)", dense)
    T = _tensor("T", operand)
    C = einsums.zeros((water["nbf"], water["nbf"]), dtype="float64")

    graph = cg.Graph("thc_water")
    with cg.capture(graph):
        einsums.einsum("m,n,p,q ; p,q -> m,n", C, M, T)
    graph.annotate_tag(M, _G.ProvenanceTag.make("eri"))
    graph.annotate_dims(M, ["nbf"] * 4)
    _G.ThcFactorization.register_grid_space(graph)

    registry = _G.FactorizationRegistry()
    registry.add(_G.ThcFactorization(
        "eri", _tensor("B", water["three"]), _tensor("X", water["grid"]), 1e-2, _DROP))
    factorization = _G.FactorizationPass(registry)
    manager = cg.PassManager()
    manager.add(factorization)

    captured = graph.num_nodes()
    assert graph.apply(manager), f"the grid fit declined: {factorization.skip_reasons}"
    assert factorization.num_factorized == 1
    assert [record.pass_name for record in graph.approximations()] == ["Thc"]
    assert graph.num_nodes() > captured

    graph.apply(cg.default_pass_manager())
    graph.execute()

    # The pass's fit and the written-out one agree, which is what lets the energy
    # arms below use the written-out one.
    thc_dense, _left, _right = _thc_dense(water)
    written_out = np.einsum("mnpq,pq->mn", thc_dense, operand)
    scale = float(np.linalg.norm(exact))
    # Five digits rather than machine precision, and the reason is the grid
    # rather than the code: the least-squares metric is near-singular, so the
    # last few kept directions carry an inverse large enough to turn a different
    # order of the same arithmetic into a visible difference. See _DROP.
    assert float(np.linalg.norm(np.asarray(C) - written_out)) <= 1e-4 * scale, (
        "the shipped provider and the same algebra written out disagree")

    # And the fit's own error, on a real molecule, inside the bound the caller
    # asserted for it.
    assert float(np.linalg.norm(np.asarray(C) - exact)) <= 1e-2 * scale


def test_the_grid_fit_reproduces_the_density_fitted_integrals(water):
    """The four-index residual, which is the quantity the record is about."""
    nocc = water["nocc"]
    dense = _dense(water)[:nocc, nocc:, :nocc, nocc:]
    thc = _thc_dense(water)[0][:nocc, nocc:, :nocc, nocc:]

    relative = float(np.linalg.norm(thc - dense) / np.linalg.norm(dense))
    largest = float(np.max(np.abs(thc - dense)))
    # Measured on water/cc-pVDZ over a 2625-point Becke grid pruned to its 280
    # independent pair directions. Stated as a ceiling rather than a target: a
    # denser grid moves it down and nothing here should move it up.
    assert relative < 1e-2, f"relative residual {relative:.3e}"
    assert largest < 1e-2, f"largest deviation {largest:.3e}"


def _mp2_arms(water, epsilon=None):
    """The correlation energy through the grid fit, optionally with a quadrature.

    The grid form is the same shape the density-fitted one is: the chain
    contracts to ``sum_Q L[Q,i,a] R[Q,j,b]`` with the grid index in the place the
    auxiliary index had, so the transform rides on it unchanged.
    """
    nocc, nvir = water["nocc"], water["nvir"]
    _thc, left, right = _thc_dense(water)
    left = np.ascontiguousarray(left[:, :nocc, nocc:])
    right = np.ascontiguousarray(right[:, :nocc, nocc:])
    shape = [nocc, nvir, nocc, nvir]

    energies = water["orbital_energies"]
    occupied = _tensor("eps_occ", energies[:nocc])
    virtual = _tensor("eps_vir", energies[nocc:])

    denominator = einsums.create_zero_tensor("D", shape)
    la.outer_sum(denominator, [occupied, virtual, occupied, virtual], [1.0, -1.0, 1.0, -1.0])
    la.element_transform(denominator, lambda x: 1.0 / x)

    L = _tensor("L", left)
    R = _tensor("R", right)
    energy = einsums.create_zero_tensor("E_corr", [1])

    graph = cg.Graph(f"thc_mp2_{epsilon}")
    K = graph.scratch("K", shape, "float64")
    T = graph.scratch("T", shape, "float64")
    again = graph.scratch("K_again", shape, "float64")
    exchange = graph.scratch("K_exchange", shape, "float64")
    combination = graph.scratch("Kbar", shape, "float64")
    with cg.capture(graph):
        einsums.einsum("Q,i,a ; Q,j,b -> i,a,j,b", K, L, R)
        la.direct_product(1.0, K, denominator, 0.0, T)
        einsums.einsum("Q,i,a ; Q,j,b -> i,a,j,b", again, L, R)
        einsums.permute("iajb <- ibja", exchange, again)
        la.axpby(2.0, again, 0.0, combination)
        la.axpby(-1.0, exchange, 1.0, combination)
        la.dot(energy, combination, T)

    captured = graph.num_nodes()
    transform = None
    if epsilon is not None:
        graph.annotate_tag(denominator, _G.LaplaceTransform.denominator_tag(
            ["eps_occ", "eps_vir", "eps_occ", "eps_vir"], "+-+-"))
        transform = cg.LaplaceTransform()
        transform.set_epsilon(epsilon)
        transform.add_energy("eps_occ", occupied)
        transform.add_energy("eps_vir", virtual)
        manager = cg.PassManager()
        manager.add(transform)
        assert graph.apply(manager), f"the transform declined: {transform.skip_reasons}"

    graph.apply(cg.default_pass_manager())
    graph.execute()
    return float(np.asarray(energy)[0]), captured, graph.num_nodes(), transform


def test_the_three_arms_agree_with_the_exact_energy(water):
    """Density fitting alone, the grid fit, and the grid fit with a quadrature."""
    nocc, nvir = water["nocc"], water["nvir"]
    dense = _dense(water)[:nocc, nocc:, :nocc, nocc:]
    energies = water["orbital_energies"]
    gaps = 1.0 / (energies[:nocc, None, None, None] - energies[None, nocc:, None, None]
                  + energies[None, None, :nocc, None] - energies[None, None, None, nocc:])

    def correlation(integrals):
        exchange = integrals.transpose(0, 3, 2, 1)
        return float(np.einsum("iajb,iajb->", 2.0 * integrals - exchange, integrals * gaps))

    df_energy = correlation(dense)
    # The density-fitted arm IS psi4's own DF-MP2, so the two arms below are
    # measured against the right quantity rather than against a reimplementation.
    assert df_energy == pytest.approx(water["reference"], abs=1e-9)

    thc_energy, captured, emitted, _none = _mp2_arms(water, epsilon=None)
    assert emitted >= captured
    thc_error = abs(thc_energy - df_energy) / abs(df_energy)
    assert thc_error < 1e-3, f"grid-fit energy error {thc_error:.3e}"

    # And with the quadrature on top. The transform's own error is bounded by
    # what it recorded; the grid fit's is what it is, and the two compose.
    for epsilon in (1e-4, 1e-6):
        joint, _captured, _emitted, transform = _mp2_arms(water, epsilon=epsilon)
        assert transform.num_transformed == 1
        bound = transform.last_measured_error + thc_error + transform.last_measured_error * thc_error
        assert abs(joint - df_energy) <= (bound + 1e-12) * abs(df_energy), (
            f"epsilon={epsilon:g}: error {abs(joint - df_energy) / abs(df_energy):.3e} "
            f"against a composed bound of {bound:.3e}")
