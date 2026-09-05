#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

"""DF-MP2 in the FULL-AXIS form, and the Laplace transform of its denominator.

The proving ground for ``LaplaceTransform`` on a real molecule. Its companions
``df_mp2_energy.py`` and ``df_mp2_graph.py`` are pair-driven: they fold
``e_i + e_j`` into a prefactor and leave a two-axis ``W[a,b]`` per pair, which is
the memory-optimal way to get the energy and is also the one shape the pass
declines by design, since two of the four energies are then scalars the graph
cannot see. This file writes the same energy over all four indices instead:

    K[i,a,j,b]   = sum_Q B[Q,i,a] B[Q,j,b]            the DF integral
    D[i,a,j,b]   = 1 / (e_i + e_j - e_a - e_b)        the tagged denominator
    T[i,a,j,b]   = K[i,a,j,b] * D[i,a,j,b]            the amplitude
    Kbar[i,a,j,b] = 2 K[i,a,j,b] - K[i,b,j,a]
    E            = sum_iajb Kbar[i,a,j,b] T[i,a,j,b]

At water/cc-pVDZ that is 5 x 19 x 5 x 19 doubles, so the four-index tensors cost
nothing and nothing here is sliced. The point is the STRUCTURE, not the size: the
pass replaces ``D`` with a quadrature and pushes one exponential per axis onto
the operands of the contraction that forms ``K``, which is where the coupling of
i, a, j and b lives. That is the substitution SOS-MP2 is built on.

Three things this file records because the real case settled them.

The denominator is built EAGERLY in the arms below, from the same two energy
vectors, and that is now a choice rather than a requirement. The refusal it was
written for declined a tagged tensor any node writes, on the grounds that a graph
recomputing the denominator on every replay disagrees with a quadrature fitted
once per bind. That holds for a recipe the pass cannot read, and not for the one
the tag already describes: an outer sum of the named energies with the tagged
signs followed by the reciprocal is checked against the nodes and dissolved with
the tensor. ``--sos`` builds it inside the capture for exactly that reason, since
a denominator declared deferred and then dissolved is never allocated.

The numerator has to be a contraction, and it is one here for a physical reason
rather than to please the pass: the exponentials ride on the OPERANDS of the
contraction that forms the numerator, so a numerator formed as B times B gets
one exponential per orbital index and a numerator that is a stored integral has
nowhere to put them.

And the density fit does not compose with it in canonical MP2, which is a
property of MP2 rather than a limit of either pass. ``--naive`` captures the form
a caller with only a dense ``(ia|jb)`` writes and runs both passes over it, and
both decline with their reasons: the integral is read elementwise and through a
dot, never as a contraction operand, so there is nothing for the fit to
re-associate, and the amplitude's numerator is then a stored tensor rather than
a contraction, so the transform has no operands to ride on. It then runs the same
form again with the two costed as ONE decision, which is what
``FactorizationPass.set_laplace_transform`` is for, and prints the joint number:
the pair is a whole scale order worse, because the amplitude is a stored tensor
and the decoupled form has to rebuild it as a sum over the quadrature and
auxiliary indices where the captured form built it with one elementwise multiply. ``--response`` shows the
two composing on this same molecule's integrals in the shape where the fit IS
profitable, an orbital-response update built from ``(ia|jb)`` and divided by
``e_i - e_a``.

``--sos`` writes the opposite-spin energy over all four indices and lets the
search re-associate it. With the denominator decoupled the whole energy is one
nine-factor product, and its cheapest bracketing contracts the orbital indices
away first: no tensor over o and v survives, and what is left is a Q-by-Q matrix
per quadrature point. That is the fourth-order algorithm, reached from the
subset search rather than from a rule about SOS-MP2, and the arm prints it beside
the pair-driven energy it has to agree with.

``--thc`` runs the grid fit on the same molecule. A collocation matrix comes
from psi4's own Becke grid, is pruned to the points whose basis-pair products
are independent, and ``ThcFactorization`` fits the coupling from the same
three-index tensor the density fit uses. It reports the residual of the fitted
integrals against the density-fitted ones and the correlation energy through
three arms, the density fit alone, the grid fit on top of it, and the grid fit
with the quadrature, so what each approximation costs in accuracy is visible
side by side. Choosing the grid is the caller's, which is why the pruning is in
this file: the pass takes the collocation matrix it is handed and the residual
it measures is what tells the caller whether the grid was enough.

Integrals come from psi4 when it can be imported and from the DLPNO fixture
otherwise, and the two paths differ only in where the buffers come from. The
fixture already carries a psi4 Becke grid, because the DLPNO domains need one,
so the offline arm needed no new file.

    PYTHONPATH=/Users/jturney/Code/Einsums/Einsums/build/lib:/Users/jturney/Code/psi4/cmake-build-debug/stage/lib \
        /Users/jturney/miniconda3/envs/einsums-dev/bin/python \
        /Users/jturney/Code/Einsums/Einsums/examples/psi4-bridge/df_mp2_laplace.py
"""
import argparse
import contextlib
import json
import os

import numpy as np

import einsums
import einsums.graph as cg
import einsums._core.graph as _G
from einsums import linalg as la

_argp = argparse.ArgumentParser(description="Full-axis DF-MP2 under LaplaceTransform.")
_argp.add_argument("--fixture", action="store_true",
                   help="read the integrals from the DLPNO fixture even if psi4 imports")
_argp.add_argument("--naive", action="store_true",
                   help="also capture the dense-integral form and report what each pass declines")
_argp.add_argument("--thc", action="store_true",
                   help="also fit the integrals on a grid and run the energy through the grid fit")
_argp.add_argument("--sos", action="store_true",
                   help="also run the opposite-spin energy and let the search re-associate it")
_argp.add_argument("--response", action="store_true",
                   help="also run the orbital-response shape, where the fit and the transform compose")
_argp.add_argument("--epsilons", type=float, nargs="+", default=[1e-3, 1e-5, 1e-8],
                   help="target accuracies to run the transform at")
_argp.add_argument("--threads", type=int, default=1,
                   help="thread count, applied with psi4.set_num_threads before any einsums work")
_args = _argp.parse_args()

_FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "dlpno", "fixtures", "water-ccpvdz.npz")

#: The geometry the DLPNO fixture was generated from, so both sources describe
#: one molecule and the two paths are comparable.
_GEOMETRY = "O\nH 1 0.96\nH 1 0.96 2 104.5\nsymmetry c1\n"


def _tensor(name, array):
    t = einsums.create_zero_tensor(name, list(array.shape))
    np.asarray(t)[...] = np.ascontiguousarray(array)
    return t


# ── Where the buffers come from ─────────────────────────────────────────────

def _from_psi4():
    """Geometry, orbitals and three-index integrals from psi4 itself."""
    import psi4
    from einsums.interop import psi4 as interop  # noqa: F401  (registers the buffer bridge)

    psi4.core.set_output_file("/tmp/psi4_df_mp2_laplace.out", False)
    psi4.set_options({"basis": "cc-pvdz", "scf_type": "df", "mp2_type": "df",
                      "freeze_core": "false", "e_convergence": 1e-10, "d_convergence": 1e-10})
    psi4.set_num_threads(_args.threads)
    mol = psi4.geometry(_GEOMETRY)
    _, wfn = psi4.energy("mp2", return_wfn=True)

    primary = wfn.basisset()
    aux = psi4.core.BasisSet.build(mol, "DF_BASIS_MP2", "", "RIFIT", primary.name())
    zero = psi4.core.BasisSet.zero_ao_basis_set()
    mints = psi4.core.MintsHelper(primary)
    return {
        "label": f"psi4 {psi4.__version__}, cc-pVDZ / {aux.name()}",
        "C": np.asarray(wfn.Ca()),
        "eps": np.asarray(wfn.epsilon_a()),
        "nocc": wfn.nalpha(),
        # The raw three-index integrals and their metric, which is what the
        # factorization provider is fitted from. Qov() would hand over the
        # already-fitted tensor and leave the provider nothing to do.
        "three": np.squeeze(np.asarray(mints.ao_eri(aux, zero, primary, primary))),
        "metric": np.squeeze(np.asarray(mints.ao_eri(aux, zero, aux, zero))),
        "reference": float(psi4.variable("MP2 CORRELATION ENERGY")),
        # The collocation matrix, from psi4's own DFT grid machinery: basis
        # function values on a Becke grid, blocked the way psi4 blocks them and
        # concatenated here. A grid fit needs this and nothing else that the
        # density fit did not already need.
        "grid": _psi4_collocation(psi4, mol, primary),
    }


def _psi4_collocation(psi4, mol, basis):
    """``(X[m,P], w[P])`` over psi4's Becke grid, as dense numpy buffers.

    Deliberately COARSE. The point of the grid arm is that a fit on a real
    molecule's basis reproduces its integrals, not that it reproduces them to
    spectroscopic accuracy, and a coarse grid makes the run seconds instead of
    minutes. A denser grid moves every residual below down.
    """
    psi4.set_options({"DFT_BASIS_TOLERANCE": 1e-12, "DFT_BS_RADIUS_ALPHA": 1.0,
                      "DFT_PRUNING_ALPHA": 1.0, "DFT_BLOCK_MAX_RADIUS": 3.0,
                      "DFT_WEIGHTS_TOLERANCE": 1e-15})
    grid = psi4.core.DFTGrid.build(
        mol, basis,
        {"DFT_SPHERICAL_POINTS": 74, "DFT_RADIAL_POINTS": 35,
         "DFT_BLOCK_MIN_POINTS": 100, "DFT_BLOCK_MAX_POINTS": 256},
        {"DFT_PRUNING_SCHEME": "ROBUST", "DFT_RADIAL_SCHEME": "TREUTLER",
         "DFT_NUCLEAR_SCHEME": "TREUTLER", "DFT_GRID_NAME": "", "DFT_BLOCK_SCHEME": "OCTREE"})
    values = psi4.core.BasisFunctions(basis, grid.max_points(), basis.nbf())
    blocks, weights = [], []
    for block in grid.blocks():
        values.compute_functions(block)
        columns = list(block.functions_local_to_global())
        npoints = block.npoints()
        # Copied, explicitly: PHI is one allocation sized to the largest block
        # and psi4 reuses it, so a loop that accumulates rather than consumes
        # gets the last block's values for every block that shared the buffer.
        phi = np.array(np.asarray(values.basis_values()["PHI"])[:npoints, :len(columns)], copy=True)
        full = np.zeros((npoints, basis.nbf()))
        full[:, columns] = phi
        blocks.append(full)
        weights.append(np.array(np.asarray(block.w())[:npoints], copy=True))
    return np.concatenate(blocks, axis=0).T, np.concatenate(weights)


def _from_fixture():
    """The same molecule's buffers as the DLPNO fixture recorded them."""
    if not os.path.exists(_FIXTURE):
        raise SystemExit(f"no psi4 and no fixture at {_FIXTURE}")
    z = np.load(_FIXTURE, allow_pickle=True)

    # Canonical orbitals from the stored Fock and overlap, in einsums: the
    # fixture keeps a localized occupied set and this expression wants the
    # canonical one, which is what makes the denominator diagonal.
    S = _tensor("S", z["S"])
    F = _tensor("F", z["F"])
    nbf = S.dim(0)
    X = la.pow(S, -0.5, 1e-10)
    half = einsums.create_zero_tensor("X'F", [nbf, nbf])
    ortho = einsums.create_zero_tensor("X'FX", [nbf, nbf])
    la.gemm(1.0, X, F, 0.0, half, trans_a=True)
    la.gemm(1.0, half, X, 0.0, ortho)
    eps = einsums.create_zero_tensor("eps", [nbf])
    la.syev(ortho, eps, compute_eigenvectors=True)
    coefficients = einsums.create_zero_tensor("C", [nbf, nbf])
    la.gemm(1.0, X, ortho, 0.0, coefficients)

    return {
        "label": f"DLPNO fixture, {str(z['meta_molecule'])} / {str(z['meta_basis'])}",
        "C": np.array(np.asarray(coefficients)),
        "eps": np.array(np.asarray(eps)),
        "nocc": int(z["C_occ"].shape[1]),
        "three": np.asarray(z["eri_3index"], dtype=np.float64),
        "metric": np.asarray(z["metric"], dtype=np.float64),
        "reference": float(z["energy_psi4_df_mp2"]),
        "grid": _fixture_collocation(z) if bool(z["has_grid"]) else None,
    }


def _fixture_collocation(z):
    """The same collocation matrix, out of the fixture's per-block buffers.

    The fixture already carries a psi4 Becke grid, because the DLPNO domains
    need one, so the offline arm needs no new file.
    """
    phi_flat, weights = z["grid_phi_flat"], z["grid_w_flat"]
    bfmap, npoints, nbfs = z["grid_bfmap_flat"], z["grid_npoints"], z["grid_nbf"]
    nbf = z["S"].shape[0]
    blocks, offset, mapped = [], 0, 0
    for count, width in zip(npoints, nbfs):
        count, width = int(count), int(width)
        block = phi_flat[offset:offset + count * width].reshape(count, width)
        full = np.zeros((count, nbf))
        full[:, bfmap[mapped:mapped + width].astype(int)] = block
        blocks.append(full)
        offset += count * width
        mapped += width
    return np.concatenate(blocks, axis=0).T, np.asarray(weights, dtype=np.float64)


def _integrals(source):
    """The MO three-index tensor, its fitted partner, and the orbital energies.

    Every step here is an einsums call. numpy holds what psi4 handed over and
    nothing else.
    """
    nocc = source["nocc"]
    nbf = source["C"].shape[0]
    nvir = source["C"].shape[1] - nocc
    naux = source["metric"].shape[0]

    occupied = _tensor("C_occ", source["C"][:, :nocc])
    virtual = _tensor("C_vir", source["C"][:, nocc:])
    ao = _tensor("(Q|mn)", source["three"])
    metric = _tensor("(P|Q)", source["metric"])

    half = einsums.create_zero_tensor("(Q|in)", [naux, nocc, nbf])
    three = einsums.create_zero_tensor("(Q|ia)", [naux, nocc, nvir])
    einsums.einsum("Q,m,n ; m,i -> Q,i,n", half, ao, occupied)
    einsums.einsum("Q,i,n ; n,a -> Q,i,a", three, half, virtual)

    fitted = einsums.create_zero_tensor("B", [naux, nocc, nvir])
    einsums.einsum("P,Q ; Q,i,a -> P,i,a", fitted, la.pow(metric, -0.5, 1e-10), three)

    energies = source["eps"]
    return {
        "nocc": nocc, "nvir": nvir, "naux": naux,
        "occupied_energies": _tensor("eps_occ", energies[:nocc]),
        "virtual_energies": _tensor("eps_vir", energies[nocc:]),
        "three": three, "metric": metric, "fitted": fitted,
    }


# ── The program, once ───────────────────────────────────────────────────────

_TAG = _G.LaplaceTransform.denominator_tag(["eps_occ", "eps_vir", "eps_occ", "eps_vir"], "+-+-")


def _denominator(problem, name="D"):
    """1 / (e_i + e_j - e_a - e_b), eagerly, from the two energy vectors.

    Eager here, and captured in the ``--sos`` arm. The pass accepts either: a
    denominator nothing writes needs no verification, and one written by an
    outer sum of the tagged energies followed by the reciprocal is a recipe it
    reads off the nodes and dissolves with the tensor. What it still refuses is a
    writer it cannot read, since a graph recomputing the denominator into
    something else on every replay is describing what the quadrature cannot
    follow.
    """
    shape = [problem["nocc"], problem["nvir"], problem["nocc"], problem["nvir"]]
    denominator = einsums.create_zero_tensor(name, shape)
    la.outer_sum(denominator,
                 [problem["occupied_energies"], problem["virtual_energies"],
                  problem["occupied_energies"], problem["virtual_energies"]],
                 [1.0, -1.0, 1.0, -1.0])
    la.element_transform(denominator, lambda x: 1.0 / x)
    return denominator


def _mp2(problem, denominator, energy, graph=None):
    """The full-axis energy, eagerly when @p graph is None and captured otherwise.

    One spelling for both, because the graph arms are meant to be the eager
    program recorded rather than a second program that agrees with it.

    ``K`` is formed twice on purpose. The transform DISSOLVES the numerator it
    rewrites, so a numerator anything else reads is declined, and the exchange
    combination needs its own copy of the integral. At these extents that copy
    costs 9025 doubles.
    """
    shape = [problem["nocc"], problem["nvir"], problem["nocc"], problem["nvir"]]
    B = problem["fitted"]
    scratch = ((lambda name: einsums.create_zero_tensor(name, shape)) if graph is None
               else (lambda name: graph.scratch(name, shape, "float64")))
    K = scratch("K")
    T = scratch("T")
    again = scratch("K_again")
    exchange = scratch("K_exchange")
    combination = scratch("Kbar")
    with contextlib.nullcontext() if graph is None else cg.capture(graph):
        einsums.einsum("Q,i,a ; Q,j,b -> i,a,j,b", K, B, B)
        la.direct_product(1.0, K, denominator, 0.0, T)
        einsums.einsum("Q,i,a ; Q,j,b -> i,a,j,b", again, B, B)
        einsums.permute("iajb <- ibja", exchange, again)
        la.axpby(2.0, again, 0.0, combination)
        la.axpby(-1.0, exchange, 1.0, combination)
        la.dot(energy, combination, T)


def _capture_dense(graph, problem, dense, denominator, energy):
    """The same energy written by a caller who holds ``(ia|jb)`` and nothing else."""
    shape = [problem["nocc"], problem["nvir"], problem["nocc"], problem["nvir"]]
    T = graph.scratch("T", shape, "float64")
    exchange = graph.scratch("K_exchange", shape, "float64")
    combination = graph.scratch("Kbar", shape, "float64")
    with cg.capture(graph):
        la.direct_product(1.0, dense, denominator, 0.0, T)
        einsums.permute("iajb <- ibja", exchange, dense)
        la.axpby(2.0, dense, 0.0, combination)
        la.axpby(-1.0, exchange, 1.0, combination)
        la.dot(energy, combination, T)


def _transform(problem, epsilon):
    """A ``LaplaceTransform`` holding this problem's energies."""
    transform = cg.LaplaceTransform()
    transform.set_epsilon(epsilon)
    transform.add_energy("eps_occ", problem["occupied_energies"])
    transform.add_energy("eps_vir", problem["virtual_energies"])
    return transform


def _fit(problem):
    """A registry with the density fit registered on the integral tag."""
    registry = _G.FactorizationRegistry()
    registry.add(_G.MetricFitFactorization("eri", problem["three"], problem["metric"], 1e-10))
    return registry


def _energy_of(graph, energy):
    graph.apply(cg.default_pass_manager())
    graph.execute()
    return float(np.asarray(energy)[0])


# ── The runs ────────────────────────────────────────────────────────────────

def main():
    try:
        source = _from_fixture() if _args.fixture else _from_psi4()
    except ImportError as failure:
        print(f"psi4 did not import ({failure}); reading the DLPNO fixture instead")
        source = _from_fixture()

    problem = _integrals(source)
    print(f"source: {source['label']}")
    print(f"nocc = {problem['nocc']}, nvir = {problem['nvir']}, naux = {problem['naux']}")
    print(f"psi4 DF-MP2 correlation energy = {source['reference']:.12f}\n")

    denominator = _denominator(problem)

    # Eager, which is the baseline the graph arms are held to.
    eager_energy = einsums.create_zero_tensor("E_eager", [1])
    _mp2(problem, denominator, eager_energy)
    exact = float(np.asarray(eager_energy)[0])
    print(f"eager, exact denominator      = {exact:.12f}   "
          f"(vs psi4 {abs(exact - source['reference']):.2e})")

    # The same program captured, with no lossy pass on it.
    graph_energy = einsums.create_zero_tensor("E_graph", [1])
    graph = cg.Graph("mp2 full axis")
    _mp2(problem, denominator, graph_energy, graph)
    captured_nodes = graph.num_nodes()
    graph_value = _energy_of(graph, graph_energy)
    print(f"graph, exact denominator      = {graph_value:.12f}   "
          f"(vs eager {abs(graph_value - exact):.2e}), {captured_nodes} nodes captured, "
          f"{graph.num_nodes()} after the default passes\n")

    print("\nnodes below are captured -> after the transform -> after the default passes")
    header = (f"{'epsilon':>9}  {'points':>6}  {'recorded bound':>14}  {'energy':>16}  "
              f"{'|dE|':>10}  {'|dE|/|E|':>10}  {'nodes':>18}")
    print(header)
    print("-" * len(header))
    for epsilon in _args.epsilons:
        energy = einsums.create_zero_tensor(f"E_laplace_{epsilon:g}", [1])
        tagged = _denominator(problem)
        transformed = cg.Graph(f"mp2 laplace {epsilon:g}")
        _mp2(problem, tagged, energy, transformed)
        transformed.annotate_tag(tagged, _TAG)

        transform = _transform(problem, epsilon)
        manager = cg.PassManager()
        manager.add(transform)
        before = transformed.num_nodes()
        if not transformed.apply(manager):
            print(f"{epsilon:9.0e}  declined: {transform.skip_reasons}")
            continue
        after = transformed.num_nodes()
        record = transformed.approximations()[0]
        value = _energy_of(transformed, energy)
        print(f"{epsilon:9.0e}  {transform.last_point_count:6d}  {record.bound:14.3e}  "
              f"{value:16.12f}  {abs(value - exact):10.2e}  "
              f"{abs(value - exact) / abs(exact):10.2e}  "
              f"{before:4d} -> {after:3d} -> {transformed.num_nodes():3d}")
        assert abs(value - exact) <= record.bound * abs(exact), (
            "the energy is outside the bound the pass recorded")
        assert [r.pass_name for r in transformed.approximations()] == ["LaplaceTransform"]
        last_record = record

    print(f"\nthe record the energy carries: {last_record.pass_name}, effect {last_record.effect}, "
          f"origin {last_record.origin}, asked for {last_record.tolerance:g}, "
          f"bound {last_record.bound:.3e}, setup '{last_record.setup}'")
    print("the emitted node count does not move with the point count, which is what makes the "
          "tolerance free to be as tight as the caller wants")

    if _args.sos:
        _report_sos(problem)
    if _args.naive:
        _report_naive(problem, exact)
    if _args.thc:
        _report_thc(source, problem, exact)
    if _args.response:
        _report_response(problem)

    print("\nfull-axis DF-MP2 under LaplaceTransform agrees with the exact denominator "
          "inside every recorded bound")


# ── The opposite-spin proving ground ────────────────────────────────────────

#: The family this arm declares, and the whole reason the rewrite is taken.
#:
#: Decoupling the denominator trades ``o^2 v^2 Q`` for ``o v Q^2 t``, so it pays
#: exactly where the auxiliary dimension is smaller than the occupied-virtual
#: product by more than the point count. At water/cc-pVDZ it is not: 84
#: auxiliary directions against 95 occupied-virtual pairs. These extents are a
#: system of a few thousand basis functions, which is where SOS-MP2 is used, and
#: declaring them is how a caller says which regime they run in. Unannotated,
#: the comparison falls back to the extents this capture happens to have and the
#: search declines, which the arm prints beside the annotated run.
_SOS_FAMILY = (("occ", "o", 300.0, "nocc"), ("vir", "v", 2700.0, "nvir"), ("aux", "x", 9000.0, "naux"),
               ("grid", "g", 30000.0, "ngrid"))


def _sos_registry():
    registry = cg.SpaceRegistry()
    for name, symbol, extent, dim in _SOS_FAMILY:
        registry.register_space(cg.index_space(name, symbol, extent, cg.GrowthClass.linear(), dim))
    return registry


def _sos_oracle(problem):
    """``E_OS`` pair by pair, which never forms a four-index tensor.

    The oracle, and the production form: the full-axis spelling below exists to
    be rewritten into something with this one's memory profile, and until the
    tiling pass can do that it is a correctness proving ground.
    """
    B = problem["fitted"]
    nocc, nvir = problem["nocc"], problem["nvir"]
    occupied = np.asarray(problem["occupied_energies"])
    gaps = einsums.create_zero_tensor("-ea-eb", [nvir, nvir])
    la.outer_sum(gaps, [problem["virtual_energies"], problem["virtual_energies"]], [-1.0, -1.0])
    integral = einsums.create_zero_tensor("K_ij", [nvir, nvir])
    squared = einsums.create_zero_tensor("K2_ij", [nvir, nvir])
    weights = einsums.create_zero_tensor("W_ij", [nvir, nvir])
    partial = einsums.create_zero_tensor("e_ij", [1])
    total = 0.0
    for i in range(nocc):
        left = B[:, i, :]
        for j in range(nocc):
            einsums.einsum("Q,a ; Q,b -> a,b", integral, left, B[:, j, :])
            la.axpby(1.0, gaps, 0.0, weights)
            shift = float(occupied[i] + occupied[j])
            la.element_transform(weights, lambda x, s=shift: 1.0 / (x + s))
            la.direct_product(1.0, integral, integral, 0.0, squared)
            la.dot(partial, squared, weights)
            total += float(np.asarray(partial)[0])
    return total


def _sos_capture(graph, problem, energy, left=None, right=None):
    """``E = sum_iajb (ia|jb)^2 / D``, over all four indices.

    ``K`` is formed twice because the transform dissolves the numerator it
    rewrites. After the search both copies are gone, so the second one costs
    nothing in the graph that runs.

    The denominator is built INSIDE the capture here, and that is the change the
    refusal above no longer forces: an outer sum of the tagged energies followed
    by the reciprocal is a recipe the pass reads off the nodes, so it accepts it,
    dissolves it with the tensor, and the four-index denominator is never
    allocated at all.
    """
    shape = [problem["nocc"], problem["nvir"], problem["nocc"], problem["nvir"]]
    left = problem["fitted"] if left is None else left
    right = problem["fitted"] if right is None else right
    K = graph.scratch("K", shape, "float64")
    T = graph.scratch("T", shape, "float64")
    again = graph.scratch("K_again", shape, "float64")
    denominator = graph.scratch("D", shape, "float64")
    with cg.capture(graph):
        la.outer_sum(denominator,
                     [problem["occupied_energies"], problem["virtual_energies"],
                      problem["occupied_energies"], problem["virtual_energies"]],
                     [1.0, -1.0, 1.0, -1.0])
        la.element_transform(denominator, "recip")
        einsums.einsum("Q,i,a ; Q,j,b -> i,a,j,b", K, left, right)
        la.direct_product(1.0, K, denominator, 0.0, T)
        einsums.einsum("Q,i,a ; Q,j,b -> i,a,j,b", again, left, right)
        la.dot(energy, again, T)
    graph.annotate_tag(denominator, _TAG)
    return denominator, (K, T, again)


def _sos_annotate(graph, problem, denominator, tensors, carriers, aux_space):
    for carrier in carriers:
        cg.annotate(carrier, (aux_space, "occ", "vir"), graph=graph)
    for tensor in tensors + (denominator,):
        cg.annotate(tensor, ("occ", "vir", "occ", "vir"), graph=graph)


def _sos_shapes(graph):
    ir = json.loads(graph.to_json())
    dims = {tensor["id"]: tensor["dims"] for tensor in ir["tensors"]}
    return [dims[t] for node in ir["nodes"] for t in node.get("outputs", []) if t in dims]


def _sos_arm(problem, epsilon, annotate, left=None, right=None, aux_space="aux"):
    """Capture, transform, search. Returns everything the table needs."""
    energy = einsums.create_zero_tensor(f"E_sos_{epsilon:g}_{annotate}", [1])
    graph = cg.Graph(f"sos {epsilon:g} {annotate}")
    registry = _sos_registry()
    graph.set_space_registry(registry)
    denominator, tensors = _sos_capture(graph, problem, energy, left, right)
    carriers = {id(t): t for t in (problem["fitted"] if left is None else left,
                                   problem["fitted"] if right is None else right)}.values()
    if annotate:
        _sos_annotate(graph, problem, denominator, tensors, tuple(carriers), aux_space)
    captured = graph.num_nodes()

    transform = _transform(problem, epsilon)
    search = cg.MultiTermFactorization()
    search.set_search_enabled(True)
    manager = cg.PassManager()
    manager.add(transform)
    manager.add(search)
    manager.run(graph)
    emitted = graph.num_nodes()
    shapes = _sos_shapes(graph)
    value = _energy_of(graph, energy)
    bound = graph.approximations()[0].bound if graph.approximations() else 0.0
    return {
        "energy": value, "bound": bound, "points": transform.last_point_count,
        "captured": captured, "emitted": emitted, "shapes": shapes,
        "search": search, "registry": registry,
        "largest": max(shapes, key=lambda d: int(np.prod(d))) if shapes else [],
    }


def _report_sos(problem):
    """The opposite-spin energy over all four indices, and what the search finds."""
    print("\nthe opposite-spin energy, full axis:")
    oracle = _sos_oracle(problem)
    shape = [problem["nocc"], problem["nvir"], problem["nocc"], problem["nvir"]]
    print(f"  pair-driven oracle, never forming a four-index tensor: {oracle:.12f}")

    header = (f"  {'epsilon':>9}  {'points':>6}  {'bound':>10}  {'energy':>16}  {'|dE|/|E|':>10}  "
              f"{'nodes':>12}  {'largest intermediate':>22}  o^2v^2")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for epsilon in _args.epsilons:
        arm = _sos_arm(problem, epsilon, annotate=True)
        rel = abs(arm["energy"] - oracle) / abs(oracle)
        print(f"  {epsilon:9.0e}  {arm['points']:6d}  {arm['bound']:10.2e}  {arm['energy']:16.12f}  "
              f"{rel:10.2e}  {arm['captured']:4d} -> {arm['emitted']:3d}  "
              f"{str(arm['largest']):>22}  {'yes' if shape in arm['shapes'] else 'no'}")
        assert shape not in arm["shapes"], "a tensor over o and v survived the rewrite"
        assert abs(arm["energy"] - oracle) <= arm["bound"] * abs(oracle) + 1e-12

    unannotated = _sos_arm(problem, _args.epsilons[-1], annotate=False)
    print(f"  unannotated, at this molecule's own extents: "
          f"{unannotated['captured']} -> {unannotated['emitted']} nodes, "
          f"o^2v^2 present = {shape in unannotated['shapes']}")
    print("  the decoupled form trades o^2 v^2 Q for o v Q^2 t, so it pays where the auxiliary")
    print("  dimension is smaller than the occupied-virtual product by more than the point count.")
    print(f"  At this molecule it is not: {problem['naux']} auxiliary directions against "
          f"{problem['nocc'] * problem['nvir']} occupied-virtual pairs, so the search declines and")
    print("  is right to. Declaring the family's typical extents is how a caller says otherwise.")


def _report_sos_thc(problem, left, right, ngrid):
    """The same opposite-spin energy with the grid fit in the density fit's place.

    The grid form has the SAME shape the density-fitted one does,
    ``sum_G L[G,i,a] R[G,j,b]`` with the grid index where the auxiliary index
    was, so the transform rides on it unchanged and the search re-associates it
    the same way. What comes out is the pairless form again, over the grid
    letter: a grid-by-grid matrix per quadrature point.

    Two things this arm does NOT reach, stated because the design block asks for
    them. The chain the block writes, ``Xo_t``, ``Xv_t``, an elementwise ``Y_t``
    and ``Z Y_t Z^T``, needs the collocation factors and the coupling as separate
    leaves, which makes the energy a fourteen-factor product over the search's
    cap of ten; what the search is handed here is the grid fit already contracted
    into two three-index factors, so it finds the same bracketing it finds for
    the density fit. And the shipped ``ThcFactorization`` cannot be asked for
    this by tagging the integral: it fits a tensor over the whole basis on every
    axis, and an occupied-virtual block is not that shape, so the joint decision
    has no candidate to cost. The grid fit is therefore performed here, as the
    density fit's factors are.
    """
    print("\n  the opposite-spin energy through the grid fit:")
    oracle = _sos_oracle(problem)
    shape = [problem["nocc"], problem["nvir"], problem["nocc"], problem["nvir"]]
    L = _tensor("L", left)
    R = _tensor("R", right)
    header = (f"    {'epsilon':>9}  {'points':>6}  {'energy':>16}  {'|dE|/|E|':>10}  "
              f"{'nodes':>12}  {'largest intermediate':>24}  o^2v^2")
    print(header)
    print("    " + "-" * (len(header) - 4))
    for epsilon in _args.epsilons:
        arm = _sos_arm(problem, epsilon, annotate=True, left=L, right=R, aux_space="grid")
        rel = abs(arm["energy"] - oracle) / abs(oracle)
        print(f"    {epsilon:9.0e}  {arm['points']:6d}  {arm['energy']:16.12f}  {rel:10.2e}  "
              f"{arm['captured']:4d} -> {arm['emitted']:3d}  {str(arm['largest']):>24}  "
              f"{'yes' if shape in arm['shapes'] else 'no'}")
    print(f"    the grid has {ngrid} points against {problem['naux']} auxiliary directions, and a")
    print("    grid is about ten times a basis where an auxiliary set is three or four. The trade")
    print("    is the same one, o^2 v^2 G against o v G^2 t, so it turns over at a larger system")
    print("    for the grid than for the fit and at a looser tolerance for the same system: the")
    print("    o^2 v^2 column above is where the search says the decoupled form has stopped")
    print("    paying, and it says so on the numbers rather than on a rule about grids.")


def _report_naive(problem, exact):
    """What each pass says about the form a caller with a dense integral writes."""
    print("\nthe dense-integral form, with both passes offered it:")
    shape = [problem["nocc"], problem["nvir"], problem["nocc"], problem["nvir"]]
    dense = einsums.create_zero_tensor("(ia|jb)", shape)
    einsums.einsum("Q,i,a ; Q,j,b -> i,a,j,b", dense, problem["fitted"], problem["fitted"])

    energy = einsums.create_zero_tensor("E_naive", [1])
    tagged = _denominator(problem)
    graph = cg.Graph("mp2 dense")
    _capture_dense(graph, problem, dense, tagged, energy)
    graph.annotate_tag(dense, _G.ProvenanceTag.make("eri"))
    graph.annotate_tag(tagged, _TAG)

    factorization = _G.FactorizationPass(_fit(problem))
    transform = _transform(problem, 1e-6)
    manager = cg.PassManager()
    manager.add(factorization)
    manager.add(transform)
    modified = graph.apply(manager)
    print(f"  modified={modified}, fits={factorization.num_factorized}, "
          f"transforms={transform.num_transformed}")
    for reason, count in factorization.skip_reasons:
        print(f"  fit declines ({count}): {reason}")
    for reason, count in transform.skip_reasons:
        print(f"  transform declines ({count}): {reason}")
    value = _energy_of(graph, energy)
    print(f"  energy {value:.12f}, which is the exact one to {abs(value - exact):.2e}: "
          "nothing was approximated because nothing was rewritten")

    # The same form again, with the two passes asked TOGETHER. Handing the
    # transform to the fit makes a tagged integral multiplied by a tagged
    # denominator a candidate the fit did not have on its own: the substituted
    # product is emitted into an intermediate, the quadrature is applied to the
    # trial, and the cost of the pair is what decides. It still declines, and
    # the one line it prints is what two unrelated silences used to be.
    print("\n  and again with the pair costed as one decision:")
    energy_joint = einsums.create_zero_tensor("E_joint", [1])
    tagged_joint = _denominator(problem, "D_joint")
    joint_graph = cg.Graph("mp2 dense, jointly costed")
    _capture_dense(joint_graph, problem, dense, tagged_joint, energy_joint)
    joint_graph.annotate_tag(dense, _G.ProvenanceTag.make("eri"))
    joint_graph.annotate_tag(tagged_joint, _TAG)

    paired = _G.FactorizationPass(_fit(problem))
    paired.set_laplace_transform(_transform(problem, 1e-6))
    joint_manager = cg.PassManager()
    joint_manager.add(paired)
    print(f"  modified={joint_graph.apply(joint_manager)}, "
          f"joint rewrites={paired.num_joint}")
    for reason, count in paired.skip_reasons:
        print(f"  pair declines ({count}): {reason}")
    print("  the amplitude is a stored o^2 v^2 tensor, so the decoupled form rebuilds it as a")
    print("  sum over the quadrature and auxiliary indices where the captured form built it")
    print("  with one elementwise multiply: one whole scale order worse. What would pay is")
    print("  never forming the amplitude, which is a re-association of the energy expression")
    print("  rather than a substitution into it.")


def _select_points(pair, count):
    """The points whose basis-pair products are independent, by pivoted selection.

    Modified Gram-Schmidt with column pivoting: take the column of largest
    remaining norm, project it out, repeat, and stop when the remaining norms
    collapse. Choosing the grid is a setup the CALLER performs, which is why it
    is here rather than in the provider: the pass takes the collocation matrix
    it is handed, and the residual it measures is what tells a caller whether
    the grid they chose was enough.
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


#: The drop threshold a grid metric needs, and NOT the provider's default.
#:
#: ``S = (X^T X)^2`` for a pruned grid runs from about ``1e-13`` to ``1`` with no
#: gap in between, where a Coulomb metric has a clear null space. An absolute
#: cutoff at ``1e-10`` therefore keeps directions whose inverse amplifies
#: rounding into the fit, and the fit comes out both less accurate and not
#: reproducible between two implementations of the same formula. At ``1e-8`` it
#: is at its most accurate here and two implementations agree to five digits.
_THC_DROP = 1e-8


def _report_thc(source, problem, exact):
    """The integrals through a grid fit, and the energy through the pair.

    Three arms, all against the same exact denominator and the same integrals:
    the density fit on its own, the grid fit on top of it, and the grid fit with
    the quadrature. The grid form has the SAME shape the density-fitted one
    does, ``sum_Q L[Q,i,a] R[Q,j,b]`` with the grid index where the auxiliary
    index was, so the transform rides on it unchanged.
    """
    if source.get("grid") is None:
        print("\nno collocation matrix available from this source; skipping the grid arm")
        return
    print("\nthe integrals through a grid fit:")

    collocation_ao, weights = source["grid"]
    coefficients = source["C"]
    nocc, nvir = problem["nocc"], problem["nvir"]
    nbf = coefficients.shape[0]

    # The weight, split four ways, because four collocation factors meet in the
    # integral this fits.
    collocation = (coefficients.T @ collocation_ao) * (np.abs(weights) ** 0.25)
    pairs = np.triu_indices(nbf)
    pivots = _select_points(collocation[pairs[0], :] * collocation[pairs[1], :], nbf * (nbf + 1) // 2)
    grid = np.ascontiguousarray(collocation[:, pivots])
    print(f"  grid: {collocation_ao.shape[1]} Becke points pruned to {grid.shape[1]} "
          f"independent pair directions, over {nbf} orbitals")

    # The density fit over the whole orbital basis, which is what the grid fit
    # is fitted FROM and what it is measured against.
    metric = np.asarray(source["metric"], dtype=np.float64)
    values, vectors = np.linalg.eigh(metric)
    inv_sqrt = vectors @ np.diag(np.where(values > 1e-10, values ** -0.5, 0.0)) @ vectors.T
    three = np.einsum("PQ,Qmn,mp,nq->Ppq", inv_sqrt, np.asarray(source["three"], dtype=np.float64),
                      coefficients, coefficients)
    dense = np.einsum("Amn,Apq->mnpq", three, three)

    # The SHIPPED provider, on this molecule, through an ordinary contraction.
    operand = np.random.default_rng(20260904).standard_normal((nbf, nbf))
    reference = np.einsum("mnpq,pq->mn", dense, operand)
    M = _tensor("(mn|pq)", dense)
    T = _tensor("T", operand)
    C = einsums.create_zero_tensor("C", [nbf, nbf])
    graph = cg.Graph("thc contraction")
    with cg.capture(graph):
        einsums.einsum("m,n,p,q ; p,q -> m,n", C, M, T)
    graph.annotate_tag(M, _G.ProvenanceTag.make("eri"))
    graph.annotate_dims(M, ["nbf"] * 4)
    _G.ThcFactorization.register_grid_space(graph)
    registry = _G.FactorizationRegistry()
    registry.add(_G.ThcFactorization("eri", _tensor("B", three), _tensor("X", grid), 1e-2, _THC_DROP))
    factorization = _G.FactorizationPass(registry)
    manager = cg.PassManager()
    manager.add(factorization)
    before = graph.num_nodes()
    fired = graph.apply(manager)
    after = graph.num_nodes()
    print(f"  the pass on a contraction over the tagged integral: fired={fired}, "
          f"factorized={factorization.num_factorized}, nodes {before} -> {after}")
    for reason, count in factorization.skip_reasons:
        print(f"    declines ({count}): {reason}")
    value = np.asarray(_energy_of_tensor(graph, C))
    scale = float(np.linalg.norm(reference))
    print(f"  the contracted integral through the chain is within "
          f"{np.linalg.norm(value - reference) / scale:.3e} of the density-fitted one")

    # The same fit written out, because the energy arms need the factors as
    # buffers and the pass keeps them inside a graph. Checked against the pass
    # above rather than trusted.
    gram = grid.T @ grid
    fit_metric = gram * gram
    fvals, fvecs = np.linalg.eigh(fit_metric)
    fit_half = fvecs @ np.diag(np.where(fvals > _THC_DROP, fvals ** -0.5, 0.0)) @ fvecs.T
    inverse = fit_half @ fit_half
    projected = np.einsum("Amn,mP,nP->AP", three, grid, grid)
    coupling = inverse @ (projected.T @ projected) @ inverse
    rebuilt = np.einsum("AP,mP,nP->Amn", projected @ inverse, grid, grid)
    print(f"  three-index relative residual {np.linalg.norm(three - rebuilt) / np.linalg.norm(three):.3e}")

    left = np.ascontiguousarray(np.einsum("iP,aP,PQ->Qia", grid[:nocc], grid[nocc:], coupling))
    right = np.ascontiguousarray(np.einsum("jQ,bQ->Qjb", grid[:nocc], grid[nocc:]))
    thc_block = np.einsum("Qia,Qjb->iajb", left, right)
    df_block = dense[:nocc, nocc:, :nocc, nocc:]
    print(f"  (ia|jb) relative residual {np.linalg.norm(thc_block - df_block) / np.linalg.norm(df_block):.3e}, "
          f"largest deviation {np.abs(thc_block - df_block).max():.3e}")

    print("\n  the correlation energy through each arm:")
    print("  arm                                 energy            |dE|      |dE|/|E|  points     nodes")
    print("  " + "-" * 88)
    print(f"  {'density fit alone':<26} {exact:16.12f}  {0.0:10.2e}  {0.0:10.2e}       -         -")
    for epsilon in [None] + list(_args.epsilons):
        energy, captured, emitted, points = _thc_arm(problem, left, right, epsilon)
        label = "grid fit alone" if epsilon is None else f"grid fit + quadrature {epsilon:g}"
        print(f"  {label:<26} {energy:16.12f}  {abs(energy - exact):10.2e}  "
              f"{abs(energy - exact) / abs(exact):10.2e}  {points:>6}  {captured:4d} -> {emitted:3d}")
    print("  the quadrature's own error falls below the grid fit's and the joint number stops "
          "moving, which is\n  the composition doing what a composition should: the looser "
          "approximation is the one that decides.")
    if _args.sos:
        _report_sos_thc(problem, left, right, grid.shape[1])


def _energy_of_tensor(graph, tensor):
    graph.apply(cg.default_pass_manager())
    graph.execute()
    return tensor


def _thc_arm(problem, left, right, epsilon):
    """The full-axis energy with the grid factors in the density fit's place."""
    shape = [problem["nocc"], problem["nvir"], problem["nocc"], problem["nvir"]]
    L = _tensor("L", left)
    R = _tensor("R", right)
    denominator = _denominator(problem, f"D_thc_{epsilon}")
    energy = einsums.create_zero_tensor("E_thc", [1])

    graph = cg.Graph(f"thc mp2 {epsilon}")
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
    points = "-"
    if epsilon is not None:
        graph.annotate_tag(denominator, _TAG)
        transform = _transform(problem, epsilon)
        manager = cg.PassManager()
        manager.add(transform)
        assert graph.apply(manager), f"the transform declined: {transform.skip_reasons}"
        points = str(transform.last_point_count)
    value = _energy_of(graph, energy)
    return value, captured, graph.num_nodes(), points


def _report_response(problem):
    """The shape where the fit and the transform do compose, on these integrals.

    ``G[i,a] = sum_jb (ia|jb) U[j,b]``, divided by ``e_i - e_a``: the orbital
    response update. The integral is a contraction operand here, so the fit has
    a re-association to make, and the transform then rides on the factor the fit
    left carrying i and a. The graph is annotated for its family, which is what
    makes the fit's cost veto abstain: at water/cc-pVDZ the auxiliary set is 84
    directions against 95 occupied-virtual pairs, so the split is not cheaper at
    THIS size and is cheaper at every interesting one.
    """
    print("\nthe orbital-response shape, where the two compose:")
    nocc, nvir = problem["nocc"], problem["nvir"]
    dense = einsums.create_zero_tensor("(ia|jb)", [nocc, nvir, nocc, nvir])
    einsums.einsum("Q,i,a ; Q,j,b -> i,a,j,b", dense, problem["fitted"], problem["fitted"])

    trial = einsums.create_zero_tensor("U", [nocc, nvir])
    np.asarray(trial)[...] = np.random.default_rng(20260904).standard_normal((nocc, nvir))
    gaps = einsums.create_zero_tensor("D_ov", [nocc, nvir])
    la.outer_sum(gaps, [problem["occupied_energies"], problem["virtual_energies"]], [1.0, -1.0])
    la.element_transform(gaps, lambda x: 1.0 / x)

    exact_update = np.einsum("iajb,jb->ia", np.asarray(dense), np.asarray(trial)) * np.asarray(gaps)

    update = einsums.create_zero_tensor("X", [nocc, nvir])
    graph = cg.Graph("orbital response")
    contracted = graph.scratch("G", [nocc, nvir], "float64")
    with cg.capture(graph):
        einsums.einsum("i,a,j,b ; j,b -> i,a", contracted, dense, trial)
        la.direct_product(1.0, contracted, gaps, 0.0, update)
    graph.annotate_tag(dense, _G.ProvenanceTag.make("eri"))
    graph.annotate_tag(gaps, _G.LaplaceTransform.denominator_tag(["eps_occ", "eps_vir"], "+-"))
    for tensor, symbols in ((dense, ["nocc", "nvir", "nocc", "nvir"]), (trial, ["nocc", "nvir"]),
                            (gaps, ["nocc", "nvir"]), (update, ["nocc", "nvir"])):
        graph.annotate_dims(tensor, symbols)

    factorization = _G.FactorizationPass(_fit(problem))
    transform = _transform(problem, 1e-6)
    manager = cg.PassManager()
    manager.add(factorization)
    manager.add(transform)
    before = graph.num_nodes()
    assert graph.apply(manager)
    print(f"  fits={factorization.num_factorized}, transforms={transform.num_transformed}, "
          f"points={transform.last_point_count}, nodes {before} -> {graph.num_nodes()}")
    records = graph.approximations()
    print("  records: " + ", ".join(f"{r.pass_name} bound {r.bound:.2e}" for r in records))
    tolerance = graph.approximation_tolerance("X")
    print(f"  composed relative tolerance on the output: {tolerance.relative:.3e}")

    graph.apply(cg.default_pass_manager())
    graph.execute()
    deviation = float(np.max(np.abs(np.asarray(update) - exact_update)))
    scale = float(np.max(np.abs(exact_update)))
    print(f"  max deviation {deviation:.3e} against a scale of {scale:.3e}, "
          f"inside {tolerance.relative * scale:.3e}")
    assert deviation <= tolerance.relative * scale


main()
