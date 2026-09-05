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

The denominator is built EAGERLY, from the same two energy vectors, rather than
inside the capture. A tagged denominator some node writes is refused: a graph
that recomputes it declares an intention to recompute, while the quadrature is
fitted per bind, so the two would silently disagree in exactly the case the
refusal is aimed at. Building it outside the capture is what the caller has to
write, and it costs one ``outer_sum`` and one ``element_transform``.

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
a contraction, so the transform has no operands to ride on. Substituting the fit
anyway is what would unlock the decoupling, and that substitution is not itself
profitable, which is what the fit's cost veto refuses. ``--response`` shows the
two composing on this same molecule's integrals in the shape where the fit IS
profitable, an orbital-response update built from ``(ia|jb)`` and divided by
``e_i - e_a``.

Integrals come from psi4 when it can be imported and from the DLPNO fixture
otherwise, and the two paths differ only in where the buffers come from.

    PYTHONPATH=/Users/jturney/Code/Einsums/Einsums/build/lib:/Users/jturney/Code/psi4/cmake-build-debug/stage/lib \
        /Users/jturney/miniconda3/envs/einsums-dev/bin/python \
        /Users/jturney/Code/Einsums/Einsums/examples/psi4-bridge/df_mp2_laplace.py
"""
import argparse
import contextlib
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
    }


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
    }


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


def _denominator(problem):
    """1 / (e_i + e_j - e_a - e_b), eagerly, from the two energy vectors.

    Eager and not captured. The pass refuses a denominator the graph writes,
    and the refusal is right: the quadrature is fitted once per bind from the
    energies, so a graph that recomputes the denominator every replay is
    describing something the substitution cannot follow.
    """
    shape = [problem["nocc"], problem["nvir"], problem["nocc"], problem["nvir"]]
    denominator = einsums.create_zero_tensor("D", shape)
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

    if _args.naive:
        _report_naive(problem, exact)
    if _args.response:
        _report_response(problem)

    print("\nfull-axis DF-MP2 under LaplaceTransform agrees with the exact denominator "
          "inside every recorded bound")


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
