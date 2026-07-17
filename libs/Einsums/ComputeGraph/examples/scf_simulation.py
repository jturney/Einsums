# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""Closed-shell restricted Hartree-Fock SCF expressed entirely as a captured
ComputeGraph in Python.

The Python surface for ComputeGraph (built up over a series of 2026-05 commits)
is the C++ counterpart's twin — every linear-algebra primitive is graph-aware,
``RuntimeTensorView`` is a first-class capture citizen, and ``__getitem__``
auto-dispatches to ``cg.view`` inside a capture. That makes it possible to
write SCF as one ``with cg.capture():`` block, hand the recorded graph to the
optimization passes, and execute it. No numpy fall-through inside the loop
body.

The only piece this example doesn't supply is **AO integrals** — those come
from a basis-set integral library in a real calculation. We hardcode two
STO-3G systems so the example is self-contained:

  * H2 at R = 1.4 bohr  → converges in 1 SCF iteration (D∞h symmetry
                           pre-determines the MO coefficients; total energy
                           -1.117 Ha matches Szabo & Ostlund).

  * HeH+ at R = 1.4632 bohr → converges in ~10 iterations (asymmetric H_core,
                              so the Fock matrix's eigenvectors actually
                              depend on the density; total energy ≈ -2.86 Ha,
                              consistent with the literature value of -2.84).

Run with::

    PYTHONPATH=<einsums-build>/lib python scf_simulation.py

**MP2 correction** uses only ops you've already seen here plus
``linalg.outer_sum`` and ``linalg.element_transform`` — see
``tests/unit/test_outer_sum_python.py`` and ``test_einsum_views_python.py``
for the canonical patterns (Δ_{ijab} = ε_i + ε_j − ε_a − ε_b, the (ia|jb)
slice via ``eri_mo[:nocc, nocc:, :nocc, nocc:]``, and the full-reduction
einsum into a rank-0 scalar).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import einsums
import einsums.graph as cg


# ──────────────────────────────────────────────────────────────────────────
# 1. AO integrals for two STO-3G systems.
#    Real calculations source these from an integral library; here they're
#    hardcoded so the example is self-contained.
# ──────────────────────────────────────────────────────────────────────────


@dataclass
class System:
    name: str
    nbf: int
    nocc: int
    enuc: float
    S: np.ndarray         # overlap matrix
    H: np.ndarray         # core Hamiltonian (kinetic + nuclear attraction)
    ERI: np.ndarray       # 4-index electron repulsion integrals (chemist's notation)


def _h2_sto3g() -> System:
    """H2 at R = 1.4 bohr (Szabo & Ostlund Table 3.5)."""
    nbf = 2
    S = np.array([[1.0, 0.6593], [0.6593, 1.0]])
    H = np.array([[-1.1204, -0.9584], [-0.9584, -1.1204]])
    ERI = np.zeros((nbf, nbf, nbf, nbf))
    ERI[0, 0, 0, 0] = ERI[1, 1, 1, 1] = 0.7746
    ERI[0, 0, 1, 1] = ERI[1, 1, 0, 0] = 0.5697
    for m, n, lam, sig in [(0, 0, 0, 1), (0, 0, 1, 0), (0, 1, 0, 0), (1, 0, 0, 0),
                           (0, 1, 1, 1), (1, 0, 1, 1), (1, 1, 0, 1), (1, 1, 1, 0)]:
        ERI[m, n, lam, sig] = 0.4441
    for m, n, lam, sig in [(0, 1, 0, 1), (1, 0, 0, 1), (0, 1, 1, 0), (1, 0, 1, 0)]:
        ERI[m, n, lam, sig] = 0.2970
    return System(name="H2/STO-3G (R=1.4 bohr)",
                  nbf=nbf, nocc=1, enuc=1.0 / 1.4, S=S, H=H, ERI=ERI)


def _heh_plus_sto3g() -> System:
    """HeH+ at R = 1.4632 bohr. Index 0 = He 1s, index 1 = H 1s.
    Standard STO-3G integrals from Szabo & Ostlund."""
    nbf = 2
    S = np.array([[1.0, 0.4508], [0.4508, 1.0]])
    H = np.array([[-2.6527, -1.3472], [-1.3472, -1.7322]])
    ERI = np.zeros((nbf, nbf, nbf, nbf))
    # Same-orbital diagonals
    ERI[0, 0, 0, 0] = 1.3072          # (He He | He He)
    ERI[1, 1, 1, 1] = 0.7746          # (H  H  | H  H)
    # Pair-pair
    ERI[0, 0, 1, 1] = ERI[1, 1, 0, 0] = 0.6057
    # 4 entries at (He,He | He,H)-type
    for m, n, lam, sig in [(0, 0, 0, 1), (0, 0, 1, 0), (0, 1, 0, 0), (1, 0, 0, 0)]:
        ERI[m, n, lam, sig] = 0.4373
    # 4 entries at (H,H | He,H)-type
    for m, n, lam, sig in [(0, 1, 1, 1), (1, 0, 1, 1), (1, 1, 0, 1), (1, 1, 1, 0)]:
        ERI[m, n, lam, sig] = 0.3118
    # 4 entries at (He,H | He,H)-type
    for m, n, lam, sig in [(0, 1, 0, 1), (1, 0, 0, 1), (0, 1, 1, 0), (1, 0, 1, 0)]:
        ERI[m, n, lam, sig] = 0.1773
    return System(name="HeH+/STO-3G (R=1.4632 bohr)",
                  nbf=nbf, nocc=1, enuc=2.0 / 1.4632, S=S, H=H, ERI=ERI)


# ──────────────────────────────────────────────────────────────────────────
# 2. Captured-graph SCF.
# ──────────────────────────────────────────────────────────────────────────


def run_scf(system: System, *, max_iter: int = 50, etol: float = 1e-9, dtol: float = 1e-7) -> float:
    """Build the captured-graph SCF for the given system, execute it, and
    return the total energy."""

    NBF = system.nbf
    NOCC = system.nocc

    # Two tensor lifetimes are in play:
    #
    #   * **Eager** ``create_zero_tensor`` for anything that needs a numpy
    #     fill before the graph runs (the system data S/H/ERI, the
    #     orthogonalizer X = S^{-1/2} that's computed once outside capture,
    #     and the initial Fock F = H).
    #
    #   * **Workspace-declared** for every pure intermediate the loop body
    #     reads and writes. These are *shells* — metadata only, no backing
    #     data — until ``ws.materialize_all()`` (also covered by the
    #     Materialization pass in ``default_pass_manager``) flips them on.
    #     Declaring them this way lets the optimization passes plan
    #     allocations from the recorded dependency graph rather than from
    #     whatever was created eagerly.
    S = einsums.create_zero_tensor("S", [NBF, NBF])
    H = einsums.create_zero_tensor("H", [NBF, NBF])
    ERI = einsums.create_zero_tensor("ERI", [NBF, NBF, NBF, NBF])
    X = einsums.create_zero_tensor("X", [NBF, NBF])
    F = einsums.create_zero_tensor("F", [NBF, NBF])

    # Scalar destinations of ``dot`` / ``norm`` — the Python bindings for
    # those reductions validate the destination's element count at capture
    # time, so they need real allocations even before the Materialization
    # pass runs.
    E_elec  = einsums.create_zero_tensor("E_elec",  [1])
    E_old   = einsums.create_zero_tensor("E_old",   [1])
    delta_E = einsums.create_zero_tensor("delta_E", [1])
    rms_D   = einsums.create_zero_tensor("rms_D",   [1])

    ws = cg.Workspace("scf_intermediates")
    D       = ws.declare_zero_tensor("D",       [NBF, NBF])
    D_old   = ws.declare_zero_tensor("D_old",   [NBF, NBF])
    C       = ws.declare_zero_tensor("C",       [NBF, NBF])
    F_prime = ws.declare_zero_tensor("F_prime", [NBF, NBF])
    tmp_NN  = ws.declare_zero_tensor("tmp_NN",  [NBF, NBF])
    J       = ws.declare_zero_tensor("J",       [NBF, NBF])
    K       = ws.declare_zero_tensor("K",       [NBF, NBF])
    sum_HF  = ws.declare_zero_tensor("sum_HF",  [NBF, NBF])
    diff_D  = ws.declare_zero_tensor("diff_D",  [NBF, NBF])
    eps     = ws.declare_zero_tensor("eps",     [NBF])

    np.asarray(S)[...]   = system.S
    np.asarray(H)[...]   = system.H
    np.asarray(ERI)[...] = system.ERI
    np.asarray(F)[...]   = system.H        # initial Fock = core Hamiltonian

    # Symmetric orthogonalization: X = S^{-1/2}. One-shot, computed eagerly
    # because ``cg.pow``'s returning form throws inside capture.
    np.asarray(X)[...] = np.asarray(einsums.linalg.pow(S, -0.5))

    g = cg.Graph(f"scf:{system.name}")

    def converged(iteration: int) -> bool:
        dE   = float(np.asarray(delta_E)[0])
        rmsD = float(np.asarray(rms_D)[0])
        e    = float(np.asarray(E_elec)[0])
        print(f"  iter {iteration:3d}  E_elec = {e:16.10f}  "
              f"dE = {abs(dE):.3e}  rms(dD) = {rmsD:.3e}")
        # First iteration's metrics are spuriously zero (E_old/D_old were
        # also zero), so don't allow convergence on iteration 0.
        if iteration == 0:
            return True
        return not (abs(dE) < etol and rmsD < dtol)

    body = g.add_loop("scf_iterations", max_iter, converged)

    with cg.capture(body):
        # Snapshot the previous iteration's energy and density.
        einsums.linalg.axpby(1.0, E_elec, 0.0, E_old)
        einsums.linalg.axpby(1.0, D,      0.0, D_old)

        # Orthogonalize F: F' = X^T F X.
        einsums.linalg.gemm(1.0, X,      F, 0.0, tmp_NN,  trans_a=True)
        einsums.linalg.gemm(1.0, tmp_NN, X, 0.0, F_prime)

        # Diagonalize F'. In-place: F_prime gets eigenvectors, eps gets eigenvalues.
        einsums.linalg.syev(F_prime, eps, compute_eigenvectors=True)

        # Backtransform MO coefficients: C = X F'.
        einsums.linalg.gemm(1.0, X, F_prime, 0.0, C)

        # Density matrix D = 2 C_occ C_occ^T. ``C[:, :NOCC]`` auto-dispatches
        # to cg.view inside capture, returning a graph-aware view that
        # aliases C.
        einsums.linalg.gemm(2.0, C[:, :NOCC], C[:, :NOCC], 0.0, D, trans_b=True)

        # Fock build: J(p,q) = sum_{rs} (pq|rs) D(r,s)
        #             K(p,q) = sum_{rs} (pr|qs) D(r,s)
        #             F      = H + J - 0.5 K.
        # (With D = 2 sum_i C_i C_i^T the Fock prefactors are 1 and -0.5.)
        einsums.einsum("pq <- pqrs ; rs", J, ERI, D)
        einsums.einsum("pq <- prqs ; rs", K, ERI, D)

        einsums.linalg.axpby(1.0, H, 0.0, F)
        einsums.linalg.axpy(1.0,  J, F)
        einsums.linalg.axpy(-0.5, K, F)

        # Electronic energy E_elec = 1/2 sum_pq D(p,q) (H(p,q) + F(p,q)).
        einsums.linalg.axpby(1.0, H, 0.0, sum_HF)
        einsums.linalg.axpy(1.0,  F, sum_HF)
        einsums.linalg.dot(E_elec, D, sum_HF)
        einsums.linalg.scale(0.5, E_elec)

        # Convergence metrics.
        einsums.linalg.axpby(1.0, E_elec, 0.0, delta_E)
        einsums.linalg.axpy(-1.0, E_old, delta_E)

        einsums.linalg.axpby(1.0, D, 0.0, diff_D)
        einsums.linalg.axpy(-1.0, D_old, diff_D)
        einsums.linalg.norm(rms_D, einsums.linalg.Norm.FROBENIUS, diff_D)

    # Report what the optimization passes did. ``Graph.apply()`` returns
    # True iff at least one pass touched the graph; node/tensor counts on
    # the loop body give a coarser before/after fingerprint without having
    # to inspect each pass's individual counters. (The top-level graph
    # holds just the single loop node; the work lives in ``body``.) The
    # parent-level node count is also worth watching — Materialization
    # hoists Alloc/Initialize nodes from the body up to the parent here.
    parent_nodes_before                    = g.num_nodes()
    body_nodes_before, body_tensors_before = body.num_nodes(), body.num_tensors()
    pm = cg.default_pass_manager()
    modified = g.apply(pm)
    body_nodes_after, body_tensors_after = body.num_nodes(), body.num_tensors()
    parent_nodes_after                   = g.num_nodes()
    print(f"  passes: parent nodes {parent_nodes_before} -> {parent_nodes_after}, "
          f"body nodes {body_nodes_before} -> {body_nodes_after}, "
          f"body tensors {body_tensors_before} -> {body_tensors_after}, "
          f"modified={modified}")

    # Note: ``ws.materialize_all()`` is no longer required here — the
    # loop-aware Materialization pass hoists allocation *and*
    # zero-initialization of body-resident workspace tensors to the parent
    # graph just before the loop, so each tensor's storage is allocated and
    # zeroed once per outer execution. ``declare_zero_tensor``'s init kind
    # rides through capture on the tensor itself, so the hoisted lifecycle
    # includes the Initialize (zero) node.
    g.execute()

    return float(np.asarray(E_elec)[0]) + system.enuc


# ──────────────────────────────────────────────────────────────────────────
# 3. Run both systems.
# ──────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Each entry: (system, human-readable reference, expected total HF energy).
    # The expected energy is asserted so this example doubles as a
    # correctness regression test — a bad optimization pass that silently
    # corrupts the captured graph (e.g. CSE merging two distinct
    # mutable-output ops) makes the run *diverge*, which a plain exit-code
    # check would miss.
    cases = [
        (_h2_sto3g(),       "-1.117 Ha (Szabo & Ostlund)",          -1.1167529403),
        (_heh_plus_sto3g(), "≈ -2.84 Ha (HeH+/STO-3G literature)",  -2.8605882865),
    ]

    failures = []
    for system, reference, expected in cases:
        print(f"\n=== {system.name} ===\n")
        e_total = run_scf(system)
        print(f"\nTotal HF energy : {e_total:16.10f}")
        print(f"Reference       : {reference}")
        if abs(e_total - expected) > 1e-6:
            failures.append(f"{system.name}: got {e_total:.10f}, expected {expected:.10f}")

    if failures:
        raise SystemExit("SCF energy mismatch (graph optimization regression?):\n  " + "\n  ".join(failures))
