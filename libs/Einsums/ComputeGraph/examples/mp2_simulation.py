# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""Closed-shell MP2 correlation energy expressed as a captured ComputeGraph
in Python.

This is the natural successor to ``scf_simulation.py``: it reuses the same
H2 and HeH+ STO-3G fixtures, recomputes the SCF reference as a captured
graph, then builds the MP2 correction as a *second* captured graph that
consumes the converged orbitals and AO integrals.

What MP2 needs that pure SCF didn't:

  * **AO → MO transform** of the rank-4 ERI tensor. A single einsum can't
    contract four C indices at once, so the example does the canonical
    four-step transform — each step is one ``einsums.einsum`` node in the
    graph. After the final step ``ERI_MO[i,j,k,l]`` holds (ij|kl) in MO
    basis.

  * The **OVOV slice**, ``(ia|jb) = ERI_MO[i, nocc+a, j, nocc+b]``. We use
    ``einsums.linalg.block_copy`` to materialize the OVOV block as its
    own tensor inside the graph — that keeps the contraction kernels
    operating on owning tensors (the einsum bindings don't yet accept
    arbitrary views).

  * The **orbital-energy denominator** Δ(i,a,j,b) = ε_i − ε_a + ε_j − ε_b.
    We build it with ``einsums.linalg.outer_sum`` — exactly the
    rank-N-from-rank-1-vectors primitive that landed alongside this
    example. ``element_transform(inv_delta, lambda x: 1/x)`` produces
    1/Δ in place.

The closed-shell spin-summed MP2 energy is::

    E_MP2 = Σ_{ijab} (ia|jb) · [2 (ia|jb) − (ib|ja)] / Δ(i,a,j,b)

which the graph evaluates as

  1. ``I_weighted = (ia|jb) ⊙ (1/Δ)``  via ``direct_product``
  2. ``numerator  = 2 (ia|jb) − (ib|ja)`` via ``axpby`` + ``axpy`` (the
     exchange permutation ``(ib|ja)`` comes from ``permute``)
  3. ``E_mp2[0]   = Frobenius⟨numerator, I_weighted⟩``  via ``dot``

Run with::

    PYTHONPATH=<einsums-build>/lib python mp2_simulation.py

Reference values (Szabo & Ostlund / standard STO-3G):

  * H2/STO-3G   : E_HF ≈ −1.117 Ha,  E_corr ≈ −0.013 Ha,  E_MP2 ≈ −1.130 Ha
  * HeH+/STO-3G : E_HF ≈ −2.86  Ha,  E_corr ≈ −0.005 Ha,  E_MP2 ≈ −2.87  Ha
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import einsums
import einsums.graph as cg

from scf_simulation import System, _h2_sto3g, _heh_plus_sto3g


@dataclass
class SCFResult:
    """Converged HF data needed to drive the MP2 graph."""
    E_total: float
    C: np.ndarray       # MO coefficients (nbf × nbf), column k = ψ_k
    eps: np.ndarray     # orbital energies (nbf,)


# ──────────────────────────────────────────────────────────────────────────
# 1. SCF as a captured graph — returns the converged C and eps that MP2
#    needs. The kernel is the same as scf_simulation.run_scf; the only
#    differences are that this version returns the full orbital bundle and
#    keeps the logging compact (the SCF mechanics are documented in detail
#    in the sibling example).
# ──────────────────────────────────────────────────────────────────────────


def run_scf(system: System, *, max_iter: int = 50, etol: float = 1e-9, dtol: float = 1e-7) -> SCFResult:
    NBF = system.nbf
    NOCC = system.nocc

    # Eager allocations: tensors that need a numpy-side fill before the
    # graph runs.
    S = einsums.create_zero_tensor("S", [NBF, NBF])
    H = einsums.create_zero_tensor("H", [NBF, NBF])
    ERI = einsums.create_zero_tensor("ERI", [NBF, NBF, NBF, NBF])
    X = einsums.create_zero_tensor("X", [NBF, NBF])
    F = einsums.create_zero_tensor("F", [NBF, NBF])

    # Scalar destinations of ``dot`` / ``norm`` need eager allocation —
    # those bindings validate the destination's element count at capture.
    E_elec  = einsums.create_zero_tensor("E_elec",  [1])
    E_old   = einsums.create_zero_tensor("E_old",   [1])
    delta_E = einsums.create_zero_tensor("delta_E", [1])
    rms_D   = einsums.create_zero_tensor("rms_D",   [1])

    # Pure intermediates declared as workspace shells. The Materialization
    # pass in default_pass_manager (and the explicit materialize_all() guard
    # before execute()) allocates them — the graph itself drives layout.
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
    np.asarray(F)[...]   = system.H

    np.asarray(X)[...] = np.asarray(einsums.linalg.pow(S, -0.5))

    g = cg.Graph(f"scf:{system.name}")

    def keep_going(iteration: int) -> bool:
        dE   = float(np.asarray(delta_E)[0])
        rmsD = float(np.asarray(rms_D)[0])
        e    = float(np.asarray(E_elec)[0])
        print(f"  iter {iteration:3d}  E_elec = {e:16.10f}  dE = {abs(dE):.3e}  rms(dD) = {rmsD:.3e}")
        if iteration == 0:
            return True
        return not (abs(dE) < etol and rmsD < dtol)

    body = g.add_loop("scf_iterations", max_iter, keep_going)

    with cg.capture(body):
        einsums.linalg.axpby(1.0, E_elec, 0.0, E_old)
        einsums.linalg.axpby(1.0, D,      0.0, D_old)

        einsums.linalg.gemm(1.0, X,      F, 0.0, tmp_NN,  trans_a=True)
        einsums.linalg.gemm(1.0, tmp_NN, X, 0.0, F_prime)
        einsums.linalg.syev(F_prime, eps, compute_eigenvectors=True)
        einsums.linalg.gemm(1.0, X, F_prime, 0.0, C)
        einsums.linalg.gemm(2.0, C[:, :NOCC], C[:, :NOCC], 0.0, D, trans_b=True)

        einsums.einsum("pq <- pqrs ; rs", J, ERI, D)
        einsums.einsum("pq <- prqs ; rs", K, ERI, D)

        einsums.linalg.axpby(1.0, H, 0.0, F)
        einsums.linalg.axpy(1.0,  J, F)
        einsums.linalg.axpy(-0.5, K, F)

        einsums.linalg.axpby(1.0, H, 0.0, sum_HF)
        einsums.linalg.axpy(1.0,  F, sum_HF)
        einsums.linalg.dot(E_elec, D, sum_HF)
        einsums.linalg.scale(0.5, E_elec)

        einsums.linalg.axpby(1.0, E_elec, 0.0, delta_E)
        einsums.linalg.axpy(-1.0, E_old, delta_E)
        einsums.linalg.axpby(1.0, D, 0.0, diff_D)
        einsums.linalg.axpy(-1.0, D_old, diff_D)
        einsums.linalg.norm(rms_D, einsums.linalg.Norm.FROBENIUS, diff_D)

    # SCF graph holds a single loop node at the top level; the operations
    # all live in ``body``. Report counts there so the before/after deltas
    # actually reflect what the passes did. Parent-level node count is
    # also instructive — the loop-aware Materialization pass hoists
    # Alloc/Initialize for body-resident workspace tensors up to the
    # parent just before the Loop.
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

    # ``ws.materialize_all()`` no longer needed — Materialization hoists
    # workspace tensor lifecycle (allocate + zero) out of the loop body to
    # the parent graph. The zero-init rides through capture because
    # declare_zero_tensor tags the tensor's pending-init policy.
    g.execute()

    return SCFResult(
        E_total=float(np.asarray(E_elec)[0]) + system.enuc,
        C=np.asarray(C).copy(),
        eps=np.asarray(eps).copy(),
    )


# ──────────────────────────────────────────────────────────────────────────
# 2. MP2 correlation energy as a captured graph.
# ──────────────────────────────────────────────────────────────────────────


def run_mp2(system: System, scf: SCFResult) -> float:
    NBF = system.nbf
    NOCC = system.nocc
    NVIRT = NBF - NOCC

    # ── Inputs (filled eagerly from the converged SCF) ──────────────────
    C     = einsums.create_zero_tensor("C",     [NBF, NBF])
    ERI   = einsums.create_zero_tensor("ERI",   [NBF, NBF, NBF, NBF])
    eps_o = einsums.create_zero_tensor("eps_o", [NOCC])
    eps_v = einsums.create_zero_tensor("eps_v", [NVIRT])
    np.asarray(C)[...]     = scf.C
    np.asarray(ERI)[...]   = system.ERI
    np.asarray(eps_o)[...] = scf.eps[:NOCC]
    np.asarray(eps_v)[...] = scf.eps[NOCC:]

    # The four nbf^4 transform scratches, the OVOV block, and the MP2
    # working tensors are pure intermediates — declare them as workspace
    # shells and let the Materialization pass / materialize_all() allocate.
    ws = cg.Workspace("mp2_intermediates")
    T1     = ws.declare_zero_tensor("T1",     [NBF, NBF, NBF, NBF])
    T2     = ws.declare_zero_tensor("T2",     [NBF, NBF, NBF, NBF])
    T3     = ws.declare_zero_tensor("T3",     [NBF, NBF, NBF, NBF])
    ERI_MO = ws.declare_zero_tensor("ERI_MO", [NBF, NBF, NBF, NBF])

    iajb_dims = [NOCC, NVIRT, NOCC, NVIRT]
    I          = ws.declare_zero_tensor("iajb",       iajb_dims)
    I_exch     = ws.declare_zero_tensor("ibja",       iajb_dims)
    Delta      = ws.declare_zero_tensor("Delta",      iajb_dims)
    inv_Delta  = ws.declare_zero_tensor("inv_Delta",  iajb_dims)
    I_weighted = ws.declare_zero_tensor("I_weighted", iajb_dims)
    numerator  = ws.declare_zero_tensor("numerator",  iajb_dims)

    # E_corr is the dot destination — eager (binding validates size).
    E_corr = einsums.create_zero_tensor("E_corr", [1])

    g = cg.Graph(f"mp2:{system.name}")
    with cg.capture(g):
        # Four-step AO → MO transform of the rank-4 ERI tensor. Each einsum
        # contracts one AO index at a time, so we never materialize anything
        # larger than nbf^4.
        einsums.einsum("ibcd <- ai ; abcd", T1,     C, ERI)
        einsums.einsum("ijcd <- bj ; ibcd", T2,     C, T1)
        einsums.einsum("ijkd <- ck ; ijcd", T3,     C, T2)
        einsums.einsum("ijkl <- dl ; ijkd", ERI_MO, C, T3)

        # OVOV slice: I[i,a,j,b] = ERI_MO[i, NOCC+a, j, NOCC+b]
        einsums.linalg.block_copy(
            I, ERI_MO,
            [0, 0, 0, 0],
            [0, NOCC, 0, NOCC],
            [NOCC, NVIRT, NOCC, NVIRT],
        )
        # Exchange counterpart: I_exch[i,a,j,b] = I[i,b,j,a]
        einsums.permute("iajb <- ibja", I_exch, I)

        # Δ(i,a,j,b) = ε_i − ε_a + ε_j − ε_b
        einsums.linalg.outer_sum(Delta, [eps_o, eps_v, eps_o, eps_v],
                                 [+1.0, -1.0, +1.0, -1.0])
        # inv_Delta = 1/Δ in place. Copy Δ first because element_transform
        # rewrites its argument and we want to keep Δ around for clarity.
        einsums.linalg.axpby(1.0, Delta, 0.0, inv_Delta)
        einsums.linalg.element_transform(inv_Delta, lambda x: 1.0 / x)

        # I_weighted = (ia|jb) ⊙ (1/Δ)  (element-wise)
        einsums.linalg.direct_product(1.0, I, inv_Delta, 0.0, I_weighted)

        # numerator = 2 (ia|jb) − (ib|ja)
        einsums.linalg.axpby(2.0, I,      0.0, numerator)
        einsums.linalg.axpy(-1.0, I_exch, numerator)

        # E_corr = Σ_{iajb} numerator · I_weighted  (Frobenius inner product)
        einsums.linalg.dot(E_corr, numerator, I_weighted)

    nodes_before, tensors_before = g.num_nodes(), g.num_tensors()
    pm = cg.default_pass_manager()
    modified = g.apply(pm)
    nodes_after, tensors_after = g.num_nodes(), g.num_tensors()
    print(f"  passes: nodes {nodes_before} -> {nodes_after}, "
          f"tensors {tensors_before} -> {tensors_after}, modified={modified}")

    # No loop here, so Materialization handled the workspace shells via
    # the in-graph path (same code path the pass has always taken for
    # flat graphs). No explicit ws.materialize_all() needed.
    g.execute()

    return float(np.asarray(E_corr)[0])


# ──────────────────────────────────────────────────────────────────────────
# 3. Run both systems: SCF → MP2.
# ──────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    # Each entry: (system, reference text, expected E_HF, expected E_MP2 total).
    # The expected energies are asserted so the example doubles as a
    # correctness regression test — a graph-optimization bug that corrupts
    # the captured SCF/MP2 would change these, which a plain exit-code check
    # would miss.
    cases = [
        (_h2_sto3g(),       "E_HF ≈ −1.117 Ha,  E_MP2 ≈ −1.130 Ha", -1.1167529403, -1.1299033127),
        (_heh_plus_sto3g(), "E_HF ≈ −2.86 Ha,  E_MP2 ≈ −2.87 Ha",   -2.8605882865, -2.8751351018),
    ]

    failures = []
    for system, reference, expected_hf, expected_mp2 in cases:
        print(f"\n=== {system.name} ===\n")

        print("[SCF]")
        scf = run_scf(system)

        print("\n[MP2]  (transforming ERIs, building Δ, contracting)")
        e_corr = run_mp2(system, scf)
        e_mp2  = scf.E_total + e_corr

        print(f"\n  E_HF              : {scf.E_total:16.10f} Ha")
        print(f"  E_corr (MP2)      : {e_corr:16.10f} Ha")
        print(f"  E_MP2 total       : {e_mp2:16.10f} Ha")
        print(f"  Reference         : {reference}")

        if abs(scf.E_total - expected_hf) > 1e-6:
            failures.append(f"{system.name} E_HF: got {scf.E_total:.10f}, expected {expected_hf:.10f}")
        if abs(e_mp2 - expected_mp2) > 1e-6:
            failures.append(f"{system.name} E_MP2: got {e_mp2:.10f}, expected {expected_mp2:.10f}")

    if failures:
        raise SystemExit("Energy mismatch (graph optimization regression?):\n  " + "\n  ".join(failures))
