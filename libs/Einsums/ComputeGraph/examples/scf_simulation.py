# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""Closed-shell restricted Hartree-Fock SCF for H2/STO-3G, expressed entirely
as a captured ComputeGraph in Python.

The Python surface for ComputeGraph (built up over a series of 2026-05 commits)
is the C++ counterpart's twin — every linear-algebra primitive is graph-aware,
``RuntimeTensorView`` is a first-class capture citizen, and ``__getitem__``
auto-dispatches to ``cg.view`` inside a capture. That makes it possible to
write SCF as one ``with cg.capture():`` block, hand the recorded graph to the
optimization passes, and execute it. No numpy fall-through inside the loop
body.

The only piece this example doesn't supply is **AO integrals** — those come
from a basis-set integral library in a real calculation. We hardcode the
H2/STO-3G values from Szabo & Ostlund's *Modern Quantum Chemistry* (Table
3.5, R = 1.4 bohr) so the example is self-contained and produces the
textbook RHF energy of ≈ -1.117 Ha.

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

import numpy as np

import einsums
import einsums.graph as cg


# ──────────────────────────────────────────────────────────────────────────
# 1. H2/STO-3G integrals at R = 1.4 bohr (Szabo & Ostlund, ch. 3).
#    Two basis functions (one 1s on each H), one doubly-occupied MO.
# ──────────────────────────────────────────────────────────────────────────

NBF = 2
NOCC = 1
ENUC = 1.0 / 1.4              # nuclear repulsion: 1/R for H2
MAX_ITER = 50
ETOL = 1e-10
DTOL = 1e-8

S_np = np.array([
    [1.0,    0.6593],
    [0.6593, 1.0   ],
])

H_np = np.array([
    [-1.1204, -0.9584],
    [-0.9584, -1.1204],
])

ERI_np = np.zeros((NBF, NBF, NBF, NBF))
# (11|11) = (22|22) = 0.7746          (4 entries)
ERI_np[0, 0, 0, 0] = ERI_np[1, 1, 1, 1] = 0.7746
# (11|22) = (22|11) = 0.5697          (2 entries)
ERI_np[0, 0, 1, 1] = ERI_np[1, 1, 0, 0] = 0.5697
# (11|12)-type: 8 entries at 0.4441
for m, n, lam, sig in [(0, 0, 0, 1), (0, 0, 1, 0), (0, 1, 0, 0), (1, 0, 0, 0),
                       (0, 1, 1, 1), (1, 0, 1, 1), (1, 1, 0, 1), (1, 1, 1, 0)]:
    ERI_np[m, n, lam, sig] = 0.4441
# (12|12)-type: 4 entries at 0.2970
for m, n, lam, sig in [(0, 1, 0, 1), (1, 0, 0, 1), (0, 1, 1, 0), (1, 0, 1, 0)]:
    ERI_np[m, n, lam, sig] = 0.2970


# ──────────────────────────────────────────────────────────────────────────
# 2. Allocate every tensor the SCF loop touches. Each one is a regular
#    RuntimeTensor; the graph refers to them by stable address. Tensors
#    that won't change after the initial setup (S, H, ERI, X) are
#    populated eagerly. Working tensors start at zero.
# ──────────────────────────────────────────────────────────────────────────

S       = einsums.create_zero_tensor("S",       [NBF, NBF])
H       = einsums.create_zero_tensor("H",       [NBF, NBF])
ERI     = einsums.create_zero_tensor("ERI",     [NBF, NBF, NBF, NBF])
X       = einsums.create_zero_tensor("X",       [NBF, NBF])
F       = einsums.create_zero_tensor("F",       [NBF, NBF])
D       = einsums.create_zero_tensor("D",       [NBF, NBF])
D_old   = einsums.create_zero_tensor("D_old",   [NBF, NBF])
C       = einsums.create_zero_tensor("C",       [NBF, NBF])
F_prime = einsums.create_zero_tensor("F_prime", [NBF, NBF])
tmp_NN  = einsums.create_zero_tensor("tmp_NN",  [NBF, NBF])
J       = einsums.create_zero_tensor("J",       [NBF, NBF])
K       = einsums.create_zero_tensor("K",       [NBF, NBF])
sum_HF  = einsums.create_zero_tensor("sum_HF",  [NBF, NBF])
diff_D  = einsums.create_zero_tensor("diff_D",  [NBF, NBF])
eps     = einsums.create_zero_tensor("eps",     [NBF])

# Scalar holders (rank-1, length 1) so we can compose with graph ops.
E_elec  = einsums.create_zero_tensor("E_elec",  [1])
E_old   = einsums.create_zero_tensor("E_old",   [1])
delta_E = einsums.create_zero_tensor("delta_E", [1])
rms_D   = einsums.create_zero_tensor("rms_D",   [1])

# Populate the read-only inputs.
np.asarray(S)[...]   = S_np
np.asarray(H)[...]   = H_np
np.asarray(ERI)[...] = ERI_np

# Initial Fock = core Hamiltonian (zero-density guess).
np.asarray(F)[...] = H_np


# ──────────────────────────────────────────────────────────────────────────
# 3. Symmetric orthogonalization: X = S^{-1/2}. One-shot, computed eagerly
#    because ``cg.pow``'s returning form throws inside capture (no place
#    to put the scalar return). The result is copied into our pre-allocated
#    X tensor so the graph captures a stable address.
# ──────────────────────────────────────────────────────────────────────────

np.asarray(X)[...] = np.asarray(einsums.linalg.pow(S, -0.5))


# ──────────────────────────────────────────────────────────────────────────
# 4. The SCF loop, captured into a Graph.
#
#    ``add_loop`` takes a Python ``cond(iter) -> bool`` callable: called
#    once after each body iteration with the just-completed index, returns
#    True to keep going and False to stop. The callback reads the scalar
#    convergence metrics (E_elec, delta_E, rms_D) directly from the rank-1
#    holder tensors — those are kept in sync by the graph nodes inside
#    the loop body.
# ──────────────────────────────────────────────────────────────────────────

g = cg.Graph("hartree_fock_scf")


def converged(iteration: int) -> bool:
    dE   = float(np.asarray(delta_E)[0])
    rmsD = float(np.asarray(rms_D)[0])
    e    = float(np.asarray(E_elec)[0])
    print(f"  iter {iteration:3d}  E_elec = {e:16.10f}  "
          f"dE = {abs(dE):.3e}  rms(dD) = {rmsD:.3e}")
    # First iteration's metrics are spuriously zero (E_old / D_old were
    # also zero), so don't allow convergence on iteration 0.
    if iteration == 0:
        return True
    return not (abs(dE) < ETOL and rmsD < DTOL)


body = g.add_loop("scf_iterations", MAX_ITER, converged)

with cg.capture(body):
    # ── Snapshot the previous iteration's energy and density.
    # ``axpby(1.0, src, 0.0, dst)`` is the captured-graph equivalent of
    # ``dst[:] = src``.
    einsums.linalg.axpby(1.0, E_elec, 0.0, E_old)
    einsums.linalg.axpby(1.0, D,      0.0, D_old)

    # ── Orthogonalize F: F' = X^T F X.
    einsums.linalg.gemm(1.0, X,      F, 0.0, tmp_NN,  trans_a=True)
    einsums.linalg.gemm(1.0, tmp_NN, X, 0.0, F_prime)

    # ── Diagonalize F'. In-place: F_prime gets the eigenvectors,
    # eps gets the eigenvalues.
    einsums.linalg.syev(F_prime, eps, compute_eigenvectors=True)

    # ── Backtransform MO coefficients: C = X F'.
    einsums.linalg.gemm(1.0, X, F_prime, 0.0, C)

    # ── Density matrix D = 2 C_occ C_occ^T.
    # ``C[:, :NOCC]`` uses the auto-cg.view path — inside capture, slicing
    # a RuntimeTensor produces a graph-aware view that aliases C, so the
    # gemm reads the same underlying storage and the optimization passes
    # see the dependency edge.
    einsums.linalg.gemm(2.0, C[:, :NOCC], C[:, :NOCC], 0.0, D, trans_b=True)

    # ── Fock build: J(p,q) = sum_{rs} (pq|rs) D(r,s)
    #                K(p,q) = sum_{rs} (pr|qs) D(r,s)
    #                F      = H + J - 0.5 K.
    # (With D = 2 sum_i C_i C_i^T — the full electron density — the
    # Fock prefactors are 1 and -0.5. The alternative "F = H + 2J - K"
    # convention applies when D is the one-electron density P = D_full / 2.)
    einsums.einsum("pq <- pqrs ; rs", J, ERI, D)
    einsums.einsum("pq <- prqs ; rs", K, ERI, D)

    einsums.linalg.axpby(1.0, H, 0.0, F)           # F  = H
    einsums.linalg.axpy(1.0,  J, F)                # F += J
    einsums.linalg.axpy(-0.5, K, F)                # F -= 0.5 K

    # ── Electronic energy E_elec = 1/2 * sum_pq D(p,q) (H(p,q) + F(p,q)).
    # Build sum_HF = H + F, dot it with D into the scalar holder, halve.
    einsums.linalg.axpby(1.0, H, 0.0, sum_HF)
    einsums.linalg.axpy(1.0,  F, sum_HF)
    einsums.linalg.dot(E_elec, D, sum_HF)
    einsums.linalg.scale(0.5, E_elec)

    # ── Convergence metrics.
    einsums.linalg.axpby(1.0, E_elec, 0.0, delta_E)
    einsums.linalg.axpy(-1.0, E_old, delta_E)        # delta_E = E - E_old

    einsums.linalg.axpby(1.0, D, 0.0, diff_D)
    einsums.linalg.axpy(-1.0, D_old, diff_D)         # diff_D  = D - D_old
    einsums.linalg.norm(rms_D, einsums.linalg.Norm.FROBENIUS, diff_D)


# ──────────────────────────────────────────────────────────────────────────
# 5. Apply the default optimization passes and run.
# ──────────────────────────────────────────────────────────────────────────

pm = cg.default_pass_manager()
g.apply(pm)

print(f"\n--- Starting SCF (H2/STO-3G, R=1.4 bohr) ---\n")
g.execute()

E_elec_final = float(np.asarray(E_elec)[0])
E_total = E_elec_final + ENUC

print(f"\n--- SCF Complete ---")
print(f"Electronic energy : {E_elec_final:16.10f}")
print(f"Nuclear repulsion : {ENUC:16.10f}")
print(f"Total HF energy   : {E_total:16.10f}")
print(f"Reference (HF/STO-3G H2 at 1.4 bohr) : -1.117 Ha")
