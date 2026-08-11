#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------
"""Locate a deviation between the local CCSD port and canonical CCSD.

With every truncation off, local CCSD is an exact rotation of canonical CCSD:
each pair's PNO space is a full-rank rotation of the virtual space, so the two
calculations are the same calculation in two bases and every amplitude of one is
a rotation of an amplitude of the other. That makes a much sharper comparison
available than comparing two energies, and this script makes it:

* run the port untruncated, and the oracle (``dlpno/canonical_ccsd.py``) in the
  port's own localized occupied basis;
* build the rotation between the two virtual bases and carry each side's
  converged amplitudes into the other's;
* evaluate each side's RESIDUAL at the other side's converged amplitudes. A
  residual is zero at its own solution, so what comes back is a direct measure
  of how far apart the two sets of equations are, in the units the equations
  are written in rather than in the units of an energy several contractions
  downstream;
* evaluate BOTH residuals at the same third set of amplitudes, which is
  sharper still. Near a common solution a defective term is small because the
  amplitudes are nearly right, not because the term is nearly right; at a probe
  point it is not hidden. The probes here are the MP2 guess and a random
  amplitude set, and the random one is swept over a scale factor so that the
  ORDER in T of any disagreement can be read off its slope;
* bisect on the singles, by clamping ``T1 = 0`` on both sides, which separates
  the T1-dressing terms of the Jiang formulation from the pure doubles ones in
  one run.

The port's singles are clamped by monkeypatching
``LCCSDSolver.compute_R_ia`` to return zero rather than by editing it: nothing
under ``dlpno/`` is touched by this script.

    PYTHONPATH=/path/to/Einsums/build/lib:. python examples/dlpno/check_ccsd_defect.py

**What it found.** The two sides agree to 1e-13 at every probe, including random
amplitudes an order of magnitude larger than the physical ones, where the
residual itself is of order 10. That is machine precision on a relative scale,
so the port's equations and canonical CCSD are the same equations and there is
no defect to localize.

The 2.92e-8 the port was thought to carry is in the number it was being compared
to. psi4's fnocc DF-CCSD uses TWO auxiliary bases in one calculation: the
amplitude integrals come from ``DF_BASIS_CC`` (RIFIT, so cc-pvdz-ri here) while
``DFCoupledCluster::T1Fock`` rebuilds the T1-dressed Fock matrix from the SCF's
saved three-index integrals over ``DF_BASIS_SCF`` (JKFIT). At ``T1 = 0`` the two
descriptions of the Fock coincide, which is why psi4's DF-MP2 and the oracle's
agree to 1e-12; the dressed part does not, and the inconsistency is worth
2.92e-8 in the correlation energy. Setting ``df_basis_cc cc-pvdz-jkfit`` so that
one basis does both makes psi4 and this oracle agree to 4.3e-13, which is what
identifies the mechanism. See the report in the commit or ``DESIGN-ccsd.md``.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import dlpno.ccsd as ccsd_module
from dlpno.canonical_ccsd import CanonicalCCSD
from dlpno.ccsd import DLPNOCCSD
from dlpno.reference_io import load_reference
from dlpno.thresholds import Thresholds

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")

#: psi4's fnocc DF-CCSD for water/cc-pVDZ, ``cc_type df``, cc-pvdz-ri,
#: all-electron. The number this whole exercise started from.
PSI4_DF_CCSD = -0.213602211142


# -- the bridge between the two bases --------------------------------------


class Bridge:
    """The rotation between the oracle's virtuals and each pair's PNO basis.

    Untruncated, pair ``ij``'s PNOs span the whole virtual space, so

        C_pno[ij] = C_pao[:, paos_ij] X_pno[ij]

    over AOs is one orthonormal basis of it and the oracle's ``C_vir`` is
    another. The rotation between them is therefore

        U[ij] = C_vir^T S C_pno[ij]

    with ``S`` the AO overlap, which the Reference carries. That is the route
    taken here: through the AO metric, rather than through any shared
    intermediate basis, because it needs nothing but the two coefficient
    matrices and it is checkable - ``U`` is orthogonal to machine precision or
    the premise (that the PNO space is the full virtual space) is false, and
    :meth:`check` asserts exactly that.
    """

    def __init__(self, cc, basis, tol=1e-12):
        S = np.asarray(cc.ref.S)
        C_pao = np.asarray(cc.C_pao)
        self.cc = cc
        self.basis = basis
        self.U = []
        self.worst = 0.0
        for ij in range(cc.n_lmo_pairs):
            C_pno = C_pao[:, cc.lmopair_to_paos[ij]] @ np.asarray(cc.X_pno[ij])
            U = basis.C_vir.T @ S @ C_pno
            self.U.append(U)
            self.worst = max(self.worst,
                             np.abs(U @ U.T - np.eye(U.shape[0])).max())
        self.tol = tol

    def check(self):
        if self.worst > self.tol:
            raise RuntimeError(
                f"the PNO bases are not full-rank rotations of the virtual space "
                f"(worst deviation from orthogonality {self.worst:.3e}); this "
                f"comparison is only meaningful untruncated")
        return self.worst

    # -- amplitudes ----------------------------------------------------------

    def to_oracle(self, solver):
        """The port's converged amplitudes in the oracle's basis."""
        cc = self.cc
        nocc, nvir = self.basis.nocc, self.basis.nvir
        t1 = np.zeros((nocc, nvir))
        t2 = np.zeros((nocc, nocc, nvir, nvir))
        for i in range(nocc):
            ii = int(cc.i_j_to_ij[i, i])
            t1[i] = (self.U[ii] @ solver.T_ia[i]).ravel()
        for ij, (i, j) in enumerate(cc.ij_to_i_j):
            t2[i, j] = self.U[ij] @ solver.T2(ij) @ self.U[ij].T
        return t1, t2

    def to_port(self, solver, t1, t2):
        """Write oracle-basis amplitudes into the port's stores."""
        cc = self.cc
        for i in range(self.basis.nocc):
            ii = int(cc.i_j_to_ij[i, i])
            solver.T_ia[i][...] = (self.U[ii].T @ t1[i]).reshape(-1, 1)
        for ij, (i, j) in enumerate(cc.ij_to_i_j):
            block = self.U[ij].T @ t2[i, j] @ self.U[ij]
            solver.T2(ij)[...] = block
            solver.Tt2(ij)[...] = 2.0 * block - block.T

    def residual_to_oracle(self, R_ia, R_iajb):
        """A port residual in the oracle's basis, laid out like the oracle's."""
        cc = self.cc
        nocc, nvir = self.basis.nocc, self.basis.nvir
        r1 = np.zeros((nocc, nvir))
        r2 = np.zeros((nocc, nocc, nvir, nvir))
        for i in range(nocc):
            ii = int(cc.i_j_to_ij[i, i])
            r1[i] = (self.U[ii] @ R_ia[i]).ravel()
        for ij, (i, j) in enumerate(cc.ij_to_i_j):
            r2[i, j] = self.U[ij] @ R_iajb[ij] @ self.U[ij].T
        return r1, r2


def port_residuals(solver, bridge, t1, t2):
    """The port's residual at the amplitudes ``(t1, t2)``, in the oracle's basis.

    Everything before the Jacobi step is a pure function of the amplitudes, so
    injecting amplitudes into the stores and replaying the residual phases
    evaluates the port's equations at a point of our choosing. That the port is
    captured rather than eager changes nothing here: the graphs read the stores
    the bridge just wrote.
    """
    bridge.to_port(solver, t1, t2)
    return bridge.residual_to_oracle(*solver.evaluate_residuals())


# -- reporting -------------------------------------------------------------


def norms(label, r1, r2):
    print(f"    {label:<34} |R1|max = {np.abs(r1).max():.3e}  "
          f"|R1|rms = {np.sqrt(np.mean(r1 ** 2)):.3e}   "
          f"|R2|max = {np.abs(r2).max():.3e}  "
          f"|R2|rms = {np.sqrt(np.mean(r2 ** 2)):.3e}")


def per_pair(r2, ij_to_i_j, limit=8):
    """The largest per-pair doubles norms, diagonal pairs flagged."""
    rows = []
    for (i, j) in ij_to_i_j:
        rows.append((np.abs(r2[i, j]).max(), i, j))
    rows.sort(reverse=True)
    print(f"    largest per-pair |R2|max (of {len(rows)} pairs):")
    for value, i, j in rows[:limit]:
        kind = "diagonal" if i == j else "off-diagonal"
        print(f"      ({i},{j}) {kind:<12} {value:.3e}")


# -- the runs ---------------------------------------------------------------


def run_port(reference, singles=True, verbose=False):
    """The port, untruncated, optionally with its singles clamped at zero.

    The solver has a ``clamp_singles`` flag - with a zero singles residual the
    singles never leave their zero starting guess, and DIIS extrapolates a
    subspace whose singles block is identically zero, so they stay zero through
    the whole iteration. Reaching it is a patch of the name ``dlpno.ccsd`` looks
    the solver up under rather than an edit: nothing under ``dlpno/`` is touched
    by this script.
    """
    original = ccsd_module.LCCSDSolver
    if not singles:
        ccsd_module.LCCSDSolver = (
            lambda cc, **kw: original(cc, clamp_singles=True, **kw))
    try:
        cc = DLPNOCCSD(reference, Thresholds.untruncated(), verbose=verbose)
        cc.compute_energy()
        return cc
    finally:
        ccsd_module.LCCSDSolver = original


def probe_amplitudes(basis, seed, scale):
    """A random amplitude pair with the closed-shell pair symmetry.

    ``t2[i,j,a,b] = t2[j,i,b,a]`` is assumed by both sets of equations - the
    port stores pair ``ji`` as the transpose of pair ``ij`` - so a probe without
    it would measure the two implementations' handling of an amplitude neither
    can represent.
    """
    rng = np.random.default_rng(seed)
    t1 = scale * rng.standard_normal((basis.nocc, basis.nvir))
    t2 = scale * rng.standard_normal((basis.nocc, basis.nocc, basis.nvir, basis.nvir))
    return t1, 0.5 * (t2 + t2.transpose(1, 0, 3, 2))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("fixture", nargs="?",
                        default=os.path.join(FIXTURES, "water-ccpvdz.npz"),
                        help=".npz written by dump_reference.py")
    parser.add_argument("--verbose", action="store_true",
                        help="let the port and the oracle print their iterations")
    args = parser.parse_args()

    reference, extras = load_reference(args.fixture)
    print(f"fixture: {args.fixture}")
    print(f"  nbf = {reference.nbf}, naux = {reference.naux}, "
          f"active occ = {reference.naocc}")

    # -- the two calculations ----------------------------------------------
    print("\n=== converged energies, untruncated ===")
    cc = run_port(reference, verbose=args.verbose)
    solver = cc.lccsd
    oracle = CanonicalCCSD(reference, occupied="localized")
    e_oracle = oracle.solve(verbose=args.verbose)
    canonical = CanonicalCCSD(reference, occupied="canonical")
    e_canonical = canonical.solve(verbose=args.verbose)

    mp2 = extras["energies"].get("psi4_df_mp2")
    print(f"    port, untruncated                  {cc.e_corr:.12f}  "
          f"({solver.n_iterations} iterations)")
    print(f"    oracle, localized occupied         {e_oracle:.12f}  "
          f"({oracle.n_iterations} iterations)")
    print(f"    oracle, canonical occupied         {e_canonical:.12f}  "
          f"({canonical.n_iterations} iterations)")
    print(f"    psi4 fnocc DF-CCSD (recorded)      {PSI4_DF_CCSD:.12f}")
    print(f"    oracle MP2 guess                   {canonical.mp2_energy():.12f}"
          + (f"   vs psi4 DF-MP2 {mp2:.12f}" if mp2 is not None else ""))
    print()
    print(f"    port - oracle(localized)           {cc.e_corr - e_oracle:+.3e}")
    print(f"    oracle(localized) - oracle(canon)  {e_oracle - e_canonical:+.3e}")
    print(f"    port - psi4                        {cc.e_corr - PSI4_DF_CCSD:+.3e}")
    print(f"    oracle - psi4                      {e_canonical - PSI4_DF_CCSD:+.3e}")

    # -- the bridge --------------------------------------------------------
    bridge = Bridge(cc, oracle.basis)
    print("\n=== the rotation between the two virtual bases ===")
    print(f"    route: U(ij) = C_vir^T S C_pno(ij), with S from the Reference")
    print(f"    worst |U U^T - 1| over {cc.n_lmo_pairs} pairs = {bridge.check():.3e}")

    t1_port, t2_port = bridge.to_oracle(solver)
    print(f"    oracle energy at the PORT's amplitudes    "
          f"{oracle.energy(t1_port, t2_port):.12f}")
    print(f"    port energy (its own expression)          {cc.e_lccsd:.12f}")

    # -- residuals at each other's solutions -------------------------------
    print("\n=== each side's residual at the other side's converged amplitudes ===")
    r1, r2 = oracle.residuals(t1_port, t2_port)
    norms("oracle residual at port amplitudes", r1, r2)
    p1, p2 = port_residuals(solver, bridge, oracle.t1, oracle.t2)
    norms("port residual at oracle amplitudes", p1, p2)
    o1, o2 = oracle.residuals(oracle.t1, oracle.t2)
    norms("oracle residual at its own       ", o1, o2)
    per_pair(r2, cc.ij_to_i_j)

    # -- both residuals at the same probe amplitudes -----------------------
    print("\n=== both residuals at the SAME amplitudes (the sharp test) ===")
    print("    A defect that cancels near the common solution does not cancel here.")
    probes = [("MP2 guess, T1 = 0",
               np.zeros((oracle.nocc, oracle.nvir)),
               oracle.basis.block("oovv") / oracle.D_ijab)]
    for scale in (1e-3, 1e-2, 1e-1):
        t1, t2 = probe_amplitudes(oracle.basis, 5, scale)
        probes.append((f"random, scale {scale:.0e}", t1, t2))
    probes.append(("random, T1 = 0, scale 1e-1",
                   np.zeros((oracle.nocc, oracle.nvir)),
                   probe_amplitudes(oracle.basis, 5, 1e-1)[1]))
    probes.append(("random, T2 = 0, scale 1e-1",
                   probe_amplitudes(oracle.basis, 5, 1e-1)[0],
                   np.zeros((oracle.nocc, oracle.nocc, oracle.nvir, oracle.nvir))))

    worst = 0.0
    for label, t1, t2 in probes:
        a1, a2 = oracle.residuals(t1, t2)
        b1, b2 = port_residuals(solver, bridge, t1, t2)
        d1 = np.abs(a1 - b1).max() / max(np.abs(a1).max(), 1e-30)
        d2 = np.abs(a2 - b2).max() / max(np.abs(a2).max(), 1e-30)
        worst = max(worst, d1, d2)
        print(f"    {label:<28} "
              f"d|R1|max = {np.abs(a1 - b1).max():.3e} (of {np.abs(a1).max():.3e})   "
              f"d|R2|max = {np.abs(a2 - b2).max():.3e} (of {np.abs(a2).max():.3e})")
    print(f"    worst RELATIVE disagreement over all probes: {worst:.3e}")

    # -- the singles bisection ---------------------------------------------
    print("\n=== bisection: T1 clamped to zero on both sides ===")
    cc0 = run_port(reference, singles=False, verbose=args.verbose)
    oracle0 = CanonicalCCSD(reference, occupied="localized")
    e0 = oracle0.solve(singles=False, verbose=args.verbose)
    print(f"    port, doubles only                 {cc0.e_corr:.12f}")
    print(f"    oracle, doubles only               {e0:.12f}")
    print(f"    difference                         {cc0.e_corr - e0:+.3e}")
    print(f"    singles contribution, port         {cc.e_corr - cc0.e_corr:+.9f}")
    print(f"    singles contribution, oracle       {e_oracle - e0:+.9f}")

    bridge0 = Bridge(cc0, oracle0.basis)
    bridge0.check()
    t1_0, t2_0 = bridge0.to_oracle(cc0.lccsd)
    d1, d2 = oracle0.residuals(t1_0, t2_0)
    # R1 is large here by construction and says nothing: with the singles
    # clamped, neither side is solving the singles equation, so only R2 is a
    # statement about agreement.
    norms("oracle residual at port doubles   ", d1, d2)

    # -- verdict -----------------------------------------------------------
    print("\n=== verdict ===")
    if worst < 1e-10:
        print(f"    The two sets of equations agree to {worst:.1e} RELATIVE at every")
        print("    probe amplitude, so they are the same equations and there is no")
        print("    term-level defect in the port.")
        print(f"    Both differ from psi4's fnocc DF-CCSD by {cc.e_corr - PSI4_DF_CCSD:+.3e},")
        print("    which is psi4's own two-auxiliary-basis inconsistency: the amplitude")
        print("    integrals come from DF_BASIS_CC (RIFIT) and the T1-dressed Fock from")
        print("    DF_BASIS_SCF (JKFIT). Run psi4 with 'df_basis_cc cc-pvdz-jkfit' and")
        print("    the two agree to 4e-13. See the module docstring.")
        return 0
    print(f"    The two residuals DISAGREE by {worst:.3e} relative at a probe point.")
    print("    Read the per-probe lines above: which family, and whether it survives")
    print("    T1 = 0, is the localization.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
