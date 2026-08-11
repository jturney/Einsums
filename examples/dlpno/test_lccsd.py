#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------
"""The captured local CCSD iteration, checked without psi4.

``dlpno/lccsd.py`` is thirty-odd contractions recorded into ComputeGraphs and
replayed, and the two things that can go wrong with it are different in kind.

*The equations can be wrong.* The check for that is not the converged energy -
two implementations reach the same fixed point from different DIIS trajectories,
so agreement there is bounded by the convergence tolerance and a term that is
small near the solution hides in it. The check is the RESIDUAL, evaluated by
both the port and an independent numpy oracle at the same arbitrary amplitudes,
including unphysically large ones. A defective term cannot cancel at a point
that is nobody's solution.

*The capture can be wrong while the equations are right.* A captured iteration
carries state between replays that an eager one does not: intermediates that
must be cleared, scratch shared between pairs, and views that have to outlive
the graph. Every one of those failure modes shows up as a replay whose answer
depends on what ran before it, so the test for them is idempotence - replaying
at fixed amplitudes twice has to give the same answer bit for bit - and that is
a test the eager implementation had no need of and no way to fail.

    PYTHONPATH=/path/to/Einsums/build/lib python -m pytest examples/dlpno/test_lccsd.py
"""

import os
import sys
from dataclasses import replace

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from check_ccsd_defect import Bridge, port_residuals, probe_amplitudes
from dlpno.canonical_ccsd import CanonicalCCSD
from dlpno.ccsd import DLPNOCCSD
from dlpno.reference_io import load_reference
from dlpno.thresholds import Thresholds

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
WATER = os.path.join(FIXTURES, "water-ccpvdz.npz")
DIMER = os.path.join(FIXTURES, "water-dimer-ccpvdz.npz")

#: Canonical DF-CCSD for water/cc-pVDZ, all-electron, cc-pvdz-ri. Four
#: independent calculations land here: ``dlpno/canonical_ccsd.py`` in both
#: occupied bases, psi4's ``dfocc`` DF-CCSD, and this port untruncated. See
#: ``test_canonical_ccsd.py``, which pins the oracle before anything is judged
#: against it.
CANONICAL_DF_CCSD = -0.213602181936


class _Solved:
    """One untruncated solve, plus everything derived from it before anything
    disturbs it.

    The converged amplitudes are snapshotted here rather than read back later
    because evaluating the residual at a probe point WRITES the amplitude
    stores - that is how the probe gets in - so after any such test the stores
    hold the last probe and not the solution. Sharing one solve across the
    module is worth a few seconds; sharing it without this snapshot makes the
    tests order-dependent.
    """

    def __init__(self, cc):
        self.cc = cc
        self.solver = cc.lccsd
        self.e_lccsd = cc.e_lccsd
        self.e_corr = cc.e_corr
        self.oracle = CanonicalCCSD(cc.ref, occupied="localized")
        self.bridge = Bridge(cc, self.oracle.basis)
        self.bridge.check()
        self.t1, self.t2 = self.bridge.to_oracle(self.solver)


@pytest.fixture(scope="module")
def solved():
    """The port run untruncated, one decade tighter than its default.

    Tighter because this is the one gate whose tolerance is set by convergence
    rather than by arithmetic: at the default ``r_convergence`` of 1e-8 the port
    stops 3.6e-10 from the oracle, and one decade takes that to 1e-12 and then
    plateaus. Gating at the default would be gating on where the DIIS trajectory
    happened to stop.
    """
    reference, _ = load_reference(WATER)
    cut = Thresholds.untruncated()
    cut = replace(cut, r_convergence=cut.r_convergence * 0.1,
                  e_convergence=cut.e_convergence * 0.1, maxiter=100)
    cc = DLPNOCCSD(reference, cut, verbose=False)
    cc.compute_energy()
    return _Solved(cc)


# => the equations <= #


def test_untruncated_local_ccsd_is_canonical_ccsd(solved):
    """With every truncation off, the local method IS the canonical one.

    Each pair's PNO space is then a full-rank rotation of the virtual space, so
    the two calculations are the same calculation in two bases. This is the
    check that pins the port down; nothing later is trustworthy without it.
    """
    assert solved.e_corr == pytest.approx(CANONICAL_DF_CCSD, abs=1e-11)


def test_residuals_match_the_canonical_oracle_at_probe_amplitudes(solved):
    """The sharp test: both sides' residuals at the SAME arbitrary amplitudes.

    Near a common solution a defective term is small because the amplitudes are
    nearly right, not because the term is right. At a probe point it is not
    hidden, and the probes here run to amplitudes an order of magnitude larger
    than the physical ones, where the residual itself is of order ten.
    """
    oracle, bridge = solved.oracle, solved.bridge

    worst = 0.0
    probes = [("MP2 guess",
               np.zeros((oracle.nocc, oracle.nvir)),
               oracle.basis.block("oovv") / oracle.D_ijab)]
    for scale in (1e-3, 1e-2, 1e-1):
        probes.append((f"random {scale:.0e}", *probe_amplitudes(oracle.basis, 5, scale)))

    for _label, t1, t2 in probes:
        want1, want2 = oracle.residuals(t1, t2)
        got1, got2 = port_residuals(solved.solver, bridge, t1, t2)
        d1 = np.abs(want1 - got1).max() / max(np.abs(want1).max(), 1e-30)
        d2 = np.abs(want2 - got2).max() / max(np.abs(want2).max(), 1e-30)
        worst = max(worst, d1, d2)
    assert worst < 1e-10, f"port and oracle residuals differ by {worst:.3e} relative"


# => the capture <= #


def test_replaying_the_residual_twice_gives_the_same_answer(solved):
    """A replay must depend on the amplitudes and on nothing else.

    Everything a captured iteration accumulates into has to be cleared at the
    top of the phase that fills it, and every buffer shared between pairs has to
    be fully written before it is read. Both failures leave the second replay
    disagreeing with the first, and neither is visible in a converged energy -
    an iteration that quietly adds last iteration's Eq. 77 to this one's still
    converges, to the wrong place, smoothly.
    """
    solver = solved.solver
    first_s, first_d = solver.evaluate_residuals()
    second_s, second_d = solver.evaluate_residuals()
    for a, b in zip(first_s, second_s):
        assert np.array_equal(a, b), "singles residual changed on replay"
    for ij in first_d:
        assert np.array_equal(first_d[ij], second_d[ij]), (
            f"doubles residual for pair {ij} changed on replay")


def test_the_energy_expression_reproduces_the_oracle_at_the_ports_amplitudes(
        solved):
    """Eq. 45 as captured, against the oracle's own energy expression.

    Separates a wrong energy from wrong amplitudes: the oracle is handed the
    port's converged amplitudes and asked what they are worth. A disagreement
    here is in Eq. 45 - the ``t_i^a t_j^b`` term above all, which is the only
    place the singles enter the energy - and not in the residual.
    """
    assert solved.oracle.energy(solved.t1, solved.t2) == pytest.approx(
        solved.e_lccsd, abs=1e-11)


def test_weak_pairs_never_enter_the_doubles_residual():
    """A weak pair's residual is exactly zero and its amplitudes do not move.

    Weak pairs keep their converged LMP2 doubles and contribute ``de_weak``;
    the residual skips them. "Exactly" is the point - a term that leaked into a
    weak pair would move its amplitude by something tiny and be booked twice,
    once as an estimate and once as a solve.
    """
    reference, _ = load_reference(DIMER)
    cc = DLPNOCCSD(reference, Thresholds.preset("NORMAL", method="cc"),
                   verbose=False)
    cc.compute_energy()
    weak = [ij for ij in range(cc.n_lmo_pairs)
            if cc.n_pno[ij] and not cc.lccsd.is_strong(ij)]
    assert weak, "the dimer is supposed to have weak pairs at NORMAL"

    before = {ij: cc.lccsd.T2(ij).copy() for ij in weak}
    _, R = cc.lccsd.evaluate_residuals()
    for ij in weak:
        assert not np.any(R[ij]), f"weak pair {ij} has a nonzero residual"
        assert np.array_equal(before[ij], cc.lccsd.T2(ij))


def test_the_plan_classifies_every_work_list_it_records():
    """Every hot work list carries shape classes, which is what M8 groups on.

    The classes are not read by the emitters yet, so nothing else would notice
    a work list that stopped being classified - and the point of deriving them
    at M4 is that the batched emitter can be written against them without
    re-deriving a single index map.
    """
    reference, _ = load_reference(WATER)
    cc = DLPNOCCSD(reference, Thresholds.preset("NORMAL", method="cc"),
                   verbose=False)
    cc.compute_energy(method="mp2")
    from dlpno.lccsd import LCCSDSolver

    plan = LCCSDSolver(cc, verbose=False).plan()
    report = plan.class_report()
    assert report, "the plan classified nothing"
    for name, (records, classes) in report.items():
        assert records > 0, f"{name} is empty"
        assert 0 < classes <= records, (
            f"{name} has {classes} shape classes over {records} records")
