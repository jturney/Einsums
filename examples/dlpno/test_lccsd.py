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
from dlpno.lccsd import LCCSDSolver
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


def test_the_merged_iteration_graph_is_the_phase_sequence_it_replaced(solved):
    """The fold's whole claim: one graph, and the same numbers to the last bit.

    The iteration used to be fourteen graphs replayed in sequence by the host,
    and each boundary ordered everything before it against everything after it.
    Merging them hands that ordering to the hazard scan, which orders only what
    actually shares a tensor, so the schedule is free to overlap phases the
    sequence could not. Nothing else changes - the same operations in the same
    emission order - so the residual has to come out identical, and identical
    here means ``array_equal`` and not a tolerance.

    Exactness holds only on the serial replay, so that is what this test uses.
    Under the OpenMP executor a node ALONE on its level threads its kernel
    internally while a node sharing a level runs serial inside the team, and a
    grouped GEMM picks its kernel per thread count - so which nodes share a
    level decides last-ulp digits, and the merged schedule shares levels the
    phase sequence never could. That is the documented threaded ~1-ulp class,
    not a reordering defect, and the fixture gates cover the threaded path; a
    serial replay of one graph IS the phase sequence in program order, which
    restores the exact claim this test exists to pin.

    Replaying the phases one graph at a time is exactly what the sequence was,
    which is what makes this a differential test of the fold and not of the
    equations: the amplitudes are held fixed, both spellings are evaluated at
    them, and any disagreement is a reordering the merge allowed and should not
    have.
    """
    solver = LCCSDSolver(solved.cc, verbose=False, use_executor=False)
    merged_s, merged_d = solver.evaluate_residuals()
    assert len(solver._graphs) == 1, "the iteration is supposed to be one graph"

    phases = [solver.phase_graph(label)
              for label, _emit, _threadable in solver.phases()]
    assert len(phases) == 14
    assert sum(g.num_nodes() for g in phases) == solver._graphs[0].num_nodes(), (
        "the merged graph does not hold exactly the phases' nodes")
    for g in phases:
        g.execute()
    seq_s, seq_d = solver.residuals()

    for i, (a, b) in enumerate(zip(merged_s, seq_s)):
        assert np.array_equal(a, b), (
            f"LMO {i}'s singles residual differs between the merged graph and "
            f"the phase sequence")
    for ij in merged_d:
        assert np.array_equal(merged_d[ij], seq_d[ij]), (
            f"pair {ij}'s doubles residual differs between the merged graph and "
            f"the phase sequence")


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


def test_the_combine_phase_adds_Rn_and_its_transpose_over_whole_stores():
    """Eq. 19's ``R[ij] += Rn[ij] + Rn[ji]^T``, as two operations per BUCKET.

    Neither half is emitted per pair: the untransposed one adds the whole store,
    and the transposed one scatters a transposing view of the store through the
    slot permutation the pair involution induces. That is sound only because the
    padding of both containers is identically zero and a weak pair's ``Rn`` block
    is never written, so both assertions are made here before the arithmetic is.

    Replaying the phase a second time over the ``Rn`` the first replay left in
    place is what makes the phase's own contribution checkable: the second replay
    starts from a known ``R`` and a known ``Rn``. The comparison is BIT FOR BIT,
    which is the whole claim - the whole-store form adds the same numbers to the
    same elements in the same order as the pair loop it replaces, so it cannot
    move a result at all. Note the reference below applies the two halves in the
    phase's order rather than pre-adding them: ``(R + Rn[ij]) + Rn[ji]^T`` is not
    ``R + (Rn[ij] + Rn[ji]^T)`` in floating point, and the phase computes the
    first.
    """
    reference, _ = load_reference(DIMER)
    cc = DLPNOCCSD(reference, Thresholds.preset("NORMAL", method="cc"),
                   verbose=False)
    cc.compute_energy(method="mp2")
    from dlpno.lccsd import LCCSDSolver

    solver = LCCSDSolver(cc, verbose=False)
    solver.evaluate_residuals()
    live = solver._plan.live
    assert len(cc.layout.bucket_dims) > 1, "the dimer should bucket into several"

    weak = [ij for ij in live if not solver.is_strong(ij)]
    assert weak, "the dimer is supposed to have weak pairs at NORMAL"
    Rn = {ij: cc.pair_block(solver.Rn_all, ij).copy() for ij in live}
    for ij in weak:
        assert not np.any(Rn[ij]), f"weak pair {ij} was written into Rn"
    for ij in live:
        n = cc.n_pno[ij]
        assert not np.any(Rn[ij][n:, :]) and not np.any(Rn[ij][:, n:]), (
            f"pair {ij}'s Rn padding is not zero")

    before = {ij: cc.pair_block(solver.R_all, ij).copy() for ij in live}
    # The iteration is one graph, so the phase is re-recorded on its own to be
    # replayed on its own; ``phase_graph`` is what that costs and why it exists.
    combine = solver.phase_graph("Eq. 19 combine")
    combine.execute()

    for ij in live:
        ji = int(cc.ij_to_ji[ij])
        want = (before[ij] + Rn[ij]) + Rn[ji].T
        assert np.array_equal(cc.pair_block(solver.R_all, ij), want), (
            f"pair {ij} did not receive Rn[ij] + Rn[ji]^T exactly")


def test_the_combine_phases_slot_map_is_the_pair_involution():
    """The permutation Eq. 19's transposed half scatters through.

    The whole-store form of ``R[ji] += Rn[ij]^T`` rests on pair ``ji`` sharing
    ``ij``'s bucket, which it does because the PNO transform mirrors the upper
    triangle onto the lower and the two therefore carry the same PNO count. What
    that buys is a permutation of the bucket's slots, and this checks it is one -
    a slot map that was not a bijection would drop some pairs' contribution and
    double others', which converges smoothly to the wrong answer.

    It also checks the map covers the pair family the per-pair emission named,
    so the two spellings describe the same sum and neither the plan's pair list
    nor the permutation can drift from the other.
    """
    reference, _ = load_reference(DIMER)
    cc = DLPNOCCSD(reference, Thresholds.preset("NORMAL", method="cc"),
                   verbose=False)
    cc.compute_energy(method="mp2")
    from dlpno.lccsd import LCCSDSolver

    plan = LCCSDSolver(cc, verbose=False).plan()
    layout = cc.layout
    assert len(plan.r_combine_perm) == len(layout.bucket_dims)

    covered = set()
    for b, perm in enumerate(plan.r_combine_perm):
        members = layout.bucket_members[b]
        assert sorted(perm) == list(range(len(members))), (
            f"bucket {b}'s slot map is not a permutation of its slots")
        for t, ij in enumerate(members):
            ji = int(cc.ij_to_ji[ij])
            assert layout.bucket_of[ji] == b, (
                f"pair {ij} and its transpose {ji} are in different buckets")
            assert members[perm[t]] == ji
            covered.add((ji, ij))
    for ij, _i, _j, ji in plan.r_combine:
        assert (ij, ji) in covered, (
            f"the strong pair {ij} takes no Rn[{ji}]^T from any bucket's slot map")


# => the batches <= #


def _all_batches(solver):
    """Every grouped GEMM the solver will emit, with the family it belongs to."""
    return [(name, index, batch)
            for name, group in sorted(solver._bat.items())
            for index, batch in enumerate(group) if batch.a]


def _batched_solver(path):
    """A planned, allocated solver on ``path``, with nothing captured yet."""
    reference, _ = load_reference(path)
    cc = DLPNOCCSD(reference, Thresholds.preset("NORMAL", method="cc"),
                   verbose=False)
    cc.compute_energy(method="mp2")
    from dlpno.lccsd import LCCSDSolver

    solver = LCCSDSolver(cc, verbose=False)
    solver.plan()
    solver._allocate()
    return solver


def test_no_grouped_batch_writes_one_destination_twice():
    """The invariant every batch in ``lccsd.py`` rests on, checked by writing.

    The members of one grouped GEMM may run CONCURRENTLY inside the node, and the
    graph's hazard scan cannot see inside a node, so two members that write the
    same memory are a race that no level checker and no serial replay will ever
    report. The residual's families accumulate thousands of terms into a few
    hundred blocks, so getting this wrong is one line away at all times: batch a
    walk over records instead of over one STEP of it, and every pair's chain
    races itself.

    The check is not "are the destination objects distinct" - two views can be
    distinct objects and the same memory. Each destination is filled with a
    marker through numpy and then read back, which catches any overlap however it
    arises, including a padded slice of a store and a reshaped slice of a flat
    one.

    The dimer at NORMAL is the fixture because it is the smallest case with
    several PNO buckets and with weak pairs, so the store-backed accumulators and
    the packed ones are both exercised.
    """
    from dlpno import tensors as ten

    solver = _batched_solver(DIMER)
    batches = _all_batches(solver)
    assert batches, "the solver built no batches at all"

    for name, index, batch in batches:
        for slot, dest in enumerate(batch.c):
            ten.view(dest)[...] = float(slot + 1)
        for slot, dest in enumerate(batch.c):
            assert np.all(ten.view(dest) == float(slot + 1)), (
                f"{name}[{index}] member {slot} of {len(batch.c)} had its "
                f"destination overwritten by another member of the same batch")


def test_no_grouped_batch_touches_the_shared_scratch_pool():
    """The pooled rank-3 scratch may not appear in a batch, on either side.

    ``_shared`` hands out one of about ten buffers per role, round-robin over a
    phase's records, and relies on the hazard scan to serialize the records that
    land on the same buffer. A batch of hundreds of members over ten buffers
    would be that serialization removed: dozens of members writing one buffer at
    once, inside a node, invisibly. So the rule is that no batched family uses
    the pool, and the emitters that do use it are the einsum ones, which are
    still a pair at a time.
    """
    solver = _batched_solver(WATER)
    pooled = {id(view) for key, view in solver._view_cache.items()
              if isinstance(key, tuple) and key and key[0] == "shared"}
    assert pooled, "the solver handed out no pooled scratch, so this proves nothing"

    for name, index, batch in _all_batches(solver):
        for role, operands in (("a", batch.a), ("b", batch.b), ("c", batch.c)):
            for slot, operand in enumerate(operands):
                assert id(operand) not in pooled, (
                    f"{name}[{index}] takes pooled scratch as operand {role} of "
                    f"member {slot}; a batch cannot share a pool buffer")


def test_stepping_a_walk_preserves_each_owners_record_order():
    """``_by_step`` may reorder owners against each other and nothing else.

    That is the whole numerical claim of the batched emitters: a neighbour walk's
    records accumulate into their owner's intermediate, transposing the walk
    makes each step a legal batch, and the sum each owner computes still runs in
    the order the unbatched emitter ran it. If the second half were not true the
    energies would move, and they are measured not to.
    """
    from dlpno.lccsd import _by_step

    records = [("a", 0), ("a", 1), ("b", 0), ("a", 2), ("c", 0), ("b", 1)]
    steps = _by_step(records, 0)
    assert [len(s) for s in steps] == [3, 2, 1]
    assert sum(len(s) for s in steps) == len(records)

    flat = [r for step in steps for r in step]
    for owner in ("a", "b", "c"):
        assert ([r for r in flat if r[0] == owner]
                == [r for r in records if r[0] == owner]), (
            f"owner {owner}'s records were reordered against each other")
        for step in steps:
            assert sum(1 for r in step if r[0] == owner) <= 1, (
                f"owner {owner} appears twice in one step, so the step's members "
                f"would race on that owner's destination")


def test_the_residual_walks_are_emitted_as_batches_and_not_per_record():
    """The node count of the hot phases, against the record count they walk.

    A regression that reverted a family to one dispatch per record would keep
    every energy exactly right - the unbatched form is bit-identical, which is
    how the batching was checked - and cost the iteration its whole margin. So
    the shape of the emission is asserted rather than inferred: each of the four
    fattest phases replays far fewer nodes than its work list has records.

    Counted per phase through ``phase_graph``, since the iteration is one graph:
    a per-record regression in one phase would be invisible in the total, which
    is the number the fold left behind.
    """
    solver = _batched_solver(DIMER)
    nodes = {label: solver.phase_graph(label).num_nodes()
             for label, _emit, _threadable in solver.phases()}
    plan = solver._plan

    # Per phase: the records it walks, and the ceiling its node count must clear.
    # The ceilings are generous - what they exclude is one node per record.
    for phase, records in (("Eq. 83 gamma", len(plan.gamma_tail)),
                           ("Eq. 84 delta", len(plan.delta_tail)),
                           ("Eq. 75-77, 80, 85", len(plan.r_sym_kl)),
                           ("Eq. 78-79, 81",
                            sum(len(r[3]) + len(r[4]) for r in plan.r_non)),
                           ("Eq. 70 projections", len(plan.projections))):
        assert records > 50, f"{phase} walks only {records} records here"
        assert nodes[phase] < records / 2, (
            f"{phase} replays {nodes[phase]} nodes over {records} records, "
            f"which is per-record emission rather than batched")


def test_the_scalar_families_are_emitted_as_runs_and_not_per_record():
    """The dot-and-accumulate families, against the record counts they walk.

    The same regression this file already guards for the batched GEMMs, in the
    family that outnumbered them. A revert to one ``la.dot`` and one ``la.axpby``
    per record would keep every energy exactly right - the runs are bit-identical
    to the loop by construction, which is why they were adoptable at all - and
    hand back thousands of nodes per iteration. So the shape of the emission is
    asserted rather than inferred.

    Each phase's ceiling is generous: what it excludes is one node per record.
    """
    solver = _batched_solver(DIMER)
    plan = solver._plan
    runs = solver._run
    assert runs, "the solver built no runs at all"

    # Per run: the work list whose records it must hold one entry per term of.
    for name, records, per_record in (("fock_occ_dot", plan.fock_occupied, 2),
                                      ("fock_occ_acc", plan.fock_occupied, 2),
                                      ("fock_kj_dot", plan.fock_kj, 1),
                                      ("fock_kj_acc", plan.fock_kj, 1),
                                      ("fkj2_dot", plan.fkj2, 1),
                                      ("fkj2_acc", plan.fkj2, 1),
                                      ("r_ia_a2_dot", plan.r_ia_a2, 1),
                                      ("energy_dot", plan.energy, 1)):
        assert len(runs[name]) == per_record * len(records), (
            f"run {name} holds {len(runs[name])} entries over "
            f"{len(records)} records")

    nodes = {label: solver.phase_graph(label).num_nodes()
             for label, _emit, _threadable in solver.phases()}
    for phase, records in (("Eq. 98 F_bar (occ)", len(plan.fock_occupied)),
                           ("Eq. 86 F''", len(plan.fkj2)),
                           ("Eq. 87-90 R_ia", len(plan.r_ia_a2))):
        assert records > 50, f"{phase} walks only {records} records here"
        assert nodes[phase] < records / 2, (
            f"{phase} replays {nodes[phase]} nodes over {records} records, "
            f"which is per-record emission rather than grouped")


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
