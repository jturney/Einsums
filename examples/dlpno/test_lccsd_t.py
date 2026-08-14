#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------
"""The iterative triples, checked without psi4.

``dlpno/lccsd_t.py`` restores the one thing (T0) drops: the off-diagonal
occupied Fock coupling between triplets, Eq. 111's three ``- f_il T_ljk`` sums.
That single difference is what this suite is about, and it makes the gate
sharper than (T0)'s rather than looser.

**(T) is invariant to a rotation of the occupied orbitals and (T0) is not.**
``test_canonical_triples.py`` asserts the second half of that, because it is
what forced the (T0) gate to be run in two bases. Here it pays off: untruncated
in the port's OWN localized basis, (T) must reproduce canonical DF-CCSD(T)
exactly, with no canonicalized rerun needed. The number it has to hit is the one
(T0) misses by 1.5e-4 Eh, so a port that quietly computed (T0) here would fail by
five percent of the correction rather than by something subtle.

That also means this milestone needed no new oracle. The value was already
pinned by two independent implementations in ``canonical_triples.py``.

    PYTHONPATH=/path/to/Einsums/build/lib python -m pytest examples/dlpno/test_lccsd_t.py
"""

import os
import sys
from dataclasses import replace

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import einsums
import einsums.graph as cg
from dlpno import tensors as ten
from dlpno.lccsd_t import (_ACCUMULATE_FLAGS, _DIRECTION, _ROTATE_FLAGS,
                           LCCSDT, chain_of, permuter_spec, rotation_stages)
from dlpno.reference_io import load_reference
from dlpno.thresholds import Thresholds
from dlpno.triples import DLPNOCCSDT

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
WATER = os.path.join(FIXTURES, "water-ccpvdz.npz")
DIMER = os.path.join(FIXTURES, "water-dimer-ccpvdz.npz")

#: Canonical DF-CCSD(T) for water/cc-pVDZ, all-electron, cc-pvdz-ri. Pinned in
#: ``test_canonical_triples.py`` by two independent implementations agreeing to
#: 3e-18, and reached here by the iterative (T) in the LOCALIZED basis.
CANONICAL_DF_T = -0.003072147091

#: What (T0) gives in the localized basis, from ``test_lccsd_t0.py``. The
#: distance between the two is the whole of the iterative correction.
LOCALIZED_T0 = -0.002917162965


@pytest.fixture(scope="module")
def solved():
    """One untruncated iterative (T), in the port's own localized basis."""
    reference, _ = load_reference(WATER)
    cut = Thresholds.untruncated()
    cut = replace(cut, r_convergence=cut.r_convergence * 0.1,
                  e_convergence=cut.e_convergence * 0.1, maxiter=100)
    cc = DLPNOCCSDT(reference, cut, verbose=False)
    cc.compute_energy()
    return cc


def test_untruncated_iterative_t_is_canonical_df_ccsd_t(solved):
    """The gate, and it needs no canonicalized rerun.

    Eq. 111's coupling terms are exactly what makes the correction invariant to
    the occupied basis, so untruncated in the localized basis this IS the
    textbook (T). (T0) cannot reach it from here by any tolerance.
    """
    correction = solved.e_lccsd_t - solved.e_lccsd
    assert correction == pytest.approx(CANONICAL_DF_T, abs=1e-10)


def test_the_iteration_recovers_what_t0_was_missing(solved):
    """The correction is the (T0)-to-(T) gap, not a small refinement.

    Worth asserting as a size rather than only as a final value: it is five
    percent of the correction, and a port that skipped the coupling entirely
    would still look plausible on a total energy.
    """
    assert solved.e_t0 == pytest.approx(LOCALIZED_T0, abs=1e-9)
    correction = solved.e_lccsd_t - solved.e_lccsd
    assert abs(correction - solved.e_t0) > 1e-4


def test_the_residual_actually_converges(solved):
    """The iteration converged rather than ran out of patience.

    ``iterate`` raises on exceeding ``maxiter``, so reaching here at all says
    it stopped on the convergence test; this pins that it did so in a sane
    number of passes, which is what would degrade first if a coupling term
    were wrong in sign.
    """
    assert 1 < solved.lccsd_t.n_iterations < 100


# => the permuter <= #


def test_the_permuter_spec_matches_psi4s_selection():
    """psi4's ``triples_permuter`` index chain, case by case.

    An amplitude block is stored once in its own ``i <= j <= k`` ordering and
    read in whatever ordering the requesting triplet needs, so this mapping is
    the whole of the cross-triplet bookkeeping. The two CYCLIC orderings are
    the ones psi4 carries a ``reverse`` flag for, because permuting a tensor
    and permuting the request run in opposite directions for a 3-cycle - which
    is exactly where an off-by-one-permutation bug would hide.
    """
    expected = {
        (0, 1, 2): "abc", (0, 2, 1): "acb", (1, 0, 2): "bac",
        (1, 2, 0): "cab", (2, 0, 1): "bca", (2, 1, 0): "cba",
    }
    labels = (10, 20, 30)
    for order, spec in expected.items():
        assert permuter_spec(*(labels[p] for p in order)) == spec


def test_the_permuter_spec_is_a_genuine_permutation():
    """Applying the spec to a labelled cube reproduces the requested ordering.

    The mapping above is asserted against psi4's table; this asserts it against
    what it is supposed to MEAN, so the two cannot drift into agreeing on a
    consistent mistake.
    """
    labels = (10, 20, 30)
    block = np.arange(27.0).reshape(3, 3, 3)
    for order in ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)):
        spec = permuter_spec(*(labels[p] for p in order))
        got = np.einsum(f"{spec}->abc", block)
        # Reading the block in the requested ordering means axis n of the
        # result is axis order[n] of the original.
        assert np.array_equal(got, block.transpose(order))


# => the batched rotation chain <= #


def _unbatched(A, S, spec, weight):
    """The coupling as ``dlpno`` wrote it before it was batched.

    Permute the partner's block into the requesting triplet's index order, then
    rotate one index at a time. This is the definition the batched chain has to
    reproduce, and writing it out here is what makes the comparison a test of
    :data:`~dlpno.lccsd_t._CHAIN` rather than of itself.
    """
    t1 = np.einsum(f"xa,{spec}->xbc", S, A)
    t2 = np.einsum("xbc,yb->xyc", t1, S)
    return weight * np.einsum("xyc,zc->xyz", t2, S)


def _batched(A, S, spec, weight, nt, npr):
    """The same coupling through :func:`~dlpno.lccsd_t.rotation_stages`."""
    odd, family = chain_of(spec)
    block = ten.from_numpy("T", A)
    source = block
    if odd:
        # A'[a1, a2, a3] = A[a1, a3, a2]: the one transposed copy that turns
        # every odd spec into a cyclic read.
        source = ten.zeros("T'", [npr, npr, npr])
        einsums.permute("abc <- acb", source, block)
    S_v = ten.from_numpy("S", S).reshape_view([nt, npr])
    S_w = ten.from_numpy("w S", weight * S).reshape_view([nt, npr])
    b1 = ten.empty("b1", [nt * npr * npr])
    b2 = ten.empty("b2", [nt * nt * npr])
    R = ten.zeros("R", [nt, nt, nt])
    stages = rotation_stages(family, source, S_v, S_w, b1, b2,
                             R.reshape_view([nt * nt, nt]),
                             R.reshape_view([nt, nt * nt]), nt, npr)
    rotate = _ROTATE_FLAGS[_DIRECTION[family]]
    for stage, (trans_a, trans_b) in enumerate((rotate, rotate,
                                                _ACCUMULATE_FLAGS[family])):
        a, b, c = stages[stage]
        cg.grouped_batched_gemm(1.0, [a], [b], 1.0 if stage == 2 else 0.0, [c],
                                trans_a=trans_a, trans_b=trans_b)
    return ten.view(R)


#: Every string ``permuter_spec`` can return, in a fixed order, so the seed
#: below is a function of the case and not of the interpreter's hash salt.
SPECS = ["abc", "acb", "bac", "bca", "cab", "cba"]


@pytest.mark.parametrize("spec", SPECS)
@pytest.mark.parametrize("nt,npr", [(5, 4), (4, 6), (3, 3)])
def test_the_batched_chain_reproduces_the_permuted_rotation(spec, nt, npr):
    """Three plain GEMMs against the permute-then-rotate chain, spec by spec.

    This is the whole of what batching the phase cost, and it is the one part of
    it that cannot be checked by looking at it. The rotation of a block by the
    same overlap in all three indices commutes with a permutation of those
    indices, so the permutation can be absorbed either into which layout the
    last GEMM writes (the three cyclic specs) or into one transposed copy of the
    source (the three odd ones). Get the layout wrong and the result is a
    transposed block, which is a perfectly plausible-looking energy.

    All six specs, both source parities, and rectangular shapes both ways
    round, because the merged-axis views are where a square block would hide a
    swapped pair of dimensions.
    """
    rng = np.random.default_rng(97 * nt + 13 * npr + SPECS.index(spec))
    A = rng.standard_normal((npr, npr, npr))
    S = rng.standard_normal((nt, npr))
    weight = float(rng.standard_normal())
    got = _batched(A, S, spec, weight, nt, npr)
    want = _unbatched(A, S, spec, weight)
    assert np.allclose(got, want, rtol=0, atol=1e-12 * np.abs(want).max())


def test_every_spec_the_permuter_can_return_has_a_chain():
    """No spec falls off :data:`~dlpno.lccsd_t._CHAIN`.

    ``permuter_spec`` returns one of six strings and the emitter looks every one
    of them up. Five appear in the fixtures; ``cba`` does not, which is exactly
    why its entry needs a test rather than a run.
    """
    for order in ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)):
        spec = permuter_spec(*(10 * (p + 1) for p in order))
        odd, family = chain_of(spec)
        assert isinstance(odd, bool)
        assert family in _ACCUMULATE_FLAGS


# => the block width <= #


def test_the_block_partition_covers_every_triplet_once():
    """A block is a skip gate, so the blocks have to be a partition.

    A triplet in two blocks would have its residual accumulated twice into one
    ``R``; a triplet in none would never be updated. Neither shows up as
    anything but a wrong energy.
    """
    solver = LCCSDT.__new__(LCCSDT)
    solver.block = 4
    solver._plan = [dict(ijk=n, entries=[0] * (n % 7)) for n in range(23)]
    blocks = solver._make_blocks(True)
    assert [len(b) for b in blocks] == [4, 4, 4, 4, 4, 3]
    assert sorted(r["ijk"] for b in blocks for r in b) == list(range(23))
    # Sorted by coupling count, so a block's calls track its members' couplings
    # rather than its widest member's.
    assert [len(r["entries"]) for r in blocks[0]] == [6, 6, 6, 5]
    # Skipping off is one block over everything, which is the widest batch.
    assert len(solver._make_blocks(False)) == 1


def test_the_block_width_moves_the_energy_only_within_convergence(monkeypatch):
    """Gating in blocks recomputes settled triplets, and that is not an error.

    A settled triplet sharing a block with a moving one is recomputed rather
    than frozen, so the trajectory differs from the per-triplet gate's. What it
    must not do is move the converged answer by more than the iteration's own
    energy tolerance, and comparing the two widths is the only way to see that
    separately from the psi4 comparison.

    A block of one is also the width at which the gate is per triplet, so this
    is what pins that the two ends of the lever solve the same equation.
    """
    import dlpno.triples as triples

    reference, _ = load_reference(WATER)
    cut = replace(Thresholds.preset("NORMAL", method="cc"),
                  t0_approximation=False)
    energies, passes = [], []
    for block in (1, 1 << 20):
        monkeypatch.setattr(triples, "LCCSDT",
                            lambda *a, _b=block, **k: LCCSDT(*a, block=_b, **k))
        cc = DLPNOCCSDT(reference, cut, verbose=False)
        cc.compute_energy(method="ccsd(t)")
        energies.append(cc.e_t_raw)
        passes.append(cc.lccsd_t.n_iterations)
    assert energies[0] == pytest.approx(energies[1], abs=1e-8)
    # One batch over every triplet is the widest the phase gets, and the pass
    # count stays within one of the per-triplet gate's.
    assert abs(passes[0] - passes[1]) <= 1


# => the wide replay and the thread plan <= #


def test_the_wide_replay_is_taken_whenever_it_is_eligible():
    """No size cutoff decides between the two captures any more.

    The phase carries two spellings of the same residual - one batch over every
    triplet, and the same work behind a conditional per block - and it used to
    choose between them per solve from a cutoff on the plan's mean TNO count,
    calibrated from two measured geometries. The cutoff existed only because a
    grouped node was threaded or serial depending on whether it landed alone on
    an execution level, so a fat plan did better under a team over blocks. A
    planned thread width is that choice made per node from a cost model, so the
    wide capture is simply used whenever it is eligible: every pass in which no
    triplet has settled yet.

    Both paths have to actually be taken here, or the assertion is vacuous: a
    truncated solve on the dimer skips triplets partway through, so the passes
    split between the two.
    """
    reference, _ = load_reference(DIMER)
    cut = replace(Thresholds.preset("NORMAL", method="cc"),
                  t0_approximation=False)
    cc = DLPNOCCSDT(reference, cut, verbose=False)
    cc.compute_energy(method="ccsd(t)")

    solver = cc.lccsd_t
    assert solver.n_wide_passes > 0, "the wide capture was never replayed"
    assert solver.n_skipped > 0, (
        "nothing settled, so the gated capture was never replayed and this "
        "test cannot tell the two apart")
    assert not hasattr(solver, "wide_max_tno"), (
        "the size cutoff is back; widths are supposed to have subsumed it")


def test_the_replayed_graphs_carry_a_thread_plan():
    """Every graph the iteration replays is planned, and for THIS machine.

    Widths are a share of a known number of threads, so a plan is only usable
    beside the count it was made for - the executor compares the two and falls
    back to width one when they disagree. Asserting the recorded count is the
    machine's is therefore the check that the plan will actually be honored,
    and it is a check that holds at any thread count, including the one this
    suite runs at.

    Whether the planner finds a width worth having is a cost-model question and
    a property of the geometry, not something to pin here; what is pinned is
    that every replayed graph went through it and that a declined graph keeps a
    parallel executor rather than quietly turning serial.
    """
    reference, _ = load_reference(WATER)
    cut = replace(Thresholds.preset("NORMAL", method="cc"),
                  t0_approximation=False)
    cc = DLPNOCCSDT(reference, cut, verbose=False)
    cc.compute_energy(method="ccsd(t)")

    solver = cc.lccsd_t
    assert set(solver.moldable) >= {"residual", "step"}, (
        "the residual and the step are supposed to be planned")
    assert all(isinstance(on, bool) for on in solver.moldable.values())


def test_two_planned_solves_agree_bit_for_bit():
    """The determinism the width plan is supposed to buy, end to end.

    A width fixes a kernel's internal partition - its BLAS blocking, its loop
    split, which of two GEMM kernels a grouped call picks - so with the widths
    fixed the arithmetic is fixed, and scheduling order may vary freely across
    replays without moving a bit. What used to be true instead was that which
    nodes shared an execution level decided which kernels threaded, so the last
    bit moved with scheduling accidents.

    Two full solves in one process, then, at the same thread count: same plan,
    same widths, same energy to the last bit. Bit-identical and not a tolerance,
    because a tolerance here would pass on exactly the drift this is about.

    One replan is armed by default and fires after the first replay of each
    graph, which both runs do identically, so this spans that boundary rather
    than dodging it.
    """
    reference, _ = load_reference(WATER)
    cut = replace(Thresholds.preset("NORMAL", method="cc"),
                  t0_approximation=False)
    energies = []
    for _ in range(2):
        cc = DLPNOCCSDT(reference, cut, verbose=False)
        cc.compute_energy(method="ccsd(t)")
        energies.append(cc.e_t_raw)
    assert energies[0] == energies[1]


def test_the_serial_replay_is_left_unplanned():
    """``use_executor=False`` still means what it said: no executor, no plan.

    The bisection this knob exists for only works if off is genuinely the
    serial replay, so planning must not leak into it - a width is honored by
    the dataflow executor alone, but binding one is exactly what off must not
    do.
    """
    import dlpno.triples as triples

    reference, _ = load_reference(WATER)
    cut = replace(Thresholds.preset("NORMAL", method="cc"),
                  t0_approximation=False)
    cc = DLPNOCCSDT(reference, cut, verbose=False)
    original = triples.LCCSDT
    try:
        triples.LCCSDT = (lambda *a, **k:
                          original(*a, use_executor=False, **k))
        cc.compute_energy(method="ccsd(t)")
    finally:
        triples.LCCSDT = original

    assert cc.lccsd_t.moldable == {}, (
        "a serial run planned widths for something")


# => the truncated path <= #


def test_the_strong_weak_triplet_split_fires():
    """``sort_triplets`` classifies, and both classes are populated.

    The split is the only thing ``tno_scale`` reads, and it decides the size of
    the space the iteration runs in. A run where every triplet came out strong
    would still converge to a plausible number in a space psi4 never used.
    """
    reference, _ = load_reference(DIMER)
    cc = DLPNOCCSDT(reference, Thresholds.preset("NORMAL", method="cc"),
                    verbose=False)
    cc.compute_energy()

    strong = sum(cc.is_strong_triplet)
    assert 0 < strong < cc.n_lmo_triplets, "the triplets did not split"
    # Strong triplets get the tighter of the two scales, so their TNO spaces
    # are the larger ones; the whole point of the split is that both exist.
    assert min(cc.n_tno) < max(cc.n_tno)
    assert cc.de_t != 0.0, "the iterative pass contributed nothing"


def test_the_iterative_pass_refuses_a_budget_it_cannot_meet():
    """Design decision 10: in core, with a MEASURED failure.

    The iterative (T) is the first phase in the port whose stores are not
    bounded by a chunk: it keeps W, V and the amplitudes for every triplet at
    once, because Eq. 111 reads its neighbours'. Bounding only the chunk is how
    this came to page instead of refusing - an ethanol/cc-pVTZ run sat at 12%
    CPU with 28 GB of swap in use, which looks exactly like a slow benchmark
    and is not one.

    psi4 spills to disk at that point. There is no disk path here, so what is
    asserted is that the refusal happens, names the requirement, and says what
    to change - because "in core only" is a defensible scope decision only if
    the point where it stops working arrives as a number.
    """
    reference, _ = load_reference(DIMER)
    cut = replace(Thresholds.preset("NORMAL", method="cc"),
                  in_core_memory=20 * 2 ** 20)
    cc = DLPNOCCSDT(reference, cut, verbose=False)
    with pytest.raises(MemoryError, match="iterative \\(T\\) needs"):
        cc.compute_energy()


def test_stopping_at_t0_needs_no_per_triplet_storage():
    """The escape the refusal names has to actually work.

    (T0) retains nothing per triplet, so the same budget that refuses the
    iterative pass must let the semicanonical one through. A refusal message
    offering an option that also fails would be worse than no message.
    """
    reference, _ = load_reference(DIMER)
    cut = replace(Thresholds.preset("NORMAL", method="cc"),
                  in_core_memory=20 * 2 ** 20, t0_approximation=True)
    cc = DLPNOCCSDT(reference, cut, verbose=False)
    cc.compute_energy()
    assert cc.e_t0 != 0.0
