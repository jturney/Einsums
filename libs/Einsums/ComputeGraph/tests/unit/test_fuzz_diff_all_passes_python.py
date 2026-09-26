# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""Every default pass in a random order, over programs built to make each one fire.

The rich random-pipeline shard shuffles the passes that act on operators, views
and scratch. Eleven more passes of the default pipeline had never run in any
order but the curated one: ProvenancePropagation, DeltaElimination, the four
antisymmetrizer passes, SymmetrizedAccumulation, the two sum-factoring folds,
MultiTermFactorization and ContractionPlanning. This shard adds them to the
shuffle (``_ALL_PASSES``), always closing with Materialization as every real
pipeline above O0 does, and compares every tensor a caller can observe against
the numpy oracle.

A shuffle over the ordinary corpus would run those passes without them ever
doing anything, since each recognizes a shape that corpus never draws. So the
programs come from ``allpass_program``, whose motifs build one shape per
recognizer: a contraction against a declared identity, a contraction over two
disjoint spaces, antisymmetrized operands contracted to a scalar (rank two and
the coset form on rank three), an antisymmetrized sum, a transposed
accumulation, a 2J-K pair, a shared-operand sum, two chains sharing a factor
and a chain written in its expensive bracketing. They land inside loops and
conditionals like any other statement, in all four dtypes.

The guards at the bottom assert each added pass fires on a floor share of a
fixed corpus, so the arm cannot quietly go vacuous. The pinned cases below
them are the defects the arm found, each reduced to a fixed program and kept
as the guard for its fix (the one marked xfail is still open). The motifs
draw each of those shapes too, as documented where they are made in
``_fuzz_diff_common.py``, and every drawn pass order runs as drawn.

The shared harness lives in _fuzz_diff_common.py.
"""

from __future__ import annotations

import os

# Every PassManager checks the graph after every pass, so a pass that leaves a
# malformed graph fails the trial naming itself rather than surfacing later as
# a wrong number or not at all. Set before einsums reads its options.
os.environ.setdefault("EINSUMS_PASS_VERIFY", "1")

import numpy as np
import pytest
from _pytest.outcomes import Skipped

import einsums
import einsums.graph as cg
from _fuzz_diff_common import *  # shared fuzz/differential harness

ALL_FUZZ_DTYPES = ["float64", "float32", "complex128", "complex64"]


def _trial(base, seed, dtype, depth, max_stmts, runs=1):
    rng = np.random.default_rng(base + seed)
    prog = allpass_program(rng, depth, max_stmts)
    arrays = allpass_seed_arrays(rng, dtype)
    return check_program_all_passes(prog, arrays, f"ap{base}_{seed}", rng, dtype=dtype, runs=runs)


@pytest.mark.parametrize("dtype", ALL_FUZZ_DTYPES)
@pytest.mark.parametrize("seed", fuzz_seeds(150))
def test_fuzz_all_passes(seed, dtype):
    _trial(410_000, seed, dtype, depth=2, max_stmts=6)


@pytest.mark.parametrize("dtype", ["float64", "complex128"])
@pytest.mark.parametrize("seed", fuzz_seeds(150))
def test_fuzz_all_passes_flat(seed, dtype):
    """Long flat programs, where a region holds several motifs at once."""
    _trial(420_000, seed, dtype, depth=0, max_stmts=10)


@pytest.mark.parametrize("dtype", ["float64", "complex64"])
@pytest.mark.parametrize("seed", fuzz_seeds(100))
def test_fuzz_all_passes_replay(seed, dtype):
    """Deeper nesting, executed twice: a rewrite that holds for one execute only shows here."""
    _trial(430_000, seed, dtype, depth=3, max_stmts=6, runs=2)


# ──────────────────────────────────────────────────────────────────────────
# Guards: a shuffled pass that never fires has been shuffled, not tested.
# ──────────────────────────────────────────────────────────────────────────

_GUARD_SEEDS = range(120)

#: The guard corpus is both shapes of program: nested (depth two) and flat
#: (ten statements, no control flow). Each carries passes the other starves:
#: ProvenancePropagation fires on a tag crossing into a loop body, and the
#: re-bracketing passes and the fold on a region holding a whole motif.
_GUARD_CORPORA = ((440_000, 2, 6), (450_000, 0, 10))

#: Floor per added pass, as a share of usable guard trials. Measured over the
#: guard corpus (240 float64 programs) once the motifs drew every pinned shape:
#: ProvenancePropagation 0.14, DeltaElimination 0.21, AntisymmetryDetection
#: 0.39, AntisymmetrizerLinearity 0.10, AntisymmetryInference 0.41,
#: AntisymmetrizerFolding 0.06, SymmetrizedAccumulation 0.16,
#: LinearCombinationContractionFolding 0.14, DistributiveFactoring 0.14,
#: MultiTermFactorization 0.12, ContractionPlanning 0.10. The floors sit near
#: half of the first measurement, taken before the motifs drew the shapes a
#: pass must decline on, so they catch a motif that stopped producing its
#: shape rather than ordinary drift. AntisymmetrizerFolding is low by construction: it
#: needs AntisymmetryInference ahead of it and AntisymmetrizerExpansion behind
#: it, one shuffled order in six.
_FIRE_FLOORS = {
    "ProvenancePropagation": 0.065,
    "DeltaElimination": 0.09,
    "AntisymmetryDetection": 0.23,
    "AntisymmetrizerLinearity": 0.06,
    "AntisymmetryInference": 0.22,
    "AntisymmetrizerFolding": 0.03,
    "SymmetrizedAccumulation": 0.11,
    "LinearCombinationContractionFolding": 0.11,
    "DistributiveFactoring": 0.09,
    "MultiTermFactorization": 0.09,
    "ContractionPlanning": 0.04,
}


def _guard_rates():
    fired = {name: 0 for name in _ALL_PASSES_ADDED}
    usable = 0
    for base, depth, max_stmts in _GUARD_CORPORA:
        for seed in _GUARD_SEEDS:
            try:
                got = _trial(base, seed, "float64", depth=depth, max_stmts=max_stmts)
            except Skipped:
                continue
            usable += 1
            for name in got & set(fired):
                fired[name] += 1
    return usable, {name: count / max(usable, 1) for name, count in fired.items()}


def test_every_added_pass_fires_under_a_shuffled_order():
    assert set(_FIRE_FLOORS) == set(_ALL_PASSES_ADDED)
    if len(fuzz_seeds(len(_GUARD_SEEDS))) < len(_GUARD_SEEDS):
        pytest.skip("the floors are shares of the full guard corpus, which a capped (sanitizer) run does not draw")
    usable, rates = _guard_rates()
    total = len(_GUARD_SEEDS) * len(_GUARD_CORPORA)
    assert usable >= 0.9 * total, f"only {usable} of {total} guard trials were usable"
    low = {name: round(rate, 3) for name, rate in rates.items() if rate < _FIRE_FLOORS[name]}
    assert not low, f"passes firing below their floor: {low}; all rates: {rates}"


def test_every_motif_is_drawn():
    """Every opcode and every extra-pool role the motifs rely on appears in the guard corpus."""
    opcodes = set()
    slots = set()

    def walk(stmts):
        for st in stmts:
            if st[0] == "loop":
                walk(st[2])
            elif st[0] == "cond":
                walk(st[2])
                walk(st[3])
            else:
                opcodes.add(st[0])

    for base, depth, max_stmts in _GUARD_CORPORA:
        for seed in _GUARD_SEEDS:
            prog = allpass_program(np.random.default_rng(base + seed), depth, max_stmts)
            walk(prog)
            slots |= allpass_named_slots(prog)
    for opcode in ("aperm", "dot", "ddiv", "leinsum", "perm", "axpby", "einsum"):
        assert opcode in opcodes, f"no program in the guard corpus holds a {opcode!r} statement"
    for role, by_shape in ALLPASS_XM_BY.items():
        role_slots = {s for group in by_shape.values() for s in group}
        assert role_slots & slots, f"no program in the guard corpus names a {role!r} slot"


# ──────────────────────────────────────────────────────────────────────────
# Pinned defects. Each is the reduced form of a fuzz failure, in fixed pool
# slots, run in the one order that exposed it. Every one but the case marked
# xfail is fixed; each stays as the guard for its defect.
# ──────────────────────────────────────────────────────────────────────────


class _FixedOrder:
    """Stands in for the rng's shuffle so a pinned case runs one exact order."""

    def __init__(self, order):
        self.order = list(order)

    def shuffle(self, x):
        x[:] = self.order


def _pinned(prog, order, label, dtype="float64", runs=1, seed=3):
    arrays = allpass_seed_arrays(np.random.default_rng(seed), dtype)
    check_program_all_passes(prog, arrays, label, _FixedOrder(order), dtype=dtype, runs=runs)


def _xm1(role, shape, k=0):
    return ALLPASS_XM_BY[role][shape][k]


_M33 = MAT_BY_SHAPE[(3, 3)]
_A32, _B23 = MAT_BY_SHAPE[(3, 2)][0], MAT_BY_SHAPE[(2, 3)][0]
_OP_IJ = ([["i"], ["j"]], 0, ["i", "j"])


def _symacc_site(tmp, tmpP, r2, s=0.5):
    return [("einsum", "ij <- ik ; kj", 1.0, _A32, _B23, 0.0, tmp, False, False),
            ("axpby", s, tmp, 1.0, r2),
            ("perm", 1.0, 0.0, tmp, tmpP),
            ("axpby", s, tmpP, 1.0, r2)]


def test_symmetrized_accumulation_keeps_a_caller_held_transpose():
    """``tmpP = tmp^T`` into a USER tensor, then both halves accumulated into r2.

    Defends against the rewrite pointing the transpose at r2 and deleting the
    second accumulate, so that tmpP was never written and the caller read its
    seed values. The pass checked tmpP's readers inside its own generation and
    nothing else: it never asked whether tmpP is a graph-owned intermediate, as
    CSE, PermuteFusion and DeadNodeElimination do before removing a write.
    """
    tmp = _xm1("scratch", (3, 3))
    _pinned(_symacc_site(tmp, _M33[0], _M33[1]), ["SymmetrizedAccumulation"], "sa_user_tmpP")


def test_symmetrized_accumulation_sees_a_later_accumulate_into_the_transpose():
    """The site, then ``tmpP += X`` and a read of tmpP.

    Defends against the pass ending tmpP's generation at the next node listing
    tmpP as an OUTPUT (``next_write_after``), which skipped the read that same
    node makes, so that ``tmpP += X`` read a transpose the rewrite no longer
    wrote.
    """
    tmp, tmpP = _xm1("scratch", (3, 3)), _xm1("symacc", (3, 3))
    prog = _symacc_site(tmp, tmpP, _M33[1]) + [("axpy", 1.0, _M33[2], tmpP), ("axpy", 1.0, tmpP, _M33[0])]
    _pinned(prog, ["SymmetrizedAccumulation"], "sa_rmw_after")


def test_symmetrized_accumulation_sees_a_loop_between_the_halves():
    """The transpose, a loop that reads r2, then the two accumulates.

    Defends against the rewrite moving the second half's contribution up to
    the transpose, ahead of the loop, so that the loop read a half-symmetrized
    r2. The same read written flat was rejected; a Loop or Conditional node
    lists none of its body's operands, and the guard read only the node's own
    lists. A shuffled Reorder or LoopInvariantHoisting produces exactly this
    placement.
    """
    tmp, tmpP = _xm1("scratch", (3, 3)), _xm1("symacc", (3, 3))
    e, a1, p, a2 = _symacc_site(tmp, tmpP, _M33[1])
    prog = [e, p, ("loop", 1, [("axpy", 1.0, _M33[1], _M33[0])]), a1, a2]
    _pinned(prog, ["SymmetrizedAccumulation"], "sa_loop_between")


def test_delta_elimination_after_symmetrized_accumulation_keeps_the_accumulate():
    """A symmetrized accumulation, then any contraction against a declared identity.

    Defends against SymmetrizedAccumulation turning the transpose into
    ``r2 += s2 * tmp^T`` by swapping in a hand-built executor while updating
    only the descriptor's snapshot ``beta``: the live ``params`` still said
    overwrite with alpha one, and ``s2`` lived in the closure alone.
    DeltaElimination lowered the region and rebuilt the node from its
    descriptor, which computed ``r2 = tmp^T``. The default order runs
    DeltaElimination first, which is why it was green there.
    """
    tmp, tmpP = _xm1("scratch", (3, 3)), _xm1("symacc", (3, 3))
    delta = _xm1("delta", (3, 3))
    prog = _symacc_site(tmp, tmpP, _M33[1]) + [("einsum", "ij <- ik ; kj", 1.0, _M33[2], delta, 0.0, _M33[0],
                                                 False, False)]
    _pinned(prog, ["SymmetrizedAccumulation", "DeltaElimination"], "sa_then_delta")


@pytest.mark.parametrize("shape", ["overwritten", "scaled", "accumulated"])
def test_provenance_propagation_tags_only_a_plain_copy(shape):
    """``S = delta^T`` is not tagged an identity when S is not one.

    Defends against the pass carrying the tag across any Permute node onto its
    output tensor, checking only the node kind and the shapes. A tag is a
    claim about the tensor for the whole program, so it was wrong when the
    permute scales (``S = delta^T / 2``), accumulates (``S += delta^T``), or
    when a later node overwrites S, and DeltaElimination then dropped a
    contraction against S as if it were the identity. The default order runs
    both, provenance first.
    """
    delta = _xm1("delta", (3, 3))
    S = _xm1("scratch", (3, 3))
    A, C = MAT_BY_SHAPE[(2, 3)][0], MAT_BY_SHAPE[(2, 3)][1]
    use = ("einsum", "ij <- ik ; kj", 1.0, A, S, 0.0, C, False, False)
    if shape == "overwritten":
        overwrite = ("einsum", "ij <- ik ; kj", 0.5, _M33[0], _M33[1], 0.0, S, False, False)
        prog = [("perm", 1.0, 0.0, delta, S), overwrite, use]
    elif shape == "scaled":
        prog = [("perm", 0.5, 0.0, delta, S), use]
    else:
        prog = [("axpy", 1.0, _M33[0], S), ("perm", 1.0, 1.0, delta, S), use]
    _pinned(prog, ["ProvenancePropagation", "DeltaElimination"], f"prov_{shape}")


def test_multi_term_factorization_sizes_a_reused_letter_by_its_own_statement():
    """``T = B D`` then ``R = A T``, both spelled ``ij <- ik ; kj``.

    Letters are scoped to a statement, and here ``i`` is 12 in the first and 3
    in the second. Defends against the pass recording extents in one table per
    region, first observation winning, and sizing the intermediate it
    introduces from that table, so that ``(A B)`` was declared 12 x 3 instead
    of 3 x 3. The verifier rejected the graph (EINSUMS_PASS_VERIFY); without it
    the execute threw a TensorCompatError.
    """
    s, L = ALLPASS_BS, ALLPASS_BL
    A, D = _xm1("mtfsrc", (s, L), 0), _xm1("mtfsrc", (s, L), 1)
    B = _xm1("mtfsrc", (L, s))
    T = _xm1("scratch", (L, L))
    prog = [("einsum", "ij <- ik ; kj", 1.0, B, D, 0.0, T, False, False),
            ("einsum", "ij <- ik ; kj", 0.5, A, T, 0.0, _xm1("mtfout", (s, L)), False, False)]
    _pinned(prog, ["MultiTermFactorization"], "mtf_letters")


def test_multi_term_factorization_shares_after_a_factor_is_written():
    """``B *= 1/2``, then two chains sharing ``A B``.

    Defends against the shared ``A B`` being emitted at the front of the
    region, before the scale that writes B, so that both results used the
    unscaled factor.
    """
    s, L = ALLPASS_BS, ALLPASS_BL
    A, C, D = (_xm1("mtfsrc", (s, L), k) for k in range(3))
    B = _xm1("mtfsrc", (L, s))
    T1, T1b = _xm1("scratch", (s, s), 0), _xm1("scratch", (s, s), 1)
    R1, R2 = _xm1("mtfout", (s, L), 0), _xm1("mtfout", (s, L), 1)
    prog = [("scale", 0.5, B),
            ("einsum", "pr <- pq ; qr", 1.0, A, B, 0.0, T1, False, False),
            ("einsum", "ps <- pr ; rs", 0.7, T1, C, 1.0, R1, False, False),
            ("einsum", "pr <- pq ; qr", 1.0, A, B, 0.0, T1b, False, False),
            ("einsum", "ps <- pr ; rs", -0.4, T1b, D, 1.0, R2, False, False)]
    _pinned(prog, ["MultiTermFactorization"], "mtf_write_before_share")


def test_antisymmetrizer_folding_keeps_the_operator_prefactor():
    """``W = 2 P(i/j) X``, ``V = 3 P(i/j) Y``, ``r = <W, V>``.

    Defends against the fold repointing the dot at the permute's source and
    scaling by the term count without reading the permute's own ``alpha``,
    which left the result off by exactly that factor.
    """
    W, V = _xm1("foldscr", (3, 3), 0), _xm1("foldscr", (3, 3), 1)
    prog = [("aperm", "m", _OP_IJ, 2.0, _M33[0], 0.0, W),
            ("aperm", "m", _OP_IJ, 3.0, _M33[1], 0.0, V),
            ("dot", "m", VEC_BY_LEN[1][0], W, V)]
    _pinned(prog, ["AntisymmetryInference", "AntisymmetrizerFolding"], "fold_alpha")


def test_antisymmetrizer_folding_sees_its_source_overwritten():
    """``W = P X``, ``V = P Y``, ``Y *= 1/2``, ``r = <V, W>``.

    Defends against the fold replacing V by its source Y in the dot, which then
    read Y after the scale. The pass checked that each dot operand has a single
    writer and never that the source stays unwritten between the operator and
    the dot; a shuffled Reorder moves the dot past such a write in the fuzz.
    """
    W, V = _xm1("foldscr", (3, 3), 0), _xm1("foldscr", (3, 3), 1)
    prog = [("aperm", "m", _OP_IJ, 1.0, _M33[0], 0.0, W),
            ("aperm", "m", _OP_IJ, 1.0, _M33[1], 0.0, V),
            ("scale", 0.5, _M33[1]),
            ("dot", "m", VEC_BY_LEN[1][0], V, W)]
    _pinned(prog, ["AntisymmetryInference", "AntisymmetrizerFolding"], "fold_source_written")


def test_antisymmetrizer_linearity_sees_a_write_between_the_operators():
    """``Y = a P(A)``, ``B *= 1/2``, ``X = c P(B)``, ``Y += s X``.

    Defends against the merged sum ``a A + s c B`` being built at Y's operator,
    before the scale, so that it read the old B. The pass checked the
    destination and the operator groups, and nothing about writes to the other
    source between the two operators; a shuffled Reorder is what puts one there
    in the fuzz.
    """
    X, Y = _xm1("scratch", (3, 3), 0), _xm1("scratch", (3, 3), 1)
    prog = [("aperm", "m", _OP_IJ, 0.8, _M33[0], 0.0, Y),
            ("scale", 0.5, _M33[1]),
            ("aperm", "m", _OP_IJ, 0.25, _M33[1], 0.0, X),
            ("axpby", 0.75, X, 1.0, Y),
            ("dot", "m", VEC_BY_LEN[1][0], Y, _M33[2])]
    _pinned(prog, ["AntisymmetrizerLinearity"], "lin_write_between")


def test_distributive_factoring_sees_a_view_write_between_the_members():
    """``C += A1^T B``, a write into a block of A2, ``C += A2^T B``.

    Defends against the sum ``A1 + A2`` being built at the first member, before
    the block write. A plain write to A2 in the same place was declined, but
    the gate (``span_interferes``) compared raw tensor ids, and the view's id
    is not A2's.
    """
    A1, A2 = MAT_BY_SHAPE[(3, 2)][1], MAT_BY_SHAPE[(3, 2)][0]
    B, C, src = MAT_BY_SHAPE[(3, 4)][0], MAT_BY_SHAPE[(2, 4)][0], MAT_BY_SHAPE[(1, 2)][0]
    prog = [("einsum", "ij <- ki ; kj", 0.8, A1, B, 1.0, C, False, False),
            ("vaxpy", 0.75, src, A2, 0, 1, 0, 2),
            ("einsum", "ij <- ki ; kj", -0.5, A2, B, 1.0, C, False, False)]
    _pinned(prog, ["DistributiveFactoring"], "df_view_between")


def test_distributive_factoring_keeps_two_groups_in_program_order():
    """Two groups into one output, the second right after the first, both headed by ``C *= 1/2``.

    Defends against the replacements landing out of program order: once the
    first group's members were erased, both replacements mapped to one
    position in ``Graph::replace_nodes``, and their order fell back to the
    order the pass visited the groups in, which here is the reverse of the
    program's. With a unit head either order gives the same sum.
    """
    S1, S2 = _xm1("dfsrc", (2, 2), 0), _xm1("dfsrc", (3, 2), 0)
    X1, X2 = _xm1("dfsrc", (3, 2), 1), S2
    Y1, Y2, Y3 = (_xm1("dfsrc", (3, 3), k) for k in range(3))
    C = _xm1("dfout", (3, 2))
    prog = [("einsum", "ij <- ik ; jk", -0.6, X1, S1, 0.5, C, False, False),
            ("einsum", "ij <- ik ; jk", 0.45, X2, S1, 1.0, C, False, False),
            ("einsum", "ij <- ik ; kj", -0.4, Y1, S2, 0.5, C, False, False),
            ("einsum", "ij <- ik ; kj", 0.06, Y2, S2, 1.0, C, False, False),
            ("einsum", "ij <- ik ; kj", 0.99, Y3, S2, 1.0, C, False, False)]
    _pinned(prog, ["DistributiveFactoring"], "df_two_groups")


def test_linear_combination_folding_sees_a_loop_between_the_members():
    """``C = a J``, a loop that rescales C, ``C += b K``.

    Defends against the fold computing ``C = a J + b K`` at the first member,
    after which the loop rescaled the whole sum. The pass called
    ``span_interferes`` with ``reject_control_flow=false``, and a Loop node
    lists none of its body's operands, so the write was invisible to it.
    """
    T = ALLPASS_XT_BY["lccft"][(2, 3, 3)][0]
    vec = ALLPASS_XV_BY["lccfv"][(2,)][0]
    C = _xm1("lccfout", (3, 3))
    prog = [("leinsum", "kij", 0.7, vec, T, 0.0, C),
            ("loop", 1, [("scale", 0.5, C)]),
            ("leinsum", "kji", -0.4, vec, T, 1.0, C)]
    _pinned(prog, ["LinearCombinationContractionFolding"], "lccf_loop_between")


def test_antisymmetrizer_folding_guard_reads_only_materialized_tensors():
    """A rank-2 fold, and a transposed accumulation of ``tmp = A A``, after an early Materialization.

    Defends against AntisymmetrizerFolding's "antisymmetry premise guard", a
    Setup node at the front of the graph, reading the site's deferred transpose
    ``tmpP``. SymmetrizedAccumulation had left it with no reader or writer but
    with the hint SymmetryPropagation derived, so it passed for a detected leaf,
    and the earlier Materialization materializes it further down. Execute threw
    "still deferred" from the Setup; a leaf now needs a reader. It needs both operands of ``tmp``'s
    product to be one tensor and SymmetryPropagation in the order; with two
    different operands, or without that pass, the trial passes.
    """
    W, V = _xm1("foldscr", (2, 2), 0), _xm1("foldscr", (2, 2), 1)
    tmp, tmpP, r2 = _xm1("scratch", (2, 2)), _xm1("symacc", (2, 2)), _xm1("symout", (2, 2))
    A = MAT_BY_SHAPE[(2, 2)][0]
    prog = [("aperm", "m", _OP_IJ, 1.0, _xm1("anti", (2, 2)), 0.0, W),
            ("aperm", "m", _OP_IJ, 1.0, _xm1("linsrc", (2, 2)), 0.0, V),
            ("dot", "m", VEC_BY_LEN[1][0], W, V),
            ("einsum", "ij <- ik ; kj", 0.5, A, A, 0.0, tmp, False, False),
            ("perm", 1.0, 0.0, tmp, tmpP),
            ("axpby", 0.5, tmp, 1.0, r2),
            ("axpby", 0.5, tmpP, 1.0, r2)]
    _pinned(prog, ["Materialization", "SymmetryPropagation", "AntisymmetryInference", "SymmetrizedAccumulation",
                   "AntisymmetrizerFolding"], "fold_guard_deferred")


@pytest.mark.parametrize("where", ["loop", "cond"])
def test_a_second_materialization_keeps_a_tensor_a_loop_body_reads(where):
    """``D += X`` at the top, then a loop (or branch) reading D, materialized twice.

    The first Materialization puts D's Materialize and zero Initialize ahead of
    the write. Defends against a second one finding the body's handle for D
    still deferred and putting another Materialize and Initialize in front of
    the loop, which wiped what the write left. Every pipeline that closes with
    Materialization after one that already ran it did this. Written against
    the graph directly, so the shape does not depend on what the arm's builder
    happens to draw.
    """
    x = np.full((2, 2), 3.0)
    X = einsums.create_zero_tensor("mat2_X", [2, 2], dtype="float64")
    R = einsums.create_zero_tensor("mat2_R", [2, 2], dtype="float64")
    np.asarray(X)[...] = x
    g = cg.Graph(f"mat2_{where}")
    D = g.declare_zero_tensor("mat2_D", [2, 2], intermediate=True, dtype="float64")
    with cg.capture(g):
        einsums.linalg.axpy(1.0, X, D)
    if where == "loop":
        body = g.add_loop("mat2_loop", 1, lambda it: False)
    else:
        body, _ = g.add_conditional("mat2_cond", lambda: True)
    with cg.capture(body):
        einsums.linalg.axpy(1.0, D, R)
    for _ in range(2):
        pm = cg.PassManager()
        pm.add(cg.Materialization())
        g.apply(pm)
    g.execute()
    np.testing.assert_allclose(np.asarray(R), x)


@pytest.mark.skipif(os.environ.get("EINSUMS_PASS_VERIFY", "1").lower() in ("0", "false", "off", "no"),
                    reason="only the per-pass graph verifier can see this defect")
def test_a_second_materialization_names_a_tensor_the_graph_holds():
    """A deferred tensor only a loop body touches, in a body that also makes two views.

    Defends against the second Materialization hoisting the tensor's
    Initialize into the parent under the id the tensor has in the BODY. The
    views gave the body more tensors than the parent, so that id named nothing
    there and the verifier rejected the graph. The saved IR renumbers ids, so
    only the verifier sees it; without the verifier this instance happened to
    execute correctly, and nothing guaranteed the id was unused in general.
    """
    ones = np.ones((3, 3))
    g = cg.Graph("mat2v")
    D = g.declare_zero_tensor("mat2v_D", [2, 2], intermediate=True, dtype="float64")
    R = einsums.create_zero_tensor("mat2v_R", [2, 2], dtype="float64")
    views = []
    for k in range(2):
        M = einsums.create_zero_tensor(f"mat2v_M{k}", [3, 3], dtype="float64")
        np.asarray(M)[...] = ones
        views.append(M)
    body = g.add_loop("mat2v_loop", 1, lambda it: False)
    with cg.capture(body):
        for M in views:
            einsums.linalg.scale(0.5, cg.view(M, [(0, 1), (0, 1)]))
        einsums.linalg.axpy(1.0, D, R)
    for _ in range(2):
        pm = cg.PassManager()
        pm.add(cg.Materialization())
        g.apply(pm)  # the verifier rejected the second one before the fix
    g.execute()


def test_a_single_materialization_keeps_a_tensor_a_loop_body_reads():
    """The control for the case above: one Materialization is right."""
    x = np.full((2, 2), 3.0)
    X = einsums.create_zero_tensor("mat1_X", [2, 2], dtype="float64")
    R = einsums.create_zero_tensor("mat1_R", [2, 2], dtype="float64")
    np.asarray(X)[...] = x
    g = cg.Graph("mat1")
    D = g.declare_zero_tensor("mat1_D", [2, 2], intermediate=True, dtype="float64")
    with cg.capture(g):
        einsums.linalg.axpy(1.0, X, D)
    body = g.add_loop("mat1_loop", 1, lambda it: False)
    with cg.capture(body):
        einsums.linalg.axpy(1.0, D, R)
    pm = cg.PassManager()
    pm.add(cg.Materialization())
    g.apply(pm)
    g.execute()
    np.testing.assert_allclose(np.asarray(R), x)
