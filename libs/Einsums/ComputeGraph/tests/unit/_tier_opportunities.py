# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""One opportunity generator per structural-algebraic pass.

Sorting the passes into their Part 5.1 tiers means running each one alone and
measuring how far its answer moves. That needs programs the pass actually
rewrites, and the differential corpus does not supply them: 480 single-pass
trials over the flat random generator produced zero rewrites across all twelve,
because that generator draws from a pool of multi-writer USER buffers while
these passes' guards want single-writer graph-owned scratch, a delta, a loop
body, or a chain worth re-parenthesizing.

So each pass gets a generator that builds the shape it responds to. A generator
returns ``(prog, m_arrays, v_arrays, t_arrays)``, where ``prog`` is either a
statement list or a ``(graph, m, v, t, name)`` callable; see
``_fuzz_diff_common._build_for_measurement``.

Two rules these generators live by, both learned by breaking them.

A generator must be DETERMINISTIC in the sense that matters here: it is invoked
twice, once for the unoptimized run and once with the pass applied, and the two
must see identical inputs. Drawing from an ``rng`` inside the builder body does
not do that, because the builder runs twice against an already-advanced engine;
the first version of the symmetrized-accumulation generator did exactly that
and reported a norm-relative error of 3.5 for a pass that was behaving. Draw
outside the builder and close over the arrays.

Every operand a generator uses lives in the POOL, which the harness holds for
the life of the trial, rather than being created inside the builder. That is
partly hygiene and partly a live bug: dropping the caller's references after
capture is safe for an unrewritten graph, and segfaults in
``make_axpby_executor`` once ``DistributiveFactoring`` has folded the group.
The pool sidesteps it; it does not fix it.

And a generator is only worth having if the pass FIRES on it, which is why the
accompanying test asserts that rather than asserting the numbers come out
small. A generator that quietly stops firing turns its pass's entry in the
classification into a statement about nothing.
"""

from __future__ import annotations

import numpy as np

import einsums
import einsums.graph as cg

_SQ_SPEC = "ij <- ik ; kj"


def _r(rng, *shape):
    return rng.standard_normal(shape)


def _pool(rng, count=6, n=3):
    return [_r(rng, n, n) for _ in range(count)]


# ── Statement-list generators ──────────────────────────────────────────────
# These passes respond to shapes the ordinary pool can express.


def gen_scale_absorption(rng):
    """A scale immediately absorbed by an overwriting consumer."""
    return [("scale", 3.0, 0), ("einsum", _SQ_SPEC, 1.0, 1, 2, 0.0, 0)], _pool(rng), [], []


def gen_element_wise_fusion(rng):
    """Two consecutive scales of one tensor, which fuse into one."""
    return [("scale", 2.0, 0), ("scale", 3.0, 0)], _pool(rng), [], []


def gen_permute_fusion(rng):
    """A transpose with a single consumer, folded into that consumer's slot."""
    return [("perm", 1.0, 0.0, 0, 1), ("einsum", _SQ_SPEC, 1.0, 1, 2, 0.0, 3)], _pool(rng), [], []


# ── Builder generators ─────────────────────────────────────────────────────
# These need something the statement format cannot say.


def gen_cse(rng):
    """A duplicated producer writing graph scratch, with a live consumer.

    The duplicate's output has to be graph-owned: CSE redirects readers to the
    survivor, and it will not do that for a buffer someone outside the graph
    may be holding.
    """
    def build(g, m, v, t, name):
        s1 = g.create_zero_tensor(f"{name}_s1", [3, 3], dtype="float64")
        s2 = g.create_zero_tensor(f"{name}_s2", [3, 3], dtype="float64")
        with cg.capture(g):
            einsums.einsum(_SQ_SPEC, s1, m[0], m[1])
            einsums.einsum(_SQ_SPEC, s2, m[0], m[1])
            einsums.einsum(_SQ_SPEC, m[2], s2, m[3])
    return build, _pool(rng), [], []


def gen_dead_node_elimination(rng):
    """A node writing graph scratch that nothing reads."""
    def build(g, m, v, t, name):
        s1 = g.create_zero_tensor(f"{name}_s1", [3, 3], dtype="float64")
        with cg.capture(g):
            einsums.einsum(_SQ_SPEC, s1, m[0], m[1])
            einsums.einsum(_SQ_SPEC, m[2], m[0], m[1])
    return build, _pool(rng), [], []


def gen_contraction_planning(rng):
    """A GEMM chain written in its expensive parenthesization.

    Written left to right the first product forms a 60x60; associated the other
    way nothing exceeds 60x4. The temporary must be a graph-owned intermediate,
    since re-parenthesizing means the pass replaces it, and it will not replace
    a buffer the caller owns.
    """
    def build(g, m, v, t, name):
        tmp = g.create_zero_tensor(f"{name}_t", [60, 60], intermediate=True, dtype="float64")
        with cg.capture(g):
            einsums.einsum(_SQ_SPEC, tmp, m[0], m[1])
            einsums.einsum(_SQ_SPEC, m[3], tmp, m[2])
    return build, [_r(rng, 60, 4), _r(rng, 4, 60), _r(rng, 60, 4), np.zeros((60, 4))], [], []


def gen_distributive_factoring(rng):
    """Two terms accumulating into one target and sharing their left operand.

    Sized at 128 deliberately. The pass costs the rewrite and declines it below
    roughly this size, which is correct of it and makes a small case measure
    nothing: at n=32 it reports the group as unprofitable and rewrites nothing.
    """
    n = 128
    m_arrays = [_r(rng, n, n), _r(rng, n, n), _r(rng, n, n), np.zeros((n, n))]
    prog = [("einsum", _SQ_SPEC, 1.0, 0, 1, 1.0, 3),
            ("einsum", _SQ_SPEC, 1.0, 0, 2, 1.0, 3)]
    return prog, m_arrays, [], []


def gen_linear_combination_contraction_folding(rng):
    """The 2J minus K shape: one tensor read twice under transposed indices."""
    def build(g, m, v, t, name):
        with cg.capture(g):
            einsums.einsum("i,j <- k ; k,i,j", m[0], v[0], t[0], c_pf=0.0, ab_pf=2.0)
            einsums.einsum("i,j <- k ; k,j,i", m[0], v[0], t[0], c_pf=1.0, ab_pf=-1.0)
    return build, [np.zeros((3, 3))], [_r(rng, 4)], [_r(rng, 4, 3, 3)]


def gen_loop_invariant_hoisting(rng):
    """A loop body holding a contraction that does not depend on the iteration."""
    def build(g, m, v, t, name):
        body = g.add_loop("loop", 5, lambda it: it < 4)
        with cg.capture(body):
            einsums.einsum(_SQ_SPEC, m[2], m[0], m[1])
            einsums.linalg.axpy(1.0, m[2], m[3])
    return build, _pool(rng), [], []


def gen_delta_elimination(rng):
    """A contraction against a tagged identity, feeding a second contraction."""
    def build(g, m, v, t, name):
        delta = einsums.asarray(np.eye(3))
        cg.annotate(delta, tag="identity", graph=g)
        tmp = g.create_zero_tensor(f"{name}_tmp", [3, 3], intermediate=True, dtype="float64")
        with cg.capture(g):
            einsums.einsum("i,j <- i,k ; k,j", tmp, m[0], delta)
            einsums.einsum("i,l <- i,j ; j,l", m[1], tmp, m[2])
    return build, _pool(rng), [], []


def gen_symmetrized_accumulation(rng):
    """An outer product accumulated, then accumulated again transposed.

    The operands are drawn HERE and closed over, not drawn inside the builder.
    The builder runs twice, and an rng read inside it would hand the two runs
    different matrices.
    """
    o, vv = 2, 3

    def build(g, m, v, t, name):
        with cg.capture(g):
            einsums.einsum("i,j,a,b <- i,a ; j,b", t[1], m[0], m[1])
            einsums.linalg.axpby(0.5, t[1], 1.0, t[0])
            einsums.permute("j,i,b,a <- i,j,a,b", t[2], t[1])
            einsums.linalg.axpby(0.5, t[2], 1.0, t[0])
    return (build,
            [_r(rng, o, vv), _r(rng, o, vv)],
            [],
            [np.zeros((o, o, vv, vv)) for _ in range(3)])


def gen_constant_folding(rng):
    """A contraction whose operands are graph-owned and written by nothing.

    The constants are FILLED outside capture, so they hold real values and no
    node writes them. Both halves matter: a node writing them would make them
    non-constant, and leaving them zero would fold an all-zero contraction and
    prove nothing about the value that gets baked.
    """
    k = _r(rng, 3, 3)

    def build(g, m, v, t, name):
        konst = g.create_zero_tensor(f"{name}_k", [3, 3], intermediate=True, dtype="float64")
        folded = g.create_zero_tensor(f"{name}_f", [3, 3], intermediate=True, dtype="float64")
        np.asarray(konst)[...] = k
        with cg.capture(g):
            einsums.einsum(_SQ_SPEC, folded, konst, konst)
            einsums.linalg.axpy(1.0, folded, m[0])
    return build, [np.zeros((3, 3))], [], []


def gen_layout_assignment(rng):
    """A deferred intermediate whose captured axis order costs a copy at each end.

    ``W`` is written (i,j,x) and read with its contracted letter ``j`` between its
    two free ones, which has no flat (M,K) reading at all; the producer pays for
    it too, because C's free groups then disagree with B's. Storing W as (i,x,j)
    removes both copies, and no local rewrite of either contraction can see the
    other's cost - which is the whole reason this pass is not a peephole.

    W is DEFERRED rather than eagerly allocated, because re-laying out a live
    buffer is a data movement the pass does not perform and it declines one. The
    harness runs Materialization on both sides so the graph can execute.
    """
    i, j, x, k, y = 4, 3, 5, 2, 3
    a = _r(rng, i, k)
    b = _r(rng, k, x, j)
    d = _r(rng, j, y)

    def build(g, m, v, t, name):
        A = einsums.create_zero_tensor(f"{name}_A", [i, k], dtype="float64")
        B = einsums.create_zero_tensor(f"{name}_B", [k, x, j], dtype="float64")
        D = einsums.create_zero_tensor(f"{name}_D", [j, y], dtype="float64")
        np.asarray(A)[...] = a
        np.asarray(B)[...] = b
        np.asarray(D)[...] = d
        W = g.declare_tensor(f"{name}_W", [i, j, x], intermediate=True, dtype="float64")
        R = t[0]
        with cg.capture(g):
            einsums.einsum("i,j,x <- i,k ; k,x,j", W, A, B)
            einsums.einsum("i,x,y <- i,j,x ; j,y", R, W, D)

    return build, [], [], [np.zeros((i, x, y))]


#: Pass name to opportunity generator. A pass here is one the classification can
#: gather evidence about.
OPPORTUNITY_GENERATORS = {
    "ScaleAbsorption": gen_scale_absorption,
    "ElementWiseFusion": gen_element_wise_fusion,
    "PermuteFusion": gen_permute_fusion,
    "CSE": gen_cse,
    "DeadNodeElimination": gen_dead_node_elimination,
    "ContractionPlanning": gen_contraction_planning,
    "DistributiveFactoring": gen_distributive_factoring,
    "LinearCombinationContractionFolding": gen_linear_combination_contraction_folding,
    "LoopInvariantHoisting": gen_loop_invariant_hoisting,
    "DeltaElimination": gen_delta_elimination,
    "SymmetrizedAccumulation": gen_symmetrized_accumulation,
    "ConstantFolding": gen_constant_folding,
    "LayoutAssignment": gen_layout_assignment,
}

#: Passes with no generator, and why. Kept as data rather than left out, so the
#: gap is something a reader can see rather than something they have to notice.
#: Empty is the goal and, as of 2026-09-01, the state: ConstantFolding was the
#: last entry, and it was here because the pass could not fire rather than
#: because nobody had written it a shape.
NO_GENERATOR = {}
