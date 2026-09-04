# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""Differential fuzz for the structural-algebraic passes.

Every other shard in this directory leaves the region rewrites almost
untested. The random-pipeline fuzz permutes ten passes and none of them
restructures algebra; the region-identity shard raises and lowers with NO
rewrite; the einsum property shard runs the default pipeline but over
single-contraction programs, so a region never holds two terms and the pass
that exists to relate two terms cannot fire. What found the last three defects
in this area was one hand-written CCSD case.

So the corpus here is what those shards cannot draw: MULTI-STATEMENT
contraction programs. Several products over a shared pool of tensors, written
through author-named intermediates in a bracketing the generator chooses,
accumulating into shared outputs, with one tensor free to appear twice in one
product, and optionally an index annotated over two spaces the registry says
share no element.

Three things are asserted per trial and the third is the reason for the other
two. The numbers must agree with the same program run with no structural pass,
at the re-associating tier's bound, since folding a common factor out of a sum
changes the summation order and bit equality would be a detector of toolchains.
The numbers must also agree with numpy, so the two graphs cannot agree on a
wrong prefactor. And the rewritten graph must satisfy the storage invariants in
``_region_invariants``, which is where a rewrite that leaves a buffer behind
shows up: no arithmetic moves when a dissolved intermediate is still allocated.
"""

from __future__ import annotations

import itertools
import json
from typing import NamedTuple

import numpy as np
import pytest
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

import einsums
import einsums.graph as cg
from einsums.testing import ALL_DTYPES

from _region_invariants import assert_materialization_invariants
from _sanitizer_scaling import sanitizer_examples

_ctr = itertools.count()


def _nm(stem):
    return f"{stem}{next(_ctr)}"


# Letters the generator builds index groups from. Extents deliberately collide,
# because that is what lets one tensor serve two factors of a product: the pool
# is keyed by SHAPE, so a chain whose first and third factors have the same
# shape may draw the same tensor for both, which is the CCSD tau shape and the
# one case the pass's chain fixture never exercised.
#
# They also spread, because a cost model decides a bracketing and every extent
# being two or three makes every bracketing cost about the same: the passes then
# decline on the merits and the corpus proves nothing about them.
_EXTENTS = {"a": 2, "b": 6, "c": 2, "d": 6, "e": 3, "f": 8, "g": 3, "h": 8,
            "p": 16, "q": 64, "r": 16, "s": 64, "u": 24, "w": 96}

#: Letters a term draws its index groups from. The grouped alphabet keeps a
#: rank-four factor small enough to run; the flat one is rank two by
#: construction, so it can afford extents at which a cost model has an opinion
#: about a matrix chain at all, which is what ContractionPlanning needs to have
#: anything to restructure.
_GROUPED_LETTERS = "abcdefgh"
_FLAT_LETTERS = "pqrsuw"

#: Product prefactors. A negative one makes cancellation reachable, which is
#: where a re-association is most visible.
_AB_PFS = (1.0, -1.0, 0.5)


class Program(NamedTuple):
    """A multi-statement contraction program, operands named by key.

    ``stmts`` are binary contractions in program order, each
    ``(out, out_letters, a, a_letters, b, b_letters, c_pf, ab_pf)``. Keys
    beginning ``p`` are pool tensors the caller owns, ``t`` are author-named
    intermediates declared on the graph, ``r`` are results.
    ``disjoint`` is ``None`` or ``(statement index, summed letter)``: that
    statement's two operands get index spaces the registry declares disjoint,
    and the second operand is zeroed so the arithmetic honours the declaration.
    ``terms`` names the factors of each product, which the statements no longer
    say once a bracketing has split them across intermediates.
    """

    pool: dict
    inter: dict
    outs: dict
    stmts: tuple
    terms: tuple
    disjoint: object


def _draw_program(pick_int) -> Program:
    """Draw one program from an integer source.

    ``pick_int(lo, hi)`` returns an integer in ``[lo, hi]``. Hypothesis and a
    seeded numpy generator both supply one, so the property test and the corpus
    guard below draw from the same generator rather than from two that have to
    be kept in step.
    """
    pool: dict = {}
    pool_by_shape: dict = {}
    inter: dict = {}
    outs: dict = {}
    stmts: list = []
    terms: list = []

    def pool_key(shape):
        keys = pool_by_shape.setdefault(shape, [])
        while len(keys) < 2:
            key = f"p{len(pool)}"
            pool[key] = shape
            keys.append(key)
        return keys[pick_int(0, len(keys) - 1)]

    # Products already drawn, so a later term can be the SAME product written
    # with a different bracketing. That is the shape a shared intermediate can
    # exist in at all, and it is the CCSD tau case: two routes through two
    # author-named intermediates that flatten to one three-factor product.
    templates: list = []

    for term in range(pick_int(2, 3)):
        if templates and pick_int(0, 2):
            groups, factors, first_merge = templates[pick_int(0, len(templates) - 1)]
            factors = list(factors)
            if pick_int(0, 1):
                # One factor swapped for another tensor of the same shape, so
                # the two terms share a PAIR rather than the whole product.
                slot = pick_int(0, len(factors) - 1)
                factors[slot] = pool_key(tuple(_EXTENTS[x] for x in groups[slot] + groups[slot + 1]))
            factors = tuple(factors)
            # The OPPOSITE bracketing, not a redrawn one. Two routes through one
            # product is what makes a shared partial product exist to be found,
            # and a redraw would agree with the first term half the time and
            # leave the corpus with nothing for the search to do.
            first_merge = len(factors) - 2 - first_merge
        else:
            # Three factors mostly, since a product of two has no pair to share.
            n_factors = 2 if pick_int(0, 3) == 0 else 3
            # Each node of the chain is a GROUP of letters, so a factor is rank
            # two to four and the products have the shape a real residual term
            # does. Sometimes every group is a single letter, which is the plain
            # matrix chain ContractionPlanning is the only pass here to restructure.
            flat = pick_int(0, 2) == 0
            avail = list(_FLAT_LETTERS if flat else _GROUPED_LETTERS)
            groups = []
            for _ in range(n_factors + 1):
                size = 1 if flat else pick_int(1, 2)
                groups.append(tuple(avail.pop(pick_int(0, len(avail) - 1)) for _ in range(size)))
            groups = tuple(groups)
            factors = tuple(pool_key(tuple(_EXTENTS[x] for x in groups[t] + groups[t + 1]))
                            for t in range(n_factors))
            first_merge = pick_int(0, n_factors - 2)
            templates.append((groups, factors, first_merge))

        terms.append(factors)
        items = [[factors[t], groups[t] + groups[t + 1], t, t] for t in range(len(factors))]

        out_letters = groups[0] + groups[-1]
        out_dims = tuple(_EXTENTS[x] for x in out_letters)
        # Accumulating two terms into one output is what gives a region a SUM to
        # factor; without it every term is its own program.
        same = [k for k, v in outs.items() if v == out_dims] if pick_int(0, 1) else []
        if same:
            target, target_pf = same[pick_int(0, len(same) - 1)], 1.0
        else:
            target, target_pf = f"r{len(outs)}", 0.0
            outs[target] = out_dims

        step = 0
        while len(items) > 1:
            i = first_merge if step == 0 else pick_int(0, len(items) - 2)
            left, right = items[i], items[i + 1]
            lo, hi = left[2], right[3]
            letters = groups[lo] + groups[hi + 1]
            if len(items) == 2:
                key, c_pf = target, target_pf
            else:
                key, c_pf = f"t{term}_{step}", 0.0
                inter[key] = tuple(_EXTENTS[x] for x in letters)
            stmts.append((key, letters, left[0], left[1], right[0], right[1],
                          c_pf, _AB_PFS[pick_int(0, len(_AB_PFS) - 1)]))
            items[i:i + 2] = [[key, letters, lo, hi]]
            step += 1

    disjoint = None
    if pick_int(0, 1):
        uses: dict = {}
        for stmt in stmts:
            for key in (stmt[2], stmt[4]):
                uses[key] = uses.get(key, 0) + 1
        cands = []
        for index, stmt in enumerate(stmts):
            a, a_letters, b, b_letters = stmt[2], stmt[3], stmt[4], stmt[5]
            # Both operands must be pool tensors used nowhere else: the
            # annotation goes on the TENSOR, so a shared operand would carry one
            # statement's spaces into another's, and the zeroing would change a
            # term the declaration says nothing about.
            if not (a.startswith("p") and b.startswith("p")) or a == b:
                continue
            if uses[a] != 1 or uses[b] != 1:
                continue
            summed = [x for x in a_letters
                      if x in b_letters and x not in stmt[1]
                      and a_letters.count(x) == 1 and b_letters.count(x) == 1]
            if summed:
                cands.append((index, summed))
        if cands:
            index, summed = cands[pick_int(0, len(cands) - 1)]
            disjoint = (index, summed[pick_int(0, len(summed) - 1)])

    return Program(pool, inter, outs, tuple(stmts), tuple(terms), disjoint)


@st.composite
def _programs(draw):
    return _draw_program(lambda lo, hi: draw(st.integers(lo, hi)))


def _ccsd_tau_program() -> Program:
    """The tau term of the CCSD doubles residual, routed twice.

    Once through ``Wmnij``, contracting the virtual pair first into an o^4
    tensor, and once through ``Wabef``, contracting the occupied pair first into
    a v^4 one. Flattened through the two intermediates both routes are the same
    three-factor product with ``tau`` in it twice. Pinned because it is the case
    that found the three defects this shard exists to keep closed.
    """
    o, v = 3, 4
    return Program(
        pool={"p_tau": (o, o, v, v), "p_oovv": (o, o, v, v)},
        inter={"t_wmnij": (o, o, o, o), "t_wabef": (v, v, v, v)},
        outs={"r_t2n": (o, o, v, v)},
        stmts=(
            ("t_wmnij", ("m", "n", "i", "j"), "p_tau", ("i", "j", "e", "f"),
             "p_oovv", ("m", "n", "e", "f"), 0.0, 1.0),
            ("r_t2n", ("i", "j", "a", "b"), "p_tau", ("m", "n", "a", "b"),
             "t_wmnij", ("m", "n", "i", "j"), 0.0, 0.125),
            ("t_wabef", ("a", "b", "e", "f"), "p_tau", ("m", "n", "a", "b"),
             "p_oovv", ("m", "n", "e", "f"), 0.0, 1.0),
            ("r_t2n", ("i", "j", "a", "b"), "p_tau", ("i", "j", "e", "f"),
             "t_wabef", ("a", "b", "e", "f"), 1.0, 0.125),
        ),
        terms=(("p_tau", "p_oovv", "p_tau"), ("p_tau", "p_oovv", "p_tau")),
        disjoint=None,
    )


# ──────────────────────────────────────────────────────────────────────────
# Running a program
# ──────────────────────────────────────────────────────────────────────────

#: Passes that restructure algebra, in the relative order the default pipeline
#: gives them. Materialization closes the list because a graph holding a
#: declared intermediate cannot execute without it, and it is where a rewrite's
#: leftovers become visible.
def _region_pass_manager():
    mtf = cg.MultiTermFactorization()
    # Off by default, so a fuzz that did not switch it on would cover the pass
    # by never running it.
    mtf.set_search_enabled(True)
    delta = cg.DeltaElimination()
    passes = [
        delta,
        cg.LinearCombinationContractionFolding(),
        cg.DistributiveFactoring(),
        mtf,
        cg.LayoutAssignment(),
        cg.ContractionPlanning(),
        cg.Materialization(),
    ]
    # The region rewrites check their own cost line against the nodes they emit. Off by default
    # because it walks the node set once per rewrite; on here, because a report offering a cost
    # as evidence is exactly the kind of claim a fuzz should be checking.
    for region_pass in (delta, mtf):
        region_pass.set_verify_costs(True)
    pm = cg.PassManager()
    for p in passes:
        pm.add(p)
    return pm, passes


#: What each region pass calls a rewrite, for the corpus guard.
_FIRED_ATTR = {
    "DeltaElimination": "num_zero_blocks",
    "LinearCombinationContractionFolding": "num_eliminated",
    "DistributiveFactoring": "num_eliminated",
    "MultiTermFactorization": "num_shared",
    "LayoutAssignment": "num_relaid_out",
    "ContractionPlanning": "chains_restructured",
}


def _disjoint_registry():
    """occ and virt share no element. Everything else is aux and is unrelated."""
    registry = cg.SpaceRegistry()
    occ = registry.register_space(cg.index_space("occ", "o", 4.0))
    virt = registry.register_space(cg.index_space("virt", "v", 4.0))
    registry.register_space(cg.index_space("aux", "x", 4.0))
    registry.declare_disjoint(occ, virt)
    return registry


def _arrays(prog, dtype, seed):
    """Seed data for every pool tensor and every result, in @p dtype."""
    rng = np.random.default_rng(seed)
    dt = np.dtype(dtype)

    def gen(dims):
        out = rng.standard_normal(dims)
        if dt.kind == "c":
            out = out + 1j * rng.standard_normal(dims)
        return out.astype(dt)

    arrays = {key: gen(dims) for key, dims in prog.pool.items()}
    arrays.update({key: gen(dims) for key, dims in prog.outs.items()})
    if prog.disjoint is not None:
        # The declaration says the summed letter ranges over two spaces sharing
        # no element, so the product has no term. Held true in the data, because
        # a rewrite justified by a declaration can only be compared against
        # arithmetic that honours it.
        arrays[prog.stmts[prog.disjoint[0]][4]] *= 0
    return arrays


def _numpy_result(prog, arrays, dtype):
    dt = np.dtype(dtype)
    values = {key: array.copy() for key, array in arrays.items()}
    values.update({key: np.zeros(dims, dtype=dt) for key, dims in prog.inter.items()})
    for out, ol, a, al, b, bl, c_pf, ab_pf in prog.stmts:
        spec = f"{''.join(al)},{''.join(bl)}->{''.join(ol)}"
        term = np.einsum(spec, values[a], values[b])
        values[out] = (np.asarray(c_pf, dt) * values[out] + np.asarray(ab_pf, dt) * term).astype(dt)
    return {key: values[key] for key in prog.outs}


def _run(prog, arrays, dtype, region):
    """Build the program into a graph, optimize it, execute it.

    Returns the result arrays, the pass manager and the pass objects, so a
    caller can read the report and the counters off the same run that produced
    the numbers.
    """
    graph = cg.Graph(_nm("region" if region else "plain"))
    if prog.disjoint is not None:
        graph.set_space_registry(_disjoint_registry())

    tensors = {}
    for key in list(prog.pool) + list(prog.outs):
        tensor = einsums.create_zero_tensor(_nm(key), list(arrays[key].shape), dtype=dtype)
        np.asarray(tensor)[...] = arrays[key]
        tensors[key] = tensor
    for key, dims in prog.inter.items():
        tensors[key] = graph.declare_tensor(_nm(key), list(dims), intermediate=True, dtype=dtype)

    with cg.capture(graph):
        for out, ol, a, al, b, bl, c_pf, ab_pf in prog.stmts:
            spec = f"{','.join(ol)} <- {','.join(al)} ; {','.join(bl)}"
            einsums.einsum(spec, tensors[out], tensors[a], tensors[b], c_pf=c_pf, ab_pf=ab_pf)

    if prog.disjoint is not None:
        index, letter = prog.disjoint
        stmt = prog.stmts[index]
        # Annotated after the capture block, which is the surface a Python
        # caller has: the descriptor's capture-time space map holds nothing for
        # a program annotated afterwards, and the pass re-derives from the
        # operands' handles for exactly that reason.
        cg.annotate(tensors[stmt[2]], tuple("occ" if x == letter else "aux" for x in stmt[3]), graph=graph)
        cg.annotate(tensors[stmt[4]], tuple("virt" if x == letter else "aux" for x in stmt[5]), graph=graph)

    if region:
        pm, passes = _region_pass_manager()
    else:
        pm, passes = cg.PassManager(), [cg.Materialization()]
        pm.add(passes[0])
    pm.run(graph)
    graph.execute()
    return {key: np.asarray(tensors[key]).copy() for key in prog.outs}, graph, pm, passes


def _numpy_magnitude(prog, arrays, dtype):
    """The same program over absolute values, which is the scale an error bound
    is relative to.

    The RESULT is not that scale, and using it is the mistake this replaces. A
    generator that draws one product twice with opposite signs writes an
    output that is exactly zero and was computed from terms that are not, so a
    ratio against the result reports the cancellation rather than the rewrite:
    two orderings of a sum that cancels agree to a few ulps of the TERMS and to
    nothing at all of the answer. The sum of the magnitudes is the classical
    bound on a floating-point sum of products, and it degrades to the result
    when nothing cancels.
    """
    dt = np.dtype(dtype)
    real = np.dtype("float32") if dt.itemsize <= 8 else np.dtype("float64")
    values = {key: np.abs(array).astype(real) for key, array in arrays.items()}
    values.update({key: np.zeros(dims, dtype=real) for key, dims in prog.inter.items()})
    for out, ol, a, al, b, bl, c_pf, ab_pf in prog.stmts:
        spec = f"{''.join(al)},{''.join(bl)}->{''.join(ol)}"
        term = np.einsum(spec, values[a], values[b])
        values[out] = (abs(c_pf) * values[out] + abs(ab_pf) * term).astype(real)
    return float(np.linalg.norm(np.concatenate(
        [values[key].ravel().astype(np.float64) for key in sorted(prog.outs)])))


def _norm_gap(got, reference):
    """The norm of the difference over every result of the program.

    Taken over the concatenation rather than per tensor, so one output the
    disjointness draw zeroed does not become a measurement of its own.
    """
    a = np.concatenate([np.asarray(got[k]).ravel() for k in sorted(got)])
    b = np.concatenate([np.asarray(reference[k]).ravel() for k in sorted(reference)])
    return float(np.linalg.norm(a.astype(np.complex128) - b.astype(np.complex128)))


def _finite(values):
    return all(np.all(np.isfinite(np.asarray(v))) for v in values.values())


def _mtf_after_costs(pm):
    """The after side of every cost line MultiTermFactorization reported.

    The region report prints ``cost <before> -> <after>`` per accepted rewrite.
    The after side read ``0`` on every rewrite the pass had ever made, because
    the terms it emits were built without a cost, and nothing compared the two
    sides.
    """
    out = []
    for line in pm.explain().splitlines():
        if "MultiTermFactorization" in line and " cost " in line and " -> " in line:
            out.append(line.split(" cost ", 1)[1].split(" -> ", 1)[1].strip())
    return out


def _check(prog, dtype, seed=0):
    arrays = _arrays(prog, dtype, seed)
    expected = _numpy_result(prog, arrays, dtype)
    if not _finite(expected):
        pytest.skip("numerically degenerate program")

    plain, _plain_graph, _plain_pm, _plain_passes = _run(prog, arrays, dtype, region=False)
    rewritten, graph, pm, passes = _run(prog, arrays, dtype, region=True)

    # The storage invariants first: they hold whatever the numbers did, and a
    # rewrite that leaves a buffer behind moves none of them.
    assert_materialization_invariants(graph, f"region pipeline, dtype={dtype}")

    for region_pass in passes:
        mismatches = getattr(region_pass, "cost_mismatches", [])
        assert not mismatches, (
            f"{region_pass.name} reported a cost the nodes it emitted do not agree with\n"
            + "\n".join(mismatches)
        )

    for after in _mtf_after_costs(pm):
        assert after != "0", (
            "MultiTermFactorization reported a rewrite whose after-cost is zero, which would mean "
            f"it emitted nothing\nreport=\n{pm.explain()}"
        )

    # Re-associating, so the bar is the tier's bound rather than bit equality:
    # the rewrite sums the same products in a different order. The constant is
    # tier_bound(ReAssociating), which is 1024 epsilon.
    eps = float(np.finfo(np.dtype(dtype)).eps)
    scale = _numpy_magnitude(prog, arrays, dtype)
    bound = 1024.0 * eps * scale
    gap = _norm_gap(rewritten, plain)
    assert gap <= bound, (
        f"the region pipeline moved the answer by {gap:.3e}, past the re-associating "
        f"bound {bound:.3e} (dtype={dtype})\nprogram={prog!r}\nreport=\n{pm.explain()}"
    )

    # And against numpy, so the two graphs cannot agree on a wrong prefactor.
    # Looser than the tier bound because this compares a BLAS chain against
    # numpy's, which is a different computation rather than a reordered one and
    # may contract a multiply and an add the other keeps apart.
    oracle_gap = _norm_gap(rewritten, expected)
    assert oracle_gap <= 4096.0 * eps * scale, (
        f"the region pipeline disagrees with numpy by {oracle_gap:.3e}, past "
        f"{4096.0 * eps * scale:.3e} (dtype={dtype})\nprogram={prog!r}"
    )
    return passes


@pytest.mark.parametrize("dtype", ALL_DTYPES)
@given(prog=_programs())
@settings(max_examples=sanitizer_examples(100), deadline=None,
          suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large])
@example(prog=_ccsd_tau_program())
def test_the_region_pipeline_keeps_the_answer(prog, dtype):
    _check(prog, dtype)


# ──────────────────────────────────────────────────────────────────────────
# Corpus guards
#
# An equivalence over a corpus that never provokes the passes is vacuously
# true, and this shard exists because the region rewrites are the least
# provoked passes in the tree. So what the generator reaches is asserted rather
# than assumed, off the same draw function the property test uses.
# ──────────────────────────────────────────────────────────────────────────


def _rng_program(seed):
    rng = np.random.default_rng(seed)
    return _draw_program(lambda lo, hi: int(rng.integers(lo, hi + 1)))


def test_the_corpus_provokes_the_region_passes():
    fired = {name: 0 for name in _FIRED_ATTR}
    for seed in range(48):
        prog = _rng_program(seed)
        for pass_obj in _check(prog, "float64", seed=seed):
            attr = _FIRED_ATTR.get(pass_obj.name)
            if attr is not None and int(getattr(pass_obj, attr)):
                fired[pass_obj.name] += 1

    # MultiTermFactorization is the pass this corpus was built for, and
    # DeltaElimination's zero-block half is what the space annotation reaches.
    # The others are reported rather than demanded: a pass that finds nothing
    # here is not thereby broken, and an assertion on it would pin the
    # generator's roll rather than the pass.
    assert fired["MultiTermFactorization"] > 0, f"nothing shared an intermediate: {fired}"
    assert fired["DeltaElimination"] > 0, f"no zero block was proved: {fired}"
    assert fired["ContractionPlanning"] > 0, f"no chain was restructured: {fired}"


def test_the_generator_draws_the_shapes_the_passes_need():
    """The three properties that separate this corpus from the einsum shards."""
    repeated_operand = shared_output = annotated = False
    for seed in range(200):
        prog = _rng_program(seed)
        for factors in prog.terms:
            if len(set(factors)) != len(factors):
                repeated_operand = True
        writes: dict = {}
        for stmt in prog.stmts:
            writes[stmt[0]] = writes.get(stmt[0], 0) + 1
        if any(count > 1 for key, count in writes.items() if key.startswith("r")):
            shared_output = True
        if prog.disjoint is not None:
            annotated = True
    assert repeated_operand, "no product ever holds one tensor twice"
    assert shared_output, "no two terms ever accumulate into one output"
    assert annotated, "no program ever declares a disjointness"


def test_the_pinned_ccsd_case_still_shares_the_occupied_intermediate():
    """The pin is only a regression while it still provokes the rewrite."""
    prog = _ccsd_tau_program()
    arrays = _arrays(prog, "float64", 7)
    _out, graph, pm, passes = _run(prog, arrays, "float64", region=True)
    mtf = next(p for p in passes if p.name == "MultiTermFactorization")
    assert mtf.num_shared == 1
    assert mtf.num_rebracketed == 2

    # The v^4 intermediate is gone, and gone means unallocated: the dissolved
    # declaration stays, since the caller holds the handle.
    ir = json.loads(graph.to_json())
    dims = {t["name"]: t["dims"] for t in ir["tensors"]}
    materialized = {n["label"] for n in ir["nodes"] if n["kind"] == "Materialize"}
    v4 = next(name for name, d in dims.items() if d == [4, 4, 4, 4])
    assert f"materialize({v4})" not in materialized
    assert all(after != "0" for after in _mtf_after_costs(pm))
    assert_materialization_invariants(graph, "pinned CCSD tau terms")


# ──────────────────────────────────────────────────────────────────────────
# The lossy arm: a tagged energy denominator
#
# LaplaceTransform is a region rewrite like the others, and it is the only one
# whose oracle is a TOLERANCE rather than a re-association bound, so it draws
# its own programs rather than joining the roll above. Joining it would also
# renumber every program the shard has ever generated, which is the reason
# `rich_views` was added behind a flag in the differential shards.
#
# What varies is the shape the pass has to recognize: which operand of the
# numerator carries each axis of the denominator, the extents, the sign of each
# axis, and whether the denominator comes out positive or negative. What is
# fixed is the comparison: the transformed graph against the same program with
# the pass off, held to the error the pass itself RECORDED.
# ──────────────────────────────────────────────────────────────────────────

_LAPLACE_LETTERS = ("i", "a", "j", "b")


class LaplaceProgram(NamedTuple):
    """One drawn denominator problem.

    ``sides`` says, per axis of the denominator, which operand of the numerator
    carries that axis; ``link`` is the extent of the contracted index the two
    operands share, or zero for a numerator that is a pure outer product.
    """

    extents: tuple
    sides: tuple
    signs: tuple
    link: int
    negative: bool
    epsilon: float


@st.composite
def _laplace_programs(draw):
    rank = draw(st.integers(min_value=2, max_value=4))
    extents = tuple(draw(st.integers(min_value=2, max_value=4)) for _ in range(rank))
    # At least one axis on each side where the rank allows it, so the drawn
    # corpus actually exercises the split rather than always piling every
    # exponential onto one operand.
    sides = tuple(draw(st.integers(min_value=0, max_value=1)) for _ in range(rank))
    signs = tuple(draw(st.sampled_from((1, -1))) for _ in range(rank))
    link = draw(st.integers(min_value=0, max_value=3))
    negative = draw(st.booleans())
    epsilon = draw(st.sampled_from((1e-3, 1e-5, 1e-7)))
    return LaplaceProgram(extents, sides, signs, link, negative, epsilon)


def _laplace_arrays(prog, seed):
    """Energies whose signed sum is uniformly one sign, and the numerator's operands."""
    rng = np.random.default_rng(seed)
    energies = []
    for axis, extent in enumerate(prog.extents):
        # Positive and bounded away from zero, so the signed sum's range is set
        # by the signs rather than by an accident of the draw.
        base = np.array([0.4 + 0.35 * k for k in range(extent)])
        energies.append(base * (1.0 if prog.signs[axis] > 0 else 1.0))

    total = np.zeros(prog.extents)
    for axis, energy in enumerate(energies):
        shape = [1] * len(prog.extents)
        shape[axis] = prog.extents[axis]
        total = total + prog.signs[axis] * energy.reshape(shape)

    # A uniform shift moves the whole sum to one side of zero without touching
    # its width, which is what the quadrature needs and what a real denominator
    # of orbital energies has.
    span = float(total.max() - total.min())
    shift = (-float(total.min()) + 0.5 + span) if not prog.negative else (-float(total.max()) - 0.5 - span)
    axis0_sign = prog.signs[0]
    energies[0] = energies[0] + shift / axis0_sign
    total = total + shift

    denominator = 1.0 / total

    left = [prog.extents[k] for k in range(len(prog.extents)) if prog.sides[k] == 0]
    right = [prog.extents[k] for k in range(len(prog.extents)) if prog.sides[k] == 1]
    if prog.link:
        left = left + [prog.link]
        right = [prog.link] + right
    a = rng.standard_normal(left if left else [1])
    b = rng.standard_normal(right if right else [1])
    return energies, denominator, a, b


def _laplace_spec(prog):
    """The einsum the numerator is formed by, in the drawn letter assignment."""
    letters = _LAPLACE_LETTERS[: len(prog.extents)]
    left = [letters[k] for k in range(len(letters)) if prog.sides[k] == 0]
    right = [letters[k] for k in range(len(letters)) if prog.sides[k] == 1]
    if prog.link:
        left = left + ["q"]
        right = ["q"] + right
    if not left:
        left = ["z"]
    if not right:
        right = ["z"]
    return ",".join(left), ",".join(right), ",".join(letters)


def _run_laplace(prog, seed):
    energies, denominator, a, b, = _laplace_arrays(prog, seed)
    a_spec, b_spec, c_spec = _laplace_spec(prog)
    shape = list(prog.extents)

    def build(graph, out):
        A = einsums.asarray(np.ascontiguousarray(a))
        B = einsums.asarray(np.ascontiguousarray(b))
        D = einsums.asarray(np.ascontiguousarray(denominator))
        numerator = graph.scratch(_nm("lap_num"), shape, "float64")
        with cg.capture(graph):
            einsums.einsum(f"{a_spec} ; {b_spec} -> {c_spec}", numerator, A, B)
            einsums.linalg.direct_product(1.0, numerator, D, 0.0, out)
        return D

    exact = einsums.zeros(shape, dtype="float64")
    reference = cg.Graph(_nm("lap_ref"))
    build(reference, exact)
    reference.apply(cg.default_pass_manager())
    reference.execute()

    out = einsums.zeros(shape, dtype="float64")
    graph = cg.Graph(_nm("lap"))
    D = build(graph, out)
    tag = {"name": "laplace_denominator"}
    names = []
    for axis in range(len(prog.extents)):
        tag[f"axis{axis}"] = f"eps{axis}"
        tag[f"sign{axis}"] = "+" if prog.signs[axis] > 0 else "-"
        names.append(f"eps{axis}")
    cg.annotate(D, tag=tag, graph=graph)

    laplace = cg.LaplaceTransform()
    laplace.set_epsilon(prog.epsilon)
    for axis, name in enumerate(names):
        laplace.add_energy(name, einsums.asarray(np.ascontiguousarray(energies[axis])))
    pm = cg.PassManager()
    pm.add(laplace)
    changed = graph.apply(pm)
    assert changed, f"the pass declined a drawn program: {laplace.skip_reasons}"

    graph.apply(cg.default_pass_manager())
    graph.execute()
    assert_materialization_invariants(graph, "laplace")
    return np.asarray(out), np.asarray(exact), graph.approximations()[0], laplace


@given(prog=_laplace_programs())
@settings(max_examples=sanitizer_examples(60), deadline=None,
          suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large])
def test_the_quadrature_keeps_the_answer_inside_its_own_record(prog):
    got, want, record, laplace = _run_laplace(prog, seed=0)
    assert record.pass_name == "LaplaceTransform"
    assert record.bound <= prog.epsilon
    assert laplace.last_point_count >= 2

    scale = float(np.max(np.abs(want)))
    if scale == 0.0:
        return
    # The recorded bound, plus the double-precision rounding of an accumulation
    # over the quadrature points, which the record does not claim to cover.
    slack = 64.0 * float(np.finfo(np.float64).eps) * laplace.last_point_count
    assert float(np.max(np.abs(got - want))) <= (record.bound + slack) * scale


def test_the_drawn_corpus_reaches_both_sides_of_the_split():
    """A corpus that always piled every exponential onto one operand would prove nothing."""
    both = one_sided = negative = 0
    for seed in range(40):
        rng = np.random.default_rng(seed)
        prog = LaplaceProgram(
            extents=tuple(int(rng.integers(2, 5)) for _ in range(int(rng.integers(2, 5)))),
            sides=(),
            signs=(),
            link=int(rng.integers(0, 4)),
            negative=bool(rng.integers(0, 2)),
            epsilon=1e-5,
        )
        rank = len(prog.extents)
        prog = prog._replace(
            sides=tuple(int(rng.integers(0, 2)) for _ in range(rank)),
            signs=tuple(1 if rng.integers(0, 2) else -1 for _ in range(rank)),
        )
        if 0 in prog.sides and 1 in prog.sides:
            both += 1
        else:
            one_sided += 1
        if prog.negative:
            negative += 1
        _run_laplace(prog, seed=seed)
    assert both > 0, "no drawn program ever split the axes across the two operands"
    assert one_sided > 0, "no drawn program ever put every axis on one operand"
    assert negative > 0, "no drawn program ever had a negative denominator"
