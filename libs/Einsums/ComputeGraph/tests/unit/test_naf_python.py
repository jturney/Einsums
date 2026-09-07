# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""``NaturalAuxiliaryFactorization`` on a real molecule's integrals, offline.

The first provider that makes an index SMALLER rather than replacing one tensor
by several. What it offers is

    B[Q,i,a] = sum_Y U[Q,Y] Bt[Y,i,a]

with ``Y`` the auxiliary directions whose singular values survive a threshold,
and the whole claim is that a density-fitted term reads ``B`` twice, so both
occurrences are substituted and the two ``U`` factors meet inside one product.

Nothing here tells the pass that the two of them contract to an identity. The
bracketing search puts that contraction first because a ``Y`` by ``Y`` matrix is
the cheapest thing it can build out of them, and the cost model then ranks the
truncated form below the captured one on the extents alone. Both halves of that
sentence are asserted rather than described: the emitted algebra is read back and
checked to be the three statements the design predicts, and the rung that decided
the accept is named.

The water numbers, at cc-pVDZ with 84 auxiliary functions over the 5 by 19
occupied-virtual block, are in ``test_the_threshold_sweep_...``. The short
version is that the published claim of "about half" is met at a threshold of
1e-1, which keeps 39, and that the recorded bound is loose by one to three
orders of magnitude against the energy error, in the safe direction.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

import einsums
import einsums._core.graph as _G
import einsums.graph as cg

_HERE = os.path.dirname(os.path.abspath(__file__))
_FIXTURE = os.path.normpath(
    os.path.join(_HERE, "..", "..", "..", "..", "..", "examples", "dlpno", "fixtures", "water-ccpvdz.npz"))

#: The family the caller declares. Only the RATIOS matter to the comparison, and
#: these are a medium molecule in a triple-zeta basis: the auxiliary set is about
#: three times the basis and the truncated one is half of that, which is the
#: claim ``register_naf_space`` writes into the space's typical extent.
_FAMILY = (("occ", "o", 300.0, "nocc"), ("vir", "v", 2700.0, "nvir"), ("aux", "x", 8100.0, "naux"))


def _tensor(name, array):
    tensor = einsums.create_zero_tensor(name, list(array.shape))
    np.asarray(tensor)[...] = np.ascontiguousarray(array)
    return tensor


@pytest.fixture(scope="module")
def water():
    """Canonical orbitals and the density-fitted three-index integrals."""
    if not os.path.exists(_FIXTURE):
        pytest.skip(f"fixture not present: {_FIXTURE}")
    z = np.load(_FIXTURE, allow_pickle=False)

    overlap, fock = z["S"], z["F"]
    nbf = overlap.shape[0]
    values, vectors = np.linalg.eigh(overlap)
    orthogonalizer = vectors @ np.diag(values ** -0.5) @ vectors.T
    energies, rotation = np.linalg.eigh(orthogonalizer.T @ fock @ orthogonalizer)
    coefficients = orthogonalizer @ rotation
    nocc = int(z["C_occ"].shape[1])
    nvir = nbf - nocc

    metric = z["metric"]
    mvals, mvecs = np.linalg.eigh(metric)
    inv_sqrt = mvecs @ np.diag(np.where(mvals > 1e-10, mvals ** -0.5, 0.0)) @ mvecs.T
    three_ao = np.einsum("PQ,Qmn->Pmn", inv_sqrt, z["eri_3index"])
    three = np.einsum("Pmn,mp,nq->Ppq", three_ao, coefficients, coefficients)

    o, v = slice(0, nocc), slice(nocc, nbf)
    gaps = 1.0 / (energies[o, None, None, None] - energies[None, v, None, None]
                  + energies[None, None, o, None] - energies[None, None, None, v])
    return {
        "nocc": nocc, "nvir": nvir, "nbf": nbf, "naux": three.shape[0],
        "B_ov": np.ascontiguousarray(three[:, o, v]),
        "B_vv": np.ascontiguousarray(three[:, v, v]),
        "gaps": np.ascontiguousarray(gaps),
        "reference": float(z["energy_psi4_df_mp2"]),
    }


@pytest.fixture(scope="module", autouse=True)
def family():
    """The family declared ONCE, on the process-global registry.

    Not on a registry of each graph's own, which would be the tidier thing to
    write and is ruled out by the file: a load resolves space names against the
    process-global registry, so a saved graph annotated against a private one
    cannot be read back. The truncated space's typical extent is DERIVED from the
    space it sits inside, and a derived declaration made twice with two different
    families is a contradiction the registry rightly refuses, so it is made here,
    before any test, rather than per program.
    """
    registry = cg.global_space_registry()
    for name, symbol, extent, dim in _FAMILY:
        registry.register_space(cg.index_space(name, symbol, extent, cg.GrowthClass.linear(), dim))
    graph = cg.Graph("naf_family")
    _G.NaturalAuxiliaryFactorization.register_naf_space(graph)
    return registry


def _integral_program(problem, name, annotated=True):
    """``K[i,a,j,b] = sum_Q B[Q,i,a] B[Q,j,b]``, the shape a DF program writes.

    One tagged tensor read TWICE, which is the shape the whole method turns on:
    the two ``U`` factors only meet if both occurrences are substituted.
    """
    nocc, nvir = problem["nocc"], problem["nvir"]
    three = _tensor("B_ov", problem["B_ov"])
    integrals = einsums.create_zero_tensor("K", [nocc, nvir, nocc, nvir])

    graph = cg.Graph(name)
    with cg.capture(graph):
        einsums.einsum("Q,i,a ; Q,j,b -> i,a,j,b", integrals, three, three)
    if annotated:
        cg.annotate(three, ("aux", "occ", "vir"), graph=graph)
        cg.annotate(integrals, ("occ", "vir", "occ", "vir"), graph=graph)
    graph.annotate_tag(three, _G.ProvenanceTag.make("eri"))
    return {"graph": graph, "three": three, "integrals": integrals}


def _truncate(program, threshold, report=None):
    """Register the provider and run the pass over @p program."""
    provider = _G.NaturalAuxiliaryFactorization("eri", program["three"], threshold)
    if report is not None:
        provider.report_dropped_into(report)
    registry = _G.FactorizationRegistry()
    registry.add(provider)
    factorization = _G.FactorizationPass(registry)
    factorization.set_dump(True)
    manager = cg.PassManager()
    # A tag declared on the graph reaches the operand handles through the analysis phase.
    manager.add(cg.ProvenancePropagation())
    manager.add(factorization)
    program.update({"provider": provider, "registry": registry, "pass": factorization,
                    "fired": program["graph"].apply(manager)})
    return program


def _emitted(graph):
    """The intermediates the rewrite declared, by name and extents."""
    document = json.loads(graph.to_json())
    return {entry["name"]: entry["dims"] for entry in document["tensors"]
            if "NaturalAuxiliary" in entry["name"] and "_x" in entry["name"]}


# ──────────────────────────────────────────────────────────────────────────
# What the truncation is worth
# ──────────────────────────────────────────────────────────────────────────


def test_the_threshold_sweep_reports_the_count_the_error_and_the_bound(water):
    """The water table, and the two properties that make the record usable.

    Loosening the threshold keeps fewer directions, and at every threshold the
    error of the rewritten integrals against the untruncated replay is below the
    bound the record carries. The bound is the norm of the dropped singular
    values, which is a bound on the integrals rather than a description of them,
    so it is loose and loose in the safe direction.

    Not asserted: that the error falls monotonically with the threshold. It does
    not, and it should not be expected to. Two dropped directions can cancel in a
    contraction that neither dominates, so a tighter threshold occasionally lands
    on a slightly worse number; what the record promises is a ceiling, and the
    ceiling is what this checks.
    """
    exact = np.einsum("Qia,Qjb->iajb", water["B_ov"], water["B_ov"])
    rows = []
    for threshold in (1e-1, 3e-2, 1e-2, 1e-3):
        program = _truncate(_integral_program(water, f"naf_sweep_{threshold}"), threshold)
        assert program["fired"], f"threshold {threshold:g} declined: {dict(program['pass'].skip_reasons)}"
        program["graph"].apply(cg.default_pass_manager())
        program["graph"].execute()

        got = np.asarray(program["integrals"])
        error = np.linalg.norm(got - exact) / np.linalg.norm(exact)
        bound = program["provider"].dropped_norm
        rows.append((threshold, program["provider"].kept, error, bound))
        assert error <= bound, f"threshold {threshold:g}: error {error:.3e} above the recorded bound {bound:.3e}"

    counts = [kept for _t, kept, _e, _b in rows]
    assert counts == sorted(counts), f"a tighter threshold kept fewer directions: {rows}"
    assert all(kept < water["naux"] for _t, kept, _e, _b in rows)
    # The published claim about the method, met on this molecule at the loosest
    # threshold of the sweep: the auxiliary index is about halved.
    assert rows[0][1] == 39 and water["naux"] == 84


def test_the_correlation_energy_stays_inside_the_composed_bound(water):
    """The DF-MP2 energy through the truncated auxiliary index.

    Measured against the UNTRUNCATED replay of the same program rather than
    against psi4, because what the bound is about is the rewrite and not the
    equations; the untruncated arm is checked against psi4 first so the quantity
    being compared is the right one.
    """
    nocc, nvir = water["nocc"], water["nvir"]
    exact = np.einsum("Qia,Qjb->iajb", water["B_ov"], water["B_ov"])
    exchange = exact.transpose(0, 3, 2, 1)
    reference = float(np.einsum("iajb,iajb->", 2.0 * exact - exchange, exact * water["gaps"]))
    assert reference == pytest.approx(water["reference"], abs=1e-9)

    program = _truncate(_integral_program(water, "naf_energy"), 1e-1)
    assert program["fired"]
    graph, integrals = program["graph"], program["integrals"]

    # The energy is formed OUTSIDE the graph, out of the rewritten integrals, so
    # what this measures is the truncation's effect on the quantity the record
    # covers rather than a second program's rounding.
    graph.apply(cg.default_pass_manager())
    graph.execute()
    got = np.asarray(integrals)
    got_exchange = got.transpose(0, 3, 2, 1)
    energy = float(np.einsum("iajb,iajb->", 2.0 * got - got_exchange, got * water["gaps"]))

    tolerance = graph.approximation_tolerance()
    assert tolerance.relative == pytest.approx(program["provider"].dropped_norm)
    assert abs(energy - reference) <= tolerance.relative * abs(reference), (
        f"energy error {abs(energy - reference) / abs(reference):.3e} against a bound of {tolerance.relative:.3e}")


# ──────────────────────────────────────────────────────────────────────────
# The mechanism
# ──────────────────────────────────────────────────────────────────────────


def test_the_two_transformation_factors_are_contracted_first(water):
    """The rewrite the entry predicts, read back as algebra rather than assumed.

    Three statements: the two ``U`` factors over the full auxiliary index into a
    small square matrix, the truncated factor through that matrix, and the
    expensive contraction over the truncated index. Nothing was taught that the
    small matrix is an identity; it comes out first because it is the cheapest
    thing the search can build.
    """
    program = _truncate(_integral_program(water, "naf_algebra"), 1e-1)
    assert program["fired"]

    after = program["pass"].dump_text.split("after:")[-1]
    statements = [line.strip() for line in after.splitlines() if line.strip()]
    assert len(statements) == 3, after
    assert "NaturalAuxiliary_U[Q,Y] NaturalAuxiliary_U[Q,Y1]" in statements[0], after
    assert statements[0].endswith("2*x*y^2"), after
    # The expensive contraction is over the TRUNCATED index: the polynomial the
    # pass priced names y where the captured one named the auxiliary letter.
    assert statements[-1].endswith("2*o^2*v^2*y"), after
    assert program["pass"].num_factorized == 1


def test_every_emitted_intermediate_carries_the_truncated_index(water):
    """Extents rather than a cost line, which is what the rewrite actually buys.

    Past the contraction of the two transformation factors nothing is over the
    full auxiliary index at all: the two ``U`` matrices are the only tensors in
    the rewritten program that still carry 84, and they are the fit's own
    factors rather than intermediates the contraction pays for.
    """
    program = _truncate(_integral_program(water, "naf_extents"), 1e-1)
    assert program["fired"]
    kept, naux = program["provider"].kept, water["naux"]

    shapes = _emitted(program["graph"])
    assert shapes, "the rewrite emitted no intermediates, so this asserts nothing"
    for name, dims in shapes.items():
        assert naux not in dims, f"{name} {dims} still carries the full auxiliary index"
        assert kept in dims, f"{name} {dims} does not carry the truncated index"


def test_the_rung_that_decides_is_the_family_extents_and_not_the_tie_break(water):
    """What "symbolically cheaper" means here, and what it costs to say nothing.

    With the family declared, the accept is decided by TYPICAL EXTENTS: scale
    order alone cannot rank the two forms, because the substituted cost carries a
    term over the truncated index squared and the captured cost has only one
    auxiliary factor to match it against, and the registry does not relate the
    truncated space to the virtual one. The rung below it substitutes the
    family's own extents and answers.

    That is what ``register_naf_space`` deriving a typical extent from the space
    it sits inside is for. Without one the truncated form carries a variable the
    substitution cannot resolve, and the rung ranks an unsubstitutable polynomial
    BELOW a substitutable one, so the rewrite is declined for being unmeasurable
    rather than for being expensive.

    Unannotated, the same program falls all the way to the documented
    lexicographic tie-break, which is arbitrary on purpose. It accepts here, and
    the point of pinning it is that the phrase "symbolically cheaper" on an
    unannotated program is not the family argument a reader would take it for.
    """
    annotated = _truncate(_integral_program(water, "naf_rung_annotated"), 1e-1)
    assert annotated["fired"]
    assert annotated["pass"].accept_rung == "TypicalExtent"

    bare = _truncate(_integral_program(water, "naf_rung_bare", annotated=False), 1e-1)
    assert bare["fired"]
    assert bare["pass"].accept_rung == "Lexicographic"


def test_the_truncated_space_is_declared_inside_the_auxiliary_one(water):
    """Containment, and its first client.

    A contraction over the truncated index is a RESTRICTION of the one over the
    full auxiliary index rather than an unrelated quantity, and the registry is
    where that is written down. The scale order beside it is a separate
    statement, which the registry insists on: containment implies nothing about
    which of two spaces is larger.
    """
    graph = cg.Graph("naf_containment")
    registry = graph.space_registry
    truncated, full = registry.find("naf"), registry.find("aux")
    assert registry.is_contained(truncated, full) == cg.Tristate.Yes
    assert registry.is_less(truncated, full) == cg.Tristate.Yes
    assert registry.is_disjoint(truncated, full) == cg.Tristate.No
    assert registry.space(truncated).dim_symbol == "nnaf"
    assert registry.space(truncated).typical_extent == pytest.approx(0.5 * registry.space(full).typical_extent)

    # With nothing to sit inside, the space is still registered and no relation is
    # invented for it: a weaker comparison rather than a wrong one.
    bare = cg.Graph("naf_containment_bare")
    bare_registry = cg.SpaceRegistry()
    bare.set_space_registry(bare_registry)
    _G.NaturalAuxiliaryFactorization.register_naf_space(bare)
    assert bare_registry.find("naf") is not None
    assert bare_registry.space(bare_registry.find("naf")).typical_extent == 0.0


def test_two_families_declaring_the_truncated_space_need_two_registries():
    """A derived extent turns one name into two spaces, and the registry says so.

    ``register_naf_space`` reads the typical extent off the space the truncated
    one sits inside, so the same name carries a different number for every
    family a process declares. A registry is a namespace and holds one
    declaration per name: the second one is refused rather than absorbed, and
    the message names the field and BOTH values, because a caller that cannot
    see which two declarations collided cannot tell which of them to move.

    The answer is a registry per caller rather than a looser registry. The same
    two declarations against two private registries both stand, each carrying
    its own extent, and that is the one line a program that does not save a
    graph should be written with.
    """
    shared = cg.Graph("naf_shared_registry")
    registry = cg.private_space_registry(shared)
    registry.register_space(cg.index_space("aux_small", "x", 100.0, cg.GrowthClass.linear(), "naux"))
    registry.register_space(cg.index_space("aux_large", "x", 900.0, cg.GrowthClass.linear(), "naux"))
    _G.NaturalAuxiliaryFactorization.register_naf_space(shared, "aux_small")

    with pytest.raises(ValueError) as conflict:
        _G.NaturalAuxiliaryFactorization.register_naf_space(shared, "aux_large")
    message = str(conflict.value)
    assert "typical extent 50 against 450" in message, message
    assert "registry of its own" in message, message
    # Refused, never overwritten: what the registry holds is still the first claim.
    assert registry.space(registry.find("naf")).typical_extent == pytest.approx(50.0)

    # The same two declarations, one registry each, both stand.
    extents = []
    for outer, extent in (("aux", 100.0), ("aux", 900.0)):
        graph = cg.Graph("naf_own_registry")
        own = cg.private_space_registry(graph)
        own.register_space(cg.index_space(outer, "x", extent, cg.GrowthClass.linear(), "naux"))
        truncated = _G.NaturalAuxiliaryFactorization.register_naf_space(graph, outer)
        extents.append(own.space(truncated).typical_extent)
    assert extents == [pytest.approx(50.0), pytest.approx(450.0)]


# ──────────────────────────────────────────────────────────────────────────
# What it declines
# ──────────────────────────────────────────────────────────────────────────


def _synthetic_program(name, naux=6, rows=4, cols=5, seed=7):
    """A small full-rank three-index tensor, for the refusals water cannot pose.

    The water spectrum is rank deficient in its last direction, so a threshold
    below every surviving singular value still drops one there and the "nothing
    was dropped" refusal is unreachable on it.
    """
    generator = np.random.default_rng(seed)
    values = generator.standard_normal((naux, rows, cols))
    three = _tensor("B_small", values)
    result = einsums.create_zero_tensor("C_small", [rows, cols, rows, cols])
    graph = cg.Graph(name)
    with cg.capture(graph):
        einsums.einsum("Q,m,n ; Q,p,q -> m,n,p,q", result, three, three)
    graph.annotate_tag(three, _G.ProvenanceTag.make("eri"))
    return {"graph": graph, "three": three, "integrals": result}


def test_a_threshold_that_drops_nothing_is_declined(capfd):
    """A truncation that truncates nothing is a rename, and says so.

    Below every singular value of a full-rank spectrum every direction survives,
    and the substitution would replace one tensor with a square rotation and a
    copy of itself: strictly more arithmetic and strictly more storage for the
    same number.
    """
    program = _synthetic_program("naf_no_drop")
    provider = _G.NaturalAuxiliaryFactorization("eri", program["three"], 1e-12)
    registry = _G.FactorizationRegistry()
    registry.add(provider)
    factorization = _G.FactorizationPass(registry)
    manager = cg.PassManager()
    manager.add(cg.ProvenancePropagation())
    manager.add(factorization)
    manager.set_verbosity(3)  # the reason is the detail behind the decline

    assert not program["graph"].apply(manager)
    assert any("provider declined" in reason for reason, _count in factorization.skip_reasons), (
        factorization.skip_reasons)
    assert "rename rather than a truncation" in capfd.readouterr().err


def test_a_tensor_this_provider_was_not_given_is_declined(water, capfd):
    """The claim that the provider holds the tensor it truncates is CHECKED.

    A truncation is only sound if the spectrum it read is the spectrum of the
    tensor the pass is about to substitute away, so handing over a different
    buffer is a decline rather than a truncation of something else.
    """
    program = _integral_program(water, "naf_other_tensor")
    other = _tensor("B_other", water["B_ov"] * 2.0)
    provider = _G.NaturalAuxiliaryFactorization("eri", other, 1e-1)
    registry = _G.FactorizationRegistry()
    registry.add(provider)
    factorization = _G.FactorizationPass(registry)
    manager = cg.PassManager()
    manager.add(cg.ProvenancePropagation())
    manager.add(factorization)
    manager.set_verbosity(3)

    assert not program["graph"].apply(manager)
    assert "not the one this provider was given to truncate" in capfd.readouterr().err


def test_a_threshold_outside_the_unit_interval_is_refused(water):
    """A relative cutoff of one or more, or a negative one, is a construction error."""
    three = _tensor("B_ov", water["B_ov"])
    for bad in (-1e-12, 1.0, 2.0):
        with pytest.raises(ValueError):
            _G.NaturalAuxiliaryFactorization("eri", three, bad)
    # Zero is spellable and means "take the option", which is how every other
    # provider's tolerance is spelled.
    assert _G.NaturalAuxiliaryFactorization("eri", three, 0.0).threshold > 0.0


# ──────────────────────────────────────────────────────────────────────────
# The measurement, and the file
# ──────────────────────────────────────────────────────────────────────────


def test_the_record_is_measured_and_the_fitting_re_measures_it_per_bind(water):
    """The number in the record and the number in the tensor are the same one.

    A record is written once, at optimize time, so what it carries is the
    truncation's error over the integrals the capture held. What a bind leaves
    behind is a value in the tensor the record names, and the two agree here
    because this bind is that bind.
    """
    dropped = einsums.create_zero_tensor("naf_dropped", [1])
    program = _truncate(_integral_program(water, "naf_measured"), 1e-1, report=dropped)
    assert program["fired"]

    records = program["graph"].approximations()
    assert [r.pass_name for r in records] == ["NaturalAuxiliary"]
    assert records[0].origin == _G.ApproximationOrigin.Measured
    assert records[0].measurement == "naf_dropped"
    assert records[0].tolerance == pytest.approx(1e-1)
    assert list(records[0].spaces) == ["naf"]

    program["graph"].apply(cg.default_pass_manager())
    program["graph"].execute()
    assert float(np.asarray(dropped)[0]) == pytest.approx(records[0].bound, rel=1e-9)


def test_the_truncated_graph_saves_loads_rebinds_and_refits(water, tmp_path):
    """A file, a fresh set of buffers, and the same tensor on the other side.

    Saved BEFORE the default manager runs, because a ``Materialize`` node holds
    an allocating closure and allocation is a resource decision this design
    re-derives on load rather than storing.

    Three things the loaded graph asks for are worth naming. The three-index
    tensor arrives twice, once for the caller's algebra and once under the
    fitting's own name, which is what a fit that reads the tensor it replaces
    always costs. The rectangular truncation matrix arrives as an interface
    tensor of its own: taking the leading columns of an eigenvector matrix is not
    an operation this node set can save, so the truncation is spelled as a
    contraction against the leading columns of an identity, and that identity is
    a constant of the structure a bind supplies. And the measurement tensor
    arrives, because the refit writes it.
    """
    dropped = einsums.create_zero_tensor("naf_dropped", [1])
    program = _truncate(_integral_program(water, "naf_roundtrip"), 1e-1, report=dropped)
    assert program["fired"]
    assert program["graph"].serializability_report() == []

    path = str(tmp_path / "naf.eig")
    cg.save_graph(program["graph"], path)

    program["graph"].apply(cg.default_pass_manager())
    program["graph"].execute()
    in_process = np.array(np.asarray(program["integrals"]), copy=True)

    loaded = cg.load_graph(path)
    records = loaded.approximations()
    assert [r.pass_name for r in records] == ["NaturalAuxiliary"]
    assert records[0].origin == _G.ApproximationOrigin.Measured

    names = set(loaded.manifest_names())
    keep = _G.NaturalAuxiliaryFactorization.keep_matrix_name("NaturalAuxiliary", "B_ov")
    assert {"B_ov", "B_ov@fit", "K", "naf_dropped", keep} <= names, sorted(names)

    nocc, nvir, naux = water["nocc"], water["nvir"], water["naux"]
    kept = program["provider"].kept
    identity = np.zeros((naux, kept))
    identity[np.arange(kept), np.arange(kept)] = 1.0
    replayed = einsums.create_zero_tensor("K", [nocc, nvir, nocc, nvir])
    fresh_dropped = einsums.create_zero_tensor("naf_dropped", [1])
    fresh = {
        "B_ov": _tensor("B_ov", water["B_ov"]),
        "B_ov@fit": _tensor("B_ov", water["B_ov"]),
        keep: _tensor(keep, identity),
        "naf_dropped": fresh_dropped,
        "K": replayed,
    }
    cg.bind(loaded, {name: tensor for name, tensor in fresh.items() if name in names})
    loaded.apply(cg.default_pass_manager())
    loaded.execute()

    got = np.asarray(replayed)
    gap = np.linalg.norm(got - in_process) / np.linalg.norm(in_process)
    assert gap <= 1e-10, f"norm-relative gap {gap:.3e}"
    # REFITTED rather than replayed from stored factors: the eigen-decomposition
    # ran again on the bound integrals and wrote what it threw away.
    assert float(np.asarray(fresh_dropped)[0]) == pytest.approx(records[0].bound, rel=1e-8)


def test_a_rebind_at_different_integrals_refits_at_the_same_count(water, tmp_path):
    """The narrowing, pinned: the COUNT is optimize time and the fit is per bind.

    The emitted node set is sized by how many directions survive, so the count
    cannot move at bind and does not. What a rebind gets is the same count over
    whatever integrals it then finds, and the honest report of that is the
    measured error, which is a different number here because the integrals are.
    """
    dropped = einsums.create_zero_tensor("naf_dropped", [1])
    program = _truncate(_integral_program(water, "naf_rebind"), 1e-1, report=dropped)
    assert program["fired"]
    kept = program["provider"].kept

    path = str(tmp_path / "naf_rebind.eig")
    cg.save_graph(program["graph"], path)
    loaded = cg.load_graph(path)
    names = set(loaded.manifest_names())
    keep = _G.NaturalAuxiliaryFactorization.keep_matrix_name("NaturalAuxiliary", "B_ov")

    naux = water["naux"]
    identity = np.zeros((naux, kept))
    identity[np.arange(kept), np.arange(kept)] = 1.0
    # A different problem over the same family: the virtual block of the same
    # integrals, folded onto the occupied-virtual shape so the extents match.
    perturbed = water["B_ov"] + 0.25 * water["B_vv"][:, :water["nocc"], :]
    nocc, nvir = water["nocc"], water["nvir"]
    replayed = einsums.create_zero_tensor("K", [nocc, nvir, nocc, nvir])
    fresh_dropped = einsums.create_zero_tensor("naf_dropped", [1])
    fresh = {
        "B_ov": _tensor("B_ov", perturbed),
        "B_ov@fit": _tensor("B_ov", perturbed),
        keep: _tensor(keep, identity),
        "naf_dropped": fresh_dropped,
        "K": replayed,
    }
    cg.bind(loaded, {name: tensor for name, tensor in fresh.items() if name in names})
    loaded.apply(cg.default_pass_manager())
    loaded.execute()

    measured = float(np.asarray(fresh_dropped)[0])
    assert measured != pytest.approx(float(np.asarray(dropped)[0]), rel=1e-6)
    # And it is still a bound on the error at the new integrals, which is the
    # property that makes a refit worth doing rather than merely correct.
    exact = np.einsum("Qia,Qjb->iajb", perturbed, perturbed)
    got = np.asarray(replayed)
    assert np.linalg.norm(got - exact) / np.linalg.norm(exact) <= measured
