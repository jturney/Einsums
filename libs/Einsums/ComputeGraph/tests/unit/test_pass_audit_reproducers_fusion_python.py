# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""End-to-end reproducers for the fusion and folding defects a read-only audit claimed.

Each case captures a small program, runs it through the named pass alone and through every
public pipeline that reaches the same rewrite, executes, and compares against numpy. Every case
below reproduced before its fix and now stays as the guard for the defect its docstring names. A
case whose old failure was a crash runs in a child interpreter, so a crash is an outcome rather
than the end of the session.

Pipeline arms are listed only where the defect was observed. Where a pipeline did NOT show it,
the reason is recorded here so nobody re-derives it:

* ``P(..)`` on an einsum never reaches LinearCombinationContractionFolding or
  StreamContractionFusion through ``optimize(O2)``: AntisymmetrizerExpansion runs ahead of both
  and lowers every einsum operator site. The tuning phase has no expansion, though, so
  StreamContractionFusion does see the operator through the documented load pipeline
  (``resource_pass_manager`` then ``tuning_pass_manager``).
* AntisymmetrizerExpansion only lowers einsums, so a PERMUTE carrying ``P(..)`` reaches
  SymmetrizedAccumulation and LayoutAssignment under O2. LayoutAssignment's ``P(i/j)`` case is
  masked under O2 only because AntisymmetryInference tags the copy antisymmetric, which pins it;
  a ``P(i/jx)`` copy is not tagged and the fold happens.
* Snapshot versus live prefactors (SymmetrizedAccumulation and LayoutAssignment reading a
  permute's ``alpha``/``beta`` snapshot, ScaleAbsorption reading a Scale's ``factor`` snapshot):
  not reachable in this build. Every writer of those live blocks (AntisymmetrizerLinearity,
  ElementWiseFusion, the IR reader, every node factory) writes the snapshot in the same step,
  and ScaleAbsorption declines a complex-valued factor outright. Composing complex scales through
  ElementWiseFusion before and after ScaleAbsorption computes the right answer.
* GEMMBatching's raw-id ``span_interferes``: a write through a view of a member's operand puts
  the next member on a later dependency level (the hazard scan resolves aliases), and the pass
  only groups members that share a level, so the raw-id check is never the last line of defence.
* Type confusion on a view in SymmetrizedAccumulation: reached before the pass declared its
  features (a ``cg.view`` destination was folded; it is now declined), but the folded node reads
  its operands through the virtual ``impl()``, which dispatches to the view's own override. The
  source cannot be a view from Python, because ``einsums.permute`` binds owning runtime tensors
  only; the C++ rewrite tests fold a view source.
* TiledExpansion's tiled-dot cast (a view result, or a result of another dtype): not expressible
  from Python. The binding takes the result as a ``RuntimeTensor`` of the operands' own dtype.
* Redirect-blind reads inside one in-memory pipeline: every ``Graph::redirect_slot`` caller (CSE,
  PermuteFusion, LayoutAssignment, InplaceOptimization) also rewrites the same graph's
  ``Node::inputs``, and CSE refuses a merge a sub-graph could observe. A node still naming a
  redirected id reaches the fusion passes through ``load_graph`` of a file that carries a
  ``slot_redirects`` entry, which the validator accepts; the cases below build one.
* InplaceOptimization's byte-size match cannot be reached by capture: every elementwise binding
  takes operands of one dtype. A saved file whose scratch tensor's dtype was edited validates,
  and that is the case below.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap

import numpy as np
import pytest

import einsums
import einsums._core.graph as _G
import einsums.graph as cg
from _fuzz_diff_common import _apply_one_pass
from _permutation_operators import apply_operator, shaped_operator

_HERE = os.path.dirname(os.path.abspath(__file__))

RTOL = 1e-10
ATOL = 1e-12


def _tensor(name, array):
    t = einsums.create_zero_tensor(name, list(array.shape), dtype=str(array.dtype))
    np.asarray(t)[...] = array
    return t


def _optimize(graph, how, pass_name):
    """Run the pipeline named by @p how on @p graph.

    ``alone`` runs @p pass_name with every other pass off, then Materialization, which every
    pipeline above O0 ends in and without which deferred scratch cannot execute.
    """
    if how == "alone":
        _apply_one_pass(graph, pass_name)
        _apply_one_pass(graph, "Materialization")
    elif how == "O2":
        graph.optimize(einsums._core.OptLevel.O2)
    elif how == "load-pipeline":
        graph.apply(cg.resource_pass_manager())
        graph.apply(cg.tuning_pass_manager())
        _apply_one_pass(graph, "Materialization")
    else:  # pragma: no cover
        raise AssertionError(how)


def _check(build, expected, how, pass_name):
    """Capture with @p build, optimize, execute, compare every output to numpy."""
    graph = cg.Graph("audit")
    outputs = build(graph)
    _optimize(graph, how, pass_name)
    graph.execute()
    for got, want in zip(outputs, expected):
        np.testing.assert_allclose(np.asarray(got), want, rtol=RTOL, atol=ATOL)


def _run_isolated(code):
    """Run @p code in a fresh interpreter, so a crash is an outcome, not the end of the session."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([_HERE, env.get("PYTHONPATH", "")])
    return subprocess.run([sys.executable, "-c", textwrap.dedent(code)], capture_output=True, text=True, env=env, timeout=300)


def _check_isolated(call):
    """Run ``<this module>.<call>`` in a child; a crash fails with the child's tail."""
    result = _run_isolated(
        f"""
        import test_pass_audit_reproducers_fusion_python as t
        t.{call}
        """
    )
    assert result.returncode == 0, f"the child died (exit {result.returncode}):\n{result.stderr[-3000:]}"


# ──────────────────────────────────────────────────────────────────────────
# LinearCombinationContractionFolding
# ──────────────────────────────────────────────────────────────────────────


def _lccf_operands(seed):
    rng = np.random.default_rng(seed)
    # A fixed vector rather than a draw, so no member's weight is small enough to hide a
    # dropped term inside the tolerance.
    return np.array([1.0, -0.5, 0.25]), rng.standard_normal((3, 4, 4))


def test_lccf_drops_a_members_permutation_operator():
    """LinearCombinationContractionFolding keeps a member's permutation operator.

    Defends against the fold key and the rebuilt contraction being made from the index lists
    alone (``einspec`` built from the c/a/b lists), so that ``C += P(i/j) a_k B_kji`` joined the
    group as the unpermuted ``a_k B_kji`` and the antisymmetrizer's second term was lost. Seen
    with the pass alone; under O2 AntisymmetrizerExpansion lowers the operator first.
    """
    a, b = _lccf_operands(1)
    x = np.einsum("k,kji->ij", a, b)
    expected = 2.0 * np.einsum("k,kij->ij", a, b) - (x - x.T)

    def build(graph):
        A, B, C = _tensor("a", a), _tensor("B", b), _tensor("C", np.zeros((4, 4)))
        with cg.capture(graph):
            einsums.einsum("i,j <- k ; k,i,j", C, A, B, c_pf=0.0, ab_pf=2.0)
            einsums.einsum("i,j <- P(i/j) k ; k,j,i", C, A, B, c_pf=1.0, ab_pf=-1.0)
        return [C]

    _check(build, [expected], "alone", "LinearCombinationContractionFolding")


def lccf_view_operand(how):
    """The folded operand is a ``cg.view``; run by the crash test in a child."""
    a, big = _lccf_operands(5)
    b = big[1:, :, :]
    a = a[:2]
    expected = 2.0 * np.einsum("k,kij->ij", a, b) - np.einsum("k,kji->ij", a, b)

    def build(graph):
        A, Big, C = _tensor("a", a), _tensor("Bbig", big), _tensor("C", np.zeros((4, 4)))
        with cg.capture(graph):
            B = cg.view(Big, [(1, 3), (0, 4), (0, 4)])
            einsums.einsum("i,j <- k ; k,i,j", C, A, B, c_pf=0.0, ab_pf=2.0)
            einsums.einsum("i,j <- k ; k,j,i", C, A, B, c_pf=1.0, ab_pf=-1.0)
        return [C]

    _check(build, [expected], how, "LinearCombinationContractionFolding")


@pytest.mark.parametrize("how", ["alone", "O2"])
def test_lccf_view_operand_is_type_confused(how):
    """LinearCombinationContractionFolding handles a view operand without a type confusion.

    Defends against the old gate, ``is_runtime`` plus one dtype, which a RuntimeTensorView
    passed, after which the L-builder cast the folded operand's ``live_tensor_ptr`` to
    ``GeneralRuntimeTensor<T>`` with ``static_cast`` (``build_l``). The ``axpy`` it fed read the
    owning tensor's storage vector off a view object: SIGBUS.
    """
    _check_isolated(f"lccf_view_operand({how!r})")


@pytest.mark.parametrize("how", ["alone", "O2"])
def test_lccf_misses_a_write_through_a_view_between_members(how):
    """LinearCombinationContractionFolding sees a write through a view between two members.

    ``scale(3, B[0])`` through a view sits between the two members. Defends against
    ``span_interferes`` comparing raw TensorIds: it saw the view's id, not B's, so the group
    folded and the L-builder read B once, at the first member's position, before the scale, and
    the second member's contribution was built from the unscaled slice.
    """
    a, b = _lccf_operands(3)
    scaled = b.copy()
    scaled[0] *= 3.0
    expected = 2.0 * np.einsum("k,kij->ij", a, b) - np.einsum("k,kji->ij", a, scaled)

    def build(graph):
        A, B, C = _tensor("a", a), _tensor("B", b), _tensor("C", np.zeros((4, 4)))
        with cg.capture(graph):
            B0 = cg.view(B, [(0, 1), (0, 4), (0, 4)])
            einsums.einsum("i,j <- k ; k,i,j", C, A, B, c_pf=0.0, ab_pf=2.0)
            einsums.linalg.scale(3.0, B0)
            einsums.einsum("i,j <- k ; k,j,i", C, A, B, c_pf=1.0, ab_pf=-1.0)
        return [C]

    _check(build, [expected], how, "LinearCombinationContractionFolding")


# ──────────────────────────────────────────────────────────────────────────
# StreamContractionFusion
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("how", ["alone", "load-pipeline"])
def test_stream_fusion_drops_a_members_permutation_operator(how):
    """StreamContractionFusion keeps a member's permutation operator.

    A J/K pair over one 8^4 integral tensor (4096 elements, the stream threshold), with
    ``K = P(p/q) (pr|qs) D_rs``. Defends against candidate collection reading only the index
    lists, so that the fused stream accumulated the unpermuted ``(pr|qs) D_rs`` into K. The
    tuning phase runs this pass with no AntisymmetrizerExpansion ahead of it, so the documented
    load pipeline reached it as well as the pass alone.
    """
    rng = np.random.default_rng(2)
    n = 8
    s = rng.standard_normal((n, n, n, n))
    d = rng.standard_normal((n, n))
    x = np.einsum("prqs,rs->pq", s, d)
    expected = [np.einsum("pqrs,rs->pq", s, d), x - x.T]

    def build(graph):
        S, D = _tensor("S", s), _tensor("D", d)
        J, K = _tensor("J", np.zeros((n, n))), _tensor("K", np.zeros((n, n)))
        with cg.capture(graph):
            einsums.einsum("pq <- pqrs ; rs", J, S, D, c_pf=0.0, ab_pf=1.0)
            einsums.einsum("pq <- P(p/q) prqs ; rs", K, S, D, c_pf=0.0, ab_pf=1.0)
        return [J, K]

    _check(build, expected, how, "StreamContractionFusion")


# ──────────────────────────────────────────────────────────────────────────
# SymmetrizedAccumulation
# ──────────────────────────────────────────────────────────────────────────


def _symacc_program(operator, beta, alpha=1.0, s2=0.5, dtype="float64"):
    """``R += tmp; tmpP = alpha P(tmp); R = beta*R + s2*tmpP``, the matched site shape."""
    rng = np.random.default_rng(3)
    n = 4
    x0 = rng.standard_normal((n, n))
    r0 = rng.standard_normal((n, n))
    if dtype.startswith("complex"):
        x0 = x0 + 1j * rng.standard_normal((n, n))
        r0 = r0 + 1j * rng.standard_normal((n, n))
    permuted = x0.T
    if operator:
        permuted = apply_operator(shaped_operator(0, ("i", "j")), ["j", "i"], permuted)
    expected = beta * (r0 + x0) + s2 * alpha * permuted

    def build(graph):
        R, X = _tensor("R", r0), _tensor("X", x0)
        # Graph-owned scratch: the pass only folds a permuted result nothing outside the graph
        # reads, so caller-held tensors here would make it decline before the defended check.
        tmp, tmp_p = graph.scratch("tmp", [n, n], dtype), graph.scratch("tmpP", [n, n], dtype)
        with cg.capture(graph):
            einsums.linalg.axpby(1.0, X, 0.0, tmp)
            einsums.linalg.axpy(1.0, tmp, R)
            einsums.permute(f"j,i <- {'P(i/j) ' if operator else ''}i,j", tmp_p, tmp, c_pf=0.0, a_pf=alpha)
            einsums.linalg.axpby(s2, tmp_p, beta, R)
        return [R]

    return build, expected


def _check_symacc_folds(build, expected, how, folded=1):
    """Like @ref _check for SymmetrizedAccumulation, asserting how many sites it folded.

    ``alone`` reads the pass's counter; ``default`` looks for its line in the default
    pipeline's explanation, which is where a pass in a pipeline says what it did.
    """
    graph = cg.Graph("audit")
    outputs = build(graph)
    if how == "alone":
        sa = _G.SymmetrizedAccumulation()
        pm = cg.PassManager()
        pm.add(sa)
        graph.apply(pm)
        _apply_one_pass(graph, "Materialization")
        assert sa.num_rewritten == folded, sa.skip_reasons
    else:
        pm = cg.default_pass_manager()
        graph.apply(pm)
        report = pm.explain()
        if folded:
            assert f"SymmetrizedAccumulation: folded {folded} " in report, report
        else:
            assert "SymmetrizedAccumulation: folded" not in report, report
    graph.execute()
    for got, want in zip(outputs, expected):
        np.testing.assert_allclose(np.asarray(got), want, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("how", ["alone", "default", "O2"])
def test_symacc_drops_the_permutes_operator(how):
    """SymmetrizedAccumulation keeps the operator of ``tmpP = P(i/j) tmp^T``.

    Defends against the folded executor's spec being rebuilt from the permute's c/a index lists
    alone (``pspec``), which folded it as a plain transpose and lost the antisymmetrizer's second
    term. AntisymmetrizerExpansion lowers only einsums, so O2 reached it too. The site folds.
    """
    build, expected = _symacc_program(operator=True, beta=1.0)
    if how == "O2":
        _check(build, [expected], how, "SymmetrizedAccumulation")
    else:
        _check_symacc_folds(build, [expected], how)


@pytest.mark.parametrize("how", ["alone", "O2"])
def test_symacc_drops_the_accumulates_beta(how):
    """SymmetrizedAccumulation keeps an accumulating axpby's ``beta != 1``.

    Defends against ``is_accumulating_axpby`` only asking that the destination be read, which
    ``beta = 0.25`` does: the fold replaced ``R = 0.25*R + 0.5*P(tmp)`` with ``R += 0.5*P(tmp)``,
    dropping the rescale of everything accumulated before it.
    """
    build, expected = _symacc_program(operator=False, beta=0.25)
    _check(build, [expected], how, "SymmetrizedAccumulation")




@pytest.mark.parametrize("how", ["alone", "default"])
@pytest.mark.parametrize("operator", [False, True], ids=["transpose", "operator"])
@pytest.mark.parametrize("alpha, s2, dtype", [
    (2.0, 0.5, "float64"),
    (1.0, 0.5j, "complex128"),
    (0.25 - 1.5j, 0.5j, "complex128"),
], ids=["scaled", "complex-accumulate", "complex-both"])
def test_symacc_folds_a_scaled_permute_and_complex_scalars(how, operator, alpha, s2, dtype):
    """SymmetrizedAccumulation folds ``R += s2 * (alpha P(tmp))`` into one permute.

    The permute's own ``alpha``, its ``P(i/j)`` and a complex ``s2`` all travel into the folded
    node, which is built from the matched permute's descriptor: ``r2 += (s2 alpha) P(tmp)``.
    These used to be declined by a hand-built rebuild that knew none of them.
    """
    build, expected = _symacc_program(operator=operator, beta=1.0, alpha=alpha, s2=s2, dtype=dtype)
    _check_symacc_folds(build, [expected], how)


@pytest.mark.parametrize("how", ["alone", "default"])
def test_symacc_leaves_an_accumulate_into_a_view(how):
    """SymmetrizedAccumulation does not fold the second accumulate when it writes a view.

    ``R += tmp; tmpP = tmp^T; R += 0.5 tmpP``, with R a ``cg.view`` of a larger tensor. The pass
    reads through views and does not rewrite a node that writes one. Defends the ``understands``
    check on that accumulate: without it the pass folded it into the permute and removed it. The
    fold happens to compute the right numbers here, so what fails without the check is the pass
    audit (``EINSUMS_PASS_VERIFY``), which refuses the removal of a node writing a view.
    """
    rng = np.random.default_rng(7)
    n = 4
    x0 = rng.standard_normal((n, n))
    r0 = rng.standard_normal((n + 1, n + 1))
    expected = r0.copy()
    expected[1:n + 1, 0:n] += x0 + 0.5 * x0.T

    def build(graph):
        Rt, X = _tensor("R", r0), _tensor("X", x0)
        # Graph-owned scratch: the pass only folds a permuted result nothing outside the graph reads.
        tmp, tmp_p = graph.scratch("tmp", [n, n], "float64"), graph.scratch("tmpP", [n, n], "float64")
        with cg.capture(graph):
            R = cg.view(Rt, [(1, n + 1), (0, n)])
            einsums.linalg.axpby(1.0, X, 0.0, tmp)
            einsums.linalg.axpy(1.0, tmp, R)
            einsums.permute("j,i <- i,j", tmp_p, tmp)
            einsums.linalg.axpby(0.5, tmp_p, 1.0, R)
        return [Rt]

    _check_symacc_folds(build, [expected], how, folded=0)


# ──────────────────────────────────────────────────────────────────────────
# LayoutAssignment
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("how, shape_index, letters", [
    ("alone", 0, ("i", "j")),
    ("O2", 1, ("i", "j", "x")),
], ids=["alone-P(i/j)", "O2-P(i/jx)"])
def test_layout_assignment_folds_away_an_antisymmetrizing_permute(how, shape_index, letters):
    """LayoutAssignment keeps a permute carrying ``P(..)`` whose copy it could re-lay out.

    Defends against ``is_pure_reordering`` checking the prefactors and the letter sets but not
    ``operators``, so that ``T = P(..) X^(jxi)`` qualified. Storing T in X's order made the copy
    look like an identity, the node was deleted, and both readers saw X instead of the
    antisymmetrized copy. Under O2 the two-letter ``P(i/j)`` was masked only because
    AntisymmetryInference tags T and pins it, so the O2 arm uses ``P(i/jx)``.
    """
    rng = np.random.default_rng(4)
    n, y, k = 4, 3, 2
    x = rng.standard_normal((n, n, n))
    u = rng.standard_normal((n, n, y))
    v = rng.standard_normal((n, n, k))
    op = shaped_operator(shape_index, letters)
    groups = "/".join("".join(g) for g in op[0])
    t = apply_operator(op, list("jxi"), np.einsum("ijx->jxi", x))
    expected = [np.einsum("jxi,jxy->iy", t, u), np.einsum("jxi,jxk->ik", t, v)]

    def build(graph):
        X, U, V = _tensor("X", x), _tensor("U", u), _tensor("V", v)
        R1, R2 = _tensor("R1", np.zeros((n, y))), _tensor("R2", np.zeros((n, k)))
        T = graph.declare_tensor("T", [n, n, n], intermediate=True, dtype="float64")
        with cg.capture(graph):
            einsums.permute(f"jxi <- P({groups}) ijx", T, X)
            einsums.einsum("iy <- jxi ; jxy", R1, T, U)
            einsums.einsum("ik <- jxi ; jxk", R2, T, V)
        return [R1, R2]

    _check(build, expected, how, "LayoutAssignment")


# ──────────────────────────────────────────────────────────────────────────
# Readers that still name a redirected id (a loaded file)
# ──────────────────────────────────────────────────────────────────────────


def _redirected_file(graph, survivor, duplicate):
    """Save @p graph and hand every reader of @p survivor a redirected duplicate id.

    The file gains a tensor ``duplicate`` shaped like ``survivor``, a ``slot_redirects`` entry
    from it to the survivor, and every node but the survivor's producer reads the duplicate. By
    the redirect, that is the same program: a plain execute reads the survivor's buffer through
    the redirected slot. It is the shape a CSE merge leaves when a reader keeps the eliminated
    id, which the loader puts back from ``slot_redirects``.
    """
    scratch = tempfile.mkdtemp()
    path = os.path.join(scratch, "saved.eig.json")
    cg.save_graph(graph, path)
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    kept = next(t for t in doc["tensors"] if t["name"] == survivor)
    dup = dict(kept)
    dup["id"] = max(t["id"] for t in doc["tensors"] + doc["manifest"]) + 1
    dup["name"] = duplicate
    doc["tensors"].append(dup)
    doc["slot_redirects"] = [{"from": dup["id"], "to": kept["id"]}]
    for node in doc["nodes"]:
        if kept["id"] not in node["outputs"]:
            node["inputs"] = [dup["id"] if i == kept["id"] else i for i in node["inputs"]]
    edited = os.path.join(scratch, "redirected.eig.json")
    with open(edited, "w", encoding="utf-8") as f:
        json.dump(doc, f)
    assert cg.validate_graph_ir(edited) is None, "the redirected file must validate"
    return edited


def _run_file(path, operands, pass_name):
    loaded = cg.load_graph(path)
    for name in loaded.manifest_names():
        loaded.bind(name, operands[name])
    if pass_name is not None:
        _apply_one_pass(loaded, pass_name)
    _apply_one_pass(loaded, "Materialization")
    loaded.execute()


def _lccf_redirect_case():
    rng = np.random.default_rng(6)
    n, k, l = 3, 2, 4
    a, m, nn = np.array([1.0, -0.5]), rng.standard_normal((k, l)), rng.standard_normal((l, n, n))
    b = np.einsum("kl,lij->kij", m, nn)
    expected = 2.0 * np.einsum("k,kij->ij", a, b) - np.einsum("k,kji->ij", a, b)

    def operands():
        return {"a": _tensor("a", a), "M": _tensor("M", m), "N": _tensor("N", nn), "C": _tensor("C", np.zeros((n, n)))}

    ops = operands()
    graph = cg.Graph("redirect")
    x = graph.declare_zero_tensor("X", [k, n, n], True)
    with cg.capture(graph):
        einsums.einsum("k,i,j <- k,l ; l,i,j", x, ops["M"], ops["N"])
        einsums.einsum("i,j <- k ; k,i,j", ops["C"], ops["a"], x, c_pf=0.0, ab_pf=2.0)
        einsums.einsum("i,j <- k ; k,j,i", ops["C"], ops["a"], x, c_pf=1.0, ab_pf=-1.0)
    return _redirected_file(graph, "X", "Y"), operands, expected


def test_redirected_file_runs_unoptimized():
    """The control for the redirect cases: the file itself computes the right answer."""
    path, operands, expected = _lccf_redirect_case()
    ops = operands()
    _run_file(path, ops, None)
    np.testing.assert_allclose(np.asarray(ops["C"]), expected, rtol=RTOL, atol=ATOL)


def test_lccf_reads_a_redirected_operand_through_its_own_object():
    """LinearCombinationContractionFolding reads a redirected operand through its slot.

    Defends against the L-builder resolving the folded operand with ``live_tensor_ptr`` (the
    handle's own object), while every captured executor reads through the slot, which the
    redirect repoints at the survivor. The fold read the duplicate's never-written zeros.
    """
    path, operands, expected = _lccf_redirect_case()
    ops = operands()
    _run_file(path, ops, "LinearCombinationContractionFolding")
    np.testing.assert_allclose(np.asarray(ops["C"]), expected, rtol=RTOL, atol=ATOL)


def test_stream_fusion_reads_a_redirected_stream_through_its_own_object():
    """StreamContractionFusion reads a redirected stream through its slot.

    Defends against ``run_stream`` resolving operands through the handle's ``impl_fn``, which
    belongs to the duplicate, not through the redirected slot, so that the fused J/K stream
    walked zeros.
    """
    rng = np.random.default_rng(6)
    n = 8
    a, b, d = rng.standard_normal((n, n, 2)), rng.standard_normal((2, n, n)), rng.standard_normal((n, n))
    s = np.einsum("pqt,trs->pqrs", a, b)
    expected = [np.einsum("pqrs,rs->pq", s, d), np.einsum("prqs,rs->pq", s, d)]

    def operands():
        return {"A": _tensor("A", a), "B": _tensor("B", b), "D": _tensor("D", d),
                "J": _tensor("J", np.zeros((n, n))), "K": _tensor("K", np.zeros((n, n)))}

    ops = operands()
    graph = cg.Graph("redirect")
    x = graph.declare_zero_tensor("X", [n] * 4, True)
    with cg.capture(graph):
        einsums.einsum("pqrs <- pqt ; trs", x, ops["A"], ops["B"])
        einsums.einsum("pq <- pqrs ; rs", ops["J"], x, ops["D"])
        einsums.einsum("pq <- prqs ; rs", ops["K"], x, ops["D"])
    path = _redirected_file(graph, "X", "Y")

    ops = operands()
    _run_file(path, ops, "StreamContractionFusion")
    for name, want in zip(("J", "K"), expected):
        np.testing.assert_allclose(np.asarray(ops[name]), want, rtol=RTOL, atol=ATOL)


def test_symacc_reads_a_redirected_source_through_its_own_object():
    """SymmetrizedAccumulation reads a redirected source through its slot.

    Defends against the folded permute reading ``live_tensor_ptr(tmp)``, the duplicate's own
    object, so that the permuted half of the symmetrization was built from zeros.
    """
    rng = np.random.default_rng(6)
    n = 4
    x0, r0 = rng.standard_normal((n, n)), rng.standard_normal((n, n))
    expected = r0 + x0 + 0.5 * x0.T

    def operands():
        return {"Xs": _tensor("Xs", x0), "R": _tensor("R", r0)}

    ops = operands()
    graph = cg.Graph("redirect")
    x = graph.declare_zero_tensor("X", [n, n], True)
    tp = graph.declare_zero_tensor("TP", [n, n], True)
    with cg.capture(graph):
        einsums.linalg.axpby(1.0, ops["Xs"], 0.0, x)
        einsums.linalg.axpy(1.0, x, ops["R"])
        einsums.permute("j,i <- i,j", tp, x, c_pf=0.0, a_pf=1.0)
        einsums.linalg.axpby(0.5, tp, 1.0, ops["R"])
    path = _redirected_file(graph, "X", "Y")

    ops = operands()
    _run_file(path, ops, "SymmetrizedAccumulation")
    np.testing.assert_allclose(np.asarray(ops["R"]), expected, rtol=RTOL, atol=ATOL)


def dead_node_elimination_on_redirected_file():
    """DeadNodeElimination on the redirected file; run by the crash test in a child."""
    path, operands, expected = _lccf_redirect_case()
    ops = operands()
    _run_file(path, ops, "DeadNodeElimination")
    np.testing.assert_allclose(np.asarray(ops["C"]), expected, rtol=RTOL, atol=ATOL)


def test_dead_node_elimination_ignores_slot_redirects():
    """DeadNodeElimination counts a reader through a slot redirect as a reader of the survivor.

    Found while reducing the redirect cases above, not one of the audit's claims. Defends
    against counting readers by raw id: no node names the survivor X once every reader names the
    redirected duplicate, so X's producer looked dead and was deleted, the readers then reached
    X's never-materialized storage through the redirect, and the next contraction segfaulted.
    """
    _check_isolated("dead_node_elimination_on_redirected_file()")


# ──────────────────────────────────────────────────────────────────────────
# InplaceOptimization
# ──────────────────────────────────────────────────────────────────────────


def _node_io(graph):
    return [(n["inputs"], n["outputs"]) for n in json.loads(graph.to_json())["nodes"]]


def test_inplace_optimization_does_not_half_rewrite_on_a_dtype_mismatch():
    """InplaceOptimization leaves the graph as it found it when a merge cannot finish.

    A saved ``X -> A -> B -> Y`` axpby chain with scratch B's dtype edited to complex64 still
    validates. Defends against pairing a source and destination by dims and byte count only:
    complex64 and float64 are both 8 bytes, so ``find_merge`` planned B onto the dying A, and
    ``apply_merge`` rewrote every node's ids first and called ``redirect_slot`` last, which threw
    on the dtype mismatch, so the pass escaped with the node lists naming A and B's slot never
    redirected.
    """
    n = 3

    def operands():
        return {"X": _tensor("X", np.arange(9.0).reshape(n, n)), "Y": _tensor("Y", np.zeros((n, n)))}

    ops = operands()
    graph = cg.Graph("inplace")
    a = graph.declare_zero_tensor("A", [n, n], True)
    b = graph.declare_zero_tensor("B", [n, n], True)
    with cg.capture(graph):
        einsums.linalg.axpby(1.0, ops["X"], 0.0, a)
        einsums.linalg.axpby(2.0, a, 0.0, b)
        einsums.linalg.axpby(1.0, b, 0.0, ops["Y"])

    scratch = tempfile.mkdtemp()
    path = os.path.join(scratch, "saved.eig.json")
    cg.save_graph(graph, path)
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    for t in doc["tensors"]:
        if t["name"] == "B":
            t["dtype"] = "complex64"
    edited = os.path.join(scratch, "mixed.eig.json")
    with open(edited, "w", encoding="utf-8") as f:
        json.dump(doc, f)
    assert cg.validate_graph_ir(edited) is None

    loaded = cg.load_graph(edited)
    ops = operands()
    for name in loaded.manifest_names():
        loaded.bind(name, ops[name])
    before = _node_io(loaded)
    try:
        _apply_one_pass(loaded, "InplaceOptimization")
    except RuntimeError:
        assert _node_io(loaded) == before, "the pass raised after rewriting the node lists"


def test_inplace_optimization_leaves_a_consumer_reading_a_redirected_slot():
    """InplaceOptimization does not reuse a dying input's storage for a node that reads a redirected slot.

    A saved ``B = X; C = B * B; R = B; Y = C`` chain, edited so that the product's first operand
    and R's source name B2, a duplicate of B whose slot is redirected to B. By raw id B has one
    reader, the product, so B looks like it dies there; through the redirect R still reads B
    afterwards. Defends the pass counting reads per buffer: first by an ``understands`` check
    that kept it off redirected slots, now by counting B2's readers as B's. Without either the
    pass wrote C into B's storage, and R read ``X * X`` instead of X (max abs error 72 on this
    data).
    """
    n = 3
    x0 = np.arange(1.0, 10.0).reshape(n, n)

    def operands():
        return {"X": _tensor("X", x0), "R": _tensor("R", np.zeros((n, n))), "Y": _tensor("Y", np.zeros((n, n)))}

    ops = operands()
    graph = cg.Graph("inplace")
    b = graph.declare_zero_tensor("B", [n, n], True)
    c = graph.declare_zero_tensor("C", [n, n], True)
    with cg.capture(graph):
        einsums.linalg.axpby(1.0, ops["X"], 0.0, b)
        einsums.linalg.direct_product(1.0, b, b, 0.0, c)
        einsums.linalg.axpby(1.0, b, 0.0, ops["R"])
        einsums.linalg.axpby(1.0, c, 0.0, ops["Y"])

    scratch = tempfile.mkdtemp()
    path = os.path.join(scratch, "saved.eig.json")
    cg.save_graph(graph, path)
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    kept = next(t for t in doc["tensors"] if t["name"] == "B")
    dup = dict(kept)
    dup["id"] = max(t["id"] for t in doc["tensors"] + doc["manifest"]) + 1
    dup["name"] = "B2"
    doc["tensors"].append(dup)
    doc["slot_redirects"] = [{"from": dup["id"], "to": kept["id"]}]
    for node in doc["nodes"]:
        if node["kind"] == "DirectProduct":
            node["inputs"][0] = dup["id"]
        elif node["kind"] == "Axpby" and node["inputs"] == [kept["id"]]:
            node["inputs"] = [dup["id"]]
    edited = os.path.join(scratch, "redirected.eig.json")
    with open(edited, "w", encoding="utf-8") as f:
        json.dump(doc, f)
    assert cg.validate_graph_ir(edited) is None

    for pass_name in (None, "InplaceOptimization"):
        ops = operands()
        _run_file(edited, ops, pass_name)
        np.testing.assert_allclose(np.asarray(ops["R"]), x0, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(np.asarray(ops["Y"]), x0 * x0, rtol=RTOL, atol=ATOL)


def _inplace_redirect_case():
    """A saved ``B = X Y; C = 2 (B2 * D); R = C Y`` chain, where B2's slot is redirected to B.

    B's only reader is the product, through B2, so B dies there and C can take its storage.
    """
    rng = np.random.default_rng(34)
    n = 4
    x, y, d = rng.standard_normal((n, n)), rng.standard_normal((n, n)), 2.0 + rng.random((n, n))
    expected = (2.0 * (x @ y) * d) @ y

    def operands():
        return {"X": _tensor("X", x), "Y": _tensor("Y", y), "D": _tensor("D", d), "R": _tensor("R", np.zeros((n, n)))}

    ops = operands()
    graph = cg.Graph("redirect")
    # Declared without a zero: Materialization zero-initializes a declared zero tensor with an
    # Initialize node, and a destination carrying one is declined.
    b = graph.declare_tensor("B", [n, n], True)
    c = graph.declare_tensor("C", [n, n], True)
    with cg.capture(graph):
        einsums.einsum("ij <- ik ; kj", b, ops["X"], ops["Y"], c_pf=0.0, ab_pf=1.0)
        einsums.linalg.direct_product(2.0, b, ops["D"], 0.0, c)
        einsums.einsum("ij <- ik ; kj", ops["R"], c, ops["Y"], c_pf=0.0, ab_pf=1.0)
    return _redirected_file(graph, "B", "B2"), operands, expected


@pytest.mark.parametrize("how", ["alone", "default"])
def test_inplace_optimization_merges_through_a_redirected_slot(how):
    """InplaceOptimization reuses the storage a redirected slot reads when that storage dies.

    The product reads B only through B2, whose slot is redirected to B, and nothing else reads
    B, so C takes B's storage and the product writes where it reads, element by element. This
    shape used to be declined by the pass's feature declaration.
    """
    path, operands, expected = _inplace_redirect_case()
    ops = operands()
    loaded = cg.load_graph(path)
    for name in loaded.manifest_names():
        loaded.bind(name, ops[name])
    if how == "alone":
        inplace = _G.InplaceOptimization()
        manager = cg.PassManager()
        manager.add(inplace)
        loaded.apply(manager)
        assert inplace.num_merged == 1, inplace.skip_reasons
        _apply_one_pass(loaded, "Materialization")
    else:
        manager = cg.default_pass_manager()
        loaded.apply(manager)
        assert "InplaceOptimization: merged 1 buffer(s)" in manager.explain(), manager.explain()
    for _ in range(2):
        loaded.execute()
        np.testing.assert_allclose(np.asarray(ops["R"]), expected, rtol=RTOL, atol=ATOL)



def test_inplace_optimization_sees_a_consumer_read_its_destination_through_a_redirected_slot():
    """``D = 2 S * D2``, D2's slot redirected to D, keeps D's own storage.

    A saved ``S = X Y; D = 2 S * D; R = D`` program, edited so the product reads its own
    destination through D2, a duplicate whose slot is redirected to D. D is declared with a zero,
    so the product computes zero. Defends the destination check comparing buffers: by raw id the
    product does not read D, so the pass put D in the dying S's storage, D2 followed the redirect
    there, and the product computed ``2 S * S``.
    """
    rng = np.random.default_rng(35)
    n = 4
    x, y = rng.standard_normal((n, n)), rng.standard_normal((n, n))

    def operands():
        return {"X": _tensor("X", x), "Y": _tensor("Y", y), "R": _tensor("R", np.ones((n, n)))}

    ops = operands()
    graph = cg.Graph("redirect")
    s, d = (graph.declare_zero_tensor(name, [n, n], True) for name in ("S", "D"))
    with cg.capture(graph):
        einsums.einsum("ij <- ik ; kj", s, ops["X"], ops["Y"], c_pf=0.0, ab_pf=1.0)
        einsums.linalg.direct_product(2.0, s, d, 0.0, d)
        einsums.linalg.axpby(1.0, d, 0.0, ops["R"])

    scratch = tempfile.mkdtemp()
    path = os.path.join(scratch, "saved.eig.json")
    cg.save_graph(graph, path)
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    kept = next(tn for tn in doc["tensors"] if tn["name"] == "D")
    dup = dict(kept)
    dup["id"] = max(tn["id"] for tn in doc["tensors"] + doc["manifest"]) + 1
    dup["name"] = "D2"
    doc["tensors"].append(dup)
    doc["slot_redirects"] = [{"from": dup["id"], "to": kept["id"]}]
    product = next(node for node in doc["nodes"] if node["kind"] == "DirectProduct")
    product["inputs"] = [dup["id"] if i == kept["id"] else i for i in product["inputs"]]
    edited = os.path.join(scratch, "redirected.eig.json")
    with open(edited, "w", encoding="utf-8") as f:
        json.dump(doc, f)
    assert cg.validate_graph_ir(edited) is None

    for pass_name in (None, "InplaceOptimization"):
        ops = operands()
        _run_file(edited, ops, pass_name)
        np.testing.assert_allclose(np.asarray(ops["R"]), np.zeros((n, n)), rtol=RTOL, atol=ATOL)

