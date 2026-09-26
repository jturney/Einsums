# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""Hypothesis: a corrupted graph file is either REPORTED or harmless.

A saved graph is a file, and a file gets edited by hand, truncated by a copy,
merged by a tool that does not know the schema, or written by an older build.
The reader's promise is the one ``validate_graph_ir`` documents: a file that is
not valid is reported, with every problem it has. The dangerous failure is not
a reader that rejects too much but one that accepts a file which then runs as
a DIFFERENT program with nothing to say so.

So this takes a handful of real graphs, saves each, and mutates one integer in
the saved JSON at a time: node ids, tensor ids, ranks, dims, GEMM hint extents,
operand ids and leading dimensions, slot redirects, gate sizes, loop bounds.
Each mutation sets the value negative, huge, zero or off by one, or swaps two
id-like values. The property:

  * ``validate_graph_ir`` and ``load_graph`` agree on whether the file is
    valid, since the validator is documented to build the graph exactly as the
    loader does;
  * a negative count or id is reported, which the reader promises;
  * a file that loads either refuses the caller's operands at bind, raises at
    execute, or computes the SAME numbers the unmutated file computes, both on
    a plain execute and after the documented load pipeline
    (``resource_pass_manager`` then ``tuning_pass_manager``).

Not every mutation is a corruption. Pointing an einsum input at a different
tensor OF THE SAME SHAPE AND DTYPE, changing a parameter's value, or lowering a
loop's iteration bound writes a well-formed different program, and no validator
could or should object. Those are classified SEMANTIC and checked only for not
crashing and for negative-count reporting; everything else must be reported
or harmless. The classification is deliberately conservative: a mutation is
semantic only when the file stays self-consistent by construction.

A crash takes the whole process with it, which is itself the report: there is
no in-process way to survive a segfault, and a test that tried would hide it.
"""

from __future__ import annotations

import copy
import functools
import json
import os
import subprocess
import sys
import tempfile
import textwrap

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

import einsums
import einsums.graph as cg
from _sanitizer_scaling import sanitizer_examples

_HERE = os.path.dirname(os.path.abspath(__file__))
_GOLDEN_MIXED = os.path.join(_HERE, "goldens", "v1_0_0_mixed.eig.json")

# ──────────────────────────────────────────────────────────────────────────
# The base graphs
# ──────────────────────────────────────────────────────────────────────────


def _tensor(name, shape, seed):
    """A caller tensor with deterministic contents, so two runs are comparable."""
    t = einsums.create_zero_tensor(name, list(shape), dtype="float64")
    if shape:
        np.asarray(t)[...] = np.random.default_rng(seed).standard_normal(shape)
    return t


def _save_text(graph):
    with tempfile.TemporaryDirectory() as scratch:
        path = os.path.join(scratch, "base.eig.json")
        cg.save_graph(graph, path)
        with open(path, encoding="utf-8") as f:
            return f.read()


_SHAPES = {
    "chain": {"A": (3, 4), "B": (4, 2), "C": (3, 2), "D": (2, 3), "E": (3, 3)},
    "pair": {"A": (3, 4), "B1": (4, 2), "B2": (4, 3), "C1": (3, 2), "C2": (3, 3)},
    "permute": {"A": (3, 4), "C": (4, 3)},
    "axpby": {"X": (3, 3), "Y": (3, 3), "Z": (3, 3)},
    "cse": {"A": (3, 4), "B": (4, 3), "O": (3, 3)},
    "perm_op": {"A": (3, 4), "B": (4, 3), "C": (3, 3)},
    # The golden's rank-0 ``dot_result`` is deliberately left out. It is a bare
    # scalar slot, which takes no tensor at bind (see
    # test_rank0_bind_to_a_loaded_scalar_slot_is_refused), so the loader's own
    # placeholder takes the dot here, and only the four rank-2 tensors are
    # compared.
    "mixed": {"A": (3, 4), "B": (4, 2), "C": (3, 2), "D": (2, 3)},
}


def _operands(base):
    return {name: _tensor(name, shape, seed) for seed, (name, shape) in enumerate(_SHAPES[base].items())}


def _build_chain(ops):
    g = cg.Graph("chain")
    with cg.capture(g):
        einsums.einsum("ij <- ik ; kj", ops["C"], ops["A"], ops["B"])
        einsums.einsum("ij <- ik ; kj", ops["E"], ops["C"], ops["D"], c_pf=0.5, ab_pf=2.0)
    return g


def _build_pair(ops):
    # Two independent GEMM-shaped einsums differing only in n, so one
    # off-by-one on a hint extent is enough to put them in one batching group.
    g = cg.Graph("pair")
    with cg.capture(g):
        einsums.einsum("ij <- ik ; kj", ops["C1"], ops["A"], ops["B1"])
        einsums.einsum("ij <- ik ; kj", ops["C2"], ops["A"], ops["B2"])
    return g


def _build_permute(ops):
    g = cg.Graph("permute")
    with cg.capture(g):
        einsums.permute("ij <- ji", ops["C"], ops["A"], c_pf=0.5, a_pf=1.5)
    return g


def _build_axpby(ops):
    g = cg.Graph("axpby")
    with cg.capture(g):
        einsums.linalg.axpby(0.5, ops["X"], 2.0, ops["Y"])
        einsums.linalg.axpby(-1.0, ops["Y"], 0.25, ops["Z"])
    return g


def _build_cse(ops):
    # Two identical contractions into graph-owned scratch, which the
    # structural phase merges: the file then carries an intermediate table, a
    # slot redirect, and a GEMM hint naming the merged-away tensor.
    g = cg.Graph("cse")
    x = g.declare_zero_tensor("X", [3, 3], True)
    y = g.declare_zero_tensor("Y", [3, 3], True)
    with cg.capture(g):
        einsums.einsum("ij <- ik ; kj", x, ops["A"], ops["B"])
        einsums.einsum("ij <- ik ; kj", y, ops["A"], ops["B"])
        einsums.einsum("ij <- ik ; kj", ops["O"], x, y, c_pf=1.0)
    return g


def _build_perm_op(ops):
    g = cg.Graph("perm_op")
    with cg.capture(g):
        einsums.einsum("i,j <- P(i/j) i,k ; k,j", ops["C"], ops["A"], ops["B"])
    return g


_BUILDERS = {
    "chain": _build_chain,
    "pair": _build_pair,
    "permute": _build_permute,
    "axpby": _build_axpby,
    "cse": _build_cse,
    "perm_op": _build_perm_op,
}


@functools.cache
def _base_text(base):
    """The saved JSON of one base graph, built once per process."""
    if base == "mixed":
        # Loops and conditionals cannot be saved from Python (their predicates
        # are callables), so the loop, gate, parameter and fragment encoding
        # comes from the checked-in golden, re-saved by this build so the
        # mutations apply to the current schema.
        return _save_text(cg.load_graph(_GOLDEN_MIXED))
    g = _BUILDERS[base](_operands(base))
    if base not in _SAVED_AS_CAPTURED:
        g.apply(cg.structural_pass_manager())
    return _save_text(g)


# Bases saved without the structural phase. The permutation-operator einsum
# cannot be saved after it: the phase expands P(i/j) through graph-owned
# scratch whose Alloc node the writer refuses (see
# test_structural_phase_of_a_permutation_operator_saves), so the operator is
# mutated as captured, where it is one Einsum carrying ``operators``.
_SAVED_AS_CAPTURED = frozenset({"perm_op"})


BASES = ["chain", "pair", "permute", "axpby", "cse", "perm_op", "mixed"]

# ──────────────────────────────────────────────────────────────────────────
# Running a file
# ──────────────────────────────────────────────────────────────────────────


class Rejected(Exception):
    """The file, the bind or the execute refused, which is a report."""

    def __init__(self, stage, message):
        super().__init__(f"{stage}: {message}")
        self.stage = stage


def _validate(path):
    try:
        cg.validate_graph_ir(path)
    except (RuntimeError, MemoryError) as exc:
        # MemoryError is the validator allocating what a huge gate size asks
        # for rather than reporting it; still an exception, not a crash.
        return f"{type(exc).__name__}: {exc}"
    return None


def _run(path, base, tuned):
    """Load, bind fresh operands by name, optionally run the documented load
    pipeline, execute, and return every operand's final contents."""
    loaded = cg.load_graph(path)
    ops = _operands(base)
    try:
        for name in loaded.manifest_names():
            if name in ops:
                loaded.bind(name, ops[name])
    except (RuntimeError, ValueError, TypeError) as exc:
        raise Rejected("bind", exc) from exc
    try:
        if tuned:
            loaded.apply(cg.resource_pass_manager())
            loaded.apply(cg.tuning_pass_manager())
        loaded.execute()
    except (RuntimeError, ValueError, TypeError, IndexError, MemoryError) as exc:
        raise Rejected("execute", exc) from exc
    return {name: np.asarray(t).copy() for name, t in ops.items()}


@functools.cache
def _baseline(base, tuned):
    with tempfile.TemporaryDirectory() as scratch:
        path = os.path.join(scratch, "base.eig.json")
        with open(path, "w", encoding="utf-8") as f:
            f.write(_base_text(base))
        assert _validate(path) is None, "the unmutated base must validate"
        try:
            return _run(path, base, tuned)
        except Rejected as rejected:
            # A loaded graph with deferred scratch refuses a plain execute until
            # the load pipeline materializes it, which is documented, so the
            # CSE base has only the tuned arm.
            assert base == "cse" and not tuned, f"the unmutated {base} base failed to run: {rejected}"
            return None


# ──────────────────────────────────────────────────────────────────────────
# Mutation targets
# ──────────────────────────────────────────────────────────────────────────

# Keys whose integer names a tensor in the frame the leaf lives in (or, for
# ``outer``, the enclosing frame).
_REFERENCE_KEYS = frozenset({"inputs", "outputs", "outer", "from", "to"})

# Keys whose value is program DATA: a different non-negative value is a
# different program, not a corrupt one. Negative is still a corruption for the
# counts among them.
_SEMANTIC_COUNT_KEYS = frozenset({"max_iterations", "index", "size"})
# Signed program data: every value is a program.
_SEMANTIC_SIGNED_KEYS = frozenset({"value", "const"})

_ID_LIKE_KEYS = _REFERENCE_KEYS | {"id"}


class Leaf:
    """One integer in the document, where it is, and what it refers to."""

    __slots__ = ("path", "key", "value", "frame", "role")

    def __init__(self, path, key, value, frame, role):
        self.path = path
        self.key = key
        self.value = value
        self.frame = frame  # {id: (dtype, dims)} the value is resolved in, when it names a tensor
        self.role = role  # "ref", "table_id", "hint_id" or "plain"

    def __repr__(self):
        return f"{'.'.join(map(str, self.path))}={self.value}"


def _frame_of(tensors):
    return {t["id"]: (t.get("dtype"), tuple(t.get("dims", ()))) for t in tensors if isinstance(t.get("id"), int)}


def _leaves(doc):
    """Every integer leaf worth mutating, with the tensor frame each resolves in."""
    out = []

    def walk(node, path, frame, parent_frame, key_ctx):
        if isinstance(node, bool):
            return
        if isinstance(node, int):
            key = key_ctx
            if key in _REFERENCE_KEYS:
                role = "ref"
            elif key == "id" and len(path) >= 3 and path[-3] in ("manifest", "tensors"):
                role = "table_id"
            elif key == "id" and len(path) >= 2 and path[-2] in ("a", "b", "c"):
                role = "hint_id"
            else:
                role = "plain"
            resolve = parent_frame if key == "outer" else frame
            out.append(Leaf(path, key, node, resolve, role))
            return
        if isinstance(node, list):
            for i, item in enumerate(node):
                walk(item, path + (i,), frame, parent_frame, key_ctx)
            return
        if isinstance(node, dict):
            inner_frame, inner_parent = frame, parent_frame
            if "nodes" in node and "tensors" in node and "manifest" not in node:
                # A fragment: a loop body or a branch, with its own dense ids.
                inner_frame, inner_parent = _frame_of(node["tensors"]), frame
            for k, v in node.items():
                if k in ("provenance", "spaces", "einsums_graph_ir"):
                    continue
                walk(v, path + (k,), inner_frame, inner_parent, k)

    top = _frame_of(doc.get("manifest", []) + doc.get("tensors", []))
    walk(doc, (), top, None, None)
    return out


def _set(doc, path, value):
    target = doc
    for step in path[:-1]:
        target = target[step]
    target[path[-1]] = value


def _huge_values(leaf):
    # Allocation sizes get values whose byte count fails immediately or wraps a
    # size_t product, never one the allocator would grant and then zero: a
    # 2**31-extent tensor is 16 GiB of page faults, which tests the machine
    # rather than the reader.
    if leaf.key in ("dims", "size"):
        return [2**50, 2**62, 2**63 - 1, 2**64]
    return [2**31, 2**62, 2**63 - 1, 2**64]


def _candidates(leaf):
    v = leaf.value
    return sorted({-1, -(2**31), 0, v + 1, v - 1, *_huge_values(leaf)} - {v})


class Mutation:
    def __init__(self, description, edits):
        self.description = description
        self.edits = edits  # [(path, new value)]

    def apply(self, doc):
        out = copy.deepcopy(doc)
        for path, value in self.edits:
            _set(out, path, value)
        return out

    def __repr__(self):
        return self.description


def _set_mutation(leaf, value):
    return Mutation(f"set {leaf!r} -> {value}", [(leaf.path, value)])


def _swap_mutation(first, second):
    return Mutation(f"swap {first!r} <-> {second!r}", [(first.path, second.value), (second.path, first.value)])


# ──────────────────────────────────────────────────────────────────────────
# Is a mutated file a well-formed DIFFERENT program?
#
# Retargeting references can produce a file that is self-consistent and simply
# means something else, and the property must not call that a bug. So a
# mutation that touches only references (and program data) is checked against
# this small independent model of what the kinds in the base graphs require of
# their operands. The model is deliberately narrower than the reader: a kind
# it does not know makes the file "not known consistent", which keeps the
# mutation strict. Its only job is to excuse, never to accuse.
# ──────────────────────────────────────────────────────────────────────────


def _table(tensors):
    ids = [t.get("id") for t in tensors]
    if sorted(ids) != list(range(len(ids))):
        return None  # not dense and unique
    return {t["id"]: t for t in tensors}


def _extents_agree(letters_by_operand):
    """Each letter has one extent across every operand that names it."""
    seen = {}
    for letters, dims in letters_by_operand:
        if len(letters) != len(dims):
            return None
        for letter, extent in zip(letters, dims):
            if seen.setdefault(letter, extent) != extent:
                return None
    return seen


def _node_consistent(node, frame, parent):
    kind = node.get("kind")
    ins, outs = node.get("inputs", []), node.get("outputs", [])
    if any(i not in frame for i in ins + outs):
        return False
    desc = node.get("descriptor", {})

    def dims(i):
        return tuple(frame[i]["dims"])

    if kind == "Einsum":
        if len(ins) not in (2, 3) or len(outs) != 1 or (len(ins) == 3 and ins[2] != outs[0]):
            return False
        extents = _extents_agree([(desc["a_indices"], dims(ins[0])), (desc["b_indices"], dims(ins[1])),
                                  (desc["c_indices"], dims(outs[0]))])
        if extents is None:
            return False
        for group in desc.get("operators", []):
            if len({extents.get(letter) for part in group for letter in part}) != 1:
                return False
        # The GEMM hint is NOT consulted. It is redundant with the operands, so
        # a retarget leaves it stale, and a stale hint is not what makes the
        # retargeted program different; mutations of the hint itself are
        # strict by _classify and never reach this model.
        return True
    if kind == "Permute":
        return (len(ins) == 1 and len(outs) == 1
                and _extents_agree([(desc["a_indices"], dims(ins[0])), (desc["c_indices"], dims(outs[0]))]) is not None)
    if kind in ("Scale", "ElementTransform"):
        return len(ins) == 1 and ins == outs
    if kind == "Axpby":
        return len(ins) == 2 and outs == [ins[1]] and dims(ins[0]) == dims(ins[1])
    if kind == "Dot":
        return len(ins) == 2 and len(outs) == 1 and dims(ins[0]) == dims(ins[1]) and frame[outs[0]]["rank"] == 0
    if kind == "WriteParam":
        return not ins and not outs
    if kind == "Loop":
        return _frame_consistent(desc.get("body", {}), frame)
    if kind == "Conditional":
        return _frame_consistent(desc.get("then", {}), frame) and _frame_consistent(desc.get("else", {}), frame)
    return False


def _frame_consistent(fragment, parent, tensors=None):
    frame = _table(tensors if tensors is not None else fragment.get("tensors", []))
    if frame is None:
        return False
    for t in frame.values():
        if "outer" in t:
            outer = parent.get(t["outer"]) if parent is not None else None
            if outer is None or (outer["dtype"], outer["dims"]) != (t["dtype"], t["dims"]):
                return False
    return all(_node_consistent(node, frame, parent) for node in fragment.get("nodes", []))


def _consistent(doc):
    top = doc.get("manifest", []) + doc.get("tensors", [])
    if not _frame_consistent(doc, None, top):
        return False
    frame = _table(top)
    for redirect in doc.get("slot_redirects", []):
        a, b = frame.get(redirect["from"]), frame.get(redirect["to"])
        if a is None or b is None or (a["dtype"], a["dims"]) != (b["dtype"], b["dims"]):
            return False
    return True


def _classify(mutation, doc, leaves_by_path):
    """``"semantic"`` when the mutation writes a well-formed different program,
    ``"strict"`` when the file must be reported or harmless.

    Program data (a parameter's value, a loop bound, a flag index or count) is
    semantic at any non-negative value. A mutation that retargets a reference
    or a table id is semantic exactly when the mutated file is consistent by
    the model above; the other half of a swap may then be a node id or a hint
    operand id, which the retarget already makes moot. Anything else (a rank,
    an extent, a hint on its own) is strict: those are redundant with the
    tensors, so any change must be caught or have no effect.
    """
    leaves = [leaves_by_path[path] for path, _ in mutation.edits]
    values = [value for _, value in mutation.edits]
    if all(leaf.key in _SEMANTIC_SIGNED_KEYS for leaf in leaves):
        return "semantic"
    if all(leaf.key in _SEMANTIC_COUNT_KEYS for leaf in leaves):
        return "semantic" if all(v >= 0 for v in values) else "strict"
    if any(leaf.role in ("ref", "table_id") for leaf in leaves) and _consistent(mutation.apply(doc)):
        return "semantic"
    return "strict"


# The GEMM hint's extents and leading dimensions are read as plain integers,
# not as counts, and nothing downstream of a single-member group reads a
# negative one, so a negative value there is exercised by the numbers check
# instead of being required to produce a report.
_HINT_INT_KEYS = frozenset({"m", "n", "k", "leading_dim"})


def _negative_count(mutation, leaves_by_path):
    """Whether the mutation writes a negative value into a field that counts or
    identifies something, which the reader promises to report."""
    for path, value in mutation.edits:
        key = leaves_by_path[path].key
        if value < 0 and key not in _SEMANTIC_SIGNED_KEYS and key not in _HINT_INT_KEYS:
            return True
    return False


# ──────────────────────────────────────────────────────────────────────────
# The property
# ──────────────────────────────────────────────────────────────────────────

_STATS = {"trials": 0, "reported": 0, "bind_refused": 0, "execute_refused": 0, "harmless": 0, "semantic_ran": 0}


def _same(got, expected):
    return all(np.allclose(got[k], expected[k], rtol=1e-12, atol=1e-12, equal_nan=True) for k in expected)


def check_mutation(base, mutation):
    doc = json.loads(_base_text(base))
    leaves_by_path = {leaf.path: leaf for leaf in _leaves(doc)}
    mutated = mutation.apply(doc)
    _STATS["trials"] += 1

    with tempfile.TemporaryDirectory() as scratch:
        path = os.path.join(scratch, "mutated.eig.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(mutated, f)

        problems = _validate(path)
        try:
            cg.load_graph(path)
            load_error = None
        except (RuntimeError, MemoryError) as exc:
            load_error = str(exc)

        assert (problems is None) == (load_error is None), (
            f"validate_graph_ir and load_graph disagree on {base}: {mutation!r}\n"
            f"validate: {problems}\nload: {load_error}"
        )
        if problems is not None:
            _STATS["reported"] += 1
            return

        assert not _negative_count(mutation, leaves_by_path), (
            f"a negative count or id loaded without a report on {base}: {mutation!r}"
        )

        semantic = _classify(mutation, doc, leaves_by_path) == "semantic"
        for tuned in (False, True):
            arm = "tuned" if tuned else "plain"
            expected = _baseline(base, tuned)
            if expected is None:
                continue  # this arm cannot run the unmutated file either
            try:
                got = _run(path, base, tuned)
            except Rejected as rejected:
                _STATS[f"{rejected.stage}_refused"] += 1
                continue
            if semantic:
                _STATS["semantic_ran"] += 1
                continue
            if not _same(got, expected):
                diff = {k: float(np.max(np.abs(got[k] - expected[k]))) for k in expected if not np.array_equal(got[k], expected[k])}
                raise AssertionError(
                    f"a mutated file validated, loaded and computed DIFFERENT numbers ({arm} arm) on {base}: {mutation!r}\n"
                    f"(the mutation is not a well-formed different program by the consistency model)\n"
                    f"max |difference| per operand: {diff}"
                )
            _STATS["harmless"] += 1


@st.composite
def mutations(draw, base):
    doc = json.loads(_base_text(base))
    leaves = _leaves(doc)
    if draw(st.integers(0, 3)) == 0:
        id_like = [leaf for leaf in leaves if leaf.key in _ID_LIKE_KEYS]
        i = draw(st.integers(0, len(id_like) - 1))
        j = draw(st.integers(0, len(id_like) - 1).filter(lambda j: id_like[j].value != id_like[i].value))
        return _swap_mutation(id_like[i], id_like[j])
    leaf = leaves[draw(st.integers(0, len(leaves) - 1))]
    return _set_mutation(leaf, draw(st.sampled_from(_candidates(leaf))))


_SETTINGS = settings(
    max_examples=sanitizer_examples(80),
    deadline=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
)


@pytest.mark.parametrize("base", BASES)
@given(data=st.data())
@_SETTINGS
def test_a_mutated_file_is_reported_or_harmless(base, data):
    check_mutation(base, data.draw(mutations(base)))


def test_every_base_carries_what_the_mutations_target():
    """The bases must hold the fields the property claims to mutate.

    A schema change that dropped the GEMM hint, the slot redirect or the loop
    bound from what these graphs save would leave the property green over
    fields that no longer exist.
    """
    keys = {base: {leaf.key for leaf in _leaves(json.loads(_base_text(base)))} for base in BASES}
    everywhere = set().union(*keys.values())
    for needed in ("id", "rank", "dims", "inputs", "outputs", "m", "n", "k", "leading_dim", "from", "to", "size",
                   "max_iterations", "outer"):
        assert needed in everywhere, f"no base graph saves an integer under {needed!r}"
    assert "from" in keys["cse"], "the CSE base no longer saves a slot redirect"
    assert "m" in keys["pair"], "the batching pair no longer saves a GEMM hint"
    # The consistency model only excuses mutations; it must at least accept
    # every unmutated base, or every retarget would be judged strict.
    for base in BASES:
        assert _consistent(json.loads(_base_text(base))), f"the consistency model rejects the unmutated {base} base"


def test_mutation_corpus_is_not_all_reports():
    """Ordered last: the property must have run files, not only rejected them.

    A reader that refused everything would satisfy "reported or harmless"
    trivially.
    """
    assert _STATS["trials"] > 0, "no mutation trial ran"
    # Most mutations touch an extent, a rank or an id, and the loader checks those against each
    # other and against the GEMM hints, so reports dominate by design. What the guard needs is
    # every non-report outcome to occur: files that ran and matched, and well-formed different
    # programs that ran.
    ran = _STATS["harmless"] + _STATS["semantic_ran"]
    assert ran > 0.05 * _STATS["trials"], f"only {ran} of {_STATS['trials']} mutated files were executed: {_STATS}"
    assert _STATS["harmless"] > 0, f"no mutated file ran and matched: {_STATS}"
    assert _STATS["semantic_ran"] > 0, f"no well-formed different program ran: {_STATS}"
    assert _STATS["reported"] > 0.1 * _STATS["trials"], f"almost nothing was reported: {_STATS}"


# ──────────────────────────────────────────────────────────────────────────
# Pinned reproducers of what the property found
# ──────────────────────────────────────────────────────────────────────────


def _pinned(base, edits):
    """The mutation writing ``edits`` ({path: value}) into ``base``'s file."""
    doc = json.loads(_base_text(base))
    leaves = {leaf.path: leaf for leaf in _leaves(doc)}
    for path in edits:
        assert path in leaves, f"{base} no longer saves {path}; re-derive the reproducer"
    description = ", ".join(f"set {leaves[path]!r} -> {value}" for path, value in edits.items())
    return Mutation(description, list(edits.items()))


def _run_isolated(code):
    """Run ``code`` in a fresh interpreter, so a crash is an outcome, not the end of the session."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([_HERE, env.get("PYTHONPATH", "")])
    return subprocess.run([sys.executable, "-c", textwrap.dedent(code)], capture_output=True, text=True, env=env, timeout=300)


def _check_isolated(base, edits):
    """``check_mutation`` in a child process; a crash fails with the child's tail."""
    result = _run_isolated(
        f"""
        import test_hyp_ir_mutation_python as t
        t.check_mutation({base!r}, t._pinned({base!r}, {edits!r}))
        """
    )
    assert result.returncode == 0, f"the child died (exit {result.returncode}):\n{result.stderr[-3000:]}"


# Bugs the property found, each reduced to one deterministic mutation and
# pinned here beside the property that found it.


def test_hint_n_lowered_is_reported_or_harmless_under_batching():
    """A GEMM hint's extents are trusted by the batching pass.

    Fixed bug: the reader checks a hint's operand IDS against the frame but never
    its m / n / k against the operands' extents, and GEMMBatching keys its
    groups on, and ``gemm_batch`` runs with, the hint's extents. Lowering the
    second einsum's n from 3 to 2 puts both einsums in one group, and the
    batched call computes two of C2's three columns: the file validates,
    loads, and after the documented load pipeline silently leaves C2's last
    column unwritten.
    """
    doc = json.loads(_base_text("pair"))
    assert [n["descriptor"]["gemm_hint"]["n"] for n in doc["nodes"]] == [2, 3]
    check_mutation("pair", _pinned("pair", {("nodes", 1, "descriptor", "gemm_hint", "n"): 2}))


def test_hint_n_raised_is_reported_or_not_batched():
    """The same hole, the other direction: an out-of-bounds batched call.

    Raising the FIRST einsum's n from 2 to 3 groups it with the second, and the
    batched call writes a third column past the end of C1's 3x2 buffer and
    reads one past the end of B1's 4x2. Executing that is heap corruption,
    which crashes only some of the time (it killed about half the runs while
    this was being reduced), so this case asserts the deterministic cause
    instead and never executes: after the documented load pipeline, two
    einsums whose REAL extents differ must not share a batched call.
    """
    doc = json.loads(_base_text("pair"))
    doc["nodes"][0]["descriptor"]["gemm_hint"]["n"] = 3
    with tempfile.TemporaryDirectory() as scratch:
        path = os.path.join(scratch, "mutated.eig.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(doc, f)
        if _validate(path) is not None:
            return  # reported, which is the fix this case is waiting for
        loaded = cg.load_graph(path)
    ops = _operands("pair")
    for name in loaded.manifest_names():
        loaded.bind(name, ops[name])
    loaded.apply(cg.resource_pass_manager())
    loaded.apply(cg.tuning_pass_manager())
    labels = [(n["kind"], n["label"]) for n in json.loads(loaded.to_json())["nodes"]]
    assert all(kind != "BatchedGemm" for kind, _ in labels), (
        f"a 3x2 and a 3x3 product were batched into one call on the file's word: {labels}"
    )


def test_reversed_slot_redirect_is_reported_or_harmless():
    """A slot redirect whose target no node references.

    Fixed bug: the CSE base saves ``Y -> X`` (Y merged into X). Reversed to ``X -> Y``,
    both are still graph-owned 3x3 float64 scratch, so the file validates and
    loads. But no node references Y, so nothing materializes it, and X's
    readers and writer follow the redirect onto Y's never-allocated buffer:
    after the documented load pipeline, execute dies with SIGSEGV instead of
    reporting the unmaterialized tensor the way a plain execute does.
    """
    _check_isolated("cse", {("slot_redirects", 0, "from"): 3, ("slot_redirects", 0, "to"): 4})


def test_write_to_a_redirected_slot_is_reported_or_harmless():
    """A node whose output is the SOURCE of a slot redirect.

    Fixed bug: retargeting the first einsum of the CSE base from X to Y (same shape
    and dtype, and Y redirects to X) is a consistent file, and it validates and
    loads; after the documented load pipeline, execute dies with SIGSEGV.
    """
    _check_isolated("cse", {("nodes", 0, "outputs", 0): 4})


def test_zero_extent_intermediate_is_reported_or_harmless():
    """An intermediate declared with extent 0 where every node needs 3.

    Fixed bug: the CSE base's scratch X is 3x3; saved as 0x3 the file validates, the
    einsums that write and read it are never compared against its extents, and
    after the documented load pipeline the result O silently differs.
    """
    check_mutation("cse", _pinned("cse", {("tensors", 0, "dims", 0): 0}))


def test_scale_writing_a_different_tensor_is_reported_or_harmless():
    """A ``Scale`` whose output is not its input.

    Fixed bug: in the mixed golden, ``scale(0.5, C)`` has inputs [C] and outputs [C].
    Rewriting its output to D (2x3, where C is 3x2) is not a program any capture
    could write, and nothing reports it: the file validates, loads, and both C
    and D silently change.
    """
    check_mutation("mixed", _pinned("mixed", {("nodes", 1, "outputs", 0): 3}))


def test_axpby_with_swapped_operands_is_reported_or_harmless():
    """An ``Axpby`` whose y input is not its output.

    Fixed bug: ``axpby(0.5, X, 2.0, Y)`` saves inputs [X, Y] and outputs [Y]. With the
    inputs swapped to [Y, X] the node claims to read X as y yet write Y, which
    the executor cannot honor; the file validates and loads, and Y silently
    comes out different.
    """
    check_mutation("axpby", _pinned("axpby", {("nodes", 0, "inputs", 0): 1, ("nodes", 0, "inputs", 1): 0}))


def test_loop_body_adopting_a_different_outer_tensor_is_reported_or_harmless():
    """A loop body tensor declared 3x2 that adopts the enclosing 2x3 tensor.

    Fixed bug: the mixed golden's loop body declares C (3x2, outer id 2). Pointing its
    ``outer`` at D (2x3) instead is accepted: ``adopt_outer_tensor`` takes the
    enclosing storage without comparing its extents (or dtype) with the body's
    declaration, and the loop silently accumulates into D.
    """
    check_mutation("mixed", _pinned("mixed", {("nodes", 6, "descriptor", "body", "tensors", 0, "outer"): 3}))


def test_rank0_bind_to_a_loaded_scalar_slot_is_refused():
    """Binding a rank-0 tensor to a loaded graph's bare scalar output.

    The golden predates the rank-0 kind, so its ``dot_result`` reads as the bare
    element a dot writes through a pointer. Binding a rank-0 tensor OBJECT there
    used to succeed, and the Dot's result was written over the object itself,
    which crashed the next read of it. The bind must refuse instead.
    """
    loaded = cg.load_graph(_GOLDEN_MIXED)
    out = einsums.create_zero_tensor("dot_result", [], dtype="float64")
    with pytest.raises(ValueError, match="bare scalar"):
        loaded.bind("dot_result", out)


def test_structural_phase_of_a_permutation_operator_saves():
    """``structural_pass_manager`` is documented as the phase a saved graph persists.

    Fixed bug: over a single ``i,j <- P(i/j) i,k ; k,j`` einsum, the structural phase
    expands the operator through graph-owned scratch and leaves an ``Alloc``
    node behind, which ``save_graph`` refuses as not reconstructible. The
    documented save flow therefore cannot save the plainest antisymmetrized
    contraction.
    """
    g = _build_perm_op(_operands("perm_op"))
    g.apply(cg.structural_pass_manager())
    _save_text(g)


def test_a_saved_scalar_einsum_loads():
    """A graph whose einsum writes a rank-0 tensor saves, loads, and computes.

    The record used to say only "rank 0", and the loader rebuilt a bare element
    with no rank-erased geometry, so ``load_graph`` refused the file ``save_graph``
    had just written. A save that cannot be read back is the one outcome the
    save's refusal rule exists to prevent.
    """
    a = _tensor("A", (2, 2), 0)
    e = einsums.create_zero_tensor("e", [], dtype="float64")
    g = cg.Graph("scalar_einsum")
    with cg.capture(g):
        einsums.einsum(" <- ij ; ij", e, a, a)
    with tempfile.TemporaryDirectory() as scratch:
        path = os.path.join(scratch, "scalar.eig.json")
        cg.save_graph(g, path)
        assert _validate(path) is None
        loaded = cg.load_graph(path)
    a2 = _tensor("A2", (2, 2), 1)
    e2 = einsums.create_zero_tensor("e2", [], dtype="float64")
    loaded.bind("A", a2)
    loaded.bind("e", e2)
    loaded.execute()
    expected = float(np.sum(np.asarray(a2) * np.asarray(a2)))
    assert float(np.asarray(e2)) == pytest.approx(expected, rel=1e-12)
