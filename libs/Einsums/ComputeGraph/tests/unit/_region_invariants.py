# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""Structural invariants a rewritten graph owes, independent of its numbers.

A differential shard answers "did the answer change". It cannot see a buffer
allocated for a tensor the rewrite dissolved, or a second lifecycle emitted for
a tensor that already had one, because neither moves a number: the first spends
memory and the second re-runs an allocation that is idempotent. Both shipped,
and the CCSD case that found the first found it by asking a question about
storage rather than about arithmetic.

So the questions are asked here, off the JSON IR, and every fuzzer that applies
a resource pass calls them. Read by NAME rather than by id, which is what
``already_materialized_in`` in the pass itself does and for the same reason: a
sub-graph carries its own id for the parent's buffer, and a name is the one
handle both spellings share.
"""

from __future__ import annotations

import json

#: Node kinds that manage a tensor's storage rather than reading or writing its
#: value. Mirrors ``is_lifecycle`` in ComputeGraphTypes/Enums.hpp.
LIFECYCLE_KINDS = ("Alloc", "Free", "Materialize", "Initialize")

#: Keys under which a node carries a nested node list (loop bodies, conditional
#: branches, setup bodies).
_SUBGRAPH_KEYS = ("body", "then_body", "else_body", "nodes")


def _walk(nodes, visit):
    for node in nodes or []:
        visit(node)
        for key in _SUBGRAPH_KEYS:
            sub = node.get(key)
            if isinstance(sub, dict):
                _walk(sub.get("nodes"), visit)
            elif isinstance(sub, list):
                _walk(sub, visit)


def _name_of(label, prefix):
    """``materialize(T)`` -> ``T``; anything else -> ``None``."""
    if label.startswith(prefix + "(") and label.endswith(")"):
        return label[len(prefix) + 1:-1]
    return None


def materialization_report(graph):
    """What the graph says about deferred storage, as plain data.

    Returns ``(materialized, used)``: the list of tensor names each
    ``Materialize`` node names, in walk order and WITH duplicates, and the set
    of tensor names some non-lifecycle node reads or writes anywhere in the
    graph tree. A ``View`` node counts as a read of its parent, which is how the
    pass's own usage analysis resolves an alias.
    """
    doc = json.loads(graph.to_json())

    names = {}

    def collect_names(document):
        for tensor in document.get("tensors", []):
            if "id" in tensor and tensor.get("name"):
                names[tensor["id"]] = tensor["name"]

    collect_names(doc)

    materialized = []
    used = set()

    def visit(node):
        kind = node.get("kind", "")
        label = node.get("label", "")
        if kind == "Materialize":
            name = _name_of(label, "materialize")
            if name is not None:
                materialized.append(name)
            return
        if kind in LIFECYCLE_KINDS:
            return
        for tid in (node.get("inputs") or []) + (node.get("outputs") or []):
            if tid in names:
                used.add(names[tid])

    _walk(doc.get("nodes"), visit)
    return materialized, used


def assert_materialization_invariants(graph, label=""):
    """Every Materialize names a tensor something uses, and names it once.

    The first is what stops a dissolved intermediate keeping its buffer; the
    second is what stops two passes each emitting a lifecycle for one tensor,
    which survives only because ``materialize_fn`` happens to be idempotent.
    """
    materialized, used = materialization_report(graph)
    where = f" [{label}]" if label else ""

    seen = set()
    duplicates = sorted({name for name in materialized if name in seen or seen.add(name)})
    assert not duplicates, (
        f"a tensor has more than one Materialize node{where}: {duplicates}\n"
        f"materialize nodes={materialized}"
    )

    # A Materialize for a name nothing reads or writes is a buffer allocated for
    # a tensor the program does not have. The pass counts these and declines
    # them; one arriving here means something re-introduced it.
    stranded = sorted(name for name in materialized if name not in used)
    assert not stranded, (
        f"a Materialize node names a tensor no node uses{where}: {stranded}\n"
        f"used={sorted(used)}"
    )
