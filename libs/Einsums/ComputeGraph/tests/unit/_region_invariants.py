# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""Structural invariants a rewritten graph owes, independent of its numbers.

A differential shard answers "did the answer change". It cannot see a buffer
allocated for a tensor the rewrite dissolved, or a second lifecycle emitted for
a tensor that already had one, because neither moves a number: the first spends
memory and the second re-runs an allocation that happens to be idempotent. Both
shipped, and the CCSD case that found the first found it by asking a question
about storage rather than about arithmetic.

So the questions are asked here and every fuzzer that applies a resource pass
calls them. The audit itself is in C++, in the Materialization pass's own
header, for two reasons the JSON cannot get around: ``Graph.to_json`` does not
descend into loop bodies or conditional branches, so a scratch tensor used only
inside one reads as unused; and a use through a view needs the alias root, which
is a graph question rather than a document one.
"""

from __future__ import annotations

import einsums.graph as cg


def assert_materialization_invariants(graph, label=""):
    """Every Materialize names a tensor something uses, and names it once.

    The first is what stops a dissolved intermediate keeping its buffer; the
    second is what stops two passes each emitting a lifecycle for one tensor.
    Only graph-owned intermediates are held to the first: a caller-owned
    deferred tensor is one the caller may read after the graph runs, so
    allocating an unused one is deliberate.
    """
    where = f" [{label}]" if label else ""

    duplicates = cg.duplicate_materializations(graph)
    assert not duplicates, f"a tensor has more than one Materialize node{where}: {duplicates}"

    stranded = cg.stranded_materializations(graph)
    assert not stranded, (
        f"a Materialize node allocates a graph-owned intermediate no node uses{where}: {stranded}"
    )
