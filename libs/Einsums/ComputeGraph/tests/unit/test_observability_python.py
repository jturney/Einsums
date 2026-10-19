#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

"""What a graph reports about itself: its shape, its interface, and what its last replay did."""

import numpy as np

import einsums
import einsums.graph as cg


def _filled(name, shape, scale):
    tensor = einsums.create_zero_tensor(name, list(shape), dtype="float64")
    np.asarray(tensor, copy=False)[:] = scale * (np.arange(int(np.prod(shape)), dtype=np.float64) % 7 - 3.0).reshape(shape)
    return tensor


def _product():
    """C = A B, then D = C B^T: two contractions over caller-held tensors."""
    A = _filled("A", (4, 3), 0.5)
    B = _filled("B", (3, 5), 0.25)
    C = einsums.create_zero_tensor("C", [4, 5], dtype="float64")
    D = einsums.create_zero_tensor("D", [4, 3], dtype="float64")
    g = cg.Graph("observed")
    with cg.capture(g):
        einsums.einsum("ij <- ik ; kj", C, A, B)
        einsums.einsum("ik <- ij ; kj", D, C, B)
    return g


def test_dot_names_every_tensor_and_operation():
    g = _product()
    dot = g.to_dot()
    assert dot.startswith('digraph "observed" {')
    assert dot.rstrip().endswith("}")
    for name in ("A", "B", "C", "D"):
        assert f'label="{name}\\n' in dot
    assert dot.count("shape=ellipse") == g.num_nodes()


def test_summary_is_text_naming_the_graph():
    g = _product()
    summary = g.summary()
    assert "observed" in summary
    assert summary.count("\n") >= g.num_nodes()


def test_timing_report_has_one_entry_per_node_run():
    g = _product()
    assert g.timing_report() == []
    g.execute()
    report = g.timing_report()
    assert len(report) == g.num_nodes()
    assert all(entry.duration_ms >= 0.0 for entry in report)
    assert all(entry.label for entry in report)
    assert {entry.kind for entry in report} == {cg.OpKind.Einsum}


def test_manifest_lists_the_interface_with_its_contract():
    g = _product()
    entries = {entry.name: entry for entry in g.manifest()}
    assert sorted(entries) == sorted(g.manifest_names())
    assert entries["A"].direction == "input"
    assert entries["D"].direction == "output"
    # C is written by the first contraction and read by the second.
    assert entries["C"].direction == "inout"
    assert entries["A"].dtype == "float64"
    assert list(entries["A"].dims) == [4, 3]
    assert entries["A"].rank == 2
    assert entries["A"].scope == "graph"
