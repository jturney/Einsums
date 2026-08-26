# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""Python mirror of SymbolicExtents.cpp and SpaceShapedDeclare.cpp.

The C++ tests own the semantics. This file asserts the half that is BOUND, and
it exists because that half was missing: ``annotate_dims`` shipped with no
``APIARY_EXPOSE``, so a Python caller could save and load a graph but could not
make one rebindable, which is the entire point of symbolic extents. Every
assertion below is something no Python program could express before.
"""

from __future__ import annotations

import pytest

import einsums
import einsums.graph as cg


def _spaces():
    """A registry whose spaces carry dim symbols, which is what the sugar reads."""
    reg = cg.SpaceRegistry()
    occ = reg.register_space(cg.index_space("occ", "o", 4.0, cg.GrowthClass.linear(), "no"))
    virt = reg.register_space(cg.index_space("virt", "v", 8.0, cg.GrowthClass.linear(), "nv"))
    return reg, occ, virt


def test_index_space_carries_a_dim_symbol():
    reg, occ, _ = _spaces()

    space = reg.space(occ)
    assert space.name == "occ"
    assert space.scale_symbol == "o"
    # Distinct from both of the above: the name a symbolic EXTENT over this space goes by.
    assert space.dim_symbol == "no"


def test_a_space_may_have_no_dim_symbol():
    reg = cg.SpaceRegistry()
    grid = reg.register_space(cg.index_space("grid", "g"))
    assert reg.space(grid).dim_symbol == ""


def test_annotate_dims_is_reachable_from_python():
    reg, occ, virt = _spaces()
    A = einsums.create_random_tensor("A", [4, 8])

    g = cg.Graph("dims")
    g.set_space_registry(reg)
    g.annotate_spaces(A, [occ, virt])
    g.annotate_dims(A, ["no", "nv"])

    assert g.tensor_dim_symbols(A) == ["no", "nv"]


def test_a_symbol_tied_to_two_spaces_is_refused():
    reg, occ, virt = _spaces()
    A = einsums.create_random_tensor("A", [4, 8])
    B = einsums.create_random_tensor("B", [8, 4])

    g = cg.Graph("conflict")
    g.set_space_registry(reg)
    g.annotate_spaces(A, [occ, virt])
    g.annotate_dims(A, ["no", "nv"])

    g.annotate_spaces(B, [virt, occ])
    with pytest.raises(Exception, match="one symbol names one extent"):
        g.annotate_dims(B, ["no", "nv"])


def test_annotate_ragged_dim_is_reachable_from_python():
    reg, occ, virt = _spaces()
    A = einsums.create_random_tensor("A", [4, 8])

    g = cg.Graph("ragged_dim")
    g.set_space_registry(reg)
    g.annotate_spaces(A, [occ, virt])
    g.annotate_ragged_dim(A, 1, "virt")

    assert g.tensor_dim_symbols(A) == ["", "ragged:virt"]


def test_space_extent_is_learned_from_an_annotation():
    reg, occ, virt = _spaces()

    g = cg.Graph("learn")
    g.set_space_registry(reg)
    assert g.space_extent(occ) is None

    g.annotate_spaces(einsums.create_random_tensor("A", [4, 8]), [occ, virt])

    assert g.space_extent(occ) == 4
    assert g.space_extent(virt) == 8


def test_a_ragged_family_leaves_the_space_unpinned():
    reg, occ, virt = _spaces()

    g = cg.Graph("ragged")
    g.set_space_registry(reg)
    g.annotate_spaces(einsums.create_random_tensor("pair_one", [4, 8]), [occ, virt])
    # Legal and deliberately not an error: this is what a PNO domain looks like.
    g.annotate_spaces(einsums.create_random_tensor("pair_two", [4, 5]), [occ, virt])

    assert g.space_extent(occ) == 4
    assert g.space_extent(virt) is None


def test_space_shaped_declare_sizes_and_annotates_in_one_call():
    reg, occ, virt = _spaces()

    g = cg.Graph("declare")
    g.set_space_registry(reg)
    g.pin_space_extent(occ, 4)
    g.pin_space_extent(virt, 8)

    scratch = g.declare_zero_tensor_over("scratch", [cg.SpaceDim(occ), cg.SpaceDim(virt)])

    assert scratch.dim(0) == 4
    assert scratch.dim(1) == 8
    assert g.tensor_spaces(scratch) == [occ, virt]
    assert g.tensor_dim_symbols(scratch) == ["no", "nv"]


def test_fixed_axis_stays_literal():
    reg, occ, _ = _spaces()

    g = cg.Graph("mixed")
    g.set_space_registry(reg)
    g.pin_space_extent(occ, 4)

    # A DIIS history depth means nothing chemically and must not move with the problem.
    history = g.declare_zero_tensor_over("history", [cg.fixed(8), cg.SpaceDim(occ)])

    assert history.dim(0) == 8
    assert history.dim(1) == 4
    assert g.tensor_dim_symbols(history) == ["", "no"]


def test_an_unpinned_space_is_refused_by_name():
    reg, _, virt = _spaces()

    g = cg.Graph("unpinned")
    g.set_space_registry(reg)

    with pytest.raises(Exception, match="how big it is"):
        g.declare_zero_tensor_over("scratch", [cg.SpaceDim(virt)])


def test_tiled_declare_takes_the_space_tiling():
    reg, occ, virt = _spaces()

    g = cg.Graph("tiled")
    g.set_space_registry(reg)
    g.pin_space_tiling(occ, [2, 2])
    g.pin_space_tiling(virt, [4, 4])

    # A partition states an extent, so pinning one pins the other.
    assert g.space_extent(occ) == 4
    assert g.space_tiling(occ) == [2, 2]

    tiled = g.declare_zero_tiled_tensor_over("T", [cg.SpaceTiling(occ), cg.SpaceTiling(virt)])

    assert g.tensor_spaces(tiled) == [occ, virt]
    # A PLAIN symbol: the space fixes the total, the tiling is layout.
    assert g.tensor_dim_symbols(tiled) == ["no", "nv"]


def test_a_tiled_axis_overrides_the_canonical_tiling():
    reg, occ, virt = _spaces()

    g = cg.Graph("tiled_override")
    g.set_space_registry(reg)
    g.pin_space_tiling(occ, [2, 2])
    g.pin_space_tiling(virt, [4, 4])

    tiled = g.declare_zero_tiled_tensor_over("U", [cg.SpaceTiling(occ, [1, 3]), cg.SpaceTiling(virt)])
    assert g.tensor_spaces(tiled) == [occ, virt]


def test_a_partition_contradicting_the_space_is_refused():
    reg, occ, _ = _spaces()

    g = cg.Graph("tiled_bad")
    g.set_space_registry(reg)
    g.pin_space_tiling(occ, [2, 2])  # occ measures 4

    with pytest.raises(Exception, match="summing to 6"):
        g.declare_zero_tiled_tensor_over("bad", [cg.SpaceTiling(occ, [2, 2, 2])])


def test_create_takes_spaces_but_writes_no_dim_symbols():
    reg, occ, virt = _spaces()

    g = cg.Graph("create")
    g.set_space_registry(reg)
    g.pin_space_extent(occ, 4)
    g.pin_space_extent(virt, 8)

    T = g.create_zero_tensor_over("T", [cg.SpaceDim(occ), cg.SpaceDim(virt)])

    assert T.dim(0) == 4
    assert T.dim(1) == 8
    assert g.tensor_spaces(T) == [occ, virt]
    # Deliberately absent: this tensor is allocated now and a bind cannot resize it.
    assert g.tensor_dim_symbols(T) == []
