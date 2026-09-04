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

import numpy as np
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


def _global_spaces(tag):
    """Spaces in the PROCESS-GLOBAL registry, which is the one load_graph resolves against."""
    reg = cg.global_space_registry()
    occ = reg.register_space(cg.index_space(f"{tag}_occ", "o", 8.0, cg.GrowthClass.linear(), f"{tag}_no"))
    vir = reg.register_space(cg.index_space(f"{tag}_virt", "v", 16.0, cg.GrowthClass.linear(), f"{tag}_nv"))
    return reg, occ, vir


def _chain(tag, no, nv, occ, vir, seed):
    """One contraction chain with a graph-owned deferred intermediate."""
    rng = np.random.default_rng(seed)
    amp = einsums.create_zero_tensor("amp", [no, nv])
    out = einsums.create_zero_tensor("out", [no, no])
    np.asarray(amp)[...] = rng.standard_normal((no, nv))

    g = cg.Graph(f"chain_{no}_{nv}")
    g.annotate_spaces(amp, [occ, vir])
    g.annotate_dims(amp, [f"{tag}_no", f"{tag}_nv"])
    g.annotate_spaces(out, [occ, occ])
    g.annotate_dims(out, [f"{tag}_no", f"{tag}_no"])
    tmp = g.declare_zero_tensor_over("tmp", [cg.SpaceDim(occ), cg.SpaceDim(vir)], True)
    with cg.capture(g):
        einsums.einsum("i,a <- i,a ; i,a", tmp, amp, amp, c_pf=0.0, ab_pf=1.0)
        einsums.einsum("i,j <- i,a ; j,a", out, tmp, amp, c_pf=0.0, ab_pf=1.0)
    return g, amp, out, tmp


def test_cg_bind_moves_a_multi_slot_interface():
    """The dict form binds every slot as ONE transaction.

    A dim symbol constrains across slots, so a caller moving the problem's extents has to
    hand over the whole interface before any of it is reconciled. Binding one slot at a
    time solves the second against an interface the first already moved.
    """
    _, occ, vir = _global_spaces("pybind_tx")
    g, _, _, tmp = _chain("pybind_tx", 4, 6, occ, vir, seed=3)

    assert [tmp.dim(i) for i in range(2)] == [4, 6]

    amp2 = einsums.create_zero_tensor("amp2", [3, 5])
    out2 = einsums.create_zero_tensor("out2", [3, 3])
    np.asarray(amp2)[...] = np.random.default_rng(9).standard_normal((3, 5))

    cg.bind(g, {"amp": amp2, "out": out2})

    # The behaviour that matters: the deferred intermediate followed the symbols.
    assert [tmp.dim(i) for i in range(2)] == [3, 5]


def test_cg_bind_rejects_a_non_dict():
    _, occ, vir = _global_spaces("pybind_bad")
    g, _, _, _ = _chain("pybind_bad", 4, 6, occ, vir, seed=3)
    with pytest.raises(TypeError, match="expects a dict"):
        cg.bind(g, [("amp", None)])


def test_a_saved_graph_with_scratch_replays_at_a_new_size(tmp_path):
    """THE case the feature exists for, for a graph that has scratch.

    Capture at one size, save, load with no addresses in it, bind a different-sized
    problem, and get bitwise what a fresh capture at that size computes.
    """
    _, occ, vir = _global_spaces("pyreuse")

    g1, _, _, _ = _chain("pyreuse", 6, 10, occ, vir, seed=7)
    path = str(tmp_path / "chain.eig")
    cg.save_graph(g1, path)

    # Fresh operands for the NEW problem, used by both the replay and the reference.
    rng = np.random.default_rng(11)
    amp_new = rng.standard_normal((8, 14))

    amp_a = einsums.create_zero_tensor("amp_a", [8, 14])
    out_a = einsums.create_zero_tensor("out_a", [8, 8])
    np.asarray(amp_a)[...] = amp_new

    g2 = cg.load_graph(path)
    cg.bind(g2, {"amp": amp_a, "out": out_a})
    g2.optimize()
    g2.execute()

    # Reference: capture the same chain at the new size.
    amp_b = einsums.create_zero_tensor("amp_b", [8, 14])
    out_b = einsums.create_zero_tensor("out_b", [8, 8])
    np.asarray(amp_b)[...] = amp_new
    g3 = cg.Graph("fresh")
    g3.annotate_spaces(amp_b, [occ, vir])
    g3.annotate_dims(amp_b, ["pyreuse_no", "pyreuse_nv"])
    g3.annotate_spaces(out_b, [occ, occ])
    g3.annotate_dims(out_b, ["pyreuse_no", "pyreuse_no"])
    tmp_b = g3.declare_zero_tensor_over("tmp", [cg.SpaceDim(occ), cg.SpaceDim(vir)], True)
    with cg.capture(g3):
        einsums.einsum("i,a <- i,a ; i,a", tmp_b, amp_b, amp_b, c_pf=0.0, ab_pf=1.0)
        einsums.einsum("i,j <- i,a ; j,a", out_b, tmp_b, amp_b, c_pf=0.0, ab_pf=1.0)
    g3.optimize()
    g3.execute()

    # Not close: identical. Both run the same kernels over the same values in the same order.
    assert np.array_equal(np.asarray(out_a), np.asarray(out_b))


def test_load_graph_into_a_private_registry(tmp_path):
    """A saved graph resolves its space NAMES against a registry the caller chooses.

    load_graph used the process-global registry unconditionally, so a caller keeping
    their own was told the space "is not registered in this process" with an empty list
    of what IS registered, having registered everything.
    """
    mine = cg.SpaceRegistry()
    occ = mine.register_space(cg.index_space("priv_occ", "o", 4.0, cg.GrowthClass.linear(), "p_no"))
    vir = mine.register_space(cg.index_space("priv_virt", "v", 6.0, cg.GrowthClass.linear(), "p_nv"))

    amp = einsums.create_zero_tensor("amp", [4, 6])
    out = einsums.create_zero_tensor("out", [4, 4])
    np.asarray(amp)[...] = np.random.default_rng(1).standard_normal((4, 6))

    g = cg.Graph("priv")
    g.set_space_registry(mine)
    g.annotate_spaces(amp, [occ, vir])
    g.annotate_dims(amp, ["p_no", "p_nv"])
    g.annotate_spaces(out, [occ, occ])
    g.annotate_dims(out, ["p_no", "p_no"])
    tmp = g.declare_zero_tensor_over("tmp", [cg.SpaceDim(occ), cg.SpaceDim(vir)], True)
    with cg.capture(g):
        einsums.einsum("i,a <- i,a ; i,a", tmp, amp, amp, c_pf=0.0, ab_pf=1.0)
        einsums.einsum("i,j <- i,a ; j,a", out, tmp, amp, c_pf=0.0, ab_pf=1.0)

    path = str(tmp_path / "priv.eig")
    cg.save_graph(g, path)

    # The global registry has never heard of these names.
    with pytest.raises(Exception, match="priv_occ"):
        cg.load_graph(path)

    # Handed the right registry, the same file loads and is still rebindable.
    loaded = cg.load_graph_into(path, mine)
    assert sorted(loaded.manifest_names()) == ["amp", "out"]

    amp2 = einsums.create_zero_tensor("amp2", [3, 5])
    out2 = einsums.create_zero_tensor("out2", [3, 3])
    np.asarray(amp2)[...] = np.random.default_rng(2).standard_normal((3, 5))
    cg.bind(loaded, {"amp": amp2, "out": out2})
    loaded.optimize()
    loaded.execute()


def test_a_bound_tensor_is_addressable_by_the_object_bound_to_it():
    """The whole point of symbolic extents is a graph a caller can rebind, and
    after that rebind the caller has to be able to ASK the graph about the
    tensor it just handed over.

    Every by-object lookup on a Graph resolves through an address index, and a
    rebind used to leave that index behind: it went on naming the tensor the
    graph was captured over, so the graph reported the tensor it had just been
    bound to as one it had never seen, while happily answering about storage it
    no longer used.
    """
    reg, occ, virt = _spaces()

    A = einsums.create_random_tensor("A", [4, 8])
    out = einsums.create_zero_tensor("out", [4, 8])

    g = cg.Graph("bind_by_object")
    g.set_space_registry(reg)
    g.annotate_spaces(A, [occ, virt])
    g.annotate_dims(A, ["no", "nv"])
    with cg.capture(g):
        einsums.permute("i,j <- i,j", out, A, c_pf=0.0, a_pf=1.0)

    A2 = einsums.create_random_tensor("A", [4, 8])
    cg.bind(g, {"A": A2, "out": out})

    # A2 took A's slot, so it is the tensor the annotations are about now.
    assert g.tensor_spaces(A2) == [occ, virt]
    assert g.tensor_dim_symbols(A2) == ["no", "nv"]

    # And the graph no longer claims to know the tensor it was moved off.
    with pytest.raises(Exception, match="is not registered"):
        g.tensor_spaces(A)
