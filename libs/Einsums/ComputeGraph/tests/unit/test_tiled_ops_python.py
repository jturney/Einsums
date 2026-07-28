# ----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# ----------------------------------------------------------------------------------------------

"""Python coverage for ComputeGraph ops over TiledRuntimeTensor operands.

The tiled type is filled per tile (tile_view -> numpy); these tests then run the
graph ops (scale / axpy / einsum) over the whole tiled tensor and compare a
gathered dense view against a numpy reference.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

import einsums
import einsums.graph as cg
from einsums import linalg as la
from einsums.testing import ALL_DTYPES, assert_close

DTYPE_TO_TRT = {
    np.float32: einsums.TiledRuntimeTensorF,
    np.float64: einsums.TiledRuntimeTensorD,
    np.complex64: einsums.TiledRuntimeTensorC,
    np.complex128: einsums.TiledRuntimeTensorZ,
}

# 1-element dense result tensors for scalar reductions: same dtype for
# dot/trace, the real type for norm.
_RT = {
    np.float32: einsums.RuntimeTensorF,
    np.float64: einsums.RuntimeTensorD,
    np.complex64: einsums.RuntimeTensorC,
    np.complex128: einsums.RuntimeTensorZ,
}
_REAL_RT = {
    np.float32: einsums.RuntimeTensorF,
    np.float64: einsums.RuntimeTensorD,
    np.complex64: einsums.RuntimeTensorF,
    np.complex128: einsums.RuntimeTensorD,
}


def _make(dtype, name, grid, fill=None):
    """Build a tiled tensor over ``grid`` (list per axis), populate every tile,
    and optionally fill from a global (row, col) -> value function."""
    t = DTYPE_TO_TRT[np.dtype(dtype).type](name, grid)
    off, sz = t.tile_offsets(), t.tile_sizes()
    for ti in range(len(sz[0])):
        for tj in range(len(sz[1])):
            t.add_tile([ti, tj])
    t.materialize()
    if fill is not None:
        for ti in range(len(sz[0])):
            for tj in range(len(sz[1])):
                a = np.asarray(t.tile_view([ti, tj]))
                for lr in range(sz[0][ti]):
                    for lc in range(sz[1][tj]):
                        a[lr, lc] = fill(off[0][ti] + lr, off[1][tj] + lc)
    return t


def _gather(t, R, C):
    """Reconstruct a dense R x C array from a 2-D tiled tensor (absent -> 0)."""
    off, sz = t.tile_offsets(), t.tile_sizes()
    M = np.zeros((R, C), dtype=np.asarray(t.tile_view([0, 0])).dtype)
    for ti in range(len(sz[0])):
        for tj in range(len(sz[1])):
            if t.has_tile([ti, tj]):
                r0, c0 = off[0][ti], off[1][tj]
                M[r0 : r0 + sz[0][ti], c0 : c0 + sz[1][tj]] = np.asarray(t.tile_view([ti, tj]))
    return M


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_tiled_scale(dtype):
    ref = (1.0 + np.arange(45, dtype=dtype)).reshape(5, 9)
    t = _make(dtype, "A", [[2, 3], [4, 5]], fill=lambda r, c: ref[r, c])
    einsums.linalg.scale(2.0, t)
    assert_close(_gather(t, 5, 9), 2.0 * ref)


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_tiled_axpy(dtype):
    xref = (1.0 + np.arange(45, dtype=dtype)).reshape(5, 9)
    yref = (3.0 - np.arange(45, dtype=dtype)).reshape(5, 9)
    X = _make(dtype, "X", [[2, 3], [4, 5]], fill=lambda r, c: xref[r, c])
    Y = _make(dtype, "Y", [[2, 3], [4, 5]], fill=lambda r, c: yref[r, c])
    einsums.linalg.axpy(1.5, X, Y)
    assert_close(_gather(Y, 5, 9), yref + 1.5 * xref)


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_tiled_axpby(dtype):
    xref = (1.0 + np.arange(45, dtype=dtype)).reshape(5, 9)
    yref = (3.0 - np.arange(45, dtype=dtype)).reshape(5, 9)
    X = _make(dtype, "X", [[2, 3], [4, 5]], fill=lambda r, c: xref[r, c])
    Y = _make(dtype, "Y", [[2, 3], [4, 5]], fill=lambda r, c: yref[r, c])
    einsums.linalg.axpby(1.5, X, -0.5, Y)
    assert_close(_gather(Y, 5, 9), 1.5 * xref - 0.5 * yref)


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_tiled_dotc(dtype):
    xref = (1.0 + np.arange(45)).reshape(5, 9).astype(dtype)
    yref = (3.0 - np.arange(45)).reshape(5, 9).astype(dtype)
    if np.dtype(dtype).kind == "c":
        xref = xref + 1j * np.arange(45, dtype="float64").reshape(5, 9).astype(dtype)
    X = _make(dtype, "X", [[2, 3], [4, 5]], fill=lambda r, c: xref[r, c])
    Y = _make(dtype, "Y", [[2, 3], [4, 5]], fill=lambda r, c: yref[r, c])
    s = einsums.zeros((1,), dtype=dtype)
    einsums.linalg.dotc(s, X, Y)
    expected = np.vdot(xref, yref)  # conjugated inner product
    assert_close(np.asarray(s).reshape(1), np.array([expected], dtype=dtype))


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_tiled_zeros_like_preserves_structure(dtype):
    # A sparse tile set must survive zeros_like: densifying would silently
    # discard the sparsity that is the point of the tiled type.
    t = DTYPE_TO_TRT[np.dtype(dtype).type]("t", [[2, 3], [4, 5]])
    t.add_tile([0, 0])
    t.add_tile([1, 1])
    t.materialize()
    np.asarray(t.tile_view([0, 0]))[...] = 7.0

    z = einsums.zeros_like(t)
    assert type(z) is type(t)
    assert z.tile_sizes() == t.tile_sizes()
    for i in range(2):
        for j in range(2):
            assert z.has_tile([i, j]) == t.has_tile([i, j])
    assert np.asarray(z.tile_view([0, 0])).max() == 0.0
    assert z.dtype == t.dtype  # the minimal tiled ergonomics layer
    assert z.shape == t.shape


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_tiled_permute_transpose(dtype):
    # C[j,i] = A[i,j]: C's grid is A's grid permuted the same way its axes are.
    aref = (1.0 + np.arange(45, dtype=dtype)).reshape(5, 9)
    A = _make(dtype, "A", [[2, 3], [4, 5]], fill=lambda r, c: aref[r, c])
    C = DTYPE_TO_TRT[np.dtype(dtype).type]("C", [[4, 5], [2, 3]])
    einsums.permute("j,i <- i,j", C, A)
    assert_close(_gather(C, 9, 5), aref.T)


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_tiled_permute_accumulates_with_prefactors(dtype):
    aref = (1.0 + np.arange(45, dtype=dtype)).reshape(5, 9)
    cref = (2.0 - np.arange(45, dtype=dtype)).reshape(9, 5)
    A = _make(dtype, "A", [[2, 3], [4, 5]], fill=lambda r, c: aref[r, c])
    C = _make(dtype, "C", [[4, 5], [2, 3]], fill=lambda r, c: cref[r, c])
    einsums.permute("j,i <- i,j", C, A, c_pf=0.5, a_pf=2.0)
    assert_close(_gather(C, 9, 5), 0.5 * cref + 2.0 * aref.T)


def test_tiled_permute_sparsity_rules():
    # An absent A tile is a rigorous zero: its target C tile keeps beta*C, and
    # a stored C tile the permutation never reaches is still scaled by beta.
    A = einsums.TiledRuntimeTensorD("A", [[2, 3], [4, 5]])
    A.add_tile([0, 0])
    A.materialize()
    np.asarray(A.tile_view([0, 0]))[...] = 3.0

    C = einsums.TiledRuntimeTensorD("C", [[4, 5], [2, 3]])
    C.add_tile([1, 1])  # never a permutation target of A's stored tiles
    C.materialize()
    np.asarray(C.tile_view([1, 1]))[...] = 8.0

    einsums.permute("j,i <- i,j", C, A, c_pf=0.5, a_pf=1.0)

    assert C.has_tile([0, 0])  # created by A[0,0]'s contribution
    np.testing.assert_allclose(np.asarray(C.tile_view([0, 0])), 3.0)
    np.testing.assert_allclose(np.asarray(C.tile_view([1, 1])), 4.0)  # 0.5 * 8
    assert not C.has_tile([0, 1]) and not C.has_tile([1, 0])  # zeros stay absent


def test_tiled_permute_rank4_symmetrizer():
    # The CC symmetrizer shape: C[j,i,b,a] = A[i,j,a,b] on a rank-4 grid.
    rng = np.random.default_rng(3)
    aref = rng.standard_normal((4, 4, 5, 5))
    grid = [[2, 2], [2, 2], [2, 3], [2, 3]]
    A = _make_nd("float64", "A", grid, ref=aref)
    C = einsums.TiledRuntimeTensorD("C", [[2, 2], [2, 2], [2, 3], [2, 3]])
    einsums.permute("j,i,b,a <- i,j,a,b", C, A)
    assert_close(_gather_nd(C, (4, 4, 5, 5), "float64"), aref.transpose(1, 0, 3, 2))


def test_tiled_permute_grid_mismatch_throws():
    A = einsums.TiledRuntimeTensorD("A", [[2, 3], [4, 5]])
    C = einsums.TiledRuntimeTensorD("C", [[2, 3], [4, 5]])  # NOT permuted: wrong for a transpose
    with pytest.raises(ValueError):
        einsums.permute("j,i <- i,j", C, A)


def _node_labels(g):
    import json
    return [n.get("label", "") for n in json.loads(g.to_json()).get("nodes", [])]


def test_tiled_permute_expands_in_pipeline():
    """TiledExpansion lowers a captured tiled permute into per-tile dense
    Permute nodes (one per stored A tile, targets bijectively permuted), with
    the leftover-scale rule for stored C tiles the permutation never reaches.
    Before the TiledPermuteDescriptor existed the node was opaque and its
    cannot-expand contagion stranded every tensor it touched."""
    aref = (1.0 + np.arange(45)).reshape(5, 9).astype("float64")
    cref = (2.0 - np.arange(45)).reshape(9, 5).astype("float64")
    A = _make("float64", "A", [[2, 3], [4, 5]], fill=lambda r, c: aref[r, c])
    C = _make("float64", "C", [[4, 5], [2, 3]], fill=lambda r, c: cref[r, c])

    g = cg.Graph("tiled_permute_expand")
    with cg.capture(g):
        einsums.permute("j,i <- i,j", C, A, c_pf=0.5, a_pf=2.0)

    g.apply(cg.default_pass_manager())
    labels = _node_labels(g)
    assert any(l.startswith("tile_permute") for l in labels), labels
    assert not any(l.startswith("tiled permute") for l in labels), labels

    g.execute()
    assert_close(_gather(C, 9, 5), 0.5 * cref + 2.0 * aref.T)
    g.execute()  # replay: 0.5*(previous) + 2*A^T again
    assert_close(_gather(C, 9, 5), 0.5 * (0.5 * cref + 2.0 * aref.T) + 2.0 * aref.T)


def test_tiled_permute_expansion_no_longer_strands_einsums():
    """The motivating fix: a permute sharing tensors with tiled einsums used to
    poison the whole body out of expansion. Now both expand together and the
    result is exact."""
    rng = np.random.default_rng(31)
    aref = rng.standard_normal((6, 6))
    grid = [[3, 3], [3, 3]]
    A = _make_nd("float64", "A", grid, ref=aref)
    T = einsums.TiledRuntimeTensorD("T", grid)
    C = einsums.TiledRuntimeTensorD("C", grid)

    g = cg.Graph("permute_plus_einsum")
    with cg.capture(g):
        einsums.einsum("ij <- ik ; kj", T, A, A, c_pf=0.0, ab_pf=1.0)  # T = A@A
        einsums.permute("j,i <- i,j", C, T)                            # C = T^T
        einsums.einsum("ij <- ik ; kj", T, C, A, c_pf=0.0, ab_pf=1.0)  # T = C@A

    g.apply(cg.default_pass_manager())
    labels = _node_labels(g)
    assert not any(l.startswith("tiled einsum") or l.startswith("tiled permute") for l in labels), labels

    g.execute()
    assert_close(_gather_nd(T, (6, 6), "float64"), (aref @ aref).T @ aref)


def test_tiled_permute_expansion_sparsity():
    """Expanded form must keep the runtime's sparsity semantics: absent A
    tiles contribute nothing (their targets keep beta*C), untargeted stored C
    tiles are scaled by beta, and zeros stay absent."""
    A = einsums.TiledRuntimeTensorD("A", [[2, 3], [4, 5]])
    A.add_tile([0, 0])
    A.materialize()
    np.asarray(A.tile_view([0, 0]))[...] = 3.0
    C = einsums.TiledRuntimeTensorD("C", [[4, 5], [2, 3]])
    C.add_tile([1, 1])
    C.materialize()
    np.asarray(C.tile_view([1, 1]))[...] = 8.0

    g = cg.Graph("tiled_permute_expand_sparse")
    with cg.capture(g):
        einsums.permute("j,i <- i,j", C, A, c_pf=0.5, a_pf=1.0)
    g.apply(cg.default_pass_manager())
    g.execute()

    np.testing.assert_allclose(np.asarray(C.tile_view([0, 0])), 3.0)
    np.testing.assert_allclose(np.asarray(C.tile_view([1, 1])), 4.0)  # 0.5 * 8
    assert not C.has_tile([0, 1]) and not C.has_tile([1, 0])


def test_tile_view_per_block_contractions():
    """The tiled ladder idiom: capture dense contractions per tile block via
    cg.tile_view, writing disjoint tiles of one tiled output. Sequential and
    Dataflow executors must agree exactly (the tile-coordinate boxes prove the
    per-tile writes disjoint, so the Dataflow replay may run them wide)."""
    rng = np.random.default_rng(21)
    aref = rng.standard_normal((6, 4))
    grid_a = [[3, 3], [4]]
    A = _make_nd("float64", "A", grid_a, ref=aref)
    B = einsums.asarray(np.ascontiguousarray(rng.standard_normal((4, 4))), name="B")
    C = einsums.TiledRuntimeTensorD("C", grid_a)

    g = cg.Graph("tile_view_ladder")
    with cg.capture(g):
        for i in range(2):
            a_i = cg.tile_view(A, [i, 0])
            c_i = cg.tile_view(C, [i, 0])
            einsums.einsum("ij <- ik ; kj", c_i, a_i, B, c_pf=0.0, ab_pf=1.0)

    g.execute()
    ref = aref @ np.asarray(B)
    assert_close(_gather_nd(C, (6, 4), "float64"), ref)

    g.set_executor(cg.DataflowExecutor())
    g.execute()
    assert_close(_gather_nd(C, (6, 4), "float64"), ref)


def test_tile_view_of_deferred_scratch_parent():
    """tile_view over a graph-owned DEFERRED tiled scratch: the tile's storage
    does not exist at capture (dims come from the grid through the deferred
    sentinel), and each replay re-resolves the tile from the live parent."""
    rng = np.random.default_rng(22)
    aref = rng.standard_normal((3, 3))
    A = einsums.asarray(np.ascontiguousarray(aref), name="A")
    out = einsums.zeros((3, 3), dtype="float64")

    g = cg.Graph("tile_view_scratch")
    scr = g.declare_zero_tiled_tensor("scr", [[3], [3]], dtype="float64", intermediate=True)
    body = g.add_loop("it", 2, lambda i: True)
    with cg.capture(body):
        s = cg.tile_view(scr, [0, 0])
        einsums.einsum("ij <- ik ; kj", s, A, A, c_pf=0.0, ab_pf=1.0)  # scratch tile = A@A
        la.axpby(1.0, s, 1.0, out)                                     # accumulate into dense out

    g.apply(cg.default_pass_manager())
    g.execute()
    assert_close(np.asarray(out).copy(), 2.0 * (aref @ aref))
    g.execute()
    assert_close(np.asarray(out).copy(), 4.0 * (aref @ aref))


def test_tile_view_validation():
    A = einsums.TiledRuntimeTensorD("A", [[2, 3], [4]])
    with pytest.raises(RuntimeError):
        cg.tile_view(A, [0, 0])  # outside capture
    g = cg.Graph("tv")
    with cg.capture(g):
        with pytest.raises(ValueError):
            cg.tile_view(A, [0])  # wrong rank
        with pytest.raises(IndexError):
            cg.tile_view(A, [5, 0])  # off the grid


def test_graph_owned_tiled_scratch_in_loop_body():
    """The CCSD loop idiom on tiled operands: scratch declared DEFERRED on the
    graph (no populated tiles until ops create them), overwritten with c_pf=0
    and consumed every replay. Materialization hoists the lifecycle pair to
    the parent; einsum's leftover-scale rule zeroes stale tiles, so reuse
    across replays is exact."""
    rng = np.random.default_rng(11)
    aref = rng.standard_normal((6, 6))
    bref = rng.standard_normal((6, 6))
    grid = [[3, 3], [3, 3]]
    A = _make_nd("float64", "A", grid, ref=aref)
    B = _make_nd("float64", "B", grid, ref=bref)
    acc = _make_nd("float64", "acc", grid, ref=np.zeros((6, 6)))

    g = cg.Graph("tiled_scratch")
    tmp = g.declare_zero_tiled_tensor("tmp", grid, dtype="float64", intermediate=True)
    body = g.add_loop("it", 3, lambda i: True)
    with cg.capture(body):
        einsums.einsum("ij <- ik ; kj", tmp, A, B, c_pf=0.0, ab_pf=1.0)  # overwrite scratch
        einsums.linalg.axpy(1.0, tmp, acc)                               # consume it

    g.apply(cg.default_pass_manager())
    g.execute()
    assert_close(_gather_nd(acc, (6, 6), "float64"), 3.0 * (aref @ bref))

    g.execute()  # replay: hoisted Materialize/Initialize rerun after any Free
    assert_close(_gather_nd(acc, (6, 6), "float64"), 6.0 * (aref @ bref))


def test_tiled_permute_captured():
    aref = (1.0 + np.arange(45)).reshape(5, 9).astype("float64")
    A = _make("float64", "A", [[2, 3], [4, 5]], fill=lambda r, c: aref[r, c])
    C = DTYPE_TO_TRT[np.float64]("C", [[4, 5], [2, 3]])

    g = cg.Graph("tiled_permute")
    with cg.capture(g):
        einsums.permute("j,i <- i,j", C, A)
    g.execute()
    assert_close(_gather(C, 9, 5), aref.T)
    g.execute()  # replay: beta=0 overwrite, same result
    assert_close(_gather(C, 9, 5), aref.T)


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_tiled_einsum_gemm(dtype):
    # C[i,j] = sum_k A[i,k] B[k,j]; contracted k partition {4,5} matches in A and B.
    aref = (1.0 + np.arange(45, dtype=dtype)).reshape(5, 9)
    bref = (2.0 - np.arange(63, dtype=dtype)).reshape(9, 7)
    A = _make(dtype, "A", [[2, 3], [4, 5]], fill=lambda r, c: aref[r, c])
    B = _make(dtype, "B", [[4, 5], [3, 4]], fill=lambda r, c: bref[r, c])
    C = DTYPE_TO_TRT[np.dtype(dtype).type]("C", [[2, 3], [3, 4]])  # empty: infer-and-create

    einsums.einsum("ij <- ik ; kj", C, A, B)

    assert C.num_filled_tiles() == 4
    assert_close(_gather(C, 5, 7), aref @ bref)


# ── Rank-3 / rank-4 validation (the engine is rank-generic; CC contractions
#    are rank-3/4-dominated) ─────────────────────────────────────────────────


def _make_nd(dtype, name, grid, ref=None):
    """Build an N-D tiled tensor over ``grid`` (one tile-size list per axis),
    populate every tile, and optionally fill from a dense reference array."""
    t = DTYPE_TO_TRT[np.dtype(dtype).type](name, grid)
    sizes, offs = t.tile_sizes(), t.tile_offsets()
    counts = [range(len(s)) for s in sizes]
    for coord in itertools.product(*counts):
        t.add_tile(list(coord))
    t.materialize()
    if ref is not None:
        for coord in itertools.product(*counts):
            slc = tuple(slice(offs[ax][coord[ax]], offs[ax][coord[ax]] + sizes[ax][coord[ax]]) for ax in range(len(sizes)))
            np.asarray(t.tile_view(list(coord)))[...] = ref[slc]
    return t


def _gather_nd(t, shape, dtype):
    sizes, offs = t.tile_sizes(), t.tile_offsets()
    M = np.zeros(shape, dtype=dtype)
    for coord in itertools.product(*[range(len(s)) for s in sizes]):
        if t.has_tile(list(coord)):
            slc = tuple(slice(offs[ax][coord[ax]], offs[ax][coord[ax]] + sizes[ax][coord[ax]]) for ax in range(len(sizes)))
            M[slc] = np.asarray(t.tile_view(list(coord)))
    return M


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_tiled_einsum_rank3(dtype):
    # C[i,j,k] = sum_l A[i,j,l] B[l,k]; contracted l partition {3,4} aligns (A axis 2, B axis 0).
    ipart, jpart, lpart, kpart = [2, 3], [2], [3, 4], [2, 3]
    aref = (1.0 + np.arange(5 * 2 * 7, dtype=dtype)).reshape(5, 2, 7)
    bref = (2.0 - np.arange(7 * 5, dtype=dtype)).reshape(7, 5)
    A = _make_nd(dtype, "A", [ipart, jpart, lpart], aref)
    B = _make_nd(dtype, "B", [lpart, kpart], bref)
    C = DTYPE_TO_TRT[np.dtype(dtype).type]("C", [ipart, jpart, kpart])  # empty: infer-and-create

    einsums.einsum("ijk <- ijl ; lk", C, A, B)

    assert_close(_gather_nd(C, (5, 2, 5), dtype), np.einsum("ijl,lk->ijk", aref, bref))


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_tiled_einsum_rank4_two_contractions(dtype):
    # C[i,j,a,b] = sum_{c,d} A[i,j,c,d] B[c,d,a,b]  (CCSD-like; contract c,d).
    ipart, jpart, cpart, dpart, apart, bpart = [2, 1], [2], [2, 1], [3], [1, 2], [2]
    aref = (1.0 + np.arange(3 * 2 * 3 * 3, dtype=dtype)).reshape(3, 2, 3, 3)
    bref = (0.5 - np.arange(3 * 3 * 3 * 2, dtype=dtype)).reshape(3, 3, 3, 2)
    A = _make_nd(dtype, "A", [ipart, jpart, cpart, dpart], aref)
    B = _make_nd(dtype, "B", [cpart, dpart, apart, bpart], bref)
    C = DTYPE_TO_TRT[np.dtype(dtype).type]("C", [ipart, jpart, apart, bpart])

    einsums.einsum("ijab <- ijcd ; cdab", C, A, B)

    assert_close(_gather_nd(C, (3, 2, 3, 2), dtype), np.einsum("ijcd,cdab->ijab", aref, bref))


# ── Scalar reductions (dot / norm / trace) ──────────────────────────────────


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_tiled_dot(dtype):
    aref = (1.0 + np.arange(45, dtype=dtype)).reshape(5, 9)
    bref = (2.0 - np.arange(45, dtype=dtype)).reshape(5, 9)
    A = _make_nd(dtype, "A", [[2, 3], [4, 5]], aref)
    B = _make_nd(dtype, "B", [[2, 3], [4, 5]], bref)
    r = _RT[np.dtype(dtype).type]("r", [1])
    einsums.linalg.dot(r, A, B)
    # cg::dot is non-conjugated (sum A*B), matching np.sum(A*B).
    assert_close(np.asarray(r)[0], np.sum(aref * bref))


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_tiled_norm_frobenius(dtype):
    aref = (1.0 + np.arange(45, dtype=dtype)).reshape(5, 9)
    A = _make_nd(dtype, "A", [[2, 3], [4, 5]], aref)
    r = _REAL_RT[np.dtype(dtype).type]("r", [1])
    einsums.linalg.norm(r, einsums.linalg.Norm.FROBENIUS, A)
    assert_close(np.asarray(r)[0], np.linalg.norm(aref.ravel()))


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_tiled_element_transform(dtype):
    aref = (0.5 + np.arange(45, dtype=dtype)).reshape(5, 9)
    A = _make_nd(dtype, "A", [[2, 3], [4, 5]], aref)
    einsums.linalg.element_transform(A, lambda x: x * x)
    assert_close(_gather_nd(A, (5, 9), dtype), aref * aref)


@pytest.mark.parametrize("dtype", ALL_DTYPES)
def test_tiled_trace(dtype):
    # Square, with matching row/col tile partition so diagonal tiles are square.
    sref = (1.0 + np.arange(25, dtype=dtype)).reshape(5, 5)
    S = _make_nd(dtype, "S", [[2, 3], [2, 3]], sref)
    r = _RT[np.dtype(dtype).type]("r", [1])
    einsums.linalg.trace(r, S)
    assert_close(np.asarray(r)[0], np.trace(sref))


# ── Eigendecomposition: syev / heev (dense + tiled) ─────────────────────────


def _block_diag_full(blocks):
    n = sum(b.shape[0] for b in blocks)
    M = np.zeros((n, n), dtype=blocks[0].dtype)
    o = 0
    for b in blocks:
        s = b.shape[0]
        M[o : o + s, o : o + s] = b
        o += s
    return M


def _make_block_diag(dtype, name, blocks):
    part = [b.shape[0] for b in blocks]
    t = DTYPE_TO_TRT[np.dtype(dtype).type](name, [part, part])
    for i in range(len(part)):
        t.add_tile([i, i])
    t.materialize()
    for i in range(len(part)):
        np.asarray(t.tile_view([i, i]))[...] = blocks[i]
    return t


def _gather_vec(t, n, dtype):
    sz, off = t.tile_sizes()[0], t.tile_offsets()[0]
    v = np.zeros(n, dtype=dtype)
    for i in range(len(sz)):
        if t.has_tile([i]):
            v[off[i] : off[i] + sz[i]] = np.asarray(t.tile_view([i]))
    return v


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_dense_syev(dtype):
    M = np.array([[2, 1, 0], [1, 2, 1], [0, 1, 2]], dtype=dtype)
    A = _RT[np.dtype(dtype).type]("A", [3, 3])
    np.asarray(A)[...] = M
    W = _RT[np.dtype(dtype).type]("W", [3])
    einsums.linalg.syev(A, W)
    assert_close(np.sort(np.asarray(W)), np.linalg.eigvalsh(M))


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_dense_heev(dtype):
    rdtype = np.float32 if dtype == np.complex64 else np.float64
    M = np.array([[2, 1j], [-1j, 2]], dtype=dtype)
    A = _RT[np.dtype(dtype).type]("A", [2, 2])
    np.asarray(A)[...] = M
    W = _REAL_RT[np.dtype(dtype).type]("W", [2])
    einsums.linalg.heev(A, W)
    assert_close(np.sort(np.asarray(W)), np.linalg.eigvalsh(M))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_tiled_syev_block_diagonal(dtype):
    b0 = np.array([[2, 1], [1, 2]], dtype=dtype)
    b1 = np.array([[4, 1, 0], [1, 5, 1], [0, 1, 6]], dtype=dtype)
    A = _make_block_diag(dtype, "A", [b0, b1])
    W = DTYPE_TO_TRT[np.dtype(dtype).type]("W", [[2, 3]])
    einsums.linalg.syev(A, W)
    # Per-block eigenvalues, gathered, equal the full block-diagonal spectrum.
    assert_close(np.sort(_gather_vec(W, 5, dtype)), np.linalg.eigvalsh(_block_diag_full([b0, b1])))


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
def test_tiled_heev_block_diagonal(dtype):
    rdtype = np.float32 if dtype == np.complex64 else np.float64
    b0 = np.array([[2, 1j], [-1j, 2]], dtype=dtype)
    b1 = np.array([[3, 0, 1j], [0, 4, 0], [-1j, 0, 5]], dtype=dtype)
    A = _make_block_diag(dtype, "A", [b0, b1])
    W = DTYPE_TO_TRT[rdtype]("W", [[2, 3]])  # eigenvalues are real -> real tiled tensor
    einsums.linalg.heev(A, W)
    assert_close(np.sort(_gather_vec(W, 5, rdtype)), np.linalg.eigvalsh(_block_diag_full([b0, b1])))


# ── Pipeline compatibility ────────────────────────────────────────────────
# A tiled tensor is a tile container, not a dense buffer, so no buffer-level
# pass may treat it as one. Three independent things keep that true:
#
#   1. a tiled einsum records as OpKind::Custom, so every pass that filters on
#      OpKind::Einsum (LCCF, GEMMBatching, StreamContractionFusion,
#      DistributionPlanning) skips it structurally;
#   2. TiledRuntimeTensor does not derive from RuntimeTensorNoType, so the
#      is_runtime gate that guards GeneralRuntimeTensor casts rejects it;
#   3. it exposes no rank-erased impl, so TensorHandle::impl_fn is null and
#      Graph::make_einsum_node refuses it outright.
#
# The default pipeline now LOWERS tiled ops instead of skipping them:
# TiledExpansion replaces each with one dense node per tile, and those tiles are
# ordinary dense RuntimeTensors, so the passes above act on them legitimately.
# The three guards still matter -- they are what keeps a WHOLE tiled operand out
# of those passes, including on every path expansion declines.
def test_tiled_operands_are_lowered_by_the_default_pipeline():
    import json

    aref = (1.0 + np.arange(45, dtype=np.float64)).reshape(5, 9)
    bref = (2.0 - np.arange(63, dtype=np.float64)).reshape(9, 7)
    A = _make(np.float64, "A", [[2, 3], [4, 5]], fill=lambda r, c: aref[r, c])
    B = _make(np.float64, "B", [[4, 5], [3, 4]], fill=lambda r, c: bref[r, c])
    C = _make(np.float64, "C", [[2, 3], [3, 4]])

    import einsums.graph as cg

    # Two accumulating contractions into one output reading the same operands:
    # exactly the shape LinearCombinationContractionFolding hunts for. On dense
    # operands it would fold them; on tiled ones it must decline.
    g = cg.Graph("tiled_pipeline")
    with cg.capture(g):
        einsums.einsum("ij <- ik ; kj", C, A, B, c_pf=0.0, ab_pf=2.0)
        einsums.einsum("ij <- ik ; kj", C, A, B, c_pf=1.0, ab_pf=-1.0)

    kinds_before = [n["kind"] for n in json.loads(g.to_json())["nodes"]]
    assert kinds_before == ["Custom", "Custom"], kinds_before

    lccf = cg.LinearCombinationContractionFolding()
    pm = cg.PassManager()
    pm.add(lccf)
    g.apply(pm)
    assert lccf.num_groups == 0, "LCCF folded tiled operands"

    # The default pipeline lowers both contractions to dense nodes. Nothing opaque
    # may survive, and no whole-tiled operand may reach a buffer-level pass -- which
    # is exactly what expanding first guarantees.
    #
    # Which dense form it picks is the cost model's call. These tiles are tiny, so it
    # densifies: gather + one einsum + scatter per contraction rather than one einsum
    # per tile pair. Assert the invariant that matters -- everything lowered --
    # rather than a node count that encodes one of the two lowerings.
    g.apply(cg.default_pass_manager())
    kinds_after = [n["kind"] for n in json.loads(g.to_json())["nodes"]]
    assert "Custom" not in kinds_after, kinds_after
    assert kinds_after.count("Einsum") >= len(kinds_before), kinds_after
    assert "TileGather" in kinds_after and "TileScatter" in kinds_after, kinds_after

    g.execute()
    # 2*AB - AB == AB. The prefactors have to survive the lowering: the first
    # contraction overwrites (c_pf=0) and the second accumulates (c_pf=1), and
    # each applies exactly once per tile no matter how many k-tiles contribute.
    assert_close(_gather(C, 5, 7), aref @ bref)


def test_tiled_operands_are_never_gpu_placed():
    """A tiled tensor must never be handed to GPUPlacement.

    It has no single contiguous buffer -- data_ptr is null, storage is one dense
    tensor per tile -- so the H2D/D2H nodes placement inserts have nothing to
    move, and TransferInsertion would emit a copy from a null pointer.

    This was a live bug, not a hypothetical: `dot` is in is_gpu_capable_op, and
    TensorHandle::total_bytes reports the honest GLOBAL size regardless of tile
    sparsity, so a tiled dot over min_bytes (65536) was a valid candidate and did
    get placed. It went unnoticed because gpu::has_unified_memory makes the
    transfers no-ops on Apple Silicon and on the mock backend; a discrete
    CUDA/HIP build would memcpy from nullptr.

    float32 on purpose: the MPS backend's backend_supports_dtype accepts Float32
    ONLY, so a float64 graph is rejected at the dtype gate and would pass this
    test without ever exercising placement.
    """
    import json

    import einsums.graph as cg

    # 128x128 float32 = 65536 bytes per operand, at/over GPUPlacement's min_bytes.
    tile = 64
    grid = [[tile, tile], [tile, tile]]
    A = _make(np.float32, "A_gpu", grid, fill=lambda r, c: 1.0)
    B = _make(np.float32, "B_gpu", grid, fill=lambda r, c: 2.0)
    out = einsums.create_zero_tensor("out_gpu", [1], dtype="float32")

    g = cg.Graph("tiled_dot_gpu")
    with cg.capture(g):
        einsums.linalg.dot(out, A, B)

    g.apply(cg.default_pass_manager())

    nodes = json.loads(g.to_json())["nodes"]
    kinds = [n["kind"] for n in nodes]
    assert not [k for k in kinds if "HostToDevice" in k or "DeviceToHost" in k], kinds
    assert all(n.get("target", "CPU") == "CPU" for n in nodes), nodes

    g.execute()
    # 128*128 elements of 1.0 * 2.0
    assert_close(np.asarray(out), np.array([2.0 * 128 * 128], dtype=np.float32))
