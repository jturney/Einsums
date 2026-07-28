#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

"""The tiled parity CONTRACT: which graph ops must work on TiledRuntimeTensor.

Tiled tensors are the structured front end for chemistry workflows (a psi4
Matrix is irrep-blocked by construction), and generic code - cg.diis is the
canonical example - is only writable once when the op surface is uniform
across dense and tiled. This test makes that uniformity a CONTRACT instead of
an aspiration: every op in WORKFLOW_OPS must expose TiledRuntimeTensor
overloads, and every op in DENSE_ONLY_OPS is a documented exclusion that must
STAY dense-only until it is deliberately promoted (at which point this test
fails and the op moves lists - the lists can never silently rot).

Design context: parity is achieved by giving ops tiled overloads at the
capture layer while TiledExpansion lowers the hot kinds (einsum, scale, axpy,
direct_division - and axpby, which decomposes into scale+axpy) onto per-tile
DENSE nodes ahead of the optimizer pipeline. Dense remains the canonical IR
and storage substrate; parity here means the USER-FACING surface, not a
parallel implementation of every pass.

Known-open items tracked toward full workflow parity, not yet asserted here:
  * graph-owned tiled scratch (declare_zero_tensor / Materialization /
    FreeInsertion / the MemoryPlanning arena are dense-only, and
    InplaceOptimization gates on !is_tiled),
  * tiled views registered in capture (cg.view_indexed / permute_view and the
    disjointness-aware scheduling are dense-only; eager IndexSpace views are
    not graph-registered),
  * GPU placement and the distribution passes skip tiled-touching nodes.
"""

import pytest

import einsums
import einsums.graph as cg  # noqa: F401  (imported for parity with real usage)

#: Ops the workflow set REQUIRES on tiled operands. Additions here should come
#: with real tests in test_tiled_ops_python.py, not just an overload.
WORKFLOW_OPS = [
    "einsum",
    "permute",
    "scale",
    "axpy",
    "axpby",
    "direct_division",
    "dot",
    "dotc",
    "norm",
    "trace",
    "element_transform",
    "conj",
    "real",
    "imag",
    "abs",
    "syev",
    "heev",
]

#: Documented dense-only exclusions. gemm-shaped contractions are covered by
#: tiled einsum; block_copy by axpby(1, X, 0, Y); the LAPACK factorizations
#: and the remaining reductions follow demand rather than completeness.
DENSE_ONLY_OPS = [
    "block_copy",
    "det",
    "direct_product",
    "gemm",
    "gemv",
    "ger",
    "gerc",
    "gesv",
    "invert",
    "max",
    "outer_sum",
    "pow",
    "qr",
    "shift",
    "sum",
    "svd",
    "svd_dd",
    "truncated_svd",
    "truncated_syev",
]


def _op(name):
    """Resolve an op callable: top-level einsums first, then einsums.linalg."""
    obj = getattr(einsums, name, None)
    if obj is None or not callable(obj):
        obj = getattr(einsums.linalg, name, None)
    if obj is None or not callable(obj):
        pytest.fail(f"op {name!r} not found on einsums or einsums.linalg - if it was renamed or removed, update the contract lists")
    return obj


def _has_tiled_overload(op):
    """pybind11 lists every overload signature in __doc__; the tiled overloads
    name the registered TiledRuntimeTensor{F,D,C,Z} classes there."""
    return "TiledRuntimeTensor" in (op.__doc__ or "")


@pytest.mark.parametrize("name", WORKFLOW_OPS)
def test_workflow_op_has_tiled_overloads(name):
    assert _has_tiled_overload(_op(name)), (
        f"{name!r} is in the tiled parity contract but exposes no TiledRuntimeTensor overloads. "
        f"Either add them (with tests in test_tiled_ops_python.py) or - deliberately, with rationale - move it to DENSE_ONLY_OPS."
    )


@pytest.mark.parametrize("name", DENSE_ONLY_OPS)
def test_dense_only_op_is_still_dense_only(name):
    assert not _has_tiled_overload(_op(name)), (
        f"{name!r} gained TiledRuntimeTensor overloads but is still listed as a documented dense-only exclusion. "
        f"Move it into WORKFLOW_OPS so the contract keeps covering it."
    )


def test_contract_lists_are_disjoint():
    overlap = set(WORKFLOW_OPS) & set(DENSE_ONLY_OPS)
    assert not overlap, f"ops cannot be in both lists: {sorted(overlap)}"


def test_zeros_like_is_structure_preserving():
    """Not an overload, but part of the contract: cloning a tiled tensor must
    keep its grid and populated-tile set (details in test_tiled_ops_python)."""
    t = einsums.TiledRuntimeTensorD("t", [[2, 3], [4]])
    t.add_tile([1, 0])
    t.materialize()
    z = einsums.zeros_like(t)
    assert type(z) is type(t)
    assert z.tile_sizes() == t.tile_sizes()
    assert z.has_tile([1, 0]) and not z.has_tile([0, 0])
